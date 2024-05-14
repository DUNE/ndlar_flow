import numpy as np
import h5py
import numpy.ma as ma
from collections import defaultdict
import logging
from math import ceil

from numba import cuda

from h5flow.core import H5FlowGenerator, resources
from h5flow import H5FLOW_MPI

from proto_nd_flow.reco.light.raw_event_generator import LightEventGenerator
import proto_nd_flow.util.units as units


class LightEventGeneratorMC(H5FlowGenerator):
    '''
        Light system event builder *for simulation only*. Converts the light
        waveforms, triggers, and truth info into the same format as generated
        by the ``LightEventGenerator``

        Parameters:
         - ``wvfm_dset_name`` : ``str``, required, path to dataset to store raw waveforms
         - ``n_modules`` : ``int``, number of modules
         - ``n_adcs`` : ``int``, number of ADC serial numbers
         - ``n_channels`` : ``int``, number of channels per ADC
         - ``n_sipms_per_module`` : ``int``, total number of SiPM channels per module
         - ``adc_sn`` : ``list`` of ``int``, serial number of each ADC
         - ``channel_map``: ``list`` of ``list`` of ``int``, mapping from simulation optical detector to adc, channel, ``-1`` values indicate channel is not connected
         - ``busy_channel``: ``list`` of ``int``, channel used for busy signal on each ADC (if relevant)
         - ``busy_delay``: ``int``, number of ticks prior to busy signal for each trigger
         - ``disabled_channels``: ``list`` of ``(adc_idx, channel_idx)``, channels to zero out (optional)

        Requires RunData resource in workflow.

        Example config::

            flow:
                source: light_event_generator_mc
                stages: []

            light_event_generator_mc:
                classname: LightEventGeneratorMC
                dset_name: 'light/events'
                params:
                    wvfm_dset_name: 'light/wvfm'
                    n_modules: 4
                    n_adcs: 8
                    n_channels: 64
                    n_sipms_per_module: 96
                    adc_sn:
                     - 0
                     - 1
                    channel_map:
                     - [0,1,2,3,4,5,6,7,8,9,
                        10,11,12,13,14,15,16,17,18,19,20,
                        21,22,23,24,25,26,27,28,29,30,
                        31,32,33,34,35,36,37,38,39,40,
                        41,42,43,44,45,46,47,48,49,50,
                        51,52,53,54,55,56,57,58,59,60,
                        61,62,63]
                     - [0,1,2,3,4,5,6,7,8,9,
                        10,11,12,13,14,15,16,17,18,19,20,
                        21,22,23,24,25,26,27,28,29,30,
                        31,32,33,34,35,36,37,38,39,40,
                        41,42,43,44,45,46,47,48,49,50,
                        51,52,53,54,55,56,57,58,59,60,
                        61,62,63]
                    busy_channel:
                     - 0
                     - 0
                    busy_delay: 120
                    busy_ampl: 10000


        See ``LightEventGenerator`` for ``events`` and ``wvfm`` datatypes.

    '''
    defaults = dict(
        wvfm_dset_name='light/wvfm',
        true_wvfm_dset_name='light/true_wvfm',
        mc_truth_dset_name='mc_truth/light',
        n_modules=4,
        n_adcs=2,
        n_channels=64,
        n_sipms_per_module= 96,
        n_sipms_total= 384,
        busy_delay=123,
        busy_ampl=20e3,
        chunk_size=32,
        max_tracks=10
        )

    def true_wvfm_dtype(self): return np.dtype([
        ('trigger_id', 'i4'),
        ('samples', np.float32, (self.n_sipms_total, self.max_tracks, self.n_samples))
    ])

    def __init__(self, **params):
        super(LightEventGeneratorMC, self).__init__(**params)

        # set up parameters
        for key,val in self.defaults.items():
            setattr(self, key, params.get(key, val))

        self.adc_sn = np.array(params['adc_sn'])
        self.channel_map = np.array(params['channel_map'])
        # self.busy_channel = np.array(params.get('busy_channel',[0]*self.n_adcs))
        self.disabled_channels = np.array(params.get('disabled_channels',[]))
        self.event_dset_name = self.dset_name
        self.n_samples = 0

        # set up input file
        self.input_file = h5py.File(self.input_filename,'r')
        self.light_dat = self.input_file['light_dat']
        self.light_trig = self.input_file['light_trig']
        self.light_wvfms = self.input_file['light_wvfm']
        self.true_light_wvfms = self.input_file['light_wvfm_mc_assn']
      
        #self.segID = self.true_light_wvfms['segment_id']

        self.end_position = self.light_trig.shape[0] if self.end_position is None else self.end_position
        self.start_position = 0 if self.start_position is None else self.start_position
        self.slices = [slice(
                min(st, self.light_trig.shape[0]),
                min(st + self.chunk_size, self.light_trig.shape[0])
                ) for st in range(self.start_position + self.rank * self.chunk_size,
                    self.end_position, self.size * self.chunk_size)]
        self.slices.append(H5FlowGenerator.EMPTY)
        self.iteration = 0
        print("self.slices", self.slices)
        """
        #print(np.unique(self.true_light_wvfms['segment_id']).shape[0])
        #n_truth_seg = np.unique(self.true_light_wvfms['segment_id']).shape[0]
        #self.abc = np.unique(self.true_light_wvfms['segment_id']).shape[0] if self.abc is None else self.abc
        self.start_position_true = 0
        self.end_position_true = len(np.unique(self.true_light_wvfms['trigger_id']))
        self.slices_true = [slice(
                min(st, self.end_position_true),
                min(st + self.chunk_size, self.end_position_true)
                ) for st in range(self.start_position_true + self.rank * self.chunk_size,
                    self.end_position_true, self.size * self.chunk_size)]
        print("self.slices_true", self.slices_true)
        """

    def __len__(self):
        return len(self.slices)

    @staticmethod
    def _remap_array(channel_map, arr, axis=0):
        '''
            Remap an array of shape (..., Ni, ...) to (..., Nj, Nk, ...) using
            an array of indices

            :param channel_map: 2D array of indices into ``Ni`` to remap, shape: ``(Nj, Nk)``

            :param arr: ND array to remap, shape ``(..., Ni, ...)``
        '''
        if axis < 0:
            axis = arr.ndim + axis
        new_shape = tuple(np.r_[arr.shape[:axis], channel_map.shape, arr.shape[axis+1:]].astype(int))
        new_arr = np.zeros(new_shape, dtype=arr.dtype)
        for i in range(channel_map.shape[0]):
            np.copyto(new_arr,
                np.expand_dims(
                    np.take(arr, channel_map[i], axis=axis),
                    axis=axis),
                where=(np.indices(new_arr.shape)[axis] == i))
        return new_arr



    def init(self):
        super(LightEventGeneratorMC, self).init()

        if self.data_manager.dset_exists(self.event_dset_name):
            raise RuntimeError(f'{self.event_dset_name} already exists, refusing to append!')
        if self.data_manager.dset_exists(self.wvfm_dset_name):
            raise RuntimeError(f'{self.wvfm_dset_name} already exists, refusing to append!')
        if self.data_manager.dset_exists(self.true_wvfm_dset_name):
            raise RuntimeError(f'{self.true_wvfm_dset_name} already exists, refusing to append!')

        if not self.n_samples:
            self.n_samples = self.light_wvfms.shape[-1]

        # fix dataset dtypes
        self.event_dtype = LightEventGenerator.event_dtype(self)
        self.wvfm_dtype = LightEventGenerator.wvfm_dtype(self)
        self.true_wvfm_dtype = LightEventGeneratorMC.true_wvfm_dtype(self)

        # initialize data objects
        self.data_manager.create_dset(self.event_dset_name, dtype=self.event_dtype)
        self.data_manager.create_dset(self.wvfm_dset_name, dtype=self.wvfm_dtype)
        self.data_manager.create_dset(self.true_wvfm_dset_name, dtype=self.true_wvfm_dtype)
        print("self.wvfm_dtype", self.wvfm_dtype)
        print("self.true_wvfm_dtype", self.true_wvfm_dtype)
        self.data_manager.create_ref(self.event_dset_name, self.wvfm_dset_name)
        #self.data_manager.create_ref(self.event_dset_name, self.true_wvfm_dset_name)
        self.data_manager.set_attrs(self.event_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    n_modules=self.n_modules,
                                    n_adcs=self.n_adcs,
                                    n_channels=self.n_channels,
                                    n_sipms_per_module=self.n_sipms_per_module,
                                    n_samples=self.n_samples,
                                    chunk_size=self.chunk_size,
                                    busy_delay=self.busy_delay,
                                    adc_sn=self.adc_sn,
                                    channel_map=self.channel_map,
                                    #busy_channel=self.busy_channel,
                                    disabled_channels=self.disabled_channels,
                                    wvfm_dset_name=self.wvfm_dset_name,
                                    start_position=self.start_position,
                                    end_position=self.end_position,
                                    input_filename=self.input_filename,
                                    mc_truth_dset_name=self.mc_truth_dset_name
                                    )

        # copy and remap the truth information
        if type(self.light_dat) is h5py.Group: # We have four 96-column matrices
            if 'light_dat_allmodules' in self.light_dat:
                light_dat = self.light_dat['light_dat_allmodules']
            else:
                light_dat = self._bloat_light_dat(self.light_dat) # Now have 384 col
        else:                   # We have the old 384-column matrix
            light_dat = self.light_dat

        self.data_manager.create_dset(self.mc_truth_dset_name, dtype=light_dat.dtype, shape=self.channel_map.shape)
        truth_len = ceil(light_dat.shape[0] // self.size)
        truth_slice = slice(self.rank * truth_len, (self.rank+1) * truth_len)

        print("truth_len ", truth_len )
        print("truth_slice", truth_slice)

        print("light_dat[truth_slice]", light_dat[truth_slice].shape)
        remapped_light_dat = self._remap_array(self.channel_map, light_dat[truth_slice], axis=-1)
        print("remapped_light_dat", remapped_light_dat.shape)
        self.data_manager.reserve_data(self.mc_truth_dset_name, truth_slice)
        self.data_manager.write_data(self.mc_truth_dset_name, truth_slice, remapped_light_dat)

        mc_channel = np.indices(light_dat[0:1].shape)[1]



        #find truth waveforms
        
        trig_ids = np.unique(self.true_light_wvfms['trigger_id'])
        true_wvfm_arr = np.empty(len(trig_ids), self.true_wvfm_dtype)
        samples = true_wvfm_arr['samples']
        """
        @cuda.jit
        def fill_samples_kernel(samples):
            itrig,ichan,iseg = cuda.grid(3)

        threads_per_block = (4, 4, 4)
        blocks_per_grid_x = (len(trig_ids) + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (self.n_sipms_total + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid_z = (self.max_tracks + threads_per_block[2] - 1) // threads_per_block[2]

        fill_samples_kernel[(blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z), threads_per_block](samples)
        """



        max_runs = len(trig_ids) * self.n_sipms_total * self.max_tracks
        print("max_runs", max_runs)

        for itrig in range(len(trig_ids)):
        #for itrig in range(1):
            trig_id = trig_ids[itrig]
            true_wvfm_arr['trigger_id'][itrig] = trig_id
            op_chans = np.unique(self.true_light_wvfms[self.true_light_wvfms['trigger_id']==trig_id]['op_channel_id'])

            for ichan in range(len(op_chans)):
                chan_id = op_chans[ichan]
                seg_ids = np.unique(self.true_light_wvfms[(self.true_light_wvfms['trigger_id']==trig_id) & (self.true_light_wvfms['op_channel_id']==chan_id)]['segment_id'])
                if len(seg_ids)>10: print(trig_id, chan_id, len(seg_ids))   
                
                for iseg in range(len(seg_ids)):
                    if itrig*ichan*iseg%1000==0: print("Processing ", (itrig*ichan*iseg*100./max_runs), "%... ")

                    seg_id = seg_ids[iseg]
                    twvfm_arr = np.empty(self.n_samples)

                    tick = self.true_light_wvfms[(self.true_light_wvfms['trigger_id']==trig_id) & (self.true_light_wvfms['op_channel_id']==chan_id) & (self.true_light_wvfms['segment_id']==seg_id)]['tick'] 
                    pe_current = self.true_light_wvfms[(self.true_light_wvfms['trigger_id']==trig_id) & (self.true_light_wvfms['op_channel_id']==chan_id) & (self.true_light_wvfms['segment_id']==seg_id)]['pe_current'] 
                    #padding the truth waveform
                    for i in range(self.n_samples):
                        if i in tick:
                            twvfm_arr[i] += pe_current[list(tick).index(i)]
                        else:
                            twvfm_arr[i] = 0.

                    true_wvfm_arr['samples'][itrig,ichan,iseg] = twvfm_arr
        itrig_slice = slice(self.rank * len(trig_ids), (self.rank+1) * len(trig_ids))
        self.data_manager.reserve_data(self.true_wvfm_dset_name, itrig_slice)
        print(itrig_slice)
        self.data_manager.write_data(self.true_wvfm_dset_name, itrig_slice, true_wvfm_arr) 



    @staticmethod
    def _bloat_light_dat(light_dat_group: h5py.Group) -> np.array:
        """ HACK
        Merge the four 96-channel matrices into a 384-channel matrix like the
        one that larnd-sim formerly produced. Avoids modifying downstream code. """
        def blocks(i):
            """Return the `light_dat` for module `i` and a clone with
            `n_photons_det` set to 0."""
            dat = light_dat_group[f'light_dat_module{i}']
            nulls = np.array(dat)
            nulls['n_photons_det'] = 0
            return dat, nulls
        dat0, nulls0 = blocks(0)
        dat1, nulls1 = blocks(1)
        dat2, nulls2 = blocks(2)
        dat3, nulls3 = blocks(3)
        light_dat_wide0 = np.hstack([dat0, nulls0, nulls0, nulls0])
        light_dat_wide1 = np.hstack([nulls1, dat1, nulls1, nulls1])
        light_dat_wide2 = np.hstack([nulls2, nulls2, dat2, nulls2])
        light_dat_wide3 = np.hstack([nulls3, nulls3, nulls3, dat3])
        return np.vstack([light_dat_wide0, light_dat_wide1, light_dat_wide2, light_dat_wide3])

    def finish(self):
        super(LightEventGeneratorMC, self).finish()
        self.input_file.close()



    def next(self):

        next_sl = self.slices[self.iteration]

        # get next trigger
        next_trig = self.light_trig[next_sl]
        next_wvfms = self.light_wvfms[next_sl]
        print('next_trig.shape[0]', next_trig.shape[0])

        # get module start channel
        next_startch = [tr_ev[0][0] for tr_ev in next_trig]
        
        # convert channel map
        tmp_wvfms = np.zeros((next_wvfms.shape[0], self.n_modules*self.n_sipms_per_module, next_wvfms.shape[-1]),dtype=next_wvfms.dtype)
        for ev in range(next_wvfms.shape[0]):
            tmp_wvfms[ev,next_startch[ev]:(next_startch[ev]+next_wvfms[ev].shape[0]),:] = next_wvfms[ev]
        next_wvfms=tmp_wvfms
        del tmp_wvfms
        remapped_wvfms = self._remap_array(self.channel_map, -next_wvfms, axis=-2)
        print("remapped_wvfms", remapped_wvfms.shape)
        print("remapped_wvfms", remapped_wvfms.shape[0])
        remapped_wvfms = remapped_wvfms * (self.channel_map != -1)[np.newaxis,:,:,np.newaxis]
        print("remapped_wvfms", remapped_wvfms.shape)
        print("remapped_wvfms", remapped_wvfms.shape[0])

        # mock busy signal
        # remapped_wvfms[:, np.r_[range(self.n_adcs)], self.busy_channel, self.busy_delay:] = self.busy_ampl

        # zero out disabled channels
        remapped_wvfms[:, self.disabled_channels[...,0], self.disabled_channels[...,1]] = 0.

        # clip to ensure within datatype bounds
        remapped_wvfms = remapped_wvfms.clip(np.iinfo(self.wvfm_dtype['samples'].base).min, np.iinfo(self.wvfm_dtype['samples'].base).max)

        # write event to file
        event_slice = self.data_manager.reserve_data(self.event_dset_name, next_trig.shape[0])
        print("event_slice", event_slice)
        event_arr = np.empty(next_trig.shape[0], self.event_dtype)
        if next_trig.shape[0]:
            event_arr['id'] = np.arange(event_slice.start, event_slice.stop)
            event_arr['event'] = np.arange(next_sl.start, next_sl.stop)
            event_arr['sn'] = self.adc_sn.reshape(1,-1)
            #event_arr['ch'] = np.arange(self.n_channels).reshape(1,1,-1)
            event_arr['utime_ms'] = next_trig['ts_s'].reshape(-1,1) * units.s / units.ms
            event_arr['tai_ns'] = (next_trig['ts_sync'].reshape(-1,1) * resources['RunData'].crs_ticks + np.fmod(next_trig['ts_s'].reshape(-1,1) * units.s, resources['RunData'].crs_ticks)) / units.ns
            event_arr['wvfm_valid'] = (self.channel_map != -1)[np.newaxis,:,:]
        self.data_manager.write_data(self.event_dset_name, event_slice, event_arr)

        self.data_manager.reserve_data(self.wvfm_dset_name, event_slice)
        wvfm_arr = np.empty(next_trig.shape[0], self.wvfm_dtype)
        if next_trig.shape[0]:
            wvfm_arr['samples'] = remapped_wvfms
        self.data_manager.write_data(self.wvfm_dset_name, event_slice, wvfm_arr)

        # set up references
        #   just event -> wvfm 1:1 refs for now
        ref = np.c_[event_arr['id'], event_arr['id']]
        self.data_manager.write_ref(self.event_dset_name, self.wvfm_dset_name, ref)

        if next_trig.shape[0] == 0:
            return H5FlowGenerator.EMPTY
        self.iteration += 1
        return event_slice
        
