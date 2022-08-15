import numpy as np
import h5py
import numpy.ma as ma
from collections import defaultdict
import logging
from math import ceil

from h5flow.core import H5FlowGenerator, resources
from h5flow import H5FLOW_MPI

from module0_flow.reco.light.raw_event_generator import LightEventGenerator
import module0_flow.util.units as units


class LightEventGeneratorMC(H5FlowGenerator):
    '''
        Light system event builder *for simulation only*. Converts the light
        waveforms, triggers, and truth info into the same format as generated
        by the ``LightEventGenerator``

        Parameters:
         - ``wvfm_dset_name`` : ``str``, required, path to dataset to store raw waveforms
         - ``n_adcs`` : ``int``, number of ADC serial numbers
         - ``n_channels`` : ``int``, number of channels per ADC
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
                    n_adcs: 2
                    n_channels: 64
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
        mc_truth_dset_name='mc_truth/light',
        n_adcs=2,
        n_channels=64,
        busy_delay=123,
        busy_ampl=20e3,
        chunk_size=32
        )


    def __init__(self, **params):
        super(LightEventGeneratorMC, self).__init__(**params)

        # set up parameters
        for key,val in self.defaults.items():
            setattr(self, key, params.get(key, val))

        self.adc_sn = np.array(params['adc_sn'])
        self.channel_map = np.array(params['channel_map'])
        self.busy_channel = np.array(params.get('busy_channel',[0]*self.n_adcs))
        self.disabled_channels = np.array(params.get('disabled_channels',[]))
        self.event_dset_name = self.dset_name
        self.n_samples = 0

        # set up input file
        self.input_file = h5py.File(self.input_filename,'r')
        self.light_dat = self.input_file['light_dat']
        self.light_trig = self.input_file['light_trig']
        self.light_wvfms = self.input_file['light_wvfm']
        self.end_position = self.light_trig.shape[0] if self.end_position is None else self.end_position
        self.start_position = 0 if self.start_position is None else self.start_position
        self.slices = [slice(
                min(st, self.light_trig.shape[0]),
                min(st + self.chunk_size, self.light_trig.shape[0])
                ) for st in range(self.start_position + self.rank * self.chunk_size,
                    self.end_position, self.size * self.chunk_size)]
        self.slices.append(H5FlowGenerator.EMPTY)
        self.iteration = 0

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

        if not self.n_samples:
            self.n_samples = self.light_wvfms.shape[-1]

        # fix dataset dtypes
        self.event_dtype = LightEventGenerator.event_dtype(self)
        self.wvfm_dtype = LightEventGenerator.wvfm_dtype(self)

        # initialize data objects
        self.data_manager.create_dset(self.event_dset_name, dtype=self.event_dtype)
        self.data_manager.create_dset(self.wvfm_dset_name, dtype=self.wvfm_dtype)
        self.data_manager.create_ref(self.event_dset_name, self.wvfm_dset_name)
        self.data_manager.set_attrs(self.event_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    n_adcs=self.n_adcs,
                                    n_channels=self.n_channels,
                                    n_samples=self.n_samples,
                                    chunk_size=self.chunk_size,
                                    busy_delay=self.busy_delay,
                                    adc_sn=self.adc_sn,
                                    channel_map=self.channel_map,
                                    busy_channel=self.busy_channel,
                                    disabled_channels=self.disabled_channels,
                                    wvfm_dset_name=self.wvfm_dset_name,
                                    start_position=self.start_position,
                                    end_position=self.end_position,
                                    input_filename=self.input_filename,
                                    mc_truth_dset_name=self.mc_truth_dset_name
                                    )

        # copy and remap the truth information
        self.data_manager.create_dset(self.mc_truth_dset_name, dtype=self.light_dat.dtype, shape=self.channel_map.shape)
        truth_len = ceil(self.light_dat.shape[0] // self.size)
        truth_slice = slice(self.rank * truth_len, (self.rank+1) * truth_len)
        remapped_light_dat = self._remap_array(self.channel_map, self.light_dat[truth_slice], axis=-1)
        self.data_manager.reserve_data(self.mc_truth_dset_name, truth_slice)
        self.data_manager.write_data(self.mc_truth_dset_name, truth_slice, remapped_light_dat)

        mc_channel = np.indices(self.light_dat[0:1].shape)[1]
        

    def finish(self):
        super(LightEventGeneratorMC, self).finish()
        self.input_file.close()


    def next(self):
        next_sl = self.slices[self.iteration]

        # get next trigger
        next_trig = self.light_trig[next_sl]
        next_wvfms = self.light_wvfms[next_sl]

        # convert channel map
        remapped_wvfms = self._remap_array(self.channel_map, -next_wvfms, axis=-2)
        remapped_wvfms = remapped_wvfms * (self.channel_map != -1)[np.newaxis,:,:,np.newaxis]

        # mock busy signal
        remapped_wvfms[:, np.r_[range(self.n_adcs)], self.busy_channel, self.busy_delay:] = self.busy_ampl

        # zero out disabled channels
        remapped_wvfms[:, self.disabled_channels[...,0], self.disabled_channels[...,1]] = 0.

        # clip to ensure within datatype bounds
        remapped_wvfms = remapped_wvfms.clip(np.iinfo(self.wvfm_dtype['samples'].base).min, np.iinfo(self.wvfm_dtype['samples'].base).max)

        # write event to file
        event_slice = self.data_manager.reserve_data(self.event_dset_name, next_trig.shape[0])
        event_arr = np.empty(next_trig.shape[0], self.event_dtype)
        if next_trig.shape[0]:
            event_arr['id'] = np.arange(event_slice.start, event_slice.stop)
            event_arr['event'] = np.arange(next_sl.start, next_sl.stop)
            event_arr['sn'] = self.adc_sn.reshape(1,-1)
            event_arr['ch'] = np.arange(self.n_channels).reshape(1,1,-1)
            event_arr['utime_ms'] = next_trig['ts_s'].reshape(-1,1,1) * units.s / units.ms
            event_arr['tai_ns'] = (next_trig['ts_sync'].reshape(-1,1,1) * resources['RunData'].crs_ticks + np.fmod(next_trig['ts_s'].reshape(-1,1,1) * units.s, resources['RunData'].crs_ticks)) / units.ns
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

