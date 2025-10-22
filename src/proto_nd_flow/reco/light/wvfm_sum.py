import numpy as np
from collections import defaultdict

from h5flow.core import H5FlowStage, resources


class WaveformSum(H5FlowStage):
    '''
        Sums the signal across light detector SiPM channels.

        Parameters:
         - ``wvfm_dset_name`` : ``str``, required, input dataset path
         - ``swvfm_dset_name`` : ``str``, required, output det sum channels dataset path
         - ``stpc_wvfm_dset_name`` : ``str``, required, output tpc sum channels dataset path
         - ``schan_wvfm_dset_name``: ``str``, required, output sum channel dataset path

        ``wvfm_dset_name`` is required in the data cache.

        The Geometry resource is required in the workflow.

        Example config::

            wvfm_sum:
                classname: WaveformSum
                requires:
                    - 'light/events'
                    - 'light/deconv'
                params:
                    wvfm_dset_name: 'light/deconv'
                    swvfm_dset_name: 'light/swvfm'
                    stpc_wvfm_dset_name: 'light/stpc_wvfm'
                    schan_wvfm_dset_name: 'light/schan_wvfm'


        Uses the same dtype as the input waveform dataset(s) except with
        ``(nadc, nchannel)`` resized to be ``(ntpc, ndet)``.

    '''
    class_version = '1.0.0'

    default_detector_channels = [list(range(64))]
    default_make_schan_wvfm_dset = False
    default_make_stpc_wvfm_dset = True
    default_make_swvfm_dset = True
    default_wvfm_dset_name = 'light/wvfm'
    default_swvfm_dset_name = 'light/swvfm'
    default_stpc_wvfm_dset_name = 'light/stpc_wvfm'
    default_single_module = -1
    def swvfm_dtype(self, ntpc, ndet, nsamples):
        return np.dtype([('samples', 'f4', (ntpc, ndet, nsamples))])

    def stpc_wvfm_dtype(self, ntpc, nsamples):
        return np.dtype([('samples', 'f4', (ntpc, 2, nsamples))])
        
    def schan_wvfm_dtype(self, ntpc, ndet, nsamples):
        return np.dtype([('samples', 'f4', (ntpc, ndet, nsamples))])

    def swvfm_align_dtype(self, ntpc, ndet):
        return np.dtype([('ns', 'f8'), ('sample_idx', 'f4', (ntpc, ndet))])

    def stpc_wvfm_align_dtype(self, ntpc, ntrap):
        return np.dtype([('ns', 'f8'), ('sample_idx', 'f4', (ntpc, 2))])

    def schan_wvfm_align_dtype(self, ntpc, ndet):
        return np.dtype([('ns', 'f8'), ('sample_idx', 'f4', (ntpc, ndet))])
    
    def __init__(self, **params):
        super(WaveformSum, self).__init__(**params)

        self.wvfm_dset_name = params.get('wvfm_dset_name', self.default_wvfm_dset_name)
        self.wvfm_align_dset_name = f'{self.wvfm_dset_name}/alignment'

        self.make_swvfm_dset = params.get('make_swvfm_dset', self.default_make_swvfm_dset)
        self.swvfm_dset_name = params.get('swvfm_dset_name')
        self.swvfm_align_dset_name = f'{self.swvfm_dset_name}/alignment'

        self.make_stpc_wvfm_dset = params.get('make_stpc_wvfm_dset', self.default_make_stpc_wvfm_dset)
        self.stpc_wvfm_dset_name = params.get('stpc_wvfm_dset_name', self.default_stpc_wvfm_dset_name)
        self.stpc_wvfm_align_dset_name = f'{self.stpc_wvfm_dset_name}/alignment'

        self.make_schan_wvfm_dset = params.get('make_schan_wvfm_dset', self.default_make_schan_wvfm_dset)
        self.schan_wvfm_dset_name = params.get('schan_wvfm_dset_name', self.default_swvfm_dset_name)
        self.schan_wvfm_align_dset_name = f'{self.schan_wvfm_dset_name}/alignment'

        self.single_module = params.get('single_module', self.default_single_module)
        if not self.make_swvfm_dset and not self.make_stpc_wvfm_dset and not self.make_schan_wvfm_dset:
            raise ValueError('All waveform sum types disabled, exiting.')
        
    def init(self, source_name):
        super(WaveformSum, self).init(source_name)

        self.data_manager.set_attrs(self.swvfm_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    source_dset=source_name,
                                    wvfm_dset=self.wvfm_dset_name)

        # then set up new datasets
        tpc_ids, det_ids = resources['Geometry'].det_bounds.keys()
        _, sum_chan_ids = resources['Geometry'].sum_chan_bounds.keys()
        wvfm_dset = self.data_manager.get_dset(self.wvfm_dset_name)

        if self.make_swvfm_dset:
            # det sum channels
            self.swvfm_dtype = self.swvfm_dtype(len(np.unique(tpc_ids)),
                len(np.unique(det_ids)), wvfm_dset.dtype['samples'].shape[2])
            self.data_manager.create_dset(self.swvfm_dset_name, dtype=self.swvfm_dtype)
            self.data_manager.create_ref(source_name, self.swvfm_dset_name)

        if self.make_schan_wvfm_dset:
            # sum channels
            self.schan_wvfm_dtype = self.schan_wvfm_dtype(len(np.unique(tpc_ids)),
                len(np.unique(sum_chan_ids)), wvfm_dset.dtype['samples'].shape[2])
            self.data_manager.create_dset(self.schan_wvfm_dset_name, dtype=self.schan_wvfm_dtype)
            self.data_manager.create_ref(source_name, self.schan_wvfm_dset_name)

        if self.make_stpc_wvfm_dset:
            # tpc sum channels
            self.stpc_wvfm_dtype = self.stpc_wvfm_dtype(len(np.unique(tpc_ids)),
                                                        wvfm_dset.dtype['samples'].shape[2])
            self.data_manager.create_dset(self.stpc_wvfm_dset_name, dtype=self.stpc_wvfm_dtype)
            self.data_manager.create_ref(source_name, self.stpc_wvfm_dset_name)

        # alignments
        if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
            if self.make_swvfm_dset:
                # det sum
                self.swvfm_align_dtype = self.swvfm_align_dtype(len(np.unique(tpc_ids)), len(np.unique(det_ids)))
                self.data_manager.create_dset(self.swvfm_align_dset_name, dtype=self.swvfm_align_dtype)
                self.data_manager.create_ref(source_name, self.swvfm_align_dset_name)
            if self.make_stpc_wvfm_dset:
                # tpc sum
                self.stpc_wvfm_align_dtype = self.stpc_wvfm_align_dtype(len(np.unique(tpc_ids)), 2)
                self.data_manager.create_dset(self.stpc_wvfm_align_dset_name, dtype=self.stpc_wvfm_align_dtype)
                self.data_manager.create_ref(source_name, self.stpc_wvfm_align_dset_name)
            #if self.make_schan_wvfm_dset:
                # channel sum
            #    self.schan_wvfm_align_dtype = self.schan_wvfm_align_dtype(len(np.unique(tpc_ids)), len(np.unique(det_ids)))
            #    self.data_manager.create_dset(self.schan_wvfm_align_dset_name, dtype=self.schan_wvfm_align_dtype)
            #    self.data_manager.create_ref(source_name, self.schan_wvfm_align_dset_name)

    def run(self, source_name, source_slice, cache):
        super(WaveformSum, self).run(source_name, source_slice, cache)

        event_data = cache[source_name]
        wvfm_data = cache[self.wvfm_dset_name].reshape(event_data.shape)
        if self.make_swvfm_dset:
            swvfm_data = np.zeros(event_data.shape, dtype=self.swvfm_dtype)
        if self.make_schan_wvfm_dset:
            schan_wvfm_data = np.zeros(event_data.shape, dtype=self.schan_wvfm_dtype)
        if self.make_stpc_wvfm_dset:
            stpc_wvfm_data = np.zeros(event_data.shape, dtype=self.stpc_wvfm_dtype)

        if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
            wvfm_align_data = cache[self.wvfm_align_dset_name].reshape(event_data.shape)
            if self.make_swvfm_dset:
                swvfm_align_data = np.zeros(event_data.shape, dtype=self.swvfm_align_dtype)
            #if self.make_schan_wvfm_dset:
            #    schan_wvfm_align_data = np.zeros(event_data.shape, dtype=self.schan_wvfm_align_dtype)
            if self.make_stpc_wvfm_dset:
                stpc_wvfm_align_data = np.zeros(event_data.shape, dtype=self.stpc_wvfm_align_dtype)

        if self.single_module in [1,2,3]:
            # plot index to list of (adc, channel) combos that correspond to a full PD tile
            io0_left_y_plot_dict = {0: [(1, 15),(1, 14),(1, 13),(1, 12),(1, 11),(1, 10)], \
                                   1: [(0, 15),(0, 14),(0, 13),(0, 12),(0, 11),(0, 10)], \
                                   2: [(1, 9),(1, 8),(1, 7),(1, 6),(1, 5),(1, 4)], \
                                   3: [(0, 9),(0, 8),(0, 7),(0, 6),(0, 5),(0, 4)]}
    
            io0_right_y_plot_dict = {0: [(1, 31),(1, 30),(1, 29),(1, 28),(1, 27),(1, 26)], \
                                   1: [(0, 31),(0, 30),(0, 29),(0, 28),(0, 27),(0, 26)], \
                                   2: [(1, 25),(1, 24),(1, 23),(1, 22),(1, 21),(1, 20)], \
                                   3: [(0, 25),(0, 24),(0, 23),(0, 22),(0, 21),(0, 20)]}
            
            io1_left_y_plot_dict = {0: [(1, 63),(1, 62),(1, 61),(1, 60),(1, 59),(1, 58)], \
                                   1: [(0, 63),(0, 62),(0, 61),(0, 60),(0, 59),(0, 58)], \
                                   2: [(1, 57),(1, 56),(1, 55),(1, 54),(1, 53),(1, 52)], \
                                   3: [(0, 57),(0, 56),(0, 55),(0, 54),(0, 53),(0, 52)]}
    
            io1_right_y_plot_dict = {0: [(1, 47),(1, 46),(1, 45),(1, 44),(1, 43),(1, 42)], \
                                   1: [(0, 47),(0, 46),(0, 45),(0, 44),(0, 43),(0, 42)], \
                                   2: [(1, 41),(1, 40),(1, 39),(1, 38),(1, 37),(1, 36)], \
                                   3: [(0, 41),(0, 40),(0, 39),(0, 38),(0, 37),(0, 36)]}
        elif self.single_module == 0:
            io0_left_y_plot_dict = {0: [(0, 30),(0, 29),(0, 28),(0, 27),(0, 26),(0, 25)], \
                               1: [(0, 23),(0, 22),(0, 21),(0, 20),(0, 19),(0, 18)], \
                               2: [(0, 14),(0, 13),(0, 12),(0, 11),(0, 10),(0, 9)], \
                               3: [(0, 7),(0, 6),(0, 5),(0, 4),(0, 3),(0, 2)]}

            io0_right_y_plot_dict = {0: [(1, 62),(1, 61),(1, 60),(1, 59),(1, 58),(1, 57)], \
                                   1: [(1, 55),(1, 54),(1, 53),(1, 52),(1, 51),(1, 50)], \
                                   2: [(1, 46),(1, 45),(1, 44),(1, 43),(1, 42),(1, 41)], \
                                   3: [(1, 39),(1, 38),(1, 37),(1, 36),(1, 35),(1, 34)]}
    
            io1_left_y_plot_dict = {0: [(1, 30),(1, 29),(1, 28),(1, 27),(1, 26),(1, 25)], \
                                   1: [(1, 23),(1, 22),(1, 21),(1, 20),(1, 19),(1, 18)], \
                                   2: [(1, 14),(1, 13),(1, 12),(1, 11),(1, 10),(1, 9)], \
                                   3: [(1, 7),(1, 6),(1, 5),(1, 4),(1, 3),(1, 2)]}
    
            io1_right_y_plot_dict = {0: [(0, 62),(0, 61),(0, 60),(0, 59),(0, 58),(0, 57)], \
                                   1: [(0, 55),(0, 54),(0, 53),(0, 52),(0, 51),(0, 50)], \
                                   2: [(0, 46),(0, 45),(0, 44),(0, 43),(0, 42),(0, 41)], \
                                   3: [(0, 39),(0, 38),(0, 37),(0, 36),(0, 35),(0, 34)]}
        if self.single_module != -1:
            sum_chan_lookup_dict = {}
            sch_id = 0 
            for Dict in [io0_left_y_plot_dict, io0_right_y_plot_dict, io1_left_y_plot_dict, io1_right_y_plot_dict]:
                for tile in list(Dict.keys()):
                    for adc_ch in Dict[tile]:
                        sum_chan_lookup_dict[adc_ch] = sch_id
                    sch_id+=1
        for adc in range(wvfm_data['samples'].shape[1]):
            for chan in range(wvfm_data['samples'].shape[2]):
                tpc_id = resources['Geometry'].sipm_rel_pos[(adc,chan)][0][0]
                det_id = resources['Geometry'].det_id[(adc,chan)]
                if self.make_schan_wvfm_dset:
                    if self.single_module != -1:
                        try:
                            sum_chan_id = sum_chan_lookup_dict[(adc, chan)]
                        except:
                            continue
                    else:
                        sum_chan_id = resources['Geometry'].sum_chan_id[(adc, chan)]
                        
                    #if sum_chan_id < 0:
                    #    continue
                # skip negative indices
                if tpc_id < 0 or det_id < 0:
                    continue

                det_type = resources['Geometry'].det_to_trap_type[(tpc_id, det_id)]
                    
                #if self.make_schan_wvfm_dset and sum_chan_id < 0:
                #    continue
                
                mask = event_data['wvfm_valid'][:,adc,chan].astype(bool)
                if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
                    if self.make_swvfm_dset:
                        # det summed wvfm alignment
                        swvfm_align_data['sample_idx'][mask,tpc_id,det_id] = wvfm_align_data['sample_idx'][mask,adc,chan]
                        swvfm_align_data['ns'][mask] = wvfm_align_data['ns'][mask]
                    if self.make_stpc_wvfm_dset:
                        # tpc summed wvfm alignment
                        stpc_wvfm_align_data['sample_idx'][mask,tpc_id,det_type] = wvfm_align_data['sample_idx'][mask,adc,chan]
                        stpc_wvfm_align_data['ns'][mask] = wvfm_align_data['ns'][mask]
                    #if self.make_schan_wvfm_dset:
                        # channel summed wvfm alignment
                    #    schan_wvfm_align_data['sample_idx'][mask,tpc_id,sum_chan_id] = wvfm_align_data['sample_idx'][mask,adc,chan]
                    #    schan_wvfm_align_data['ns'][mask] = wvfm_align_data['ns'][mask]

                if self.make_swvfm_dset:
                    # det summed wvfm
                    swvfm_data['samples'][mask,tpc_id,det_id,:] += (
                        wvfm_data['samples'][mask,adc,chan].filled(0))
                if self.make_stpc_wvfm_dset:
                    # tpc summed wvfm
                    stpc_wvfm_data['samples'][mask,tpc_id,det_type,:] += (
                        wvfm_data['samples'][mask,adc,chan].filled(0))
                if self.make_schan_wvfm_dset:
                    # channel summed wvfm
                    #sum_chan_id = sum_chan_lookup_dict[(adc, chan)]
                    wvfm = wvfm_data['samples'][mask,adc,chan]
                    #if self.single_module == 0:
                    #    wvfm = wvfm - np.mean(wvfm[0:80])
                    schan_wvfm_data['samples'][mask,tpc_id,sum_chan_id,:] += (wvfm.filled(0))
                    #schan_wvfm_data['samples'][mask,tpc_id,sum_chan_id,:] += -1*((wvfm - np.mean(wvfm[0:50])).filled(0))
        # reserve new data:
        if self.make_swvfm_dset:
            # det summed wvfm
            swvfm_slice = self.data_manager.reserve_data(self.swvfm_dset_name, source_slice)
            self.data_manager.write_data(self.swvfm_dset_name, source_slice, swvfm_data)
            if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
                swvfm_align_slice = self.data_manager.reserve_data(self.swvfm_align_dset_name, source_slice)
                self.data_manager.write_data(self.swvfm_align_dset_name, swvfm_align_slice, swvfm_align_data)
            
            swvfm_ref = np.c_[source_slice, swvfm_slice]
            self.data_manager.write_ref(source_name, self.swvfm_dset_name, swvfm_ref)
            if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
                swvfm_ref = np.c_[source_slice, swvfm_align_slice]
                self.data_manager.write_ref(source_name, self.swvfm_align_dset_name, swvfm_ref)

        if self.make_stpc_wvfm_dset:
            # tpc summed wvfm
            stpc_wvfm_slice = self.data_manager.reserve_data(self.stpc_wvfm_dset_name, source_slice)
            self.data_manager.write_data(self.stpc_wvfm_dset_name, source_slice, stpc_wvfm_data)
            if(self.data_manager.dset_exists(self.stpc_wvfm_align_dset_name)):
                stpc_wvfm_align_slice = self.data_manager.reserve_data(self.stpc_wvfm_align_dset_name, source_slice)
                self.data_manager.write_data(self.stpc_wvfm_align_dset_name, stpc_wvfm_align_slice, stpc_wvfm_align_data)

            stpc_wvfm_ref = np.c_[source_slice, stpc_wvfm_slice]
            self.data_manager.write_ref(source_name, self.stpc_wvfm_dset_name, stpc_wvfm_ref)
            if(self.data_manager.dset_exists(self.stpc_wvfm_align_dset_name)):
                stpc_wvfm_ref = np.c_[source_slice, stpc_wvfm_align_slice]
                self.data_manager.write_ref(source_name, self.stpc_wvfm_align_dset_name, stpc_wvfm_ref)

        if self.make_schan_wvfm_dset:
            # sum channel summed wvfm
            schan_wvfm_slice = self.data_manager.reserve_data(self.schan_wvfm_dset_name, source_slice)
            self.data_manager.write_data(self.schan_wvfm_dset_name, source_slice, schan_wvfm_data)
            #if(self.data_manager.dset_exists(self.schan_wvfm_align_dset_name)):
            #    schan_wvfm_align_slice = self.data_manager.reserve_data(self.schan_wvfm_align_dset_name, source_slice)
            #    self.data_manager.write_data(self.schan_wvfm_align_dset_name, schan_wvfm_align_slice, schan_wvfm_align_data)

            schan_wvfm_ref = np.c_[source_slice, schan_wvfm_slice]
            self.data_manager.write_ref(source_name, self.schan_wvfm_dset_name, schan_wvfm_ref)
            #if(self.data_manager.dset_exists(self.schan_wvfm_align_dset_name)):
            #    schan_wvfm_ref = np.c_[source_slice, schan_wvfm_align_slice]
            #    self.data_manager.write_ref(source_name, self.schan_wvfm_align_dset_name, schan_wvfm_ref)
