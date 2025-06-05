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

        self.wvfm_dset_name = params.get('wvfm_dset_name')
        self.wvfm_align_dset_name = f'{self.wvfm_dset_name}/alignment'

        self.swvfm_dset_name = params.get('swvfm_dset_name')
        self.swvfm_align_dset_name = f'{self.swvfm_dset_name}/alignment'

        self.stpc_wvfm_dset_name = params.get('stpc_wvfm_dset_name')
        self.stpc_wvfm_align_dset_name = f'{self.stpc_wvfm_dset_name}/alignment'

        self.schan_wvfm_dset_name = params.get('schan_wvfm_dset_name')
        self.schan_wvfm_align_dset_name = f'{self.schan_wvfm_dset_name}/alignment'

    def init(self, source_name):
        super(WaveformSum, self).init(source_name)

        self.data_manager.set_attrs(self.swvfm_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    source_dset=source_name,
                                    wvfm_dset=self.wvfm_dset_name)

        # then set up new datasets
        tpc_ids, det_ids = resources['Geometry'].det_bounds.keys()
        wvfm_dset = self.data_manager.get_dset(self.wvfm_dset_name)

        # det sum channels
        self.swvfm_dtype = self.swvfm_dtype(len(np.unique(tpc_ids)),
            len(np.unique(det_ids)), wvfm_dset.dtype['samples'].shape[2])
        self.data_manager.create_dset(self.swvfm_dset_name, dtype=self.swvfm_dtype)
        self.data_manager.create_ref(source_name, self.swvfm_dset_name)

        # sum channels
        self.schan_wvfm_dtype = self.schan_wvfm_dtype(len(np.unique(tpc_ids)),
            len(np.unique(det_ids)), wvfm_dset.dtype['samples'].shape[2])
        self.data_manager.create_dset(self.schan_wvfm_dset_name, dtype=self.schan_wvfm_dtype)
        self.data_manager.create_ref(source_name, self.schan_wvfm_dset_name)
        
        # tpc sum channels
        self.stpc_wvfm_dtype = self.stpc_wvfm_dtype(len(np.unique(tpc_ids)),
                                                    wvfm_dset.dtype['samples'].shape[2])
        self.data_manager.create_dset(self.stpc_wvfm_dset_name, dtype=self.stpc_wvfm_dtype)
        self.data_manager.create_ref(source_name, self.stpc_wvfm_dset_name)

        # alignments
        if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
            # det sum
            self.swvfm_align_dtype = self.swvfm_align_dtype(len(np.unique(tpc_ids)), len(np.unique(det_ids)))
            self.data_manager.create_dset(self.swvfm_align_dset_name, dtype=self.swvfm_align_dtype)
            self.data_manager.create_ref(source_name, self.swvfm_align_dset_name)
            # tpc sum
            self.stpc_wvfm_align_dtype = self.stpc_wvfm_align_dtype(len(np.unique(tpc_ids)), 2)
            self.data_manager.create_dset(self.stpc_wvfm_align_dset_name, dtype=self.stpc_wvfm_align_dtype)
            self.data_manager.create_ref(source_name, self.stpc_wvfm_align_dset_name)
            # channel sum
            self.schan_wvfm_align_dtype = self.schan_wvfm_align_dtype(len(np.unique(tpc_ids)), len(np.unique(det_ids)))
            self.data_manager.create_dset(self.schan_wvfm_align_dset_name, dtype=self.schan_wvfm_align_dtype)
            self.data_manager.create_ref(source_name, self.schan_wvfm_align_dset_name)

    def run(self, source_name, source_slice, cache):
        super(WaveformSum, self).run(source_name, source_slice, cache)

        event_data = cache[source_name]
        wvfm_data = cache[self.wvfm_dset_name].reshape(event_data.shape)
        swvfm_data = np.zeros(event_data.shape, dtype=self.swvfm_dtype)
        schan_wvfm_data = np.zeros(event_data.shape, dtype=self.schan_wvfm_dtype)
        stpc_wvfm_data = np.zeros(event_data.shape, dtype=self.stpc_wvfm_dtype)

        if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
            wvfm_align_data = cache[self.wvfm_align_dset_name].reshape(event_data.shape)
            swvfm_align_data = np.zeros(event_data.shape, dtype=self.swvfm_align_dtype)
            schan_wvfm_align_data = np.zeros(event_data.shape, dtype=self.schan_wvfm_align_dtype)
            stpc_wvfm_align_data = np.zeros(event_data.shape, dtype=self.stpc_wvfm_align_dtype)

        for adc in range(wvfm_data['samples'].shape[1]):
            for chan in range(wvfm_data['samples'].shape[2]):
                tpc_id = resources['Geometry'].sipm_rel_pos[(adc,chan)][0][0]
                det_id = resources['Geometry'].det_id[(adc,chan)]
                sum_chan_id = resources['Geometry'].sum_chan_id[(tpc_id, chan)]
                #if sum_chan_id < 0:
                #    print(f"{sum_chan_id=}, {adc=}, {chan=}")
                # skip negative indices
                if tpc_id < 0 or det_id < 0 or sum_chan_id < 0:
                    continue
                # check number of channels with same det_id
                n_ch_per_det = 0
                for adc_ in range(wvfm_data['samples'].shape[1]):
                    for chan_ in range(wvfm_data['samples'].shape[2]):
                        tpc_id_ = resources['Geometry'].sipm_rel_pos[(adc_,chan_)][0][0]
                        det_id_ = resources['Geometry'].det_id[(adc_,chan_)]
                        if det_id_ == det_id and tpc_id_ == tpc_id:
                            n_ch_per_det += 1
                # if 6 channels, det type is 0 (ACL), if 2 channels, det type is 1 (LCM)
                det_type = 0
                if n_ch_per_det == 2:
                    det_type = 1
                elif n_ch_per_det != 6:
                    raise ValueError(f"Invalid number of channels for det_id {det_id}: {n_ch_per_det}")
                # skip negative indices
                if tpc_id < 0 or det_id < 0 or sum_chan_id < 0:
                    continue
                mask = event_data['wvfm_valid'][:,adc,chan].astype(bool)
                if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
                    # det summed wvfm alignment
                    swvfm_align_data['sample_idx'][mask,tpc_id,det_id] = wvfm_align_data['sample_idx'][mask,adc,chan]
                    swvfm_align_data['ns'][mask] = wvfm_align_data['ns'][mask]
                    # tpc summed wvfm alignment
                    stpc_wvfm_align_data['sample_idx'][mask,tpc_id,det_type] = wvfm_align_data['sample_idx'][mask,adc,chan]
                    stpc_wvfm_align_data['ns'][mask] = wvfm_align_data['ns'][mask]
                    # channel summed wvfm alignment
                    schan_wvfm_align_data['sample_idx'][mask,tpc_id,sum_chan_id] = wvfm_align_data['sample_idx'][mask,adc,chan]
                    schan_wvfm_align_data['ns'][mask] = wvfm_align_data['ns'][mask]

        for adc in range(wvfm_data['samples'].shape[1]):
            for chan in range(wvfm_data['samples'].shape[2]):
                tpc_id = resources['Geometry'].sipm_rel_pos[(adc,chan)][0][0]
                det_id = resources['Geometry'].det_id[(adc,chan)]
                sum_chan_id = resources['Geometry'].sum_chan_id[(tpc_id, chan)]
                # skip negative indices
                if tpc_id < 0 or det_id < 0 or sum_chan_id < 0:
                    continue
                # check number of channels with same det_id
                n_ch_per_det = 0
                for adc_ in range(wvfm_data['samples'].shape[1]):
                    for chan_ in range(wvfm_data['samples'].shape[2]):
                        tpc_id_ = resources['Geometry'].sipm_rel_pos[(adc_,chan_)][0][0]
                        det_id_ = resources['Geometry'].det_id[(adc_,chan_)]
                        if det_id_ == det_id and tpc_id_ == tpc_id:
                            n_ch_per_det += 1
                # if 6 channels, det type is 0 (ACL), if 2 channels, det type is 1 (LCM)
                det_type = 0
                if n_ch_per_det == 2:
                    det_type = 1
                elif n_ch_per_det != 6:
                    raise ValueError(f"Invalid number of channels for det_id {det_id}: {n_ch_per_det}")
                # WARNING: does not handle case where different channels on same detector are not aligned (not relevant for Module 0 data)
                mask = event_data['wvfm_valid'][:,adc,chan].astype(bool)
                # det summed wvfm
                swvfm_data['samples'][mask,tpc_id,det_id,:] += (
                    wvfm_data['samples'][mask,adc,chan].filled(0))
                # tpc summed wvfm
                stpc_wvfm_data['samples'][mask,tpc_id,det_type,:] += (
                    wvfm_data['samples'][mask,adc,chan].filled(0))
                # channel summed wvfm
                schan_wvfm_data['samples'][mask,tpc_id,sum_chan_id,:] += (
                    wvfm_data['samples'][mask,adc,chan].filled(0))

        # reserve new data:

        # det summed wvfm
        swvfm_slice = self.data_manager.reserve_data(self.swvfm_dset_name, source_slice)
        self.data_manager.write_data(self.swvfm_dset_name, source_slice, swvfm_data)
        if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
            swvfm_align_slice = self.data_manager.reserve_data(self.swvfm_align_dset_name, source_slice)
            self.data_manager.write_data(self.swvfm_align_dset_name, swvfm_align_slice, swvfm_align_data)

        # sum channel summed wvfm
        schan_wvfm_slice = self.data_manager.reserve_data(self.schan_wvfm_dset_name, source_slice)
        self.data_manager.write_data(self.schan_wvfm_dset_name, source_slice, schan_wvfm_data)
        if(self.data_manager.dset_exists(self.schan_wvfm_align_dset_name)):
            schan_wvfm_align_slice = self.data_manager.reserve_data(self.schan_wvfm_align_dset_name, source_slice)
            self.data_manager.write_data(self.schan_wvfm_align_dset_name, schan_wvfm_align_slice, schan_wvfm_align_data)

        # tpc summed wvfm
        stpc_wvfm_slice = self.data_manager.reserve_data(self.stpc_wvfm_dset_name, source_slice)
        self.data_manager.write_data(self.stpc_wvfm_dset_name, source_slice, stpc_wvfm_data)
        if(self.data_manager.dset_exists(self.stpc_wvfm_align_dset_name)):
            stpc_wvfm_align_slice = self.data_manager.reserve_data(self.stpc_wvfm_align_dset_name, source_slice)
            self.data_manager.write_data(self.stpc_wvfm_align_dset_name, stpc_wvfm_align_slice, stpc_wvfm_align_data)

        # save references:

        # det summed wvfm
        swvfm_ref = np.c_[source_slice, swvfm_slice]
        self.data_manager.write_ref(source_name, self.swvfm_dset_name, swvfm_ref)
        if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
            swvfm_ref = np.c_[source_slice, swvfm_align_slice]
            self.data_manager.write_ref(source_name, self.swvfm_align_dset_name, swvfm_ref)

        # sum channel summed wvfm
        schan_wvfm_ref = np.c_[source_slice, schan_wvfm_slice]
        self.data_manager.write_ref(source_name, self.schan_wvfm_dset_name, schan_wvfm_ref)
        if(self.data_manager.dset_exists(self.schan_wvfm_align_dset_name)):
            schan_wvfm_ref = np.c_[source_slice, schan_wvfm_align_slice]
            self.data_manager.write_ref(source_name, self.schan_wvfm_align_dset_name, schan_wvfm_ref)

        # tpc summed wvfm
        stpc_wvfm_ref = np.c_[source_slice, stpc_wvfm_slice]
        self.data_manager.write_ref(source_name, self.stpc_wvfm_dset_name, stpc_wvfm_ref)
        if(self.data_manager.dset_exists(self.stpc_wvfm_align_dset_name)):
            stpc_wvfm_ref = np.c_[source_slice, stpc_wvfm_align_slice]
            self.data_manager.write_ref(source_name, self.stpc_wvfm_align_dset_name, stpc_wvfm_ref)
