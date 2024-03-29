import numpy as np
from collections import defaultdict

from h5flow.core import H5FlowStage, resources


class WaveformSum(H5FlowStage):
    '''
        Sums the signal across light detector SiPM channels.

        Parameters:
         - ``wvfm_dset_name`` : ``str``, required, input dataset path
         - ``swvfm_dset_name`` : ``str``, required, output dataset path

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


        Uses the same dtype as the input waveform dataset(s) except with
        ``(nadc, nchannel)`` resized to be ``(ntpc, ndet)``.

    '''
    class_version = '1.0.0'

    default_detector_channels = [list(range(64))]

    def swvfm_dtype(self, ntpc, ndet, nsamples):
        return np.dtype([('samples', 'f4', (ntpc, ndet, nsamples))])

    def align_dtype(self, ntpc, ndet):
        return np.dtype([('ns', 'f8'), ('sample_idx', 'f4', (ntpc,ndet))])

    def __init__(self, **params):
        super(WaveformSum, self).__init__(**params)

        self.wvfm_dset_name = params.get('wvfm_dset_name')
        self.wvfm_align_dset_name = f'{self.wvfm_dset_name}/alignment'
        self.swvfm_dset_name = params.get('swvfm_dset_name')
        self.align_dset_name = f'{self.swvfm_dset_name}/alignment'

                
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

        self.swvfm_dtype = self.swvfm_dtype(len(np.unique(tpc_ids)),
            len(np.unique(det_ids)), wvfm_dset.dtype['samples'].shape[2])
        self.data_manager.create_dset(self.swvfm_dset_name, dtype=self.swvfm_dtype)
        self.data_manager.create_ref(source_name, self.swvfm_dset_name)

        if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
            self.align_dtype = self.align_dtype(len(np.unique(tpc_ids)), len(np.unique(det_ids)))
            self.data_manager.create_dset(self.align_dset_name, dtype=self.align_dtype)
            self.data_manager.create_ref(source_name, self.align_dset_name)

    def run(self, source_name, source_slice, cache):
        super(WaveformSum, self).run(source_name, source_slice, cache)

        event_data = cache[source_name]
        wvfm_data = cache[self.wvfm_dset_name].reshape(event_data.shape)
        swvfm_data = np.zeros(event_data.shape, dtype=self.swvfm_dtype)

        if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
            wvfm_align_data = cache[self.wvfm_align_dset_name].reshape(event_data.shape)
            align_data = np.zeros(event_data.shape, dtype=self.align_dtype)

        for adc in range(wvfm_data['samples'].shape[1]):
            for chan in range(wvfm_data['samples'].shape[2]):
                tpc_id = resources['Geometry'].sipm_rel_pos[(adc,chan)][0][0]
                det_id = resources['Geometry'].det_id[(adc,chan)]
                if tpc_id < 0 or det_id < 0:
                    continue
                mask = event_data['wvfm_valid'][:,adc,chan].astype(bool)
                if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
                    align_data['sample_idx'][mask,tpc_id,det_id] = wvfm_align_data['sample_idx'][mask,adc,chan]
                    align_data['ns'][mask] = wvfm_align_data['ns'][mask]

        for adc in range(wvfm_data['samples'].shape[1]):
            for chan in range(wvfm_data['samples'].shape[2]):
                tpc_id = resources['Geometry'].sipm_rel_pos[(adc,chan)][0][0]
                det_id = resources['Geometry'].det_id[(adc,chan)]
                if tpc_id < 0 or det_id < 0:
                    continue
                # WARNING: does not handle case where different channels on same detector are not aligned (not relevant for Module 0 data)
                mask = event_data['wvfm_valid'][:,adc,chan].astype(bool)
                swvfm_data['samples'][mask,tpc_id,det_id,:] += (
                    wvfm_data['samples'][mask,adc,chan].filled(0))

        # reserve new data
        swvfm_slice = self.data_manager.reserve_data(self.swvfm_dset_name, source_slice)
        self.data_manager.write_data(self.swvfm_dset_name, source_slice, swvfm_data)

        if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
            align_slice = self.data_manager.reserve_data(self.align_dset_name, source_slice)
            self.data_manager.write_data(self.align_dset_name, align_slice, align_data)

        # save references
        ref = np.c_[source_slice, swvfm_slice]
        self.data_manager.write_ref(source_name, self.swvfm_dset_name, ref)

        if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
            ref = np.c_[source_slice, align_slice]
            self.data_manager.write_ref(source_name, self.align_dset_name, ref)

