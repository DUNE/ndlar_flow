import numpy as np
from collections import defaultdict

from h5flow.core import H5FlowStage, resources


class WaveformSum(H5FlowStage):
    '''
        Sums the signal across light detector SiPM channels.

        Parameters:
         - ``wvfm_dset_name`` : ``str``, required, input dataset path
         - ``rms_dset_name`` : ``str``, required, input dataset path
         - ``swvfm_dset_name`` : ``str``, required, output det sum channels dataset path
         - ``srms_dset_name`` : ``str``, required, output det sum channels rms dataset path
         - ``stpc_wvfm_dset_name`` : ``str``, required, output tpc sum channels dataset path
         - ``stpc_rms_dset_name`` : ``str``, required, output tpc sum channels rms dataset path

        ``wvfm_dset_name`` is required in the data cache.

        The Geometry resource is required in the workflow.

        Example config::

            wvfm_sum:
                classname: WaveformSum
                requires:
                    - 'light/events'
                    - 'light/cwvfm'
                    - 'light/cwvfm_rms'
                params:
                    wvfm_dset_name: 'light/cwvfm'
                    rms_dset_name: 'light/cwvfm_rms'
                    swvfm_dset_name: 'light/swvfm'
                    srms_dset_name: 'light/swvfm_rms'
                    stpc_wvfm_dset_name: 'light/stpc_wvfm'
                    stpc_rms_dset_name: 'light/stpc_wvfm_rms'


        Uses the same dtype as the input waveform dataset(s) except with
        ``(nadc, nchannel)`` resized to be ``(ntpc, ndet)``. If the input
        waveforms have a ``clipped`` field, it will be propagated to the summed
        waveforms: if any channel contributing to a sum is clipped, that sum
        channel will be marked as clipped.

    '''
    class_version = '1.0.0'

    default_detector_channels = [list(range(64))]

    def swvfm_dtype(self, ntpc, ndet, nsamples):
        return np.dtype([
            ('samples', 'f4', (ntpc, ndet, nsamples)),
            ('clipped', '?', (ntpc, ndet))  # True if any contributing channel was clipped
        ])

    def stpc_wvfm_dtype(self, ntpc, nsamples):
        return np.dtype([
            ('samples', 'f4', (ntpc, 2, nsamples)),
            ('clipped', '?', (ntpc, 2))  # True if any contributing channel was clipped
        ])

        return np.dtype([('samples', 'f4', (ntpc, ndet, nsamples))])
    def swvfm_rms_dtype(self, ntpc, ndet):
        return np.dtype([('rms', 'f4', (ntpc, ndet))])

    def stpc_wvfm_dtype(self, ntpc, nsamples):
        return np.dtype([('samples', 'f4', (ntpc, 2, nsamples))])
    def stpc_wvfm_rms_dtype(self, ntpc):
        return np.dtype([('rms', 'f4', (ntpc, 2))])


    def swvfm_align_dtype(self, ntpc, ndet):
        return np.dtype([('ns', 'f8'), ('sample_idx', 'f4', (ntpc, ndet))])

    def stpc_wvfm_align_dtype(self, ntpc, ntrap):
        return np.dtype([('ns', 'f8'), ('sample_idx', 'f4', (ntpc, 2))])

    def __init__(self, **params):
        super(WaveformSum, self).__init__(**params)

        self.wvfm_dset_name = params.get('wvfm_dset_name')
        self.wvfm_align_dset_name = f'{self.wvfm_dset_name}/alignment'
        self.rms_dset_name = params.get('rms_dset_name')

        self.swvfm_dset_name = params.get('swvfm_dset_name')
        self.swvfm_align_dset_name = f'{self.swvfm_dset_name}/alignment'
        self.srms_dset_name = params.get('srms_dset_name')

        self.stpc_wvfm_dset_name = params.get('stpc_wvfm_dset_name')
        self.stpc_wvfm_align_dset_name = f'{self.stpc_wvfm_dset_name}/alignment'
        self.stpc_rms_dset_name = params.get('stpc_rms_dset_name')


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

        # rms for det sum channels
        self.srms_dtype = self.swvfm_rms_dtype(len(np.unique(tpc_ids)),
            len(np.unique(det_ids)))
        self.data_manager.create_dset(self.srms_dset_name, dtype=self.srms_dtype)
        self.data_manager.create_ref(source_name, self.srms_dset_name)

        # tpc sum channels
        self.stpc_wvfm_dtype = self.stpc_wvfm_dtype(len(np.unique(tpc_ids)),
                                                    wvfm_dset.dtype['samples'].shape[2])
        self.data_manager.create_dset(self.stpc_wvfm_dset_name, dtype=self.stpc_wvfm_dtype)
        self.data_manager.create_ref(source_name, self.stpc_wvfm_dset_name)

        # rms for tpc sum channels
        self.stpc_rms_dtype = self.stpc_wvfm_rms_dtype(len(np.unique(tpc_ids)))
        self.data_manager.create_dset(self.stpc_rms_dset_name, dtype=self.stpc_rms_dtype)
        self.data_manager.create_ref(source_name, self.stpc_rms_dset_name)

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

    def run(self, source_name, source_slice, cache):
        super(WaveformSum, self).run(source_name, source_slice, cache)

        event_data = cache[source_name]
        wvfm_data = cache[self.wvfm_dset_name].reshape(event_data.shape)
        rms_data = cache[self.rms_dset_name].reshape(event_data.shape)
        swvfm_data = np.zeros(event_data.shape, dtype=self.swvfm_dtype)
        srms_data = np.zeros(event_data.shape, dtype=self.srms_dtype)
        stpc_wvfm_data = np.zeros(event_data.shape, dtype=self.stpc_wvfm_dtype)
        stpc_rms_data = np.zeros(event_data.shape, dtype=self.stpc_rms_dtype)

        # Check if input waveforms have clipped field
        has_clipped = 'clipped' in wvfm_data.dtype.names

        if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
            wvfm_align_data = cache[self.wvfm_align_dset_name].reshape(event_data.shape)
            swvfm_align_data = np.zeros(event_data.shape, dtype=self.swvfm_align_dtype)
            stpc_wvfm_align_data = np.zeros(event_data.shape, dtype=self.stpc_wvfm_align_dtype)

        for adc in range(wvfm_data['samples'].shape[1]):
            for chan in range(wvfm_data['samples'].shape[2]):
                tpc_id = resources['Geometry'].sipm_rel_pos[(adc,chan)][0][0]
                det_id = resources['Geometry'].det_id[(adc,chan)]
                # skip negative indices
                if tpc_id < 0 or det_id < 0:
                    continue
                det_type = resources['Geometry'].det_type[(tpc_id, det_id)]
                mask = event_data['wvfm_valid'][:,adc,chan].astype(bool)
                if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
                    # det summed wvfm alignment
                    swvfm_align_data['sample_idx'][mask,tpc_id,det_id] = wvfm_align_data['sample_idx'][mask,adc,chan]
                    swvfm_align_data['ns'][mask] = wvfm_align_data['ns'][mask]
                    # tpc summed wvfm alignment
                    stpc_wvfm_align_data['sample_idx'][mask,tpc_id,det_type] = wvfm_align_data['sample_idx'][mask,adc,chan]
                    stpc_wvfm_align_data['ns'][mask] = wvfm_align_data['ns'][mask]

        for adc in range(wvfm_data['samples'].shape[1]):
            for chan in range(wvfm_data['samples'].shape[2]):
                tpc_id = resources['Geometry'].sipm_rel_pos[(adc,chan)][0][0]
                det_id = resources['Geometry'].det_id[(adc,chan)]
                # skip negative indices
                if tpc_id < 0 or det_id < 0:
                    continue
                det_type = resources['Geometry'].det_type[(tpc_id, det_id)]
                # WARNING: does not handle case where different channels on same detector are not aligned (not relevant for Module 0 data)
                mask = event_data['wvfm_valid'][:,adc,chan].astype(bool)
                # det summed wvfm
                swvfm_data['samples'][mask,tpc_id,det_id,:] += (
                    wvfm_data['samples'][mask,adc,chan].filled(0))
                # add sum rms in quadrature
                srms_data['rms'][mask,tpc_id,det_id] += rms_data['rms'][mask,adc,chan]**2
                # tpc summed wvfm
                stpc_wvfm_data['samples'][mask,tpc_id,det_type,:] += (
                    wvfm_data['samples'][mask,adc,chan].filled(0))
                # add sum tpc rms in quadrature
                stpc_rms_data['rms'][mask,tpc_id,det_type] += rms_data['rms'][mask,adc,chan]**2

                # propagate clipped flag: if any channel contributing to sum is clipped, mark sum as clipped
                if has_clipped:
                    swvfm_data['clipped'][mask,tpc_id,det_id] |= wvfm_data['clipped'][mask,adc,chan]
                    stpc_wvfm_data['clipped'][mask,tpc_id,det_type] |= wvfm_data['clipped'][mask,adc,chan]


        # Take square root to complete RMS in quadrature calculation
        srms_data['rms'] = np.sqrt(srms_data['rms'])
        stpc_rms_data['rms'] = np.sqrt(stpc_rms_data['rms'])

        # reserve new data:

        # det summed wvfm
        swvfm_slice = self.data_manager.reserve_data(self.swvfm_dset_name, source_slice)
        self.data_manager.write_data(self.swvfm_dset_name, source_slice, swvfm_data)
        if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
            swvfm_align_slice = self.data_manager.reserve_data(self.swvfm_align_dset_name, source_slice)
            self.data_manager.write_data(self.swvfm_align_dset_name, swvfm_align_slice, swvfm_align_data)
        # det summed wvfm rms
        swvfm_rms_slice = self.data_manager.reserve_data(self.srms_dset_name, source_slice)
        self.data_manager.write_data(self.srms_dset_name, swvfm_rms_slice, srms_data)

        # tpc summed wvfm
        stpc_wvfm_slice = self.data_manager.reserve_data(self.stpc_wvfm_dset_name, source_slice)
        self.data_manager.write_data(self.stpc_wvfm_dset_name, source_slice, stpc_wvfm_data)
        if(self.data_manager.dset_exists(self.stpc_wvfm_align_dset_name)):
            stpc_wvfm_align_slice = self.data_manager.reserve_data(self.stpc_wvfm_align_dset_name, source_slice)
            self.data_manager.write_data(self.stpc_wvfm_align_dset_name, stpc_wvfm_align_slice, stpc_wvfm_align_data)
        # tpc summed wvfm rms
        stpc_wvfm_rms_slice = self.data_manager.reserve_data(self.stpc_rms_dset_name, source_slice)
        self.data_manager.write_data(self.stpc_rms_dset_name, stpc_wvfm_rms_slice, stpc_rms_data)

        # save references:

        # det summed wvfm
        swvfm_ref = np.c_[source_slice, swvfm_slice]
        self.data_manager.write_ref(source_name, self.swvfm_dset_name, swvfm_ref)
        if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
            swvfm_ref = np.c_[source_slice, swvfm_align_slice]
            self.data_manager.write_ref(source_name, self.swvfm_align_dset_name, swvfm_ref)
        # det summed wvfm rms
        swvfm_rms_ref = np.c_[source_slice, swvfm_rms_slice]
        self.data_manager.write_ref(source_name, self.srms_dset_name, swvfm_rms_ref)

        # tpc summed wvfm
        stpc_wvfm_ref = np.c_[source_slice, stpc_wvfm_slice]
        self.data_manager.write_ref(source_name, self.stpc_wvfm_dset_name, stpc_wvfm_ref)
        if(self.data_manager.dset_exists(self.stpc_wvfm_align_dset_name)):
            stpc_wvfm_ref = np.c_[source_slice, stpc_wvfm_align_slice]
            self.data_manager.write_ref(source_name, self.stpc_wvfm_align_dset_name, stpc_wvfm_ref)
        # tpc summed wvfm rms
        stpc_wvfm_rms_ref = np.c_[source_slice, stpc_wvfm_rms_slice]
        self.data_manager.write_ref(source_name, self.stpc_rms_dset_name, stpc_wvfm_rms_ref)
