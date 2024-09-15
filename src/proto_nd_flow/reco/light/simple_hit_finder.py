import numpy as np
import numpy.ma as ma
from collections import defaultdict
import scipy.interpolate
from scipy.signal import butter, filtfilt

from h5flow.core import H5FlowStage, resources

import module0_flow.util.units as units


class WaveformHitFinder(H5FlowStage):
    '''
        Extracts "hits" from light events using a simple threshold. So if a maximum value in a waveform is above
        a threshold, then it is considered a hit. This class currently only uses the sum waveforms dataset, not the
        SiPM waveforms.

        Parameters:
         - ``wvfm_dset_name``: ``str``, path to input waveforms
         - ``t_ns_dset_name``: ``str``, path to corrected light PPS timestamps
         - ``hits_dset_name``: ``str``, path to output hits dataset
         - ``threshold``: ``dict`` of ``dict`` containing sets of ``tpc_index: {channel_index: threshold, ...}`` used for hit finding. A fixed global value can also be specified with a single ``float`` value

         Both ``wvfm_dset_name``, ``{wvfm_dset_name}/alignment``, and ``t_ns_dset_name`` are required in the cache.

         Requires RunData resource in workflow.

         ``hits`` datatype::

            id          u4,             unique identifier
            tpc         u1,             tpc (for sum_hit)
            det         u1,             detector (for sum_hit/sipm_hit)
            boundary    f4(3),          (x,y,z) center of det
            samples     f4(nsamples,),  waveform adc values
            amplitude   f4,             peak adc value
    '''
    class_version = '2.0.0'

    default_hits_dset_name = 'light/hits'
    default_global_threshold = 1000
    default_mask = []

    def default_threshold(self, global_threshold):
        return defaultdict(lambda: defaultdict(lambda: global_threshold))

    def hits_dtype(self, nsamples):
        return np.dtype([
                ('id', 'u4'),
                ('tpc', 'u1'),
                ('det', 'u1'),
                ('boundary', 'f4', (2,3)),
                ('samples', 'f4', (nsamples,)),
                ('samples_filtered', 'f4', (nsamples,)),
                ('amplitude', 'f4')
            ])
    def apply_filter(self, waveform, cutoff_freq, order=2):
        """
        Apply a low pass filter to the input waveform.

        Parameters:
            waveform (array_like): Input waveform array.
            cutoff_freq (float): Cutoff frequency of the low-pass filter in Hz.
            order (int): Order of the low-pass filter.

        Returns:
            array_like: Filtered waveform.
        """
        sampling_freq = 62.5e6
        normalized_cutoff_freq = cutoff_freq / (sampling_freq / 2)
        b, a = butter(order, normalized_cutoff_freq, btype='lowpass', analog=False)
        filtered_waveform = filtfilt(b, a, waveform)

        return filtered_waveform
    def __init__(self, **params):
        super(WaveformHitFinder, self).__init__(**params)
        self.wvfm_dset_name = params.get('wvfm_dset_name')
        self.t_ns_dset_name = params.get('t_ns_dset_name')
        self.hits_dset_name = params.get('hits_dset_name', self.default_hits_dset_name)
        # set hit finding thresholds (will be converted to an array later in init())
        self.threshold = params.get('threshold', self.default_global_threshold)
        
        if isinstance(self.threshold, int) or isinstance(self.threshold, float):
            # if a global threshold is specified, use the default generator
            self.threshold = self.default_threshold(
                self.threshold)
        elif isinstance(self.threshold, dict):
            # otherwise convert to a defaultdict
            new_dict = self.default_threshold(
                self.default_global_threshold)
            for key, subdict in self.threshold.items():
                for subkey, subval in subdict.items():
                    new_dict[key][subkey] = subval
            self.threshold = new_dict
            
    def init(self, source_name):
        super(WaveformHitFinder, self).init(source_name)

        wvfm_dset = self.data_manager.get_dset(self.wvfm_dset_name)

        # get convert sample rate to ns
        self.sample_rate = (resources['RunData'].lrs_ticks
                            / units.ns)

        # get waveform shape information
        self.ntpc = wvfm_dset.dtype['samples'].shape[0]
        self.ndet = wvfm_dset.dtype['samples'].shape[1]
        self.nsamples = wvfm_dset.dtype['samples'].shape[2]
        self.hits_dtype = self.hits_dtype(self.nsamples)
        
        # convert channel thresholds into an array
        threshold_array = np.zeros((self.ntpc, self.ndet, 1))
        for tpc in range(self.ntpc):
            for det in range(self.ndet):
                threshold_array[tpc,
                                det] = self.threshold[tpc][det]
        self.threshold = threshold_array

        # create datasets and references
        self.data_manager.create_dset(self.hits_dset_name,
                                      dtype=self.hits_dtype)
        self.data_manager.create_ref(source_name, self.hits_dset_name)
        self.data_manager.create_ref(self.wvfm_dset_name, self.hits_dset_name)
        self.data_manager.set_attrs(self.hits_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    wvfm_dset=self.wvfm_dset_name,
                                    thresholds=self.threshold,
                                    ntpc=self.ntpc,
                                    ndet=self.ndet,
                                    nsamples=self.nsamples
                                    )

    def run(self, source_name, source_slice, cache):
        super(WaveformHitFinder, self).run(source_name, source_slice, cache)
        wvfms = cache[self.wvfm_dset_name].reshape(cache[source_name].shape)[
            'samples']  # 1:1 relationship
        events = cache[source_name]
        
        hit_id = 0
        event_id = []
        hits_data = np.zeros((0,), dtype=self.hits_dtype)
        for i in range(len(events)):
            wvfms_event = wvfms[i]
            for tpc_id in range(self.ntpc):
                if not np.all(wvfms_event[tpc_id,0,:] != 0):
                    continue # skip tpcs without data
                for det_id in range(self.ndet):
                    hit_data = np.zeros((1,), self.hits_dtype)
                    wvfm = np.array(wvfms_event[tpc_id, det_id, :]).astype('int')
                    wvfm_filtered = self.apply_filter(wvfm, 10e6, 2)
                    if np.any(wvfm_filtered > self.threshold[tpc_id][det_id][0]):
                        event_id.append(events[i]['id'])
                        hit_data['id'] = hit_id
                        hit_data['tpc'] = tpc_id
                        hit_data['det'] = det_id
                        hit_data['samples'] = wvfm
                        hit_data['samples_filtered'] = wvfm_filtered
                        hit_data['amplitude'] = np.max(wvfm)
                        hit_data['boundary'] = np.array(resources['Geometry'].det_bounds[(tpc_id,det_id)][0])
                        hits_data = np.concatenate((hits_data, hit_data))
                        hit_id += 1
                    
        # save data
        hit_slice = self.data_manager.reserve_data(
            self.hits_dset_name, len(hits_data))
        #if len(hit_data):
        #    hit_data['id'] = np.r_[hit_slice]
        self.data_manager.write_data(self.hits_dset_name, hit_slice, hits_data)

        if len(hits_data):
            ref = np.c_[event_id, hits_data['id']]
        else:
            ref = np.empty((0, 2))
        self.data_manager.write_ref(source_name, self.hits_dset_name, ref)
        self.data_manager.write_ref(
            self.wvfm_dset_name, self.hits_dset_name, ref)