import numpy as np
import numpy.ma as ma
from collections import defaultdict
import scipy.interpolate

from h5flow.core import H5FlowStage, resources

import module0_flow.util.units as units


class WaveformHitFinder(H5FlowStage):
    '''
        Extracts "hits" from waveforms. A hit is defined as a local maxima above
        a defined threshold. Stores the nearest ±N samples around the hit,
        along with timing information and some summary information.

        To most precisely reconstruct the time of a given hit, use the
        following::

            (hits['ns'] + hits['busy_ns'] + hits['ns_spline']) * units.ns

        Parameters:
         - ``wvfm_dset_name``: ``str``, path to input waveforms
         - ``t_ns_dset_name``: ``str``, path to corrected light PPS timestamps
         - ``hits_dset_name``: ``str``, path to output hits dataset
         - ``near_samples``: ``int``, number of neighboring samples to keep
         - ``hit_level``: ``str``, "sipm" or "sum" hit finder (defines variable names)
         - ``threshold``: ``dict`` of ``dict`` containing sets of ``tpc_index: {channel_index: threshold, ...}`` used for hit finding. A fixed global value can also be specified with a single ``float`` value
         - ``mask``: ``list`` of ``int``, detectors to ignore when finding hits

         Both ``wvfm_dset_name``, ``{wvfm_dset_name}/alignment``, and ``t_ns_dset_name`` are required in the cache.

         Requires RunData resource in workflow.

         ``hits`` datatype::

            id          u4,             unique identifier
            tpc/adc     u1,             tpc/adc index (for sum_hit/sipm_hit)
            det/chan    u1,             detector/channel index (for sum_hit/sipm_hit)
            pos         f4(3),          (x,y,z) center of det/sipm 
            sample_idx  u2,             sample index of peak within waveform
            ns          f8,             PPS timestamp of waveform [ns]
            busy_ns     f8,             timestamp of peak relative to busy rising edge (aka when the waveform was triggered) [ns]
            samples     f4(2*near+1,),  sample adc value around peak
            sum         f4,             sum of sample adc values (out to ±near_samples)
            max         f4,             peak adc value
            sum_spline  f4,             integral of spline around peak (out to ±near_samples)
            max_spline  f4,             maximum of spline around peak
            ns_spline   f4,             offset from center sample for maximum of spline [ns]
            rising_spline f4,           projection of spline to rising edge zero-crossing (offset from center sample) [ns]
            rising_err_spline f4,       an estimate of the error on the rising edge zero-crossing [ns]
            fwhm_spline f4,             spline FWHM [ns]

    '''
    class_version = '2.0.0'

    default_hits_dset_name = 'light/true_hits'
    default_near_samples = 3
    default_interpolation = 256
    default_global_threshold = 2000
    default_mask = []

    def default_threshold(self, global_threshold):
        return defaultdict(lambda: defaultdict(lambda: global_threshold))

    def hits_dtype(self, near_samples):
        if self.hit_level=="sum":
            return np.dtype([
                ('id', 'u4'),
                ('tpc', 'u1'),
                ('det', 'u1'),
                ('boundary', 'f4', (2,3)),
                ('sample_idx', 'u2'),
                ('ns', 'f8'),
                ('busy_ns', 'f8'),
                ('samples', 'f4', (2 * near_samples + 1,)),
                ('sum', 'f4'),
                ('max', 'f4'),
                ('sum_spline', 'f4'),
                ('max_spline', 'f4'),
                ('ns_spline', 'f4'),
                ('rising_spline', 'f4'),
                ('rising_err_spline', 'f4'),
                ('fwhm_spline', 'f4')
            ])
        elif self.hit_level=="sipm":
            return np.dtype([
                ('id', 'u4'),
                ('adc', 'u1'),
                ('chan', 'u1'),
                ('pos', 'f4', (3,)),
                ('sample_idx', 'u2'),
                ('ns', 'f8'),
                ('busy_ns', 'f8'),
                ('samples', 'f4', (2 * near_samples + 1,)),
                ('sum', 'f4'),
                ('max', 'f4'),
                ('sum_spline', 'f4'),
                ('max_spline', 'f4'),
                ('ns_spline', 'f4'),
                ('rising_spline', 'f4'),
                ('rising_err_spline', 'f4'),
                ('fwhm_spline', 'f4')
            ])
        else:
            raise RuntimeError(f'Invalid hit level {self.hit_level}')

    def __init__(self, **params):
        super(WaveformHitFinder, self).__init__(**params)
        self.wvfm_dset_name = params.get('wvfm_dset_name')
        print(self.wvfm_dset_name)
        self.hits_dset_name = params.get('hits_dset_name',
                                         self.default_hits_dset_name)
        self.near_samples = params.get('near_samples',
                                       self.default_near_samples)
        self.hit_level = params.get('hit_level')
        self.mask = np.array(params.get('mask',
                                                self.default_mask))
        self.interpolation = params.get('interpolation',
                                        self.default_interpolation)

        # set hit finding thresholds (will be converted to an array later in init())
        self.threshold = params.get('threshold',
                                            self.default_global_threshold)
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

        self.hits_dtype = self.hits_dtype(self.near_samples)

    def init(self, source_name):
        super(WaveformHitFinder, self).init(source_name)
        wvfm_dset = self.data_manager.get_dset(self.wvfm_dset_name)
        
        # get convert sample rate to ns
        self.sample_rate = (resources['RunData'].lrs_ticks
                            / units.ns)

        # get waveform shape information
        #self.ntpc = wvfm_dset.dtype['samples'].shape[0]
        self.ntpc = 8
        self.ndet = wvfm_dset.dtype['samples'].shape[1]
        self.nsamples = wvfm_dset.dtype['samples'].shape[-1]

        # convert channel thresholds into an array
        threshold_array = np.zeros((self.ntpc, self.ndet, 1))
        for tpc in range(self.ntpc):
            for det in range(self.ndet):
                threshold_array[tpc,
                                det] = self.threshold[tpc][det]
        self.threshold = threshold_array
        #print("self.threshold", self.threshold)

        # create datasets and references
        self.data_manager.create_dset(self.hits_dset_name,
                                      dtype=self.hits_dtype)
        self.data_manager.create_ref(source_name, self.hits_dset_name)
        self.data_manager.create_ref(self.wvfm_dset_name, self.hits_dset_name)
        self.data_manager.set_attrs(self.hits_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    wvfm_dset=self.wvfm_dset_name,
                                    #t_ns_dset=self.t_ns_dset_name,
                                    near_samples=self.near_samples,
                                    thresholds=self.threshold,
                                    mask=self.mask,
                                    ntpc=self.ntpc,
                                    ndet=self.ndet,
                                    nsamples=self.nsamples,
                                    hit_level=self.hit_level
                                    )

    def run(self, source_name, source_slice, cache):
        super(WaveformHitFinder, self).run(source_name, source_slice, cache)
        wvfms = cache[self.wvfm_dset_name].reshape(cache[source_name].shape)[
            'samples']  # 1:1 relationship
        print("wvfms", wvfms.shape)

        #print("source_slice", source_slice)

        events = cache[source_name]
        wvfm_sn = events['sn']
        wvfm_det = np.broadcast_to(np.arange(wvfms.shape[-2]).reshape(1,1,-1), wvfms.shape[:-1])
        wvfm_d = np.diff(wvfms, axis=-1)
        print("wvfm_d", wvfm_d.shape)
        peaks = ((np.sign(wvfm_d[..., 1:]) * np.sign(wvfm_d[..., :-1]) < 0)
                 & (np.sign(np.diff(wvfm_d, axis=-1)) <= 0)
                 #& ~np.isin(np.arange(self.ndet), self.mask).reshape(1, 1, -1, 1)
                 #& np.all(~wvfms.mask, axis=-1, keepdims=True)
                )
        print("peaks 1", peaks, peaks.shape)
        peaks = np.where(peaks)  # tuple of (ev, tpc, det, index)
        peak_max = wvfms[..., 1:][peaks]  # waveform value at each peak
        print("peaks 2", peaks, len(peaks))
        print("peak_max", peak_max, len(peak_max))
