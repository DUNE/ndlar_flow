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
         - ``threshold``: ``dict`` of ``dict`` containing sets of ``tpc_index: {channel_index: threshold, ...}`` used for hit finding. A fixed global value can also be specified with a single ``float`` value
         - ``mask``: ``list`` of ``int``, detectors to ignore when finding hits

         Both ``wvfm_dset_name``, ``{wvfm_dset_name}/alignment``, and ``t_ns_dset_name`` are required in the cache.

         Requires RunData resource in workflow.

         ``hits`` datatype::

            id          u4,             unique identifier
            tpc         u1,             tpc index
            det         u1,             detector index
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

    default_hits_dset_name = 'light/hits'
    default_near_samples = 3
    default_interpolation = 256
    default_global_threshold = 2000
    default_mask = []

    def default_threshold(self, global_threshold):
        return defaultdict(lambda: defaultdict(lambda: global_threshold))

    def hits_dtype(self, near_samples):
        return np.dtype([
            ('id', 'u4'),
            ('tpc', 'u1'),
            ('det', 'u1'),
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

    def __init__(self, **params):
        super(WaveformHitFinder, self).__init__(**params)

        self.wvfm_dset_name = params.get('wvfm_dset_name')
        self.wvfm_align_dset_name = f'{self.wvfm_dset_name}/alignment'
        self.t_ns_dset_name = params.get('t_ns_dset_name')
        self.hits_dset_name = params.get('hits_dset_name',
                                         self.default_hits_dset_name)
        self.near_samples = params.get('near_samples',
                                       self.default_near_samples)
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
        self.ntpc = wvfm_dset.dtype['samples'].shape[0]
        self.ndet = wvfm_dset.dtype['samples'].shape[1]
        self.nsamples = wvfm_dset.dtype['samples'].shape[2]

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
                                    t_ns_dset=self.t_ns_dset_name,
                                    near_samples=self.near_samples,
                                    thresholds=self.threshold,
                                    mask=self.mask,
                                    ntpc=self.ntpc,
                                    ndet=self.ndet,
                                    nsamples=self.nsamples
                                    )

    def run(self, source_name, source_slice, cache):
        super(WaveformHitFinder, self).run(source_name, source_slice, cache)
        wvfms = cache[self.wvfm_dset_name].reshape(cache[source_name].shape)[
            'samples']  # 1:1 relationship
        wvfm_align = cache[self.wvfm_align_dset_name].reshape(cache[source_name].shape)
        t = cache[self.t_ns_dset_name].reshape(cache[source_name].shape)[
            't_ns']  # 1:1 relationship
        events = cache[source_name]
        t = ma.array(t, mask=~events['wvfm_valid'].astype(bool))
        wvfm_sn = events['sn']
        wvfm_det = np.broadcast_to(np.arange(wvfms.shape[-2]).reshape(1,1,-1), wvfms.shape[:-1])

        # find all peaks
        wvfm_d = np.diff(wvfms, axis=-1)
        peaks = ((np.sign(wvfm_d[..., 1:]) * np.sign(wvfm_d[..., :-1]) < 0)
                 & (np.sign(np.diff(wvfm_d, axis=-1)) <= 0)
                 & ~np.isin(np.arange(self.ndet), self.mask).reshape(1, 1, -1, 1)
                 & np.all(~wvfms.mask, axis=-1, keepdims=True))
        peaks = np.where(peaks)  # tuple of (ev, tpc, det, index)
        peak_max = wvfms[..., 1:][peaks]  # waveform value at each peak

        # apply threshold
        threshold_mask = peak_max >= self.threshold[peaks[1:-1]].ravel()

        if np.count_nonzero(threshold_mask):
            # hits are present in event, extract parameters
            peaks = tuple(p[threshold_mask].reshape(-1, 1) for p in peaks)
            peak_max = peak_max[threshold_mask]

            # get neighboring samples
            peak_sample_index = np.clip(peaks[-1].reshape(-1, 1)
                                        + np.arange(-self.near_samples + 1, self.near_samples + 2), 0, self.nsamples - 1)
            peak_samples = wvfms[peaks[:-1] + (peak_sample_index,)]
            peak_sum = np.sum(peak_samples.astype(int), axis=-1)

            # create hit spline
            peak_spline = scipy.interpolate.CubicSpline(
                np.arange(-self.near_samples, self.near_samples + 1),
                peak_samples, axis=-1, extrapolate=True)
            # calculate integral
            peak_sum_spline = peak_spline.integrate(-self.near_samples,
                                                    self.near_samples)
            # find max
            subsamples = np.linspace(-self.near_samples, self.near_samples,
                                     self.interpolation)
            peak_spline_subsamples = peak_spline(subsamples)
            peak_max_spline = np.max(peak_spline_subsamples, axis=-1)
            peak_ns_spline = np.expand_dims(np.take_along_axis(subsamples,
                                                               np.argmax(peak_spline_subsamples, axis=-1), axis=0), axis=-1) * self.sample_rate

            # project back to 0-crossing
            peak_spline_d = peak_spline.derivative(1)(subsamples)
            peak_rising_spline_samples = ma.array(subsamples
                                                  - peak_spline_subsamples / peak_spline_d,
                                                  mask=subsamples >= peak_ns_spline)
            rising_outlier_mask = self.find_outlier_mask(
                peak_rising_spline_samples)
            # calculate rising edge
            peak_rising_spline_samples = ma.array(peak_rising_spline_samples,
                                                  mask=rising_outlier_mask)
            peak_rising_spline = ma.mean(peak_rising_spline_samples, axis=-1,
                                         keepdims=True) * self.sample_rate
            peak_rising_err_spline = ma.std(peak_rising_spline_samples, axis=-1,
                                            keepdims=True) * self.sample_rate

            # calculate FWHM
            peak_lhm_spline_samples = ma.array(subsamples
                                               + (np.expand_dims(peak_max_spline, axis=-1) * 0.5 - peak_spline_subsamples) / peak_spline_d,
                                               mask=subsamples >= peak_ns_spline)
            peak_uhm_spline_samples = ma.array(subsamples
                                               + (np.expand_dims(peak_max_spline, axis=-1) * 0.5 - peak_spline_subsamples) / peak_spline_d,
                                               mask=subsamples <= peak_ns_spline)
            lhm_outlier_mask = self.find_outlier_mask(peak_lhm_spline_samples)
            uhm_outlier_mask = self.find_outlier_mask(peak_uhm_spline_samples)
            # calculate fwhm
            peak_lhm_spline_samples = ma.array(peak_lhm_spline_samples,
                                               mask=lhm_outlier_mask)
            peak_uhm_spline_samples = ma.array(peak_uhm_spline_samples,
                                               mask=uhm_outlier_mask)
            peak_fwhm_spline = (peak_uhm_spline_samples.mean(axis=-1)
                - peak_lhm_spline_samples.mean(axis=-1))

            hit_data = np.empty((len(peaks[-1])), dtype=self.hits_dtype)
            hit_data['tpc'] = peaks[1].ravel()
            hit_data['det'] = wvfm_det[peaks[:3]].ravel()
            hit_data['ns'] = wvfm_align['ns'][peaks[0]].ravel()
            hit_data['sample_idx'] = peaks[-1].ravel() + 1
            hit_data['busy_ns'] = (peaks[-1] + 1 - wvfm_align['sample_idx'][peaks[:3]]).ravel() * self.sample_rate
            hit_data['samples'] = peak_samples.reshape(
                -1, 2 * self.near_samples + 1)
            hit_data['sum'] = peak_sum.ravel()
            hit_data['max'] = peak_max.ravel()
            hit_data['sum_spline'] = peak_sum_spline.ravel()
            hit_data['max_spline'] = peak_max_spline.ravel()
            hit_data['ns_spline'] = peak_ns_spline.ravel()
            hit_data['rising_spline'] = peak_rising_spline.ravel()
            hit_data['rising_err_spline'] = peak_rising_err_spline.ravel()
            hit_data['fwhm_spline'] = peak_fwhm_spline.ravel()
        else:
            hit_data = np.empty((0,), dtype=self.hits_dtype)

        # save data
        hit_slice = self.data_manager.reserve_data(
            self.hits_dset_name, len(hit_data))
        if len(hit_data):
            hit_data['id'] = np.r_[hit_slice]
        self.data_manager.write_data(self.hits_dset_name, hit_slice, hit_data)

        # save references
        if len(hit_data):
            source_index = np.r_[source_slice].reshape(-1, 1, 1)
            source_index = np.broadcast_to(source_index, wvfms.shape[:-1])
            source_index = source_index[peaks[:-1]]

            ref = np.c_[source_index, hit_slice]
        else:
            ref = np.empty((0, 2))
        self.data_manager.write_ref(source_name, self.hits_dset_name, ref)
        self.data_manager.write_ref(
            self.wvfm_dset_name, self.hits_dset_name, ref)

    @staticmethod
    def find_outlier_mask(arr):
        '''
            Find outlier mask using median absolute deviation. An outlier is
            defined as::

                |arr - median(arr, axis=-1)| >
                    median(|arr - median(arr, axis=-1)|, axis=-1)

            :param arr: 2D masked array of points, ``shape: (N,M)``

            :returns: 2D boolean masked array of outliers, ``shape: (N,M)``, ``True == outlier``

        '''
        med = ma.median(arr, axis=-1, keepdims=True)
        mad = ma.median(np.abs(arr - med), axis=-1, keepdims=True)
        return np.abs(arr - med) > mad
