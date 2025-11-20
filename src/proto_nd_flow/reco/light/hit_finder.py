from functools import total_ordering
import numpy as np
import numpy.ma as ma
from collections import defaultdict
import scipy.interpolate
from scipy.ndimage import uniform_filter1d

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
         - ``near_samples``:   ``int``, number of neighboring samples to keep
         - ``hit_level``:      ``str``, "sipm" or "sum" hit finder (defines variable names)
         - ``mad_factor``:     ``float``, factor of median abs dev used to define threshold under which noise width is taken
         - ``noise_factor``:   ``float``, factor of noise width used to define threshold over which hit finder is run
         - ``n_bins_rolled``:  ``int``, number of bins over which the rolling threshold of the hit finder is defined
         - ``rt_sqrt_factor``: ``float``, factor used to scale the statistical contribution to the rolling threshold
         - ``pe_weight``:      ``float``, weight applied to the PEs in rolling threshold statistical component
         - ``rising_edge``:    ``bool``, True => hit finder tags first bin over rolling threshold as the hit
         - ``local_maxima``:   ``bool``, True => uses 5 sample window after rising edge, tags hits as argmax of those samples (otherwise uses derivative based method)
         - ``prompt_window``:  ``float``, Prompt light window in ns (fprompt caluclation input for PSD)
         - ``long_window``:    ``float``, Long light window in ns (fprompt caluclation input for PSD)
         - ``tick_duration``:  ``float``, Duration of ticks in ADC sampling
         - ``threshold``:      ``dict`` of ``dict`` containing sets of ``tpc_index: {channel_index: threshold, ...}`` used for hit finding. A fixed global value can also be specified with a single ``float`` value
         - ``mask``: ``list`` of ``int``, detectors to ignore when finding hits

         Both ``wvfm_dset_name``, ``{wvfm_dset_name}/alignment``, and ``t_ns_dset_name`` are required in the cache.

         Requires RunData resource in workflow.

         ``hits`` datatype::

            id          u4,             unique identifier
            tpc/adc     u1,             tpc/adc index (for sum_tpc_hit/sum_hit/sipm_hit)
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
            fprompt     f4,             prompt light fraction as proxy for singlet fraction in LAr scintillation
            integral    f4,             integral of a pulse

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
        if self.hit_level=="sum_tpc":
            return np.dtype([
                ('id', 'u4'),
                ('tpc', 'u1'),
                ('trap_type', 'u1'),
                #('boundary', 'f4', (2,3)),
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
                ('fwhm_spline', 'f4'),
                ('integral', 'f4'),
                ('fprompt', 'f4'),
                ('tot', 'f4'),
                ('tot_upper', 'f4'),
            ])
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
                ('fwhm_spline', 'f4'),
                ('integral', 'f4'),
                ('fprompt', 'f4'),
                ('tot', 'f4'),
                ('tot_upper', 'f4'),
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


    # function to calculate the prompt light fraction in a vectorized way
    def calculate_fprompt(self, summed_wvfm, interactions, prompt_window_ns, long_window_ns, tick_duration_ns):
        # Define regions
        prompt_bins = int(np.ceil(prompt_window_ns / tick_duration_ns))
        total_bins = int(np.ceil(long_window_ns / tick_duration_ns))
        # Take integrals (first 3 dims of interactions)
        prompt_int = np.zeros(interactions.shape[:-1])
        total_int = np.zeros(interactions.shape[:-1])
        # Loop over each event
        for i in range(interactions.shape[0]):
            # Loop over each TPC
            for j in range(interactions.shape[1]):
                # Loop over each trap type
                for k in range(interactions.shape[2]):
                    # Calculate the prompt and total integrals
                    t0_bin = np.argmax(interactions[i, j, k]) - 5
                    end_prompt = t0_bin + prompt_bins
                    end_total = t0_bin + total_bins
                    prompt_int[i, j, k] = np.sum(summed_wvfm[i, j, k, t0_bin:end_prompt])
                    total_int[i, j, k] = np.sum(summed_wvfm[i, j, k, t0_bin:end_total])
        # Calculate fprompt
        with np.errstate(divide='ignore', invalid='ignore'):
            fprompt = np.where(
            (total_int > 0) & (prompt_int > 0) & ~np.isnan(prompt_int) & ~np.isnan(total_int),
            np.divide(prompt_int, total_int),
            np.nan
            )
        return total_int, fprompt


    def get_noise_threshold(self, wvfms, n_mad_factor):
        # Initialize median and MAD
        median = np.ma.median(wvfms, axis=-1)
        mad = np.ma.median(np.abs(wvfms - median[..., np.newaxis]), axis=-1)
        # identify outliers in the waveform
        mad_factor = n_mad_factor * mad
        noise_mask = np.abs(wvfms - median[..., np.newaxis]) < mad_factor[..., np.newaxis]
        # set non mask values to nan
        noise_samples = np.where(noise_mask, wvfms, np.nan)
        # calculate noise as stddev of noise_samples
        noise = np.where(np.nansum(noise_samples, axis=-1) != 0,
                         np.nanstd(noise_samples, axis=-1),
                         np.nan)
        return  noise


    # gets ToT for threshold crossing pairs of samples (incl hysterisis)
    def pair_runs_with_peaks(self, first_bins_over_noise: np.ndarray,
                            first_bins_under_noise: np.ndarray,
                            peak_bins: np.ndarray):

        starts_raw = first_bins_over_noise
        ends_raw   = first_bins_under_noise

        # Arm 'end' events only after we have ever seen a start
        seen_start = np.maximum.accumulate(starts_raw, axis=-1)
        ends_armed = ends_raw & seen_start

        # Cumsums over time
        cs = np.cumsum(starts_raw, axis=-1, dtype=np.int32)
        ce = np.cumsum(ends_armed, axis=-1, dtype=np.int32)

        # 1) VALID STARTS: only the first start after the most recent armed end
        max_ce = np.maximum.accumulate(ce, axis=-1)
        starts_since_last_end = cs - max_ce
        valid_starts = starts_raw & (starts_since_last_end == 1)

        # 2) VALID ENDS: only the first armed end after the most recent VALID start
        ce_at_valid_start = np.where(valid_starts, ce, -1)
        ce_anchor = np.maximum.accumulate(ce_at_valid_start, axis=-1)
        ends_since_last_valid_start = ce - ce_anchor
        valid_ends = ends_armed & (ends_since_last_valid_start == 1)

        # Shapes and index grid
        T = starts_raw.shape[-1]
        idx = np.arange(T, dtype=np.int32).reshape((1,) * (starts_raw.ndim - 1) + (-1,))

        # Nearest future END index for every t
        inf = T + 1
        end_pos = np.where(valid_ends, idx, inf)
        next_end_idx = np.minimum.accumulate(end_pos[..., ::-1], axis=-1)[..., ::-1]
        has_future_end = next_end_idx < inf

        # Count peaks between start and its matched end: (start, end]
        peaks_cum = np.cumsum(peak_bins.astype(np.int32), axis=-1)
        peaks_at_end   = np.take_along_axis(peaks_cum, np.minimum(next_end_idx, T - 1), axis=-1)
        peaks_at_start = np.take_along_axis(peaks_cum, idx, axis=-1)
        peaks_between = peaks_at_end - peaks_at_start

        # Keep starts only if there is a future end and at least one peak in between
        keep_start = valid_starts & has_future_end & (peaks_between > 0)

        # Keep exactly the end that matches each kept start (optional mask you already had)
        ends_kept = np.zeros_like(valid_ends, dtype=bool)
        start_idxs = np.where(keep_start)
        if start_idxs[0].size:
            end_for_kept = next_end_idx[start_idxs]
            sel_e = (end_for_kept >= 0) & (end_for_kept < T)
            if np.any(sel_e):
                idx_tuple_e = tuple(ax[sel_e] for ax in start_idxs[:-1]) + (end_for_kept[sel_e],)
                ends_kept[idx_tuple_e] = True

        # New bit: nearest future PEAK index for every t
        peak_pos = np.where(peak_bins, idx, inf)
        next_peak_idx = np.minimum.accumulate(peak_pos[..., ::-1], axis=-1)[..., ::-1]

        # Duration values placed at the PEAK indices
        duration_bins = np.zeros_like(valid_starts, dtype=np.int32)
        if start_idxs[0].size:
            # indices for each kept start
            peak_for_kept = next_peak_idx[start_idxs]         # first peak after start
            end_for_kept  = next_end_idx[start_idxs]          # matching end for that start
            start_t       = start_idxs[-1]                    # start indices along time axis

            # sanity selection
            sel = (peak_for_kept >= 0) & (peak_for_kept < T) & (end_for_kept >= 0) & (end_for_kept < T)
            if np.any(sel):
                # duration = end - start, but write it at the peak index
                dur_vals = (end_for_kept - start_t)[sel]
                idx_tuple = tuple(ax[sel] for ax in start_idxs[:-1]) + (peak_for_kept[sel],)
                duration_bins[idx_tuple] = dur_vals

        return duration_bins


    def peak_finder(self, wvfm, noise,
                    n_noise_factor,
                    n_bins_rolled,
                    n_sqrt_rt_factor,
                    pe_weight):

        # height = flat threshold over noise (n*sigma)
        height = n_noise_factor * noise[..., np.newaxis] * np.ones(wvfm.shape[-1])
        height_below = (n_noise_factor-1) * noise[..., np.newaxis] * np.ones(wvfm.shape[-1])
        uheight = 2*n_noise_factor * noise[..., np.newaxis] * np.ones(wvfm.shape[-1])
        uheight_below = (2*n_noise_factor-1) * noise[..., np.newaxis] * np.ones(wvfm.shape[-1])

        # dynamic_threshold = rolling threshold of previous 5 bins + n*sqrt(rolling threshold)
        wvfm_rolled = np.roll(wvfm, n_bins_rolled)  # kept as in original
        rolling_average = uniform_filter1d(wvfm_rolled, size=n_bins_rolled)
        sqrt_rolling_average = np.sqrt(np.abs(rolling_average) * pe_weight**2)
        sqrt_rolling_average[sqrt_rolling_average == 0] = 1
        dynamic_threshold = rolling_average + n_sqrt_rt_factor * sqrt_rolling_average

        # find bins over noise floor
        bins_over_noise_threshold = (wvfm > height)
        first_bins_over_noise = bins_over_noise_threshold.copy()
        first_bins_over_noise[..., 1:] &= ~bins_over_noise_threshold[..., :-1]
        # find bins under noise floor - hysteresis
        bins_under_noise_threshold = (wvfm <= height_below)
        first_bins_under_noise = bins_under_noise_threshold.copy()
        first_bins_under_noise[..., 1:] &= ~bins_under_noise_threshold[..., :-1]

        # find bins over upper threshold
        bins_over_upper_threshold = (wvfm > uheight)
        first_bins_over_upper = bins_over_upper_threshold.copy()
        first_bins_over_upper[..., 1:] &= ~bins_over_upper_threshold[..., :-1]
        # find bins under upper threshold - hysteresis
        bins_under_upper_threshold = (wvfm <= uheight_below)
        first_bins_under_upper = bins_under_upper_threshold.copy()
        first_bins_under_upper[..., 1:] &= ~bins_under_upper_threshold[..., :-1]

        # find bins over dynamic threshold AND noise floor
        bins_over_dynamic_threshold = (wvfm > dynamic_threshold)
        bins_over_thresholds = bins_over_noise_threshold & bins_over_dynamic_threshold

        # rising edge of the combined thresholds
        first_bins_over = bins_over_thresholds.copy()
        first_bins_over[..., 1:] &= ~bins_over_thresholds[..., :-1]

        # check 5 bins after first_bins_over and add argmax
        peak_bins = np.zeros_like(wvfm, dtype=bool)
        first_bins_indices = np.where(first_bins_over)
        for idx in zip(*first_bins_indices):
            start_idx = idx[-1]
            end_idx = min(start_idx + 5, wvfm.shape[-1])
            peak_bin = np.argmax(wvfm[idx[:-1] + (slice(start_idx, end_idx),)])
            peak_bins[idx[:-1] + (start_idx + peak_bin,)] = True

        # noise threshold tot
        tot = self.pair_runs_with_peaks(first_bins_over_noise, first_bins_under_noise, peak_bins)
        # upper threshold tot
        tout = self.pair_runs_with_peaks(first_bins_over_upper, first_bins_under_upper, peak_bins)

        return peak_bins, tot, tout


    def __init__(self, **params):
        super(WaveformHitFinder, self).__init__(**params)
        self.wvfm_dset_name = params.get('wvfm_dset_name')
        self.wvfm_align_dset_name = f'{self.wvfm_dset_name}/alignment'
        self.t_ns_dset_name = params.get('t_ns_dset_name')
        self.hits_dset_name = params.get('hits_dset_name',
                                         self.default_hits_dset_name)
        self.near_samples = params.get('near_samples',
                                       self.default_near_samples)
        self.hit_level = params.get('hit_level')
        self.mad_factor = params.get('mad_factor')
        self.noise_factor = params.get('noise_factor')
        self.n_bins_rolled = params.get('n_bins_rolled')
        self.rt_sqrt_factor = params.get('rt_sqrt_factor')
        self.pe_weight = params.get('pe_weight')
        self.rising_edge = params.get('rising_edge')
        self.local_maxima = params.get('local_maxima')
        self.prompt_window = params.get('prompt_window')
        self.long_window = params.get('long_window')
        self.tick_duration = params.get('tick_duration')
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
                                    mask=self.mask,
                                    ntpc=self.ntpc,
                                    ndet=self.ndet,
                                    nsamples=self.nsamples,
                                    hit_level=self.hit_level
                                    )

        # For high channel counts (e.g. ND-LAr) we can't store the thresholds as
        # an attribute; it exceeds the max size that the hdf5 library allows.
        # TODO: Move to separate dataset? For now retain attribute for 2x2/FSD
        # analyzers.
        if self.ntpc * self.ndet <= 512:
            self.data_manager.set_attrs(self.hits_dset_name,
                                        thresholds=self.threshold)


    def run(self, source_name, source_slice, cache):
        super(WaveformHitFinder, self).run(source_name, source_slice, cache)

        wvfms = cache[self.wvfm_dset_name].reshape(cache[source_name].shape)[
            'samples']  # 1:1 relationship
        wvfm_align = cache[self.wvfm_align_dset_name].reshape(cache[source_name].shape)

        wvfm_det = np.broadcast_to(np.arange(wvfms.shape[-2]).reshape(1,1,-1), wvfms.shape[:-1])

        noise = self.get_noise_threshold(wvfms, self.mad_factor)

        peaks_found, tot, tot_upper = self.peak_finder(wvfms, noise,
                                      self.noise_factor,
                                      self.n_bins_rolled,
                                      self.rt_sqrt_factor,
                                      self.pe_weight)

        t0_bin = np.argmax(peaks_found, axis=-1)

        t0_mask = np.any(peaks_found, axis=-1)

        t0_bin[~t0_mask] = -1

        peaks = np.where(peaks_found)

        peak_max = wvfms[peaks]  # waveform value at each peak
        peak_tot = tot[peaks]
        peak_tot_upper = tot_upper[peaks]

        # For threshold, we need adc and channel indices only (peaks[1] and peaks[2])
        # self.threshold has shape (ntpc, ndet, 1)
        threshold_mask = peak_max >= self.threshold[peaks[1], peaks[2], 0]

        if self.hit_level=="sum_tpc" or self.hit_level=="sum":
            integrals, fprompts = self.calculate_fprompt(wvfms, peaks_found,
                                                         self.prompt_window,
                                                         self.long_window,
                                                         self.tick_duration)
            # match integrals and fprompts to peaks
            integrals = integrals[peaks[:-1]][threshold_mask]
            fprompts = fprompts[peaks[:-1]][threshold_mask]

        if np.count_nonzero(threshold_mask):
            # hits are present in event, extract parameters
            peaks = tuple(p[threshold_mask].reshape(-1, 1) for p in peaks)
            peak_max = peak_max[threshold_mask]
            peak_tot = peak_tot[threshold_mask]
            peak_tot_upper = peak_tot_upper[threshold_mask]
            # get neighboring samples
            peak_sample_index = np.clip(peaks[-1].reshape(-1, 1)
                                        + np.arange(-self.near_samples + 1, self.near_samples + 2), 0, self.nsamples - 1)
            peak_samples = wvfms[peaks[:-1] + (peak_sample_index,)]
            peak_sum = np.sum(peak_samples, axis=-1)
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

            peak_spline_d = np.where(peak_spline_d == 0, np.nan,peak_spline_d)

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

            if self.hit_level=="sum_tpc":
                hit_data['tpc'] = peaks[1].ravel()
                hit_data['trap_type'] = wvfm_det[peaks[:3]].ravel()
                hit_data['integral'] = integrals.ravel()
                hit_data['fprompt'] = fprompts.ravel()
                hit_data['tot'] = peak_tot.ravel()
                hit_data['tot_upper'] = peak_tot_upper.ravel()

            elif self.hit_level=="sum":
                hit_data['tpc'] = peaks[1].ravel()
                hit_data['det'] = wvfm_det[peaks[:3]].ravel()
                hit_data['boundary'] = [np.array(resources['Geometry'].det_bounds[(tpc,det)][0]) for tpc, det in zip(peaks[1].ravel(),wvfm_det[peaks[:3]].ravel())]
                hit_data['integral'] = integrals.ravel()
                hit_data['fprompt'] = fprompts.ravel()
                hit_data['tot'] = peak_tot.ravel()
                hit_data['tot_upper'] = peak_tot_upper.ravel()

            elif self.hit_level=="sipm":
                hit_data['adc'] = peaks[1].ravel()
                hit_data['chan'] = wvfm_det[peaks[:3]].ravel()
                hit_data['pos'] = [np.array(resources['Geometry'].sipm_abs_pos[(adc,chan)][0]) for adc, chan in zip(peaks[1].ravel(),wvfm_det[peaks[:3]].ravel())]

            hit_data['ns'] = wvfm_align['ns'][peaks[0]].ravel()
            hit_data['sample_idx'] = peaks[-1].ravel()


            # =================================================================
            # 2022-05-17 kvtsang
            # -----------------------------------------------------------------
            # The original version was designed for swvfm/alignment, which
            # assumes a shape of (n_batch, n_tpc, n_ch)
            #
            # For deconv/alignment, shape = (n_batch, n_tpc)
            # Here is a simple fix to expand the dim
            # =================================================================
            align_sample_idx = wvfm_align['sample_idx']
            if align_sample_idx.ndim == 2:
                target_shape = wvfms.shape[:-1] #(n_batch, n_tpc, n_ch)
                n_ch = target_shape[-1]
                align_sample_idx = np.reshape(
                    np.repeat(align_sample_idx, n_ch), target_shape
                )

            hit_data['busy_ns'] = (
                (peaks[-1] - align_sample_idx[peaks[:3]]).ravel()
                * self.sample_rate
            )
            hit_data['samples'] = peak_samples.reshape(-1, 2 * self.near_samples + 1)
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
