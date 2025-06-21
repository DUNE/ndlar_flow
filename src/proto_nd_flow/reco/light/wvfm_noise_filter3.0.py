import numpy as np
import logging
from scipy.interpolate import interp1d
from h5flow.core import H5FlowStage


class WaveformNoiseFilter(H5FlowStage):
    '''
        Applies a custom noise filter algorithm across specified waveform
        channels, looping on light event data.

        Coherent noise filter averages every ``modulo_param``-th sample from
        ``filter_samples[0]->filter_samples[1]``, e.g.
        ``avg[i] = 1/N * (sample[i] + sample[i+1*modulo_param] + sample[i+2*modulo_param] + ...)``.
        Then applies a subtraction across the waveform of
        ``filtered[i] = sample[i] - avg[i % modulo_param]``.

        Finally a pedestal subtraction is applied as::

            filtered[i] = filtered[i] - filtered[filter_samples[0]:filter_samples[1]].mean()

        Parameters:
         - ``fwvfm_dset_name`` : ``str``, required, output dataset path
         - ``wvfm_dset_name`` : ``str``, required, input dataset path for waveforms
         - ``filter_channels`` : ``list`` of ``int``, optional, list of channels to apply filter to (others are copied to output dataset)
         - ``filter_samples`` : ``list`` of ``int``, length of 2, min and max sample to use for filter
         - ``modulo_param`` : ``int``, repeat template after this number of samples (starting with ``filter_samples[0]``)

        ``wvfm_dset_name`` is required in the data cache.

        Example config::

            wvfm_noise_filter:
                classname: WaveformNoiseFilter
                requires:
                    - 'light/events'
                    - 'light/wvfm'
                params:
                    fwvfm_dset_name: 'light/fwvfm'
                    wvfm_dset_name: 'light/wvfm'
                    filter_channels: [ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63 ]
                    filter_samples: [ 0, 80 ]
                    modulo_param: 10
                    keep_noise: True
                    noise_dset_name: 'light/fwvfm_noise'

        Uses the same dtype as the input waveform dataset except with ``'samples'`` converted to floats.

    '''
    class_version = '1.0.0'

    default_filter_samples = (0, 80)
    default_modulo_param = 10
    default_keep_noise = False
    default_noise_dset_name = 'light/fwvfm_noise'

    def fwvfm_dtype(self, nadc, nchannels, nsamples): return np.dtype([('samples', 'f4', (nadc, nchannels, nsamples))])

    def __init__(self, **params):
        super(WaveformNoiseFilter, self).__init__(**params)

        self.fwvfm_dset_name = params.get('fwvfm_dset_name')
        self.wvfm_dset_name = params.get('wvfm_dset_name')
        self.filter_channels = np.array(params.get('filter_channels'))
        self.filter_samples = params.get('filter_samples', self.default_filter_samples)
        self.modulo_param = params.get('modulo_param', self.default_modulo_param)
        self.keep_noise = params.get('keep_noise', self.default_keep_noise)
        self.noise_dset_name = params.get('noise_dset_name', self.default_noise_dset_name)

    def init(self, source_name):
        super(WaveformNoiseFilter, self).init(source_name)

        # save all config info
        self.data_manager.set_attrs(self.fwvfm_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    source_dset=source_name,
                                    wvfm_dset=self.wvfm_dset_name,
                                    filter_channels=self.filter_channels,
                                    modulo_param=self.modulo_param
                                    )

        # then set up new datasets
        wvfm_dset = self.data_manager.get_dset(self.wvfm_dset_name)
        self.fwvfm_dtype = self.fwvfm_dtype(*wvfm_dset.dtype['samples'].shape)
        self.data_manager.create_dset(self.fwvfm_dset_name, dtype=self.fwvfm_dtype)
        self.data_manager.create_ref(source_name, self.fwvfm_dset_name)
        if self.keep_noise:
            self.data_manager.create_dset(self.noise_dset_name, dtype=wvfm_dset.dtype)
            self.data_manager.create_ref(source_name, self.noise_dset_name)

    def run(self, source_name, source_slice, cache):
        super(WaveformNoiseFilter, self).run(source_name, source_slice, cache)
        event_data = cache[source_name]
        wvfm_data = cache[self.wvfm_dset_name].reshape(event_data.shape).data  # don't worry about masked data since 1:1 references

        # flatten into individual waveforms
        wvfm_samples = wvfm_data['samples'].reshape(-1, wvfm_data['samples'].shape[-1])
        # truncate lowest 2-bits and convert to float
        wvfm_samples = (wvfm_samples - wvfm_samples % 4).astype(float)
        wvfm_mask = event_data['wvfm_valid'].astype(bool).flatten()
        wvfm_mask = wvfm_mask & \
            np.isin(np.tile(np.arange(wvfm_data['samples'].shape[-2]),wvfm_mask.shape[0] // wvfm_data['samples'].shape[-2]), self.filter_channels)

        # wrap subset of waveforms according to the modulo parameter
        subsamples = self.filter_samples[-1] - self.filter_samples[0]
        masked_wvfm = wvfm_samples[wvfm_mask, self.filter_samples[0]:self.filter_samples[-1]]
        #interpolate the masked waveform 
        interpolation_ticks = 4 #How many interpolation points is needed. The full interpolated array will be interpolation_ticks*1000 long
        original_ticks = masked_wvfm.shape[1]
        new_ticks = original_ticks*interpolation_ticks
        # Create original and new tick indices
        original_indices = np.linspace(0, original_ticks - 1, original_ticks) 
        new_indices = np.linspace(0, original_ticks - 1, new_ticks) 
        interp_func = interp1d(original_indices, masked_wvfm, kind='linear', axis=1, bounds_error=False, fill_value="extrapolate")
        interpolated_masked_wvfm = interp_func(new_indices)  # (nevent*8*64, 300)
        del interp_func
        expanded_modulo = int(self.modulo_param*interpolation_ticks) #Translate the input modulo parameter into the modulo ticks needed in the interpolated waveform
        expanded_subsamples = subsamples*interpolation_ticks #subsample value in the expanded regime 
        interpolated_masked_wvfm = interpolated_masked_wvfm[:, :expanded_subsamples - expanded_subsamples % expanded_modulo].reshape(-1, expanded_subsamples // expanded_modulo, expanded_modulo)
        # take "floating" mean to combine wrapped waveforms
        offsetted_interpolated_masked_wvfm =interpolated_masked_wvfm - np.mean(interpolated_masked_wvfm, axis=-1, keepdims=True)
        sample_length,segment_number,segment_length =interpolated_masked_wvfm.shape


        # Compute range for each segment (ptp across axis 2)
        ranges = np.ptp(offsetted_interpolated_masked_wvfm, axis=2)  # Shape [60,12]

        # Get the max range per segment per sample
        max_ranges = np.max(ranges, axis=1)  # Shape [60]

        # Compute the median excluding the largest value (per sample)
        sorted_ranges = np.sort(ranges, axis=1)  # Shape [60,12]
        median_ranges = np.median(sorted_ranges[:, :-1], axis=1)  # Shape [60] (excluding max value)

        # Determine which samples need removal of 2 segments
        remove_two = max_ranges >= 2 * median_ranges  # Shape [60] (True/False per sample)

        # Get indices of the two largest segments per sample (vectorized)
        largest_two_idx = np.argpartition(ranges, -2, axis=1)[:, -2:]  # Shape [60,2]

        # Generate indices for all 12 segments
        all_indices = np.arange(segment_number)

        # Mask: For each sample, remove the two largest segments if condition is met
        keep_mask = np.ones((sample_length,segment_number), dtype=bool)
        keep_mask[np.arange(sample_length)[:, None], largest_two_idx] = ~remove_two[:, None]  # Only remove if condition is met

        # Convert keep_mask into index arrays per batch sample
        filtered_indices = [all_indices[keep_mask[i]] for i in range(sample_length)]  # List of arrays, each with either 12 or 10 elements

        # Apply advanced indexing using `np.array` with different-sized rows (object array workaround)
        filtered_wvfm = np.array([offsetted_interpolated_masked_wvfm[i, filtered_indices[i], :] for i in range(sample_length)], dtype=object)

        # Compute mean across the kept segments
        interpolated_masked_wvfm = np.array([np.mean(filtered_wvfm[i], axis=0) for i in range(sample_length)])

        #interpolated_masked_wvfm =  np.mean( offsetted_interpolated_masked_wvfm, axis=1) #(6144,25)
        
        
        # extrapolate noise template across waveform
        noise = np.zeros_like(wvfm_samples)
        idcs = np.indices(wvfm_samples[wvfm_mask].shape)
        noise[wvfm_mask] = interpolated_masked_wvfm[idcs[0], (idcs[1]*interpolation_ticks) %(expanded_modulo)]
        # cast back into original shape
        noise = noise.reshape(wvfm_data['samples'].shape)
        # subtract noise from waveform
        fwvfm = np.empty(wvfm_data.shape, dtype=self.fwvfm_dtype)
        #print(wvfm_samples.reshape(noise.shape)[0][0][5][0:10])
        #print(noise[0][0][5][0:10])
        fwvfm['samples'] = wvfm_samples.reshape(noise.shape) - noise
        
        # subtract pedestal value
        fwvfm['samples'] = fwvfm['samples'] - fwvfm['samples'][..., self.filter_samples[0]:self.filter_samples[-1]].mean(axis=-1, keepdims=True)
        # reserve new data
        fwvfm_slice = self.data_manager.reserve_data(self.fwvfm_dset_name, source_slice)
        self.data_manager.write_data(self.fwvfm_dset_name, source_slice, fwvfm)

        # save references
        ref = np.c_[fwvfm_slice, fwvfm_slice]
        self.data_manager.write_ref(source_name, self.fwvfm_dset_name, ref)

        if self.keep_noise:
            # reserve new data
            noise_slice = self.data_manager.reserve_data(self.noise_dset_name, source_slice)
            noise_data = fwvfm.copy()
            noise_data['samples'] = noise
            self.data_manager.write_data(self.noise_dset_name, source_slice, noise_data)

            # save references
            self.data_manager.write_ref(source_name, self.noise_dset_name, ref)
