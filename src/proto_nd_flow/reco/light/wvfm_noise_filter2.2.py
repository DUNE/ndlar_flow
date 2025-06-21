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
        
        
        
        '''
        # Step 1: Compute the range for each group (max - min across 25 ticks)
        group_ranges = np.ptp(interpolated_masked_wvfm, axis=-1)  # Shape: (6144, 12)

        # Step 2: Find indices of the 2 largest and 2 smallest ranges per event
        sorted_indices = np.argsort(group_ranges, axis=1)  # Sort group indices by range
        to_remove = np.hstack((sorted_indices[:, :2], sorted_indices[:, -2:]))  # First 2 and last 2 indices

        # Step 3: Create a mask to keep only the 8 middle-range groups
        mask = np.ones_like(group_ranges, dtype=bool)
        mask[np.arange(mask.shape[0])[:, None], to_remove] = False  # Mark groups to remove

        # Step 4: Compute the mean across remaining 8 groups and all 25 ticks
        filtered_wvfm = np.where(mask[:, :, None], interpolated_masked_wvfm, np.nan)  # Set removed groups to NaN
        offset = np.nanmean(filtered_wvfm, axis=(1, 2), keepdims=True)  # Shape: (6144, 1, 1)

        # Step 5: Subtract the computed offset from all values
        interpolated_masked_wvfm -= offset
        
        
        '''
        # Step 1: Compute the range for each group (max - min across 25 ticks)
        group_ranges = np.ptp(interpolated_masked_wvfm, axis=-1)  # Shape: (6144, 12)

        # Step 2: Find indices of the 2 largest ranges per event (removing only the largest two)
        sorted_indices = np.argsort(group_ranges, axis=1)  # Sort group indices by range
        to_remove = sorted_indices[:, -2:]  # Select only the last two (largest range groups)

        # Step 3: Create a mask to keep only the remaining 10 groups
        mask = np.ones_like(group_ranges, dtype=bool)
        mask[np.arange(mask.shape[0])[:, None], to_remove] = False  # Mark groups to remove

        # Expand mask for broadcasting with (44, 12, 25)
        masked_interpolated_wvfm = np.where(mask[:, :, None], interpolated_masked_wvfm, np.nan)

        # Step 4: Compute mean across the remaining 10 groups and all 25 ticks
        offset = np.nanmean(masked_interpolated_wvfm, axis=(1, 2), keepdims=True)  # Shape: (44, 1, 1)

        # Step 5: Subtract offset from all values

        filtered_wvfm = masked_interpolated_wvfm - offset
        filtered_wvfm = np.nanmean(filtered_wvfm, axis=1)  # Mean over groups → (44, 25)
        
        '''
        # take "floating" mean to combine wrapped waveforms
        offset = np.mean(interpolated_masked_wvfm, axis=-1, keepdims=True)
        
        interpolated_masked_wvfm = np.mean(interpolated_masked_wvfm - offset, axis=1) #(6144,25)
        '''
        # extrapolate noise template across waveform
        noise = np.zeros_like(wvfm_samples)
        idcs = np.indices(wvfm_samples[wvfm_mask].shape)
        noise[wvfm_mask] = filtered_wvfm[idcs[0], (idcs[1]*interpolation_ticks) %(expanded_modulo)]
        # cast back into original shape
        noise = noise.reshape(wvfm_data['samples'].shape)
        # subtract noise from waveformls
        fwvfm = np.empty(wvfm_data.shape, dtype=self.fwvfm_dtype)

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
