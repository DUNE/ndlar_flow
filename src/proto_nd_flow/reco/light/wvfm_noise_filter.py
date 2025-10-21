import numpy as np
import logging

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
    default_segment_size = 25
    default_num_segment = 40
    default_num_means = 4
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
        self.segment_size = params.get("segment_size", self.default_segment_size)
        self.num_segment = params.get("num_segment", self.default_num_segment)
        self.num_means = params.get("num_means", self.default_num_means)

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

    def min_range_baseline(self, array, segment_size, num_segments, num_means=4):

        # Define start and end indices for segments
        indices = np.arange(num_segments + 1) * segment_size
        start_indices, end_indices = indices[:-1], indices[1:]
        # Generate index array for advanced indexing
        segment_range = np.arange(segment_size)
        index_array = start_indices[:, None] + segment_range

        # Extract data from segments using indexing
        sliced_data = array[..., index_array]

        # Compute range (peak-to-peak difference) and mean for each segment
        ranges = np.abs(np.ptp(sliced_data, axis=-1))
        means = np.mean(sliced_data, axis=-1)

        # Mask zero ranges
        mask_zero = (ranges != 0)
        ranges = np.where(mask_zero, ranges, np.nan)
        means = np.where(mask_zero, means, np.nan)

        # Find the ordering of the segments based on the smallest range
        smallest_ordering = np.argsort(ranges, axis=-1)

        # Sort means according to the ordering of smallest ranges
        sorted_means = np.take_along_axis(means, smallest_ordering, axis=-1)

        # Compute the average of the 2nd, 3rd, and 4th smallest means
        average_mean = np.mean(sorted_means[..., 1:num_means], axis=-1)

        # calculate RMS for the ranges of the smallest range segments
        rms = np.sqrt(np.mean(np.square(np.take_along_axis(ranges, smallest_ordering[..., :num_means], axis=-1)), axis=-1))

        return average_mean, rms


    def run(self, source_name, source_slice, cache):
        super(WaveformNoiseFilter, self).run(source_name, source_slice, cache)

        event_data = cache[source_name]
        wvfm_data = cache[self.wvfm_dset_name].reshape(event_data.shape).data  # don't worry about masked data since 1:1 references

        # flatten into individual waveforms
        wvfm_samples = wvfm_data['samples'].reshape(-1, wvfm_data['samples'].shape[-1])
        # truncate lowest 2-bits and convert to float
        wvfm_samples = (wvfm_samples - wvfm_samples % 4).astype(float)

        # # subtract noise from waveform
        fwvfm = np.empty(wvfm_data.shape, dtype=self.fwvfm_dtype)

        # subtract pedestal value
        pedestal, rms = self.min_range_baseline(wvfm_data['samples'], self.segment_size, self.num_segment, self.num_means)
        fwvfm['samples'] = wvfm_data['samples']  - pedestal[..., np.newaxis]

        # reserve new data
        fwvfm_slice = self.data_manager.reserve_data(self.fwvfm_dset_name, source_slice)
        self.data_manager.write_data(self.fwvfm_dset_name, source_slice, fwvfm)

        # save references
        ref = np.c_[fwvfm_slice, fwvfm_slice]
        self.data_manager.write_ref(source_name, self.fwvfm_dset_name, ref)
