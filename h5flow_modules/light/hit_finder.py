import numpy as np
import numpy.ma as ma
import logging
from collections import defaultdict

from h5flow.core import H5FlowStage
from h5flow import H5FLOW_MPI

class WaveformHitFinder(H5FlowStage):
    '''
        Extracts "hits" from waveforms. A hit is defined as a local maxima above
        a defined threshold. Stores the nearest ±N samples around the hit,
        along with timing information and some summary information.

        Parameters:
         - ``wvfm_dset_name``: ``str``, path to input waveforms
         - ``t_ns_dset_name``: ``str``, path to corrected light PPS timestamps
         - ``hits_dset_name``: ``str``, path to output hits dataset
         - ``near_samples``: ``int``, number of neighboring samples to keep
         - ``busy_channel``: ``int``, channel to extract ADC busy signal (used for timing)
         - ``sample_rate``: ``float``, sample rate of waveform [ns]
         - ``channel_threshold``: ``dict`` of ``dict`` containing sets of ``adc_index: {channel_index: adc_threshold, ...}`` used for hit finding. A fixed global value can also be specified with a single ``float`` value
         - ``channel_mask``: ``list`` of ``int``, channels to ignore when finding hits

         Both ``wvfm_dset_name`` and ``t_ns_dset_name`` are required in the cache.

         ``hits`` datatype::

            id          u4,             unique identifier
            adc         u1,             adc index (1:1 w/ sn)
            sn          u4,             serial number of adc
            ch          u1,             channel id
            ns          f8,             PPS timestamp of peak [ns]
            busy_ns     f8,             timestamp of peak relative to busy rising edge [ns]
            samples     i2(2*near+1,),  sample adc value around peak
            sum         i2,             sum of sample adc values (out to nearest local minima or edge of near_samples)
            max         i2,             peak adc value
            sum_spline  f8,             integral of spline around peak (out to nearest local minima or edge of near_samples)
            max_spline  f8,             maximum of spline around peak
            ns_spline   f8,             offset from PPS timestamp of peak for maximum of spline [ns]

    '''

    default_hits_dset_name = 'light/hits'
    default_near_samples = 3
    default_busy_channel = 0
    default_sample_rate = 10
    default_global_threshold = 500
    default_channel_threshold = lambda global_threshold : defaultdict(lambda : defaultdict(lambda : global_threshold))
    default_channel_mask = []

    hits_dtype = lambda near_samples: np.dtype([
        ('id', 'u4'),
        ('adc', 'u1'),
        ('sn', 'u4'),
        ('ch', 'u1'),
        ('ns', 'f8'),
        ('busy_ns', 'f8'),
        ('samples', f'i2({near_samples},)'),
        ('sum', 'i2'),
        ('max', 'i2'),
        ('sum_spline', 'f8'),
        ('max_spline', 'f8'),
        ('ns_spline', 'f8')
        ])

    def __init__(self, **params):
        super(WaveformHitFinder,self).__init__(**params)

        self.wvfm_dset_name = params.get('wvfm_dset_name')
        self.t_ns_dset_name = params.get('t_ns_dset_name')
        self.hits_dset_name = params.get('hits_dset_name', self.default_hits_dset_name)
        self.near_samples = params.get('near_samples', self.default_near_samples)
        self.busy_channel = params.get('busy_channel', self.default_busy_channel)
        self.sample_rate = params.get('sample_rate', self.default_sample_rate)
        self.channel_mask = np.array(params.get('channel_mask', self.default_channel_mask))

        # set channel hit finding thresholds (will be converted to an array later in init())
        self.channel_threshold = params.get('channel_threshold', self.default_global_threshold)
        if isinstance(self.channel_threshold, int) or isinstance(self.channel_threshold, float):
            # if a global threshold is specified, use the default generator
            self.channel_threshold = self.default_channel_threshold(self.channel_threshold)
        elif isinstance(self.channel_threshold, dict):
            # otherwise convert to a defaultdict
            new_dict = self.default_channel_threshold(self.default_global_threshold)
            for key,subdict in self.channel_threshold.items():
                for subkey,subval in subdict.items():
                    new_dict[key][subkey] = subval
            self.channel_threshold = new_dict

        self.hits_dtype = self.hits_dtype(self.near_samples)

    def init(self, source_name):
        wvfm_dset = self.data_manager.get_dset(self.wvfm_dset_name)

        # get waveform shape information
        self.nadc = wvfm_dset.dtype['samples'].shape[0]
        self.nchan = wvfm_dset.dtype['samples'].shape[1]
        self.nsamples = wvfm_dset.dtype['samples'].shape[2]

        # convert channel thresholds into an array
        threshold_array = np.zeros((self.nadc,self.nchan,1))
        for adc in range(self.nadc):
            for channel in range(self.nchan):
                threshold_array[adc,channel] = self.channel_threshold[adc][channel]
        self.channel_threshold = threshold_array

        # create datasets and references
        self.data_manager.create_dset(self.hits_dset_name, dtype=self.hits.dtype)
        self.data_manager.create_ref(source_name, self.hits_dset_name)
        self.data_manager.create_ref(self.wvfm_dset_name, self.hits_dset_name)
        self.data_manager.set_attrs(self.hits_dset_name,
            classname=self.classname,
            class_version=self.class_version,
            wvfm_dset=self.wvfm_dset_name,
            t_ns_dset=self.t_ns_dset_name,
            near_samples=self.near_samples,
            busy_channel=self.busy_channel,
            sample_rate=self.sample_rate,
            channel_thresholds=self.channel_threshold,
            channel_mask=self.channel_mask
        )

    def run(self, source_name, source_slice, cache):
        wvfms = cache[self.wvfm_dset_name].reshape(cache[source_name].shape)['samples'] # 1:1 relationship
        t = cache[self.t_ns_dset_name].reshape(cache[source_name].shape)['t_ns'] # 1:1 relationship
        events = cache[source_name]
        wvfm_valid = events['wvfm_valid'].astype(bool)
        wvfm_sn = events['sn']
        wvfm_ch = events['ch']
        wvfm_ns = t.reshape(-1,self.nadc,1,1)

        wvfms.mask = wvfms.mask | np.expand_dims(~wvfm_valid, axis=-1)

        # find all peaks
        wvfm_d = np.diff(wvfms, axis=-1)
        peaks = wvfm_d[...,1:]*wvfm_d[...,:-1] < 0 & # derivative crosses 0 \
            np.diff(wvfm_d, axis=-1) <= 0 & # negative curvature \
            ~np.isin(np.arange(self.nchan), self.channel_mask).reshape(1,1,-1,1) & # not masked\
            np.all(~wvfms.mask, axis=-1, keepdims=True)
        peaks = np.where(peaks) # tuple of (ev, adc, ch, index)
        peak_max = wvfms[...,1:-1][peaks] # waveform value at each peak

        # apply threshold
        threshold_mask = (peak_max >= self.channel_threshold[peaks[1:-1]])

        if np.count_nonzero(threshold_mask):
            # hits are present in event, extract parameters
            peaks = tuple(p[threshold_mask].reshape(-1,1) for p in peaks)
            peak_max = peak_max[threshold_mask]
            peak_ns = t[peaks[:2]] + peaks[-1] * self.sample_rate

            # get neighboring samples
            peak_sample_index = np.clip(
                peaks[-1].reshape(-1,1) + np.arange(-self.near_samples+1,self.near_samples+2),
                0, self.nsamples-1)
            peak_samples = wvfms[peaks[:-1]+(peak_sample_index,)]
            peak_sum = np.sum(peak_samples, axis=-1)

            # create hit spline
            peak_spline = scipy.interpolate.CubicSpline(
                np.arange(-self.near_samples,self.near_samples+1), peak_samples,
                axis=-1, extrapolate=True)
            peak_d = peak_spline.derivative(1)

            peak_extrema = peak_d.roots()
            closest_extrema = np.argmin(np.abs(peak_extrema)).reshape(-1,1) # use extrema closest to original

            peak_sum_spline = peak_spline.integrate(0,peak_samples.shape[-1])
            peak_ns_spline = peak_ns + np.take_along_axis(peak_extrema, closest_extrema, -1) * self.sample_rate
            peak_max_spline = np.max(peak_spline(np.linspace(-self.near_samples+1,self.near_samples+2,100)), axis=-1)

            # find busy signal rising edge
            busy_sig = wvfm[...,self.busy_channel,:]
            busy_d = wvfm_d[...,self.busy_channel,:]
            rising_edge = np.argmax(busy_d,axis=-1)
            # project to 0-crossing for sub-sample resolution
            rising_edge = rising_edge - np.take_along_axis(busy_sig,rising_edge,axis=-1) / np.take_along_axis(busy_d,rising_edge,axis=-1)
            peak_busy_ns = (peaks[-1] - rising_edge[peaks[:2]]) * self.sample_rate

            hit_data = np.empty((len(peaks[-1])), dtype=self.hits_dtype)
            hit_data['sn'] = wvfm_sn[peaks[:2]]
            hit_data['ch'] = wvfm_ch[peaks[:3]]
            hit_data['ns'] = peak_ns
            hit_data['busy_ns'] = peak_busy_ns
            hit_data['samples'] = peak_samples
            hit_data['sum'] = peak_sum
            hit_data['max'] = peak_max
            hit_data['sum_spline'] = peak_sum_spline
            hit_data['max_spline'] = peak_max_spline
            hit_data['ns_spline'] = peak_ns_spline
        else:
            hit_data = np.empty((0,), dtype=self.hits_dtype)

        # save data
        hit_slice = self.data_manager.reserve_data(self.hits_dset_name, len(hit_data))
        if len(hit_data):
            hit_data['id'] = np.r_[hit_slice]
        self.data_manager.write_data(self.hits_dset_name, hits_slice, hit_data)

        # save references
        if len(hit_data):
            source_index = np.r_[source_slice].reshape(-1,1,1)
            source_index = np.broadcast_to(source_index, wvfms.shape[:-1])
            source_index = source_index[peaks[:-1]]

            ref = np.c_[source_index, hit_data['id']]
        else:
            ref = np.empty((0,2))
        self.data_manager.write_ref(source_name, self.hits_dset_name, ref)
        self.data_manager.write_ref(self.wvfm_dset_name, self.hits_dset_name, ref)
