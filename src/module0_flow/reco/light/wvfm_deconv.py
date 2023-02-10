import numpy as np
import numpy.ma as ma
import logging
import os
import scipy.interpolate

from h5flow.core import H5FlowStage
from h5flow import H5FLOW_MPI


class WaveformDeconvolution(H5FlowStage):
    '''
        Applies a Wiener deconvolution filter to each waveform based on an
        electronics impulse response function, an input noise power
        spectrum, and an input signal plus noise power spectrum::

            wvfm_fft = FFT(wvfm)
            filtered_wvfm_fft = wvfm_fft * CONJ(response_fft) * |sig_fft|^2 / (|sig_fft|^2 * |response_fft|^2 + |noise_fft|^2)
            filtered_wvfm = IFFT(filtered_wvfm_fft)

        Or applies an inverse filter to each waveform base on the electronics response::

            wvfm_fft = FFT(wvfm)
            filtered_wvfm_fft = wvfm_fft * CONJ(response_fft) / |response_fft|^2

        Also can generate the signal and noise power spectra from non-PPS and
        PPS light triggers, respectively.

        Finally, can also generate an impulse response function from non-PPS
        light triggers using the rising-edge aligned waveform average

        Parameters:
         - ``wvfm_dset_name``: waveform dataset to analyze, required
         - ``filter_channels``: ``list`` of channels to apply filter on, others are copied, required
         - ``deconv_dset_name``: output dataset name, required
         - ``pps_channel``: channel to detect PPS events, default=32
         - ``pps_threshold``: threshold to detect PPS events, default=0
         - ``noise_strategy``: noise estimation strategy (for noise extraction only), either ``pps`` or ``slice``, default=``pps``
         - ``noise_slice``: samples to use for noise estimation if noise strategy is ``slice``
         - ``signal_amplitude``: bounds for waveform amplitude to include in signal and impulse function extraction, default=``(-inf,inf)``
         - ``gen_noise_spectrum``: flag to produce a noise spectrum .npz file from analyzed waveforms, default=``False``
         - ``gen_signal_spectrum``: flag to produce a signal spectrum .npz file from analyzed waveforms, default=``False``
         - ``gen_signal_impulse``: flag to produce an impulse .npz file from analyzed waveforms, default=``False``
         - ``impulse_alignment_oversampling``: factor to increase samples for aligning waveforms, default=10
         - ``do_filtering``: flag to produce inverse/wiener filtered dataset from input .npz files, , default=``True``
         - ``filter_type``: set inverse filtering strategy, either ``wiener``, ``inverse``, or ``matched``, default=``wiener``
         - ``gaus_filter_width``: set gaussian filter width applied to filtered waveforms, a value of 0 does not apply a gaussian filter, default=``0``
         - ``noise_spectrum_filename``: filename for input/output noise spectrum .npz, default=``wvfm_deconv_noise_power.npz``
         - ``signal_spectrum_filename``: filename for input/output signal spectrum .npz, default=``wvfm_deconv_signal_power.npz``
         - ``signal_impulse_filename``: filename for input/output signal impulse .npz, default=``wvfm_deconv_signal_impulse.npz``

        Example config::

            # configuration for impulse, signal, and noise extraction
            light_deconv:
                classname: WaveformDeconvolution
                requires:
                    - 'light/fwvfm'
                params:
                    wvfm_dset_name: 'light/fwvfm' # use pedestal+noise subtracted waveforms
                    deconv_dset_name: 'light/deconv_wvfm'
                    gen_noise_spectrum: True
                    gen_signal_spectrum: True
                    gen_signal_impulse: True
                    do_filtering: False
                    filter_type: Wiener #, Inverse, or Matched
                    gaus_filter_width: 2 # use a gaussian filter to reduce HF noise
                    noise_strategy: PPS # or slice
                    noise_slice: [-256, null] # last 256 samples

    '''
    class_version = '0.0.1'

    default_noise_spectrum_filename = 'wvfm_deconv_noise_power.npz'
    default_signal_spectrum_filename = 'wvfm_deconv_signal_power.npz'
    default_signal_impulse_filename = 'wvfm_deconv_signal_impulse.npz'
    default_gaus_filter_width = 0
    default_signal_amplitude = (0, np.inf)

    FILT_WIENER = 'wiener'
    FILT_INVERSE = 'inverse'
    FILT_MATCHED = 'matched'

    NOISE_PPS = 'pps'
    NOISE_SLICE = 'slice'

    def __init__(self, **params):
        super(WaveformDeconvolution, self).__init__(**params)

        self.wvfm_dset_name = params.get('wvfm_dset_name')

        self.pps_channel = params.get('pps_channel', 32)
        self.pps_threshold = params.get('pps_threshold', 0)
        self.noise_strategy = params.get('noise_strategy', self.NOISE_PPS).lower()
        if self.noise_strategy not in (self.NOISE_PPS, self.NOISE_SLICE):
            raise RuntimeError(f'Invalid noise estimation strategy: {self.noise_strategy}')
        self.noise_slice = slice(*params.get('noise_slice', (None, None)))
        self.signal_amplitude = params.get('signal_amplitude', (-np.inf, np.inf))

        self.gen_noise_spectrum = params.get('gen_noise_spectrum', False)
        self.gen_signal_spectrum = params.get('gen_signal_spectrum', False)
        self.impulse_alignment_oversampling = params.get('impulse_alignment_oversampling', 10)
        self.gen_signal_impulse = params.get('gen_signal_impulse', False)

        self.do_filtering = params.get('do_filtering', True)
        self.filter_type = params.get('filter_type', self.FILT_WIENER).lower()
        if self.filter_type not in (self.FILT_WIENER, self.FILT_INVERSE, self.FILT_MATCHED):
            raise RuntimeError(f'Invalid filter type: {self.filter_type}')
        self.gaus_filter_width = params.get('gaus_filter_width', self.default_gaus_filter_width)
        self.filter_channels = np.array(params.get('filter_channels'))

        self.deconv_dset_name = params.get('deconv_dset_name')
        self.noise_spectrum_filename = params.get('noise_spectrum_filename', self.default_noise_spectrum_filename)
        self.signal_spectrum_filename = params.get('signal_spectrum_filename', self.default_signal_spectrum_filename)
        self.signal_impulse_filename = params.get('signal_impulse_filename', self.default_signal_impulse_filename)

    def write_spectrum_or_impulse(self, name, data, spectrum=False, impulse=False, **attrs):
        if spectrum:
            dtype = np.dtype([('spectrum', data.dtype, data.shape)])
        if impulse:
            dtype = np.dtype([('impulse', data.dtype, data.shape)])

        write_data = np.empty((1,), dtype=dtype)
        write_data['spectrum' if spectrum else 'impulse'] = data
        self.data_manager.set_attrs(self.deconv_dset_name + '/' + name, **attrs)
        self.data_manager.create_dset(self.deconv_dset_name + '/' + name, dtype=dtype)
        self.data_manager.reserve_data(self.deconv_dset_name + '/' + name, slice(0, 1))
        self.data_manager.write_data(self.deconv_dset_name + '/' + name, slice(0, 1), write_data)

        return self.data_manager.get_dset(self.deconv_dset_name + '/' + name)

    def init(self, source_name):
        super(WaveformDeconvolution, self).init(source_name)

        wvfm_dset = self.data_manager.get_dset(self.wvfm_dset_name)

        if self.do_filtering:
            self.noise_spectrum = dict(np.load(self.noise_spectrum_filename))
            self.signal_spectrum = dict(np.load(self.signal_spectrum_filename))
            self.signal_impulse = dict(np.load(self.signal_impulse_filename))

            # interpolate mis-matched FFTs
            fft_shape = (2 * wvfm_dset.dtype['samples'].shape[-1]) // 2 + 1
            for spectrum in (self.noise_spectrum, self.signal_spectrum):
                s = spectrum['spectrum']
                s_shape = s.shape[-1]
                if s_shape != fft_shape:
                    if self.rank == 0:
                        logging.warning(f'Input spectrum size mismatch (in: {s_shape}, needed: {fft_shape}). '
                                        'Interpolating assuming same sample rate...')
                    spline = scipy.interpolate.CubicSpline(np.linspace(0, fft_shape, s_shape), s/s_shape, axis=-1)
                    s = spline(np.arange(fft_shape)) * fft_shape
                    s[np.isnan(s)] = 0
                    spectrum['spectrum'] = s

            impulse_shape = 2 * wvfm_dset.dtype['samples'].shape[-1]
            impulse = self.signal_impulse['impulse']
            if impulse.shape[-1] != impulse_shape:
                if self.rank == 0:
                    logging.warning(f'Input impulse function size mismatch (in: {impulse.shape[-1]}, needed: {impulse_shape}). '
                                    'Truncating to shorter length/appending zeros...')
                new_impulse = np.zeros(wvfm_dset.dtype['samples'].shape[:-1] + (impulse_shape,), dtype=wvfm_dset.dtype['samples'].base)
                valid_samples = min(impulse_shape, impulse.shape[-1])
                new_impulse[..., :valid_samples] = impulse[..., :valid_samples]
                self.signal_impulse['impulse'] = new_impulse

            if self.gaus_filter_width > 0:
                gaus = np.exp(-0.5 * np.arange(-impulse_shape // 2, impulse_shape // 2)**2 / self.gaus_filter_width**2)
                gaus /= gaus.sum()
                gaus = gaus.reshape(1, 1, 1, impulse_shape)
                self.gaus_fft = np.abs(np.fft.rfft(gaus, axis=-1))
            else:
                self.gaus_fft = None

            # save noise / signal spectra used for processing
            noise_spectrum_dset = self.write_spectrum_or_impulse('noise_spectrum',
                                                                 self.noise_spectrum['spectrum'], spectrum=True,
                                                                 filename=self.noise_spectrum_filename)
            signal_spectrum_dset = self.write_spectrum_or_impulse('signal_spectrum',
                                                                  self.signal_spectrum['spectrum'], spectrum=True,
                                                                  filename=self.signal_spectrum_filename)
            signal_impulse_dset = self.write_spectrum_or_impulse('signal_impulse',
                                                                 self.signal_impulse['impulse'], impulse=True,
                                                                 filename=self.signal_impulse_filename)

            self.data_manager.create_dset(self.deconv_dset_name, dtype=wvfm_dset.dtype)
            self.data_manager.create_ref(source_name, self.deconv_dset_name)
            self.data_manager.set_attrs(self.deconv_dset_name,
                                        classname=self.classname,
                                        class_version=self.class_version,
                                        noise_spectrum=noise_spectrum_dset.ref,
                                        signal_spectrum=signal_spectrum_dset.ref,
                                        signal_impulse=signal_impulse_dset.ref,
                                        filter_channels=self.filter_channels
                                        )
        else:
            fft_shape = (wvfm_dset.dtype['samples'].shape[-1] // 2 + 1,)
            self.noise_spectrum = dict(
                spectrum=np.zeros(wvfm_dset.dtype['samples'].shape[:-1] + fft_shape, dtype=wvfm_dset.dtype['samples'].base),
                n=np.zeros(wvfm_dset.dtype['samples'].shape[:-1] + (1,), dtype=int)
            )
            self.signal_spectrum = dict(
                spectrum=np.zeros(wvfm_dset.dtype['samples'].shape[:-1] + fft_shape, dtype=wvfm_dset.dtype['samples'].base),
                n=np.zeros(wvfm_dset.dtype['samples'].shape[:-1] + (1,), dtype=int)
            )
            self.signal_impulse = dict(
                impulse=np.zeros(wvfm_dset.dtype['samples'].shape, dtype=wvfm_dset.dtype['samples'].base),
                n=np.zeros(wvfm_dset.dtype['samples'].shape[:-1] + (1,), dtype=int)
            )

    def run(self, source_name, source_slice, cache):
        super(WaveformDeconvolution, self).run(source_name, source_slice, cache)

        wvfms = cache[self.wvfm_dset_name].reshape(cache[source_name].shape)['samples']  # 1:1 relationship
        wvfm_valid = cache[source_name]['wvfm_valid'].astype(bool)

        wvfms.mask = wvfms.mask | np.expand_dims(~wvfm_valid, axis=-1)

        # generate a noise spectrum from PPS signals
        if self.gen_noise_spectrum:
            if self.noise_strategy == self.NOISE_PPS:
                # only use events with all ADCs with valid PPS signal, use full waveforms
                pps_mask = (wvfms[:, :, self.pps_channel, :] > self.pps_threshold).any(axis=-1) \
                    & (~wvfms.mask).all(axis=-1).any(axis=-1)
                pps_mask = pps_mask.reshape(pps_mask.shape + (1, 1))

                if np.any(pps_mask):
                    pps_fft = np.fft.rfft(wvfms, axis=-1)

                    spectrum = ma.array(np.abs(pps_fft)**2, mask=~np.broadcast_to(pps_mask, pps_fft.shape))
                    spectrum = spectrum.mean(axis=0)
                    n = np.count_nonzero(pps_mask, axis=0)

                    old_n = self.noise_spectrum['n']
                    self.noise_spectrum['spectrum'] = (n * spectrum +
                                                       old_n * self.noise_spectrum['spectrum']) / (n + old_n + 1.e-9)
                    self.noise_spectrum['n'] = n + old_n
            elif self.noise_strategy == self.NOISE_SLICE:
                # use all events, but only a subset of waveform
                mask = (~wvfms.mask).all(axis=-1).any(axis=-1)
                mask = mask.reshape(mask.shape + (1, 1))

                if np.any(mask):
                    fft = np.fft.rfft(wvfms[..., self.noise_slice], axis=-1)

                    spectrum = ma.array(np.abs(fft)**2, mask=~np.broadcast_to(mask, fft.shape))
                    spectrum = spectrum.mean(axis=0)
                    spectrum[spectrum.mask] = 0
                    n = np.count_nonzero(mask, axis=0)

                    # interpolate back to "full" fft
                    fft_bins = fft.shape[-1]
                    exp_fft_bins = self.noise_spectrum['spectrum'].shape[-1]

                    spline = scipy.interpolate.CubicSpline(np.linspace(0, exp_fft_bins, fft_bins), spectrum/fft_bins, axis=-1)
                    spectrum = spline(np.arange(exp_fft_bins)) * exp_fft_bins
                    spectrum[np.isnan(spectrum)] = 0

                    old_n = self.noise_spectrum['n']
                    self.noise_spectrum['spectrum'] = (n * spectrum +
                                                       old_n * self.noise_spectrum['spectrum']) / (n + old_n + 1.e-9)
                    self.noise_spectrum['n'] = n + old_n

        # generate a signal spectrum from non-PPS signals
        if self.gen_signal_spectrum:
            # only use events with all ADCs with no PPS signal
            pps_mask = (~(wvfms[:, :, self.pps_channel, :] > self.pps_threshold).any(axis=-1)) \
                & (~wvfms.mask).all(axis=-1).any(axis=-1)
            pps_mask = pps_mask.reshape(pps_mask.shape + (1, 1))

            if np.any(pps_mask):
                # only use wvfms within signal amplitude window
                wvfm_mask = (wvfms > self.signal_amplitude[0]) & (wvfms < self.signal_amplitude[-1])
                wvfm_mask = wvfm_mask.any(axis=-1, keepdims=True)

                signal_fft = np.fft.rfft(wvfms, axis=-1)

                spectrum = ma.array(np.abs(signal_fft)**2, mask=~np.broadcast_to(pps_mask & wvfm_mask, signal_fft.shape))
                spectrum = spectrum.mean(axis=0)
                spectrum[spectrum.mask] = 0.
                n = np.count_nonzero(pps_mask & wvfm_mask, axis=0)

                old_n = self.signal_spectrum['n']
                self.signal_spectrum['spectrum'] = (n * spectrum +
                                                    old_n * self.signal_spectrum['spectrum']) / (n + old_n + 1.e-9)
                self.signal_spectrum['n'] = n + old_n

        # generate an impulse response function from non-PPS signals
        if self.gen_signal_impulse:
            # only use waveforms with no PPS signal in event
            pps_mask = (~(wvfms[:, :, self.pps_channel, :] > self.pps_threshold).any(axis=-1)) \
                & (~wvfms.mask).all(axis=-1).any(axis=-1)
            pps_mask = pps_mask.reshape(pps_mask.shape + (1, 1))

            if np.any(pps_mask):
                # only use wvfms within signal amplitude window
                wvfm_mask = (wvfms > self.signal_amplitude[0]) & (wvfms < self.signal_amplitude[-1])
                wvfm_mask = wvfm_mask.any(axis=-1, keepdims=True)

                # oversample waveform
                interpolation_samples = wvfms.shape[-1] * self.impulse_alignment_oversampling
                signal_wvfms_interp = scipy.interpolate.CubicSpline(
                    np.arange(wvfms.shape[-1]), wvfms, axis=-1, extrapolate=False)
                sample_pts = np.linspace(0, wvfms.shape[-1], interpolation_samples)
                signal_wvfms_oversampled = signal_wvfms_interp(sample_pts)
                signal_wvfms_oversampled[np.isnan(signal_wvfms_oversampled)] = 0.

                # find rising edge
                signal_wvfms_der = signal_wvfms_interp(sample_pts, 1)  # first derivative
                signal_wvfms_der[np.isnan(signal_wvfms_der)] = 0.
                rising_subsample = np.expand_dims(np.argmax(signal_wvfms_der, axis=-1), axis=-1)

                # project to zero-crossing
                with np.errstate(divide='ignore', invalid='ignore'):
                    crossing_subsample = rising_subsample - \
                        self.impulse_alignment_oversampling * \
                        np.take_along_axis(signal_wvfms_oversampled, rising_subsample, axis=-1) / \
                        np.take_along_axis(signal_wvfms_der, rising_subsample, axis=-1)
                invalid_mask = ~np.isfinite(crossing_subsample) | np.isnan(crossing_subsample)
                crossing_subsample[invalid_mask] = rising_subsample[invalid_mask]
                crossing_subsample = np.clip(crossing_subsample, 0, interpolation_samples - 1).astype(int)

                # perform alignment
                source_mask = np.arange(interpolation_samples).reshape(1, 1, 1, -1) >= crossing_subsample
                dest_mask = np.arange(interpolation_samples).reshape(1, 1, 1, -1) < source_mask.sum(axis=-1, keepdims=True)

                aligned_wvfms = np.zeros_like(signal_wvfms_oversampled)
                np.place(aligned_wvfms, dest_mask, signal_wvfms_oversampled[source_mask])

                # resample
                aligned_wvfms = aligned_wvfms[:, :, :, ::self.impulse_alignment_oversampling]

                impulse = ma.array(aligned_wvfms, mask=~np.broadcast_to(pps_mask & wvfm_mask, aligned_wvfms.shape))
                impulse = impulse.mean(axis=0)
                impulse[impulse.mask] = 0
                n = np.count_nonzero(pps_mask & wvfm_mask, axis=0)

                old_n = self.signal_impulse['n']
                self.signal_impulse['impulse'] = (n * impulse +
                                                  old_n * self.signal_impulse['impulse']) / (n + old_n + 1.e-9)
                self.signal_impulse['n'] = n + old_n

        if self.do_filtering:
            # zero-pad to remove cyclic artifacts
            padding = np.zeros_like(wvfms)
            fft = np.fft.rfft(np.concatenate((wvfms,padding), axis=-1), axis=-1)
            impulse_fft = np.fft.rfft(self.signal_impulse['impulse'])

            with np.errstate(divide='ignore', invalid='ignore'):
                if self.filter_type == self.FILT_WIENER:
                    # wiener deconvolution assuming delta-funtion (or gaussian) signal (optimizes MSE)
                    if self.gaus_fft is None:
                        sig_power = np.clip(self.signal_spectrum['spectrum'] - self.noise_spectrum['spectrum'], 0, None).mean(axis=-1, keepdims=True)
                    else:
                        sig_power = np.clip(self.signal_spectrum['spectrum'] - self.noise_spectrum['spectrum'], 0, None).sum(axis=-1, keepdims=True) * np.abs(self.gaus_fft)**2
                    filt_fft = fft * np.conj(impulse_fft) * sig_power \
                        / (sig_power * np.abs(impulse_fft)**2 + self.noise_spectrum['spectrum'])
                elif self.filter_type == self.FILT_INVERSE:
                    # inverse filter (perfect if no noise)
                    filt_fft = fft * np.conj(impulse_fft) / np.abs(impulse_fft)**2
                elif self.filter_type == self.FILT_MATCHED:
                    # general matched filter (optimizes SNR)
                    filt_fft = fft * np.conj(impulse_fft) / np.sqrt(self.noise_spectrum['spectrum'])

                # further gaussian filtering
                if self.gaus_fft is not None:
                    filt_fft *= self.gaus_fft

            filt_fft[np.isnan(filt_fft) | ~np.isfinite(filt_fft)] = 0.  # protect against invalid values
            filt_wvfms = np.fft.irfft(filt_fft, axis=-1)[..., :wvfms.shape[-1]]

            # save waveforms
            fwvfm = cache[self.wvfm_dset_name].reshape(cache[source_name].shape).copy()
            unfiltered_mask = ~np.isin(np.arange(fwvfm.dtype['samples'].shape[-2]), self.filter_channels)
            fwvfm['samples'][..., self.filter_channels, :] = filt_wvfms[..., self.filter_channels, :]
            fwvfm['samples'][..., unfiltered_mask, :] = wvfms[..., unfiltered_mask, :]

            fwvfm_slice = self.data_manager.reserve_data(self.deconv_dset_name, source_slice)
            self.data_manager.write_data(self.deconv_dset_name, source_slice, fwvfm)

            # write references
            ref = np.c_[source_slice, fwvfm_slice]
            self.data_manager.write_ref(source_name, self.deconv_dset_name, ref)

    def finish(self, source_name):
        super(WaveformDeconvolution, self).finish(source_name)

        if self.gen_noise_spectrum:
            # gather from all processes
            noise_spectra = self.comm.gather(self.noise_spectrum, root=0) if H5FLOW_MPI else [self.noise_spectrum]

            if self.rank == 0:
                # merge
                total_n = np.sum([s['n'] for s in noise_spectra], axis=0)
                total_spectrum = np.sum([s['spectrum'] * s['n'] for s in noise_spectra], axis=0) / total_n

                # save to file
                np.savez_compressed(self.noise_spectrum_filename, spectrum=total_spectrum, n=total_n)

        if self.gen_signal_spectrum:
            # gather from all processes
            signal_spectra = self.comm.gather(self.signal_spectrum, root=0) if H5FLOW_MPI else [self.signal_spectrum]

            if self.rank == 0:
                # merge
                total_n = np.sum([s['n'] for s in signal_spectra], axis=0)
                total_spectrum = np.sum([s['spectrum'] * s['n'] for s in signal_spectra], axis=0) / total_n

                # save to file
                np.savez_compressed(self.signal_spectrum_filename, spectrum=total_spectrum, n=total_n)

        if self.gen_signal_impulse:
            # gather from all processes
            signal_impulses = self.comm.gather(self.signal_impulse, root=0) if H5FLOW_MPI else [self.signal_impulse]

            if self.rank == 0:
                # merge
                total_n = np.sum([s['n'] for s in signal_impulses], axis=0)
                total_impulse = np.sum([s['impulse'] * s['n'] for s in signal_impulses], axis=0) / total_n
                # normalize to integral
                total_impulse /= total_impulse.sum(axis=-1, keepdims=True)

                # save to file
                np.savez_compressed(self.signal_impulse_filename, impulse=total_impulse, n=total_n)
