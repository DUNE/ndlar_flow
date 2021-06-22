import numpy as np
import logging
import scipy.interpolate

from h5flow.core import H5FlowStage

class WaveformDeconvolution(H5FlowStage):
    '''
        Applies a Wiener deconvolution filter to each waveform based on an
        electronics impulse response function, an input noise power
        spectrum, and an input signal plus noise power spectrum::

            wvfm_fft = FFT(wvfm)
            filtered_wvfm_fft = wvfm * CONJ(response_fft) * |sig_fft|^2 / (|sig_fft|^2 |response_fft|^2 + |noise_fft|^2)
            filtered_wvfm = IFFT(filtered_wvfm_fft)

        Also can generate the signal and noise power spectra from non-PPS and
        PPS light triggers, respectively.

        Finally, can also generate an impulse response function from non-PPS
        light triggers using the rising-edge aligned waveform average

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

    '''
    class_version = '0.0.0'

    default_noise_spectrum_filename = 'wvfm_deconv_noise_power.npz'
    default_signal_spectrum_filename = 'wvfm_deconv_signal_power.npz'
    default_signal_impulse_filename = 'wvfm_deconv_signal_impulse.npz'

    def __init__(self, **params):
        super(WaveformNoiseFilter,self).__init__(**params)

        self.wvfm_dset_name = params.get('wvfm_dset_name')

        self.pps_channel = params.get('pps_channel',32)
        self.pps_threshold = params.get('pps_threshold',0)
        self.gen_noise_spectrum = params.get('gen_noise_spectrum',False)
        self.gen_signal_spectrum = params.get('gen_signal_spectrum',False)
        self.impulse_alignment_oversampling = params.get('impulse_alignment_oversampling',10)
        self.gen_signal_impulse = params.get('gen_signal_impulse',False)

        self.do_filtering = params.get('do_filtering',True)

        self.deconv_dset_name = params.get('deconv_dset_name')
        self.noise_spectrum_filename = params.get('noise_spectrum_filename', self.default_noise_spectrum_filename)
        self.signal_spectrum_filename = params.get('signal_spectrum_filename', self.default_signal_spectrum_filename)
        self.signal_impulse_filename = params.get('signal_impulse_filename', self.default_signal_impulse_filename)

    def init(self, source_name):
        wvfm_dset = self.data_manager.get_dset(self.wvfm_dset_name)

        if self.do_filtering:
            self.noise_spectrum = np.load(self.noise_spectrum_filename)
            self.signal_spectrum = np.load(self.signal_spectrum_filename)
            self.signal_impulse = np.load(self.signal_impulse_filename)
        else:
            fft_shape = (wvfm_dset['samples'].shape[-1]//2+1,)
            self.noise_spectrum = dict(
                spectrum = np.empty(wvfm_dset['samples'].shape[1:-1]+fft_shape, dtype=wvfm_dset['samples'].dtype),
                n = np.zeros(1, dtype=int)
                )
            self.signal_spectrum = dict(
                spectrum = np.empty(wvfm_dset['samples'].shape[1:-1]+fft_shape, dtype=wvfm_dset['samples'].dtype),
                n = np.zeros(1, dtype=int)
                )
            self.signal_impulse = dict(
                impulse = np.empty(wvfm_dset['samples'].shape[1:], dtype=wvfm_dset['samples'].dtype),
                n = np.zeros(1, dtype=int)
                )

    def run(self, source_name, source_slice, cache):
        wvfms = cache[self.wvfm_dset_name]['samples']
        wvfm_valid = cache[source_name]['wvfm_valid'].astype(bool)

        wvfms.mask = wvfms.mask | np.expand_dims(~wvfm_valid, axis=-1)

        # generate a noise spectrum from PPS signals
        if self.gen_noise_spectrum:
            # only use events with all ADCs with valid PPS signal
            pps_mask = (wvfms[:,:,self.pps_channel,:] > self.pps_threshold).any(axis=-1).all(axis=-1) \
                & (~wvfms.mask).all(axis=-1).any(axis=-1).all(axis=-1)

            pps_fft = np.fft.rfft(wvfms[pps_mask], axis=-1)
            n = np.count_nonzero(pps_mask)

            old_n = self.noise_spectrum['n']
            self.noise_spectrum['spectrum'] = (n * np.abs(pps_fft)**2 + \
                old_n * self.noise_spectrum['spectrum'])/ (n + old_n)
            self.noise_spectrum['n'] = n + old_n

        # generate a signal spectrum from non-PPS signals
        if self.gen_signal_spectrum:
            # only use events with all ADCs with no PPS signal
            pps_mask = (~(wvfms[:,:,self.pps_channel,:] > self.pps_threshold).any(axis=-1)).all(axis=-1) \
                & (~wvfms.mask).all(axis=-1).any(axis=-1).all(axis=-1)

            signal_fft = np.fft.rfft(wvfms[pps_mask], axis=-1)
            n = np.count_nonzero(pps_mask)

            old_n = self.signal_spectrum['n']
            self.signal_spectrum['spectrum'] = (n * np.abs(signal_fft)**2 + \
                old_n * self.signal_spectrum['spectrum'])/ (n + old_n)
            self.signal_spectrum['n'] = n + old_n

        # generate an impulse response function from non-PPS signals
        if self.gen_signal_impulse:
            # only use events with all ADCs with no PPS signal
            pps_mask = (~(wvfms[:,:,self.pps_channel,:] > self.pps_threshold).any(axis=-1)).all(axis=-1) \
                & (~wvfms.mask).all(axis=-1).any(axis=-1).all(axis=-1)

            wvfms = wvfms[pps_mask]

            # oversample waveform
            interpolation_samples = wvfms.shape[-1] * self.impulse_alignment_oversampling
            signal_wvfms_interp = scipy.interpolate.CubicSpline(
                np.arange(wvfms.shape[-1]), wvfms, axis=-1, extrapolate=False)
            sample_pts = np.linspace(0,wvfms.shape[-1],interpolation_samples)
            signal_wvfms_oversampled = signal_wvfms_interp(sample_pts)
            signal_wvfms_oversampled[np.isnan(signal_wvfms_oversampled)] = 0.

            # find rising edge
            signal_wvfms_der = signal_wvfms_interp(sample_pts, 1) # first derivative
            signal_wvfms_der[np.isnan(signal_wvfms_der)] = 0.
            rising_subsample = np.argmax(signal_wvfms_der, axis=-1)

            # project to zero-crossing
            crossing_subsample = rising_subsample - \
                np.take_along_axis(signal_wvfms_oversampled, rising_subsample, axis=-1) / \
                np.take_along_axis(signal_wvfms_der * self.impulse_alignment_oversampling, rising_subsample, axis=-1)
            crossing_subsample = np.clip(crossing_subsample,0,interpolation_samples-1)
            valid_samples = interpolation_samples - crossing_subsample

            # perform alignment
            source_mask = (crossing_subsample.astype(int) <= np.arange(interpolation_samples)) & \
                (crossing_subsample.astype(int) > np.arange(interpolation_samples))
            dest_mask = np.arange(interpolation_samples).reshape(1,1,1,-1) < valid_samples

            aligned_wvfms = np.zeros_like(signal_wvfms_oversampled)
            np.place(aligned_wvfms[dest_mask], source_mask ,signal_wvfms_oversampled)

            # normalize and resample
            aligned_wvfms = aligned_wvfms / aligned_wvfms.sum(axis=-1, keepdims=True) * self.impulse_alignment_oversampling
            aligned_wvfms = aligned_wvfms[:,:,:,::self.impulse_alignment_oversampling]

            n = np.count_nonzero(pps_mask)

            old_n = self.signal_impulse['n']
            self.signal_impulse['impulse'] = (n * aligned_wvfms + \
                old_n * self.signal_impulse['impulse'])/ (n + old_n)
            self.signal_impulse['n'] = n + old_n

    def finish(self, source_name):
        if self.gen_noise_spectrum:
            # gather from all processes
            noise_spectra = self.comm.gather(self.noise_spectrum, root=0)

            # merge
            total_n = np.sum([s['n'] for s in noise_spectra])
            total_spectrum = np.sum([s['spectrum'] * s['n'] for s in noise_spectra]) / total_n

            # save to file
            if self.rank == 0:
                np.savez_compressed(self.noise_spectrum_filename, spectrum=total_spectrum, n=total_n)

        if self.gen_signal_spectrum:
            # gather from all processes
            signal_spectra = self.comm.gather(self.signal_spectrum, root=0)

            # merge
            total_n = np.sum([s['n'] for s in signal_spectra])
            total_spectrum = np.sum([s['spectrum'] * s['n'] for s in signal_spectra]) / total_n

            # save to file
            if self.rank == 0:
                np.savez_compressed(self.signal_spectrum_filename, spectrum=total_spectrum, n=total_n)

        if self.gen_signal_impulse:
            # gather from all processes
            signal_impulses = self.comm.gather(self.signal_impulse, root=0)

            # merge
            total_n = np.sum([s['n'] for s in signal_impulses])
            total_impulse = np.sum([s['impulse'] * s['n'] for s in signal_impulses]) / total_n

            # save to file
            if self.rank == 0:
                np.savez_compressed(self.signal_impulse_filename, spectrum=total_impulse, n=total_n)


