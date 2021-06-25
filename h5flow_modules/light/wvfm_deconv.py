import numpy as np
import logging
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
                    filter_type: Wiener # or Inverse

    '''
    class_version = '0.0.0'

    default_noise_spectrum_filename = 'wvfm_deconv_noise_power.npz'
    default_signal_spectrum_filename = 'wvfm_deconv_signal_power.npz'
    default_signal_impulse_filename = 'wvfm_deconv_signal_impulse.npz'

    WIENER = 'Wiener'
    INVERSE = 'Inverse'

    def __init__(self, **params):
        super(WaveformDeconvolution,self).__init__(**params)

        self.wvfm_dset_name = params.get('wvfm_dset_name')

        self.pps_channel = params.get('pps_channel',32)
        self.pps_threshold = params.get('pps_threshold',0)
        self.gen_noise_spectrum = params.get('gen_noise_spectrum',False)
        self.gen_signal_spectrum = params.get('gen_signal_spectrum',False)
        self.impulse_alignment_oversampling = params.get('impulse_alignment_oversampling',10)
        self.gen_signal_impulse = params.get('gen_signal_impulse',False)

        self.do_filtering = params.get('do_filtering',True)
        self.filter_type = params.get('filter_type',self.WIENER)
        if self.filter_type not in (self.WIENER, self.INVERSE):
            raise RuntimeError(f'Invalid filter type: {self.filter_type}')

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
        self.data_manager.set_attrs(self.deconv_dset_name+'/'+name,**attrs)
        self.data_manager.create_dset(self.deconv_dset_name+'/'+name, dtype=dtype)
        self.data_manager.reserve_data(self.deconv_dset_name+'/'+name, slice(0,1))
        self.data_manager.write_data(self.deconv_dset_name+'/'+name, slice(0,1), write_data)

        return self.data_manager.get_dset(self.deconv_dset_name+'/'+name)

    def init(self, source_name):
        wvfm_dset = self.data_manager.get_dset(self.wvfm_dset_name)

        if self.do_filtering:
            self.noise_spectrum = dict(np.load(self.noise_spectrum_filename))
            self.signal_spectrum = dict(np.load(self.signal_spectrum_filename))
            self.signal_impulse = dict(np.load(self.signal_impulse_filename))

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

            self.data_manager.set_attrs(self.deconv_dset_name,
                                        classname=self.classname,
                                        class_version=self.class_version,
                                        noise_spectrum=noise_spectrum_dset.ref,
                                        signal_spectrum=signal_spectrum_dset.ref,
                                        signal_impulse=signal_impulse_dset.ref
            )
            self.data_manager.create_dset(self.deconv_dset_name, dtype=wvfm_dset.dtype)
            self.data_manager.create_ref(source_name, self.deconv_dset_name)
        else:
            fft_shape = (wvfm_dset.dtype['samples'].shape[-1]//2+1,)
            self.noise_spectrum = dict(
                spectrum = np.empty(wvfm_dset.dtype['samples'].shape[:-1]+fft_shape, dtype=wvfm_dset.dtype['samples'].base),
                n = np.zeros(1, dtype=int)
                )
            self.signal_spectrum = dict(
                spectrum = np.empty(wvfm_dset.dtype['samples'].shape[:-1]+fft_shape, dtype=wvfm_dset.dtype['samples'].base),
                n = np.zeros(1, dtype=int)
                )
            self.signal_impulse = dict(
                impulse = np.empty(wvfm_dset.dtype['samples'].shape, dtype=wvfm_dset.dtype['samples'].base),
                n = np.zeros(1, dtype=int)
                )

    def run(self, source_name, source_slice, cache):
        wvfms = cache[self.wvfm_dset_name].reshape(cache[source_name].shape)['samples'] # 1:1 relationship
        wvfm_valid = cache[source_name]['wvfm_valid'].astype(bool)

        wvfms.mask = wvfms.mask | np.expand_dims(~wvfm_valid, axis=-1)

        # generate a noise spectrum from PPS signals
        if self.gen_noise_spectrum:
            # only use events with all ADCs with valid PPS signal
            pps_mask = (wvfms[:,:,self.pps_channel,:] > self.pps_threshold).any(axis=-1).all(axis=-1) \
                & (~wvfms.mask).all(axis=-1).any(axis=-1).all(axis=-1)

            if np.any(pps_mask):
                pps_fft = np.fft.rfft(wvfms[pps_mask], axis=-1)

                spectrum = (np.abs(pps_fft)**2).mean(axis=0)
                n = np.count_nonzero(pps_mask)

                old_n = self.noise_spectrum['n']
                self.noise_spectrum['spectrum'] = (n * spectrum + \
                    old_n * self.noise_spectrum['spectrum'])/ (n + old_n)
                self.noise_spectrum['n'] = n + old_n

        # generate a signal spectrum from non-PPS signals
        if self.gen_signal_spectrum:
            # only use events with all ADCs with no PPS signal
            pps_mask = (~(wvfms[:,:,self.pps_channel,:] > self.pps_threshold).any(axis=-1)).all(axis=-1) \
                & (~wvfms.mask).all(axis=-1).any(axis=-1).all(axis=-1)

            if np.any(pps_mask):
                signal_fft = np.fft.rfft(wvfms[pps_mask], axis=-1)

                spectrum = (np.abs(signal_fft)**2).mean(axis=0)
                n = np.count_nonzero(pps_mask)

                old_n = self.signal_spectrum['n']
                self.signal_spectrum['spectrum'] = (n * spectrum + \
                    old_n * self.signal_spectrum['spectrum'])/ (n + old_n)
                self.signal_spectrum['n'] = n + old_n

        # generate an impulse response function from non-PPS signals
        if self.gen_signal_impulse:
            # only use events with all ADCs with no PPS signal
            pps_mask = (~(wvfms[:,:,self.pps_channel,:] > self.pps_threshold).any(axis=-1)).all(axis=-1) \
                & (~wvfms.mask).all(axis=-1).any(axis=-1).all(axis=-1)

            if np.any(pps_mask):
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
                rising_subsample = np.expand_dims(np.argmax(signal_wvfms_der, axis=-1), axis=-1)

                # project to zero-crossing
                with np.errstate(divide='ignore', invalid='ignore'):
                    crossing_subsample = rising_subsample - \
                        self.impulse_alignment_oversampling * \
                        np.take_along_axis(signal_wvfms_oversampled, rising_subsample, axis=-1) / \
                        np.take_along_axis(signal_wvfms_der, rising_subsample, axis=-1)
                invalid_mask = ~np.isfinite(crossing_subsample) | np.isnan(crossing_subsample)
                crossing_subsample[invalid_mask] = rising_subsample[invalid_mask]
                crossing_subsample = np.clip(crossing_subsample,0,interpolation_samples-1).astype(int)

                # perform alignment
                source_mask = np.arange(interpolation_samples).reshape(1,1,1,-1) >= crossing_subsample
                dest_mask = np.arange(interpolation_samples).reshape(1,1,1,-1) < source_mask.sum(axis=-1, keepdims=True)

                aligned_wvfms = np.zeros_like(signal_wvfms_oversampled)
                np.place(aligned_wvfms, dest_mask, signal_wvfms_oversampled[source_mask])

                # resample
                aligned_wvfms = aligned_wvfms[:,:,:,::self.impulse_alignment_oversampling]

                impulse = aligned_wvfms.mean(axis=0)
                n = np.count_nonzero(pps_mask)

                old_n = self.signal_impulse['n']
                self.signal_impulse['impulse'] = (n * impulse + \
                    old_n * self.signal_impulse['impulse'])/ (n + old_n)
                self.signal_impulse['n'] = n + old_n

        if self.do_filtering:
            fft = np.fft.rfft(wvfms, axis=-1)
            impulse_fft = np.fft.rfft(self.signal_impulse['impulse'])

            # wiener deconvolution
            with np.errstate(divide='ignore', invalid='ignore'):
                if self.filter_type == self.WIENER:
                    filt_fft = fft * np.conj(impulse_fft) * self.signal_spectrum['spectrum']\
                               / (self.signal_spectrum['spectrum'] * np.abs(impulse_fft)**2 + self.noise_spectrum['spectrum'])
                elif self.filter_type == self.INVERSE:
                    # inverse filter
                    filt_fft = fft * np.conj(impulse_fft) / np.abs(impulse_fft)**2

            filt_fft[np.isnan(filt_fft) | ~np.isfinite(filt_fft)] = 0. # protect against invalid values
            filt_wvfms = np.fft.irfft(filt_fft, axis=-1)

            # save waveforms
            fwvfm = cache[self.wvfm_dset_name].reshape(cache[source_name].shape).copy()
            fwvfm['samples'] = filt_wvfms
            
            fwvfm_slice = self.data_manager.reserve_data(self.deconv_dset_name, source_slice)
            self.data_manager.write_data(self.deconv_dset_name, source_slice, fwvfm)

            # write references
            ref = np.c_[source_slice, fwvfm_slice]
            self.data_manager.write_ref(source_name, self.deconv_dset_name, ref)

    def finish(self, source_name):
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


