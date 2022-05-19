import numpy as np
import numpy.ma as ma
import scipy.ndimage as ndimage
import scipy.optimize as optimize
import scipy.interpolate as interpolate
import logging
import os

from h5flow import H5FLOW_MPI
from h5flow.core import H5FlowStage

import module0_flow.util.units as units


def f_scint(t, singlet_fraction=0.3, tau_s=7, tau_t=750, smear=10):
    t[t < 0] = -1
    f = (singlet_fraction / tau_s * np.exp(-t / tau_s) + (1 - singlet_fraction) / tau_t * np.exp(-t / tau_t))
    f[t < 0] = 0
    f = ndimage.gaussian_filter1d(f, sigma=smear, axis=-1)
    f *= np.diff(t).mean()
    return f


def f_delayed(t, prompt_a, prompt_t, delayed_a, delayed_t, *args, **kwargs):
    prompt_model = prompt_a * f_scint(t - prompt_t, *args, **kwargs)
    delayed_model = delayed_a * f_scint(t - delayed_t, *args, **kwargs)
    return prompt_model + delayed_model


class DelayedSignal(H5FlowStage):
    class_version = '0.1.0'

    defaults = dict(
        hits_dset_name='light/hits',
        wvfm_dset_name='light/deconv',
        wvfm_align_dset_name='light/deconv/align',
        fit_dset_name='analysis/time_reco/fit',
        prompt_dset_name='analysis/time_reco/prompt',
        delayed_dset_name='analysis/time_reco/delayed',
        prompt_threshold_factor=1,
        prompt_window=[-350,-250], # ns
        delayed_window=[100,10240], # ns
        delayed_hit_window=10, # ns
        fit_singlet=False,
        fit_triplet=False,
        fit_fraction=False,
        singlet_tau=7, # ns
        triplet_tau=750, # ns
        singlet_fraction=0.3, # ns
        smearing=10 # ns
        )

    @staticmethod
    def fit_dtype(fit_prompt, fit_triplet, fit_fraction):
        dtype_spec = [
            ('valid', 'u1'),
            ('prompt_a', 'f4'),
            ('prompt_ns', 'f4'),
            ('delayed_a', 'f4'),
            ('delayed_ns', 'f4')]
        cov_size = 4
        if fit_prompt:
            dtype_spec += [('tau_s', 'f4')]
            cov_size += 1
        if fit_triplet:
            dtype_spec += [('tau_t', 'f4')]
            cov_size += 1
        if fit_fraction:
            dtype_spec += [('fraction', 'f4')]
            cov_size += 1
        dtype_spec += [('cov', 'f4', (cov_size,cov_size))]
        return np.dtype(dtype_spec)

    @staticmethod
    def prompt_dtype(ntpc, ndet):
        return np.dtype([
            ('valid', 'u1'),
            ('ns', 'f8'),
            ('ampl', 'f8', (ntpc, ndet)),
            ])


    @staticmethod
    def delayed_dtype(ntpc, ndet):
        return np.dtype([
            ('ns', 'f8'),
            ('delay', 'f8'),
            ('ampl', 'f8', (ntpc, ndet)),
            ('valid', 'u1')
            ])


    def __init__(self, **params):
        super(DelayedSignal, self).__init__(**params)
        for param,default in self.defaults.items():
            setattr(self, param, params.get(param, default))


    def init(self, source_name):
        super(DelayedSignal, self).init(source_name)

        # get number of tpcs/detectors
        hit_attrs = self.data_manager.get_attrs(self.hits_dset_name)
        ntpc = hit_attrs['ntpc']
        ndet = hit_attrs['ndet']
        self.prompt_dtype = self.prompt_dtype(ntpc, ndet)
        self.delayed_dtype = self.delayed_dtype(ntpc, ndet)

        self.sample_rate = (resources['RunData'].lrs_ticks / units.ns)

        # create fit datatype
        self.fit_dtype = self.fit_dtype(self.fit_prompt, self.fit_triplet, self.fit_fraction)
            f_delayed(t, prompt_a, prompt_t, delayed_a, delayed_t,
                singlet_fraction, tau_s, tau_t, smear=self.smearing)
        fit_func = lambda prompt_a, prompt_t, delayed_a, delayed_t, singlet_fraction, tau_s, tau_t: f_delayed(t, prompt_a, prompt_t, delayed_a, delayed_t,
                singlet_fraction=singlet_fraction, tau_s=tau_s, tau_t=tau_t, smear=self.smearing)
        self.fit_func = fit_func
        if not self.fit_singlet:
            def fit_func(*args, **kwargs):
                return fit_func(*args, tau_s=self.tau_s, **kwargs)
            self.fit_func = fit_func
        if not self.fit_triplet:
            def fit_func(*args, **kwargs):
                return fit_func(*args, tau_t=self.tau_t, **kwargs)
            self.fit_func = fit_func
        if not self.fit_fraction:
            def fit_func(*args, **kwargs):
                return fit_func(*args, singlet_fraction=self.singlet_fraction, **kwargs)
            self.fit_func = fit_func

        # set prompt signal thresholds
        thresholds = hit_attrs['thresholds']
        self.prompt_thresholds = self.prompt_threshold_factor * thresholds

        # format output file
        attrs = dict(class_version=self.class_version, classname=self.classname)
        for param in self.defaults:
            attrs[param] = getattr(self, param)
        attrs['prompt_thresholds'] = self.prompt_thresholds

        self.data_manager.create_dset(self.prompt_dset_name, dtype=self.prompt_dtype)
        self.data_manager.create_dset(self.delayed_dset_name, dtype=self.delayed_dtype)
        self.data_manager.create_dset(self.fit_dset_name, dtype=self.fit_dtype)
        self.data_manager.create_ref(source_name, self.fit_dset_name)
        self.data_manager.create_ref(source_name, self.prompt_dset_name)
        self.data_manager.create_ref(self.prompt_dset_name, self.delayed_dset_name)

        self.data_manager.set_attrs(self.fit_dset_name, **attrs)
        self.data_manager.set_attrs(self.prompt_dset_name, **attrs)
        self.data_manager.set_attrs(self.delayed_dset_name, **attrs)


    def run(self, source_name, source_slice, cache):
        super(DelayedSignal, self).run(source_name, source_slice, cache)
        # load from cache
        hits = cache[self.hits_dset_name]
        wvfm = cache[self.wvfm_dset_name]
        wvfm_align = cache[self.wvfm_align_dset_name]
        if len(hits):
            hits = hits.reshape(len(np.r_[source_slice]),-1)
            wvfm = wvfm.reshape((len(np.r_[source_slice]),-1) + wvfm.shape[-3:])
            wvfm_align = wvfm_align.reshape((len(np.r_[source_slice]),-1) + wvfm_align.shape[-1:] + (1,1))
        else:
            hits = ma.array(np.empty((0,1), dtype=hits.dtype))
            wvfm = ma.array(np.empty((0,1) + wvfm.shape[-3:], dtype=wvfm.dtype))
            wvfm_align = ma.array(np.empty((0,1) + wvfm_align.shape[-1:], dtype=wvfm_align.dtype))

        # find prompt signal time and amplitude
        # definition:
        #  - hit falls within trigger window
        #  - hit occurs in first event light trigger
        #  - largest hit amplitude on each channel (meeting above criteria)
        first_trigger_ns = np.around(hits['ns']).min(axis=-1, keepdims=True)
        prompt_hit_mask = (
            (hits['busy_ns'] + hits['ns_spline'] >= self.prompt_window[0])
            & (hits['busy_ns'] + hits['ns_spline'] < self.prompt_window[1])
            & (np.around(hits['ns']) == first_trigger_ns)
            & (hits['max_spline'] >= self.prompt_thresholds[hits['tpc'].filled(0), hits['det'].filled(0)].reshape(hits.shape))
            )
        hit_ns = hits['ns'] + hits['busy_ns'] + hits['ns_spline']
        hit_ns.mask = hit_ns.mask | ~prompt_hit_mask
        prompt_ns = ma.average(hit_ns, axis=-1, weights=hits['sum_spline'])
        prompt_ampl = np.zeros(hits.shape[0:1] + self.prompt_dtype['ampl'].shape)
        for i in range(prompt_ampl.shape[1]):
            for j in range(prompt_ampl.shape[2]):
                hit_submask = (hits['tpc'] == i) & (hits['det'] == j) & prompt_hit_mask
                if np.any(hit_submask):
                    prompt_ampl[:,i,j] = (hit_submask * hits['sum_spline']).sum(axis=-1)
                    prompt_ampl[~np.any(hit_submask, axis=-1),i,j] = 0

        # calculate delayed parameters
        # definition:
        #  - relative time: time between prompt signal and hit
        #  - relative ampl: ratio of hit amplitude and prompt signal on given detector
        hit_rel_ns = np.zeros(hits.shape)
        hit_rel_ampl = np.zeros(hits.shape)
        hit_rel_valid = np.zeros(hits.shape, dtype=bool)
        for i in range(prompt_ampl.shape[1]):
            for j in range(prompt_ampl.shape[2]):
                hit_submask = ((hits['tpc'] == i) & (hits['det'] == j)
                    & ~prompt_hit_mask
                    & (hits['ns'] + hits['busy_ns'] + hits['ns_spline'] - prompt_ns[:,np.newaxis] < self.delayed_window[1])
                    & (hits['ns'] + hits['busy_ns'] + hits['ns_spline'] - prompt_ns[:,np.newaxis] >= self.delayed_window[0]))
                if np.any(hit_submask):
                    p_ns = np.broadcast_to(prompt_ns[:,np.newaxis], hit_submask.shape)
                    p_ampl = np.broadcast_to(prompt_ampl.sum(axis=-1).sum(axis=-1)[...,np.newaxis], hit_submask.shape)
                    hit_rel_ns[hit_submask] = (hits[hit_submask]['ns']
                        + hits[hit_submask]['busy_ns']
                        + hits[hit_submask]['ns_spline']
                        - p_ns[hit_submask])
                    hit_rel_ampl[hit_submask] = hits[hit_submask]['sum_spline'] / np.clip(p_ampl[hit_submask],1e-15,None)
                    hit_rel_valid[hit_submask] = (p_ampl[hit_submask] > 0)

        else:
            # look for delayed signal

            # find best delayed time interval
            # definition:
            #  - create a sliding window
            #  - find window with largest energy
            #  - use weighted mean of relative hit time of hits within window
            hit_mask = (hit_rel_valid)

            sliding_center = np.linspace(
                self.delayed_window[0]+self.delayed_hit_window,
                self.delayed_window[1]-self.delayed_hit_window,
                2*int(np.ceil((self.delayed_window[1] - self.delayed_window[0] - 2*self.delayed_hit_window)/self.delayed_hit_window)))
            sliding_center = sliding_center[np.newaxis,np.newaxis,:]
            hit_in_window = (hit_mask[...,np.newaxis]
                             & (hit_rel_ns[...,np.newaxis] < sliding_center + self.delayed_hit_window)
                             & (hit_rel_ns[...,np.newaxis] >= sliding_center - self.delayed_hit_window))
            sliding_score = np.sum(hit_in_window * hits['sum_spline'][...,np.newaxis], axis=-2)
            if len(sliding_score):
                delayed_ns_med = np.take_along_axis(sliding_center[0], np.argmax(sliding_score, axis=-1)[...,np.newaxis], axis=-1)
            else:
                delayed_ns_med = np.empty((0,1))
            delayed_ns_min = delayed_ns_med - self.delayed_hit_window
            delayed_ns_max = delayed_ns_med + self.delayed_hit_window

            # calculate delayed signal parameters
            hit_in_window = hit_mask & (hit_rel_ns < delayed_ns_max) & (hit_rel_ns >= delayed_ns_min)
            if np.any(hit_in_window):
                delayed_time = ma.average(hit_rel_ns, axis=-1, weights=hit_in_window * hits['sum_spline'])
            else:
                delayed_time = np.zeros(hit_rel_ns.shape[0])
            delayed_ns = prompt_ns + delayed_time
            delayed_valid = np.any(hit_in_window, axis=-1)
            delayed_ampl = np.zeros_like(prompt_ampl)
            for i in range(delayed_ampl.shape[1]):
                for j in range(delayed_ampl.shape[2]):
                    hit_submask = (hits['tpc'] == i) & (hits['det'] == j) & hit_in_window
                    if np.any(hit_submask):
                        delayed_ampl[:,i,j] = (hit_submask * hits['sum_spline']).sum(axis=-1)
                        delayed_ampl[~np.any(hit_submask, axis=-1),i,j] = 0

            # do fit
            # reconstruct the sample timestamp (relative to first trigger)
            wvfm_ns = wvfm_align['ns'][...,np.newaxis,np.newaxis,np.newaxis] + (wvfm_align['sample_idx'][...,np.newaxis] - np.arange(wvfm.shape[-1])) * self.sample_rate

            fit_data = np.zeros(hits.shape[0], dtype=self.fit_dtype)

            for iev in range(wvfm.shape[0]):
                if np.all(wvfm[i].mask):
                    continue
                min_ns = np.min(wvfm_ns[iev])
                n_ticks = ceil((np.max(wvfm_ns[iev]) - min_ns) / self.sample_rate)
                xdata = np.linspace(min_ns, min_ns + self.sample_rate * n_ticks, n_ticks)
                ydata = np.zeros_like(xdata)
                for itrig in range(wvfm.shape[1]):
                    for itpc in range(wvfm.shape[2]):
                        for idet in range(wvfm.shape[3]):
                            ydata += np.interp1d(xdata, wvfm_ns[iev,itrig,itpc,idet,:], wvfm[iev,itrig,itpc,idet,:], left=0, right=0)

                p0 = (p_ampl[iev], p_ns[iev], delayed_ampl[iev], delayed_ns[iev]) + self.p0
                try:
                    p,cov = optimize.curve_fit(self.fit_func, xdata, ydata, p0=p0)
                    fit_data[iev]['valid'] = True
                    fit_data[iev]['prompt_a'] = p[0]
                    fit_data[iev]['prompt_ns'] = p[1]
                    fit_data[iev]['delayed_a'] = p[2]
                    fit_data[iev]['delayed_ns'] = p[3]
                    fit_data[iev]['cov'] = cov
                    if self.fit_fraction:
                        fit_data[iev]['fraction'] = p[4]
                    if self.fit_singlet:
                        fit_data[iev]['tau_s'] = p[4 + self.fit_fraction]
                    if self.fit_triplet:
                        fit_data[iev]['tau_t'] = p[4 + self.fit_fraction + self.fit_singlet]
                except:
                    continue

            # save data to file
            prompt_data = np.zeros(hits.shape[0], dtype=self.prompt_dtype)
            delayed_data = np.zeros(hits.shape[0], dtype=self.delayed_dtype)

            if len(prompt_data):
                prompt_data['ns'] = prompt_ns
                prompt_data['valid'] = np.any(prompt_ampl > 0)
                prompt_data['ampl'] = prompt_ampl

            if len(delayed_data):
                delayed_data['ns'] = delayed_ns
                delayed_data['delay'] = delayed_time
                delayed_data['ampl'] = delayed_ampl
                delayed_data['valid'] = delayed_valid

            fit_slice = self.data_manager.reserve_data(
                self.fit_dset_name, len(fit_data))
            self.data_manager.write_data(self.fit_dset_name, fit_slice, fit_data)
            ref = np.c_[source_slice, fit_slice][fit_data['valid'].astype(bool)]
            if len(ref) == 0:
                ref = np.empty((0,2), int)
            self.data_manager.write_ref(source_name, self.fit_dset_name, ref)

            prompt_slice = self.data_manager.reserve_data(
                self.prompt_dset_name, len(prompt_data))
            self.data_manager.write_data(self.prompt_dset_name, prompt_slice, prompt_data)
            ref = np.c_[source_slice, prompt_slice][np.any(prompt_hit_mask, axis=-1)]
            if len(ref) == 0:
                ref = np.empty((0,2), int)
            self.data_manager.write_ref(source_name, self.prompt_dset_name, ref)

            delayed_slice = self.data_manager.reserve_data(
                self.delayed_dset_name, len(delayed_data))
            self.data_manager.write_data(self.delayed_dset_name, delayed_slice, delayed_data)
            ref = np.c_[prompt_slice, delayed_slice][delayed_valid]
            if len(ref) == 0:
                ref = np.empty((0,2), int)
            self.data_manager.write_ref(self.prompt_dset_name, self.delayed_dset_name, ref)
