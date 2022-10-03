import numpy as np
import numpy.ma as ma
import scipy.ndimage as ndimage
import scipy.optimize as optimize
import scipy.interpolate as interpolate
import logging
import os

import time

from h5flow import H5FLOW_MPI, resources
from h5flow.core import H5FlowStage

import module0_flow.util.units as units


def f_scint(t, singlet_fraction=0.3, tau_t=750, tau_s=7, smearing=1):

    f = np.where(t >= 0, (singlet_fraction / tau_s * np.exp(-t.clip(0,tau_s*10) / tau_s) + (1 - singlet_fraction) / tau_t * np.exp(-t.clip(0,tau_t*10) / tau_t)), 0)
    ndimage.gaussian_filter1d(f, output=f, sigma=smearing, mode='nearest', axis=-1)
    return f


def f_delayed(t, prompt_a, prompt_t, delayed_a, delayed_t, *args, **kwargs):
    prompt_model = prompt_a * f_scint(t - prompt_t, *args, **kwargs)
    delayed_model = delayed_a * f_scint(t - prompt_t - delayed_t, *args, **kwargs)
    return prompt_model + delayed_model


def loss(x, *args):
    t = args[0] # shape (trig x tpc x det, ticks)
    y = args[1] # shape (trig x tpc x det, ticks)
    prompt_acceptance = args[2] # shape (trig x tpc x det)
    delayed_acceptance = args[3] # shape (trig x tpc x det)
    noise = args[4] # shape (trig x tpc x det)
    prompt_weight = args[5] # bool (True if weight by prompt acceptance, False if weight by delayed acceptance)
    time_offset = args[6] # constant factor to add to prompt time

    prompt_f = x[0]
    prompt_t = x[1] + time_offset
    delayed_t = x[2]
    singlet_fraction = x[3]
    tau_t = x[4]
    model_args = (singlet_fraction, tau_t)

    total_sum = y.sum() * (t[0,1] - t[0,0])
    model = total_sum * f_delayed(t,
                      prompt_f * prompt_acceptance[...,np.newaxis], prompt_t,
                      (1 - prompt_f) * delayed_acceptance[...,np.newaxis], delayed_t,
                      *model_args)

    err = np.sum((model - y)**2 / np.clip(noise[...,np.newaxis],1e-15,None)**2, axis=(0,-1))
    if not prompt_weight:
        weight = 1
    else:
        weight = prompt_acceptance / np.sum(prompt_acceptance)
    return np.sum(err * weight)        


class DelayedSignal(H5FlowStage):
    class_version = '0.2.0'

    defaults = dict(
        hits_dset_name='charge/hits',
        hit_drift_dset_name='combined/hit_drift',
        stopping_sel_dset_name='analysis/stopping_muons/event_sel_reco',
        michel_id_dset_name='analysis/michel_id/michel_label',
        hit_label_dset_name='analysis/michel_id/hit_label',
        wvfm_dset_name='light/swvfm',
        wvfm_align_dset_name='light/swvfm/alignment',
        fit_dset_name='analysis/time_reco/fit',
        prompt_dset_name='analysis/time_reco/prompt',
        delayed_dset_name='analysis/time_reco/delayed',
        prompt_response_model=dict(
            data={256: 'h5flow_data/mod0_response.data.256.npz',
                  1024: 'h5flow_data/mod0_response.data.1024.npz',
                  2048: 'h5flow_data/mod0_response.data.2048.npz'},
            mc={256: 'h5flow_data/mod0_response.sim.256.npz',
                1024: 'h5flow_data/mod0_response.sim.1024.npz'}
        ),
        response_model_terms=3,
        singlet_fraction=0.3,
        triplet_time=750, # ns
        noise_factor=1,
        prompt_window=[-400,-200], # ns
        delayed_window=[100,20000], # ns
        sig_avg_window=50, # ns
        edge_effect_window=3, # samples
        acceptance_threshold=1e-3,
        noise=None,
        model_fit=False
        )

    @staticmethod
    def fit_dtype(ntpc,ndet):
        cov_size = 5
        dtype_spec = [
            ('valid', 'u1'),
            ('prompt_acc', 'f4', (ntpc,ndet)),
            ('delayed_acc', 'f4', (ntpc,ndet)),            
            ('prompt_f', 'f4'),
            ('prompt_ns', 'f8'),
            ('pe_vis', 'f4'),
            ('delayed_ns', 'f8'),
            ('fraction', 'f4'),
            ('tau_t', 'f4'),
            ('mse', 'f4'),
            ('mse0', 'f4'),
            ('ndf', 'i4'),
            ('cov', 'f4', (cov_size,cov_size))]
        return np.dtype(dtype_spec)

    @staticmethod
    def prompt_dtype(ntpc,ndet,nterms):
        return np.dtype([
            ('valid', 'u1'),
            ('ns', 'f8'),
            ('ampl', 'f4'),
            ('sum', 'f4'),
            ('sig', 'f4'),
            ('terms', 'f4', (nterms,ntpc,ndet))
        ])

    delayed_dtype = np.dtype([
        ('ns', 'f8'),
        ('delay', 'f4'),
        ('ampl', 'f4',),
        ('sum', 'f4'),
        ('sig', 'f4'),
        ('valid', 'u1')
    ])


    def __init__(self, **params):
        super(DelayedSignal, self).__init__(**params)
        for param,default in self.defaults.items():
            setattr(self, param, params.get(param, default))
        if self.noise is None:
            self.noise = np.ones(1)
        else:
            self.noise = np.array(self.noise)


    def init(self, source_name):
        super(DelayedSignal, self).init(source_name)

        self.sample_rate = (resources['RunData'].lrs_ticks / units.ns)
        self.is_mc = (resources['RunData'].is_mc)

        # load prompt response model from file
        nsamples = self.data_manager[self.wvfm_dset_name+'/data'].dtype['samples'].shape[-1]
        if self.is_mc:
            self.prompt_response_model = self.prompt_response_model['mc'][nsamples]
        else:
            self.prompt_response_model = self.prompt_response_model['data'][nsamples]
        if self.rank == 0:
            print('Using response model:', self.prompt_response_model)
        d = np.load(self.prompt_response_model)
        self.template = d['template'][:self.response_model_terms]
        self.template_align = d['alignment'][:self.response_model_terms]
        self.res_dev = d['res_dev'][self.response_model_terms-1]
        self.res_dev[self.res_dev == 0] = 1e9

        # create datatypes
        self.fit_dtype = self.fit_dtype(*self.data_manager[self.wvfm_dset_name+'/data'].dtype['samples'].shape[:-1])
        self.prompt_dtype = self.prompt_dtype(*self.data_manager[self.wvfm_dset_name+'/data'].dtype['samples'].shape[:-1], self.response_model_terms)

        # set initial fit parameters
        self.p0 = (self.singlet_fraction, self.triplet_time) # fraction, triplet time

        # format output file
        attrs = dict(class_version=self.class_version, classname=self.classname)
        for param in self.defaults:
            attrs[param] = getattr(self, param)
        #attrs['response_model_template'] = self.template
        #attrs['response_model_align'] = self.template_align
        #attrs['response_model_residual'] = self.res_dev

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
        wvfm = cache[self.wvfm_dset_name]
        wvfm_align = cache[self.wvfm_align_dset_name]
        if wvfm is not None and len(wvfm):
            wvfm = wvfm.reshape((len(np.r_[source_slice]),-1))['samples']
            wvfm_align = wvfm_align.reshape((len(np.r_[source_slice]),-1))
        else:
            wvfm = ma.array(np.empty((0,1,1,1,1), dtype=wvfm.dtype['samples']))
            wvfm_align = ma.array(np.empty((0,1), dtype=wvfm_align.dtype))
        # remove samples near end of waveform
        wvfm = wvfm[...,self.edge_effect_window:-self.edge_effect_window]
        wvfm_ns = wvfm_align['ns'][...,np.newaxis,np.newaxis,np.newaxis] - (wvfm_align['sample_idx'][...,np.newaxis] - self.edge_effect_window - np.arange(wvfm.shape[-1])) * self.sample_rate

        # prep output arrays
        prompt_data = np.zeros(wvfm.shape[0], dtype=self.prompt_dtype)
        delayed_data = np.zeros(wvfm.shape[0], dtype=self.delayed_dtype)        
        fit_data = np.zeros(wvfm.shape[0], dtype=self.fit_dtype)

        hits = cache[self.hits_dset_name]
        if hits.shape[0]:
            # calculate event acceptance
            hit_drift = cache[self.hit_drift_dset_name]
            michel_id = cache[self.michel_id_dset_name]
            hit_label = cache[self.hit_label_dset_name]
            event_sel = cache[self.stopping_sel_dset_name]

            hits = hits.reshape(hit_drift.shape)
            hit_label = hit_label.reshape(hit_drift.shape)
            event_sel = event_sel.reshape(hit_drift.shape[0])
            xyz = ma.concatenate([hits['px'], hits['py'], hit_drift['z']], axis=-1)
            stop_pt = michel_id['stop_pt'].reshape(hits.shape[0],3)

            tpc,det = np.indices(wvfm.shape[-3:-1])
            acc = resources['Geometry'].solid_angle(xyz.reshape(-1,3), tpc.ravel(), det.ravel()).reshape(hits.shape + tpc.shape) / (4 * np.pi) # (ev, hits, 1, tpc, det)
            stop_pt_acc = resources['Geometry'].solid_angle(stop_pt, tpc.ravel(), det.ravel()).reshape(stop_pt.shape[:1] + tpc.shape) / (4 * np.pi)
            acc *= hits['iogroup'][...,np.newaxis,np.newaxis]-1 == tpc[np.newaxis,np.newaxis,np.newaxis] # assert same tpc only FIXME
            prompt_acc = ((acc * (hit_label['muon_flag'] * hits['q'])[...,np.newaxis,np.newaxis]).sum(axis=(1,2))
                          / (hit_label['muon_flag'] * hits['q']).sum(axis=(1,2),keepdims=True))
            delayed_acc = ((acc * (hit_label['michel_flag'] * hits['q'])[...,np.newaxis,np.newaxis]).sum(axis=(1,2))
                           / np.clip((hit_label['michel_flag'] * hits['q']).sum(axis=(1,2), keepdims=True), 1, None))
            delayed_acc[np.sum(delayed_acc, axis=(1,2)) == 0] = stop_pt_acc[np.sum(delayed_acc, axis=(1,2)) == 0]

            for iev in range(wvfm.shape[0]):
                if (np.all(wvfm[iev].mask)
                    or np.all(wvfm_ns[iev].mask)
                    or np.all(prompt_acc[iev] == 0)
                    or ~event_sel[iev]['sel'].astype(bool)):
                    continue

                # align waveforms to overlapping region only
                sample_idx = ma.array(wvfm_align[iev]['sample_idx'] - self.edge_effect_window,
                                      mask=wvfm_align[iev]['sample_idx']==0)
                last_trig_sample_idx = sample_idx.min(axis=(-1,-2,-3))
                first_trig_sample_idx = sample_idx.max(axis=(-1,-2,-3))

                trig_end_clip = wvfm.shape[-1] - (first_trig_sample_idx[...,np.newaxis,np.newaxis]- sample_idx).astype(int)
                trig_start_clip = (sample_idx - last_trig_sample_idx[...,np.newaxis,np.newaxis]).astype(int)
                overlapping_samples = int(np.min(trig_end_clip - trig_start_clip))

                iclip = np.indices(wvfm[iev].shape[:-1] + (overlapping_samples,))[-1] + trig_start_clip[...,np.newaxis]
                wvfm_clipped = np.take_along_axis(wvfm[iev], iclip, axis=-1)
                wvfm_ns_clipped = np.take_along_axis(wvfm_ns[iev], iclip, axis=-1)
                wvfm_clipped_align_sample = (first_trig_sample_idx - np.max(trig_start_clip)).astype(int)

                # fit prompt component from pca
                template_align_sample = (self.template_align/32).astype(int)
                first_trig_sample_idx = np.maximum(wvfm_clipped_align_sample, template_align_sample[np.newaxis])
                last_trig_sample_idx = np.minimum(wvfm_clipped_align_sample, template_align_sample[np.newaxis])
                template_end_clip = min(self.template.shape[-1], wvfm_clipped.shape[-1]) - (first_trig_sample_idx - template_align_sample[np.newaxis])
                template_start_clip = template_align_sample - last_trig_sample_idx
                wvfm_end_clip = min(self.template.shape[-1], wvfm_clipped.shape[-1]) - (first_trig_sample_idx - wvfm_clipped_align_sample)
                wvfm_start_clip = wvfm_clipped_align_sample - last_trig_sample_idx
                overlapping_samples = int(np.min(template_end_clip - template_start_clip))

                iclip_template = np.indices(
                    wvfm.shape[1:2]
                    + self.template.shape[1:-1]
                    + (overlapping_samples,))[-1] + template_start_clip[...,np.newaxis] # (ntrig, ntpc, ndet, nsamples)
                template_clipped = np.take_along_axis(self.template[:,np.newaxis], iclip_template[np.newaxis], axis=-1) # (ntemp, ntrig, ntpc, ndet, nsamples)
                res_dev_clipped = np.take_along_axis(self.res_dev[np.newaxis], iclip_template, axis=-1) # (ntrig, ntpc, ndet, nsamples)
                template_norm = np.sum(template_clipped**2, axis=-1).clip(1e-100,None)
                template_clipped_align_sample = (template_align_sample[:,np.newaxis] - template_start_clip).astype(int)
                iclip_wvfm = np.indices(wvfm_clipped.shape[:-1] + (overlapping_samples,))[-1] + wvfm_start_clip[...,np.newaxis]
                wvfm_clipped = np.take_along_axis(wvfm_clipped, iclip_wvfm, axis=-1)
                wvfm_ns_clipped = np.take_along_axis(wvfm_ns_clipped, iclip_wvfm, axis=-1)
                wvfm_clipped_align_sample = (wvfm_clipped_align_sample - np.max(wvfm_start_clip)).astype(int)
                
                # find prompt trigger
                wvfm_align_ns = ma.array(
                    wvfm_align[iev]['ns'],
                    mask=wvfm_align[iev]['ns'] == 0)
                ifirst_trigger = np.argmin(wvfm_align_ns, axis=-1)
                first_trigger_ns = wvfm_align_ns[ifirst_trigger]

                # fit to prompt components
                res = wvfm_clipped[ifirst_trigger].copy()
                prompt_f = np.zeros(template_clipped.shape[0:1] + wvfm_clipped.shape[1:-1])
                for ic in range(template_clipped.shape[0]):
                    prompt_f[ic] = np.sum(res * template_clipped[ic,ifirst_trigger], axis=-1)
                    prompt_f[ic] /= template_norm[ic,ifirst_trigger]
                    res -= prompt_f[ic,...,np.newaxis] * template_clipped[ic,ifirst_trigger]

                prompt_signal = np.sum(prompt_f[:,...,np.newaxis] * template_clipped[:,ifirst_trigger], axis=0)
                prompt_signal_ns = wvfm_ns_clipped[ifirst_trigger]

                # calculate delayed significance
                min_ns = np.min(wvfm_ns_clipped)
                n_ticks = int(np.ceil((np.max(wvfm_ns_clipped) - min_ns) / self.sample_rate))
                xdata = np.linspace(min_ns, min_ns + self.sample_rate * (n_ticks+1), n_ticks+2)
                ydata = np.zeros_like(xdata)
                ydata_prompt_sub = np.zeros_like(xdata)
                ysig = np.zeros_like(xdata)
                ysig_prompt_sub = np.zeros_like(xdata)                
                mask = np.zeros_like(xdata, dtype=bool)
                for itrig in range(wvfm.shape[1]):
                    if itrig == ifirst_trigger:
                        prompt_trigger = True
                    else:
                        prompt_trigger = False
                    for itpc in range(wvfm.shape[2]):
                        for idet in range(wvfm.shape[3]):
                            if idet%4 == 0: # skip ArcLights
                                continue
                            if np.any(~wvfm_clipped[itrig,itpc,idet,:].mask) or np.any(~wvfm_ns_clipped[itrig,itpc,idet,:].mask):
                                subset = (xdata >= wvfm_ns_clipped[itrig,itpc,idet,0]) & (xdata < wvfm_ns_clipped[itrig,itpc,idet,-1] + self.sample_rate)
                                mask[subset] = True
                                yinterp = np.interp(xdata[subset], wvfm_ns_clipped[itrig,itpc,idet,:], wvfm_clipped[itrig,itpc,idet,:], left=0, right=0)
                                yinterp_prompt_sub = np.interp(xdata[subset], wvfm_ns_clipped[itrig,itpc,idet,:], wvfm_clipped[itrig,itpc,idet,:] - prompt_signal[itpc,idet,:] * prompt_trigger, left=0, right=0)
                                noise = res_dev_clipped[0,itpc,idet] * prompt_trigger + res_dev_clipped[0,itpc,idet].mean() * ~prompt_trigger
                                
                                ydata[subset] += yinterp
                                ydata_prompt_sub[subset] += yinterp_prompt_sub
                                ysig[subset] += (np.sign(yinterp) * (yinterp**2) / noise**2)

                                ysig_prompt_sub[subset] += (np.sign(yinterp_prompt_sub) * (yinterp_prompt_sub**2) / noise**2)
                                
                avg_samples = int(self.sig_avg_window/self.sample_rate)
                #ydata_sliding_window = np.convolve(ysig, np.ones(avg_samples)/avg_samples, 'same')
                ydata_sum = np.convolve(ydata, np.ones(avg_samples), 'same')[mask]
                ydata_prompt_sub_sum = np.convolve(ydata_prompt_sub, np.ones(avg_samples), 'same')[mask]                
                xdata = xdata[mask]
                ydata = ydata[mask]
                ydata_prompt_sub = ydata_prompt_sub[mask]
                #ysig = ysig[mask]
                ysig = np.convolve(ysig, np.ones(avg_samples)/avg_samples, 'same')[mask]
                #ysig_prompt_sub = ysig_prompt_sub[mask]
                ysig_prompt_sub = np.convolve(ysig_prompt_sub, np.ones(avg_samples)/avg_samples, 'same')[mask]
                #ydata_sliding_window = ydata_sliding_window[mask]
            
                # guess prompt signal time and amplitude
                # definition:
                #  - largest sample of sliding sum
                #  - refined with prompt-only fit
                first_trigger_ns = wvfm_align[iev]['ns'].min()
                prompt_mask = ((xdata - first_trigger_ns >= self.prompt_window[0])
                               & (xdata - first_trigger_ns < self.prompt_window[1]))
                if not np.any(prompt_mask):
                    continue
                else:
                    prompt_data[iev]['valid'] = True
                #prompt_sample = np.argmax(ydata_sliding_window[prompt_mask])
                prompt_sample = np.argmax(ysig[prompt_mask])
                prompt_ns = xdata[prompt_mask][prompt_sample]
                prompt_ampl = ydata[prompt_mask][prompt_sample]
                prompt_sum = ydata_sum[prompt_mask][prompt_sample]                
                prompt_data[iev]['ns'] = prompt_ns
                prompt_data[iev]['ampl'] = prompt_ampl
                prompt_data[iev]['sum'] = prompt_sum                
                prompt_data[iev]['sig'] = ysig[prompt_mask][prompt_sample]
                prompt_data[iev]['terms'] = prompt_f
                #prompt_data[iev]['sig'] = ydata_sliding_window[prompt_mask][prompt_sample]

                # skip fit on detectors with little expected signal (or ArcLights)
                trig,tpc,det = np.where(~np.any(wvfm_ns_clipped.mask, axis=-1) & ((prompt_acc[iev] > self.acceptance_threshold) | (delayed_acc[iev] > self.acceptance_threshold))[np.newaxis])
                mask = det%4 != 0 # exclude ArcLights from fit
                trig = trig[mask]
                tpc = tpc[mask]
                det = det[mask]

                prompt_acc_norm = prompt_acc[iev,tpc,det].sum()
                delayed_acc_norm = delayed_acc[iev,tpc,det].sum()
                pe_vis = wvfm[iev,trig,tpc,det].sum() * self.sample_rate # PE/tick -> PE
                
                # guess delayed signal time and amplitude
                # definition:
                #  - largest positive deviation from prompt guess, within delayed window
                #  - refined with prompt + delayed fit
                delayed_mask = ((xdata - prompt_ns >= self.delayed_window[0])
                                & (xdata - prompt_ns < self.delayed_window[1]))
                if not np.any(delayed_mask):
                    continue
                else:
                    delayed_data[iev]['valid'] = True

                '''
                delayed_sig = np.zeros_like(ydata)[delayed_mask]
                for itrig in range(wvfm.shape[1]):
                    for itpc in range(wvfm.shape[2]):
                        for idet in range(wvfm.shape[3]):
                            if idet%4 == 0:
                                continue
                            if np.any(~wvfm[iev,itrig,itpc,idet,:].mask) or np.any(~wvfm_ns[iev,itrig,itpc,idet,:].mask):
                                # prompt model assumes 100% of waveform energy contained in prompt signal
                                subset = (xdata >= wvfm_ns[iev,itrig,itpc,idet,0]) & (xdata <= wvfm_ns[iev,itrig,itpc,idet,-1])
                                yinterp = (np.interp(xdata[delayed_mask], wvfm_ns[iev,itrig,itpc,idet,:].compressed(), wvfm[iev,itrig,itpc,idet,:].compressed(), left=0, right=0)
                                           - f_delayed(xdata[delayed_mask], pe_vis * prompt_acc[iev,itpc,idet]/prompt_acc_norm, prompt_ns, 0, 0))
                                delayed_sig += np.sign(yinterp) * (yinterp)**2 / (self.noise[itpc,idet] * self.noise_factor if self.noise.ndim > 1 else self.noise*self.noise_factor)**2
                delayed_sig = np.convolve(delayed_sig, np.ones(avg_samples)/avg_samples, 'same')
                '''
                delayed_sample = np.argmax(ysig_prompt_sub[delayed_mask])
                #delayed_sample = np.argmax(delayed_sig)
                delayed_ns = xdata[delayed_mask][delayed_sample]
                delayed_ampl = ydata_prompt_sub[delayed_mask][delayed_sample]
                delayed_sum = ydata_prompt_sub_sum[delayed_mask][delayed_sample]
                #delayed_ampl = ydata[delayed_mask][delayed_sample]                
                delayed_data[iev]['ns'] = delayed_ns
                delayed_data[iev]['ampl'] = delayed_ampl
                delayed_data[iev]['sum'] = delayed_sum                
                delayed_data[iev]['delay'] = delayed_ns - prompt_ns
                delayed_data[iev]['sig'] = ysig_prompt_sub[delayed_mask][delayed_sample]
                #delayed_data[iev]['sig'] = delayed_sig[delayed_sample]                

                t_offset_ns = xdata.min()
                p0 = (prompt_ampl / (prompt_ampl + delayed_ampl), prompt_ns - t_offset_ns, delayed_ns - prompt_ns) + self.p0
                fit_args = (np.array(wvfm_ns[iev,trig,tpc,det,:], dtype='f8'),
                    np.array(wvfm[iev,trig,tpc,det,:], dtype='f4'),
                    prompt_acc[iev,tpc,det]/prompt_acc_norm,
                    delayed_acc[iev,tpc,det]/delayed_acc_norm,
                    self.noise[tpc,det] * self.noise_factor if self.noise.ndim > 1
                        else self.noise*self.noise_factor,
                    False,
                    t_offset_ns
                )
                if self.model_fit:
                    fit_result = optimize.minimize(
                        loss, p0, args=fit_args,
                        method='Powell',
                        #method='Nelder-Mead',
                        bounds=[(0,1)]+[(0,np.inf) for _ in p0[1:-2]] + [(0,1), (0,np.inf)],
                        options=dict(disp=False)
                    )
                    p = tuple(fit_result.x)
                    pnull = list(fit_result.x)
                else:
                    p = tuple(p0)
                    pnull = list(p)
                pnull[2] = np.inf # push delayed time to infinity
                mse0 = loss(p0, *fit_args) if fit_args[0].size else 0
                    
                if self.model_fit and not fit_result.success:
                    print('fit failed on ',np.r_[source_slice][iev],'because',fit_result.message)

                fit_data[iev]['prompt_acc'] = prompt_acc[iev] * (np.arange(wvfm.shape[-2]) % 4 != 0) # exclude arclight
                fit_data[iev]['delayed_acc'] = delayed_acc[iev] * (np.arange(wvfm.shape[-2]) % 4 != 0) # exclude arclight
                fit_data[iev]['valid'] = fit_result.success if self.model_fit else False
                fit_data[iev]['prompt_f'] = p[0]
                fit_data[iev]['prompt_ns'] = p[1] + t_offset_ns
                fit_data[iev]['delayed_ns'] = p[2]
                fit_data[iev]['pe_vis'] = pe_vis
                #fit_data[iev]['cov'] = fit_result.jac.T @ fit_result.jac * fit_result.fun**2
                fit_data[iev]['mse'] = fit_result.fun if self.model_fit else loss(p, *fit_args)
                fit_data[iev]['mse0'] = mse0
                fit_data[iev]['ndf'] = wvfm_ns[iev,trig,tpc,det,:].size - len(p)
                fit_data[iev]['fraction'] = p[3]
                fit_data[iev]['tau_t'] = p[4]


                ## FOR DEBUG ONLY!
                # plots each analyzed event
                if False: #(not fit_result.success and resources['RunData'].is_mc):
                    print('prompt', prompt_data[iev])
                    print('delayed', delayed_data[iev])
                    print('fit', fit_data[iev])

                    imax0,imax1 = ma.argsort((wvfm[iev] * delayed_acc[iev,...,np.newaxis] * (np.arange(wvfm.shape[-2]) % 4 != 0)[...,np.newaxis]).sum(axis=(0,-1)).ravel())[-2:].tolist()
                    import matplotlib.pyplot as plt
                    plt.ion()
                    
                    # plot prompt/delayed significance
                    fig,axes = plt.subplots(2, 1, num=3, dpi=100, sharex='all')
                    axes[0].plot(xdata[prompt_mask], ydata[prompt_mask], color='k', label='sum')
                    axes[0].plot(xdata[~prompt_mask], ydata[~prompt_mask], color='k', ls=':')
                    axes[0].plot(prompt_signal_ns.mean(axis=(0,1)), prompt_signal.sum(axis=(0,1)), label='prompt signal', color='r')
                    axes[0].plot(xdata[delayed_mask], ydata_prompt_sub[delayed_mask], color='b', label='prompt subtracted')
                    axes[0].plot(xdata[~delayed_mask], ydata_prompt_sub[~delayed_mask], color='b', ls=':')
                    axes[1].plot(xdata[prompt_mask], ysig[prompt_mask], color='r', label='prompt significance')
                    axes[1].plot(xdata[~prompt_mask], ysig[~prompt_mask], color='r', ls=':')
                    axes[1].plot(xdata[delayed_mask], ysig_prompt_sub[delayed_mask], color='b', label='delayed significance')
                    axes[1].plot(xdata[~delayed_mask], ysig_prompt_sub[~delayed_mask], color='b', ls=':')                    
                    axes[1].set_xlabel('time [ticks]')
                    axes[0].legend()
                    axes[1].legend()                    
                    axes[1].set_yscale('log')
                    
                    fig,axes = plt.subplots(1,2,num=2, dpi=100, sharey='all')
                    mask = hit_label[iev]['michel_flag'].astype(bool).ravel()
                    axes[0].scatter(xyz[iev][mask,0].compressed(), xyz[iev][mask,1].compressed(), c='r', s=1, marker='s')
                    axes[1].scatter(xyz[iev][mask,2].compressed(), xyz[iev][mask,1].compressed(), c='r', s=1, marker='s')                    
                    mask = hit_label[iev]['muon_flag'].astype(bool).ravel()
                    axes[0].scatter(xyz[iev][mask,0].compressed(), xyz[iev][mask,1].compressed(), c='b', s=1, marker='s')
                    axes[1].scatter(xyz[iev][mask,2].compressed(), xyz[iev][mask,1].compressed(), c='b', s=1, marker='s')                    
                    axes[0].set_xlabel('x [mm]')
                    axes[1].set_xlabel('z [mm]')                    
                    axes[1].set_ylabel('y [mm]')
                    axes[0].set_aspect('equal')
                    axes[1].set_aspect('equal')                    
                    plt.show()

                    fig,axes = plt.subplots(2, 4, num=1, dpi=100, figsize=(12,6), sharex='all')
                    if 'mc_truth' in self.data_manager['/']:
                        true_parent = self.data_manager['analysis/muon_capture/truth_labels/stopping_track','mc_truth/tracks', np.r_[source_slice][iev]]['t0'].ravel()[0]
                        true_decay = self.data_manager['analysis/muon_capture/truth_labels/michel_track','mc_truth/tracks', np.r_[source_slice][iev]]['t0'].ravel()[0]
                        if true_decay > 0:
                            for ax in axes:
                                for a in ax:
                                    a.axvline((true_decay - true_parent)*1e3 + p[1] + t_offset_ns, color='k', ls=':', label='true decay')
                    for itpc in range(prompt_acc[iev].shape[0]):
                        for idet in range(prompt_acc[iev].shape[1]):
                            pinit = (p0[0] * prompt_acc[iev,itpc,idet]/prompt_acc_norm, p0[1] + t_offset_ns, (1-p0[0]) * delayed_acc[iev,itpc,idet]/delayed_acc_norm, p0[2], p0[3], p0[4])
                            pbest = (p[0] * prompt_acc[iev,itpc,idet]/prompt_acc_norm, p[1] + t_offset_ns, (1-p[0]) * delayed_acc[iev,itpc,idet]/delayed_acc_norm, p[2], p[3], p[4])
                            t = wvfm_ns[iev,:,itpc,idet,:].reshape(-1,wvfm_ns.shape[-1]).compressed()
                            y = wvfm[iev,:,itpc,idet,:].reshape(-1,wvfm_ns.shape[-1]).compressed()
                            order = np.argsort(t)
                            t = t[order]
                            y = y[order]

                            offset = idet//4
                            
                            axes[itpc,idet%4].plot(t, y, '.', color=f'C{offset}', label='largest waveform')
                            axes[itpc,idet%4].plot(t, pe_vis * f_delayed(t, *pinit), ':', color=f'C{offset}', label='fit initialization')
                            axes[itpc,idet%4].plot(t, pe_vis * f_delayed(t, *pbest), color=f'C{offset}', label='fit (both)')
                            axes[itpc,idet%4].axvline(prompt_data[iev]['ns'], color=f'C{offset}', lw=1, ls=':')
                            axes[itpc,idet%4].axvline(delayed_data[iev]['ns'], color=f'C{offset}', lw=1, ls=':')
                            axes[itpc,idet%4].axvline(fit_data[iev]['prompt_ns'], color=f'C{offset}', lw=1, ls='-')
                            axes[itpc,idet%4].axvline(fit_data[iev]['prompt_ns']+fit_data[iev]['delayed_ns'], color=f'C{offset}', lw=1, ls='-')
                            axes[itpc,idet%4].axvline(first_trigger_ns + self.prompt_window[0], color='k', lw=1, ls='--', label='prompt window')
                            axes[itpc,idet%4].axvline(first_trigger_ns + self.prompt_window[1], color='k', lw=1, ls='--')

                    fig,axes = plt.subplots(2, 4, num=4, dpi=100, figsize=(12,6), sharex='all')
                    if 'mc_truth' in self.data_manager['/']:
                        true_parent = self.data_manager['analysis/muon_capture/truth_labels/stopping_track','mc_truth/tracks', np.r_[source_slice][iev]]['t0'].ravel()[0]
                        true_decay = self.data_manager['analysis/muon_capture/truth_labels/michel_track','mc_truth/tracks', np.r_[source_slice][iev]]['t0'].ravel()[0]
                        if true_decay > 0:
                            for ax in axes:
                                for a in ax:
                                    a.axvline((true_decay - true_parent)*1e3 + p[1] + t_offset_ns, color='k', ls=':', label='true decay', zorder=np.inf)
                    for itpc in range(prompt_acc[iev].shape[0]):
                        for idet in range(prompt_acc[iev].shape[1]):
                            pinit = (p0[0] * prompt_acc[iev,itpc,idet]/prompt_acc_norm, p0[1] + t_offset_ns, (1-p0[0]) * delayed_acc[iev,itpc,idet]/delayed_acc_norm, p0[2], p0[3], p0[4])
                            pbest = (p[0] * prompt_acc[iev,itpc,idet]/prompt_acc_norm, p[1] + t_offset_ns, (1-p[0]) * delayed_acc[iev,itpc,idet]/delayed_acc_norm, p[2], p[3], p[4])
                            t = wvfm_ns[iev,:,itpc,idet,:].reshape(-1,wvfm_ns.shape[-1]).compressed()
                            y = wvfm[iev,:,itpc,idet,:].reshape(-1,wvfm_ns.shape[-1]).compressed()
                            order = np.argsort(t)
                            t = t[order]
                            y = y[order]

                            offset = idet//4
                            axes[itpc,idet%4].plot(t, np.cumsum(y), '.', color=f'C{offset}', label='largest waveform')
                            axes[itpc,idet%4].plot(t, np.cumsum(pe_vis * f_delayed(t, *pinit)), ':', color=f'C{offset}', label='fit initialization')
                            axes[itpc,idet%4].plot(t, np.cumsum(pe_vis * f_delayed(t, *pbest)), color=f'C{offset}', label='fit (both)')

                    plt.show(block=True)

        # save data to file
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
        ref = np.c_[source_slice, prompt_slice][prompt_data['valid'].astype(bool)]
        if len(ref) == 0:
            ref = np.empty((0,2), int)
        self.data_manager.write_ref(source_name, self.prompt_dset_name, ref)

        delayed_slice = self.data_manager.reserve_data(
            self.delayed_dset_name, len(delayed_data))
        self.data_manager.write_data(self.delayed_dset_name, delayed_slice, delayed_data)
        ref = np.c_[prompt_slice, delayed_slice][delayed_data['valid'].astype(bool)]
        if len(ref) == 0:
            ref = np.empty((0,2), int)
        self.data_manager.write_ref(self.prompt_dset_name, self.delayed_dset_name, ref)
