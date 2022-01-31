import numpy as np
import numpy.ma as ma
import scipy.ndimage as ndimage
import logging
import os

from h5flow import H5FLOW_MPI
from h5flow.core import H5FlowStage


def fill_hist(adc, det, rel_time, rel_ampl, bkg_hist, *bins):
    pairs = np.concatenate([np.expand_dims(adc,-1), np.expand_dims(det,-1), np.expand_dims(rel_time,-1), np.expand_dims(rel_ampl,-1)], axis=-1)
    return np.histogramdd(pairs.reshape(-1,4), bins=bins)[0] + bkg_hist


def normalize_hist(hist, sigma=1):
    cum_norm_hist = np.cumsum(hist, axis=-1)
    cum_norm_hist /= np.clip(np.sum(hist, axis=-1, keepdims=True), 1e-15, None)
    cum_norm_hist = 1 - cum_norm_hist
    if sigma is not None:
        cum_norm_hist = ndimage.gaussian_filter1d(cum_norm_hist, sigma, axis=-1, mode='nearest')
    return cum_norm_hist


def score_delayed(adc, det, rel_time, rel_ampl, bkg_cum_norm_hist, *bins):
    i_adc = np.clip(np.digitize(adc, bins=bins[0])-1,0,len(bins[0])-2)
    i_det = np.clip(np.digitize(det, bins=bins[1])-1,0,len(bins[1])-2)
    i_time = np.clip(np.digitize(rel_time, bins=bins[2])-1,0,len(bins[2])-2)
    i_ampl = np.clip(np.digitize(rel_ampl, bins=bins[3])-1,0,len(bins[3])-2)

    return 1 - bkg_cum_norm_hist[i_adc, i_det, i_time, i_ampl]


class DelayedSignal(H5FlowStage):
    class_version = '0.0.0'

    defaults = dict(
        hits_dset_name='light/hits',
        prompt_dset_name='analysis/muon_capture/prompt',
        delayed_dset_name='analysis/muon_capture/delayed',
        prompt_threshold_factor=1,
        prompt_window=[-350,-250], # ns
        delayed_window=[100,1600], # ns
<<<<<<< Updated upstream
        delayed_hit_window=10, # ns
        delayed_likelihood_cut=1.,
=======
        delayed_hit_window=20, # ns
        delayed_likelihood_cut=0.10,
>>>>>>> Stashed changes
        calibration_flag=False,
        delayed_bkg_file='h5flow_data/delayed_bkg-{class_version}.npz',
        bkg_bins=(np.linspace(100,1600,150), np.geomspace(1e-4,2,50)),
        )


    @staticmethod
    def prompt_dtype(nadc, ndet):
        return np.dtype([
            ('valid', 'u1'),
            ('ns', 'f8'),
            ('ampl', 'f8', (nadc, ndet)),
            ])


    @staticmethod
    def delayed_dtype(nadc, ndet):
        return np.dtype([
            ('ns', 'f8'),
            ('delay', 'f8'),
            ('ampl', 'f8', (nadc, ndet)),
            ('score', 'f8'),
            ('valid', 'u1')
            ])


    def __init__(self, **params):
        super(DelayedSignal, self).__init__(**params)
        for param,default in self.defaults.items():
            setattr(self, param, params.get(param, default))

        self.delayed_bkg_file = self.delayed_bkg_file.format(class_version=self.class_version)
        self.bkg_bins = [np.array(bins) for bins in self.bkg_bins]


    def init(self, source_name):
        super(DelayedSignal, self).init(source_name)

        # get number of adcs/detectors
        hit_attrs = self.data_manager.get_attrs(self.hits_dset_name)
        nadc = hit_attrs['nadc']
        ndet = hit_attrs['nchannels']
        self.prompt_dtype = self.prompt_dtype(nadc, ndet)
        self.delayed_dtype = self.delayed_dtype(nadc, ndet)

        # add set of bins for each detector
        self.bkg_bins = [np.arange(nadc+1), np.arange(ndet+1)] + self.bkg_bins

        # set prompt signal thresholds
        thresholds = hit_attrs['thresholds']
        self.prompt_thresholds = self.prompt_threshold_factor * thresholds

        # load / setup calibration file
        if os.path.exists(self.delayed_bkg_file) and not self.calibration_flag:
            delayed_data = np.load(self.delayed_bkg_file)
            self.bkg_bins = [delayed_data[key] for key in sorted(list(delayed_data.keys())) if 'bins' in key]
            self.bkg_cum_norm_hist = delayed_data['hist']
        else:
            self.bkg_cum_norm_hist = np.ones([len(bins)-1 for bins in self.bkg_bins])

        self.bkg_cum_norm_hist = normalize_hist(self.bkg_cum_norm_hist)

        self.bkg_hist = np.zeros([len(bins)-1 for bins in self.bkg_bins])

        # format output file
        if not self.calibration_flag:
            attrs = dict(class_version=self.class_version, classname=self.classname)
            for param in self.defaults:
                attrs[param] = getattr(self, param)
            attrs['prompt_thresholds'] = self.prompt_thresholds
            del attrs['bkg_bins']
            for i in range(len(self.bkg_bins)):
                attrs[f'bkg_bins{i}'] = self.bkg_bins[i]

            self.data_manager.create_dset(self.prompt_dset_name, dtype=self.prompt_dtype)
            self.data_manager.create_dset(self.delayed_dset_name, dtype=self.delayed_dtype)
            self.data_manager.create_ref(source_name, self.prompt_dset_name)
            self.data_manager.create_ref(self.prompt_dset_name, self.delayed_dset_name)

            self.data_manager.set_attrs(self.prompt_dset_name, **attrs)
            self.data_manager.set_attrs(self.delayed_dset_name, **attrs)


    def finish(self, source_name):
        super(DelayedSignal, self).finish(source_name)

        if self.calibration_flag:
            if H5FLOW_MPI:
                self.bkg_hist = self.comm.gather(self.bkg_hist, root=0)
                self.bkg_hist = np.sum(self.bkg_hist, axis=0)

            if self.rank == 0:
                logging.info(f'Saving background histogram to {self.delayed_bkg_file}...')
                np.savez_compressed(self.delayed_bkg_file,
                    bins0=self.bkg_bins[0],
                    bins1=self.bkg_bins[1],
                    bins2=self.bkg_bins[2],
                    bins3=self.bkg_bins[3],
                    hist=self.bkg_hist
                    )


    def run(self, source_name, source_slice, cache):
        super(DelayedSignal, self).run(source_name, source_slice, cache)
        # load from cache
        hits = cache[self.hits_dset_name]
        if len(hits):
            hits = hits.reshape(len(np.r_[source_slice]),-1)

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
            & (hits['max_spline'] >= self.prompt_thresholds[hits['adc'].filled(0), hits['ch'].filled(0)].reshape(hits.shape))
            )
        hit_ns = hits['ns'] + hits['busy_ns'] + hits['ns_spline']
        hit_ns.mask = hit_ns.mask | ~prompt_hit_mask
        prompt_ns = ma.average(hit_ns, axis=-1, weights=hits['max_spline'])
        prompt_ampl = np.zeros(hits.shape[0:1] + self.prompt_dtype['ampl'].shape)
        for i in range(prompt_ampl.shape[1]):
            for j in range(prompt_ampl.shape[2]):
                hit_submask = (hits['adc'] == i) & (hits['ch'] == j) & prompt_hit_mask
                if np.any(hit_submask):
                    prompt_ampl[:,i,j] = (hit_submask * hits['max_spline']).sum(axis=-1)
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
                hit_submask = ((hits['adc'] == i) & (hits['ch'] == j)
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
                    hit_rel_ampl[hit_submask] = hits[hit_submask]['max_spline'] / np.clip(p_ampl[hit_submask],1e-15,None)
                    hit_rel_valid[hit_submask] = (p_ampl[hit_submask] > 0)

        if self.calibration_flag:
            # fill likelihood histogram
            self.bkg_hist = fill_hist(
                hits['adc'][hit_rel_valid], hits['ch'][hit_rel_valid],
                hit_rel_ns[hit_rel_valid], hit_rel_ampl[hit_rel_valid],
                self.bkg_hist, *self.bkg_bins)

        else:
            # look for delayed signal
            hit_score = score_delayed(
                hits['adc'], hits['ch'],
                hit_rel_ns * hit_rel_valid, hit_rel_ampl * hit_rel_valid,
                self.bkg_cum_norm_hist, *self.bkg_bins)

            # find best delayed time interval
            # definition:
            #  - create a sliding window
            #  - find window with largest energy
            #  - use weighted mean of relative hit time of hits within window
            hit_mask = ((hit_score < self.delayed_likelihood_cut)
                        & (hit_rel_valid))

            sliding_center = np.linspace(
                self.delayed_window[0]+self.delayed_hit_window,
                self.delayed_window[1]-self.delayed_hit_window,
                2*int(np.ceil((self.delayed_window[1] - self.delayed_window[0] - 2*self.delayed_hit_window)/self.delayed_hit_window)))
            sliding_center = sliding_center[np.newaxis,np.newaxis,:]
            hit_in_window = (hit_mask[...,np.newaxis]
                             & (hit_rel_ns[...,np.newaxis] < sliding_center + self.delayed_hit_window)
                             & (hit_rel_ns[...,np.newaxis] >= sliding_center - self.delayed_hit_window))
            sliding_score = np.sum(hit_in_window * hits['max_spline'][...,np.newaxis], axis=-2)
            if len(sliding_score):
                delayed_ns_med = np.take_along_axis(sliding_center[0], np.argmax(sliding_score, axis=-1)[...,np.newaxis], axis=-1)
            else:
                delayed_ns_med = np.empty((0,1))
            delayed_ns_min = delayed_ns_med - self.delayed_hit_window
            delayed_ns_max = delayed_ns_med + self.delayed_hit_window

            # calculate delayed signal parameters
            hit_in_window = hit_mask & (hit_rel_ns < delayed_ns_max) & (hit_rel_ns >= delayed_ns_min)
            if np.any(hit_in_window):
                delayed_time = ma.average(hit_rel_ns, axis=-1, weights=hit_in_window * hits['max_spline'])
            else:
                delayed_time = np.zeros(hit_rel_ns.shape[0])
            delayed_ns = prompt_ns + delayed_time
            delayed_score = -np.sum(hit_in_window * np.log(np.clip(hit_score,1e-15,None)), axis=-1)
            delayed_valid = np.any(hit_in_window, axis=-1)
            delayed_ampl = np.zeros_like(prompt_ampl)
            for i in range(delayed_ampl.shape[1]):
                for j in range(delayed_ampl.shape[2]):
                    hit_submask = (hits['adc'] == i) & (hits['ch'] == j) & hit_in_window
                    if np.any(hit_submask):
                        delayed_ampl[:,i,j] = (hit_submask * hits['max_spline']).sum(axis=-1)
                        delayed_ampl[~np.any(hit_submask, axis=-1),i,j] = 0

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
                delayed_data['score'] = delayed_score
                delayed_data['valid'] = delayed_valid

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
