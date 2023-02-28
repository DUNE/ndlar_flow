import numpy as np
import numpy.ma as ma
import os
import scipy.ndimage as ndimage
import logging

from h5flow import H5FLOW_MPI
from h5flow.core import H5FlowStage, resources

from module0_flow.util import units
from module0_flow.analysis.michel_id import load_likelihood_pdf


def fill_likelihood_pdf(cos_mu, d, sig_label, pdf_sig, pdf_bkg,
                        cos_mu_bins, d_bins):
    v = np.concatenate(
        [np.expand_dims(cos_mu.ravel(), -1), np.expand_dims(d.ravel(), -1)], axis=-1)

    sig = pdf_sig + np.histogramdd(v[sig_label], (cos_mu_bins, d_bins))[0]
    bkg = pdf_bkg + np.histogramdd(v[~sig_label], (cos_mu_bins, d_bins))[0]

    return sig, bkg


def bkg_likelihood_score(cos_mu, d, pdf_sig, pdf_bkg,
                            cos_mu_bins, d_bins):
    cos_mu_i = np.clip(np.digitize(cos_mu, cos_mu_bins)-1, 0, len(cos_mu_bins)-2)
    d_i = np.clip(np.digitize(d, d_bins)-1, 0, len(d_bins)-2)

    p_sig = pdf_sig[cos_mu_i, d_i]
    p_bkg = pdf_bkg[cos_mu_i, d_i]

    with np.errstate(divide='ignore', invalid='ignore'):
        score = np.log(p_sig) - np.log(p_bkg)
        if np.any(p_bkg > 0):
            score[p_bkg == 0] = (np.log(p_sig)-np.log(p_bkg[p_bkg>0].min()))[p_bkg == 0]
        if np.any(p_sig > 0):
            score[p_sig == 0] = (np.log(p_sig[p_sig > 0].min())-np.log(p_bkg))[p_sig == 0]
        score[(p_bkg == 0) & (p_sig == 0)] = 0

    return score


class BackgroundID(H5FlowStage):
    class_version = '0.0.0'

    defaults = dict(
        hits_dset_name='charge/hits',
        charge_dset_name='charge/hits',
        drift_dset_name='combined/hit_drift',
        profile_dset_name='analysis/stopping_muons/event_profile',
        stopping_sel_dset_name='analysis/stopping_muons/event_sel_reco',
        hit_prof_dset_name='analysis/stopping_muons/hit_profile',

        hit_label_dset_name='analysis/michel_id/hit_bkg_label',
        bkg_label_dset_name='analysis/michel_id/bkg_label',

        score_cut=0.0,
        likelihood_pdf_filename='bkg_pdf-{version}.npz',
        generate_likelihood_pdf=False,
        update_likelihood=True
        )

    bkg_label_dtype = np.dtype([
        ('bkg_nhit', 'i4'),
        ('bkg_q', 'f4'),
        ('muon_dir', 'f8', (3,)),
        ('stop_pt', 'f8', (3,))
    ])

    hit_label_dtype = np.dtype([
        ('score', 'f8'),
        ('bkg_flag', 'u1'),
        ('muon_flag', 'u1')
    ])

    likelihood_bins = (np.linspace(-1, 1, 50), np.geomspace(2, 1000, 50))

    def __init__(self, **params):
        super(BackgroundID, self).__init__(**params)
        for name, default in self.defaults.items():
            setattr(self, name, params.get(name, default))

        self.likelihood_pdf_filename = self.likelihood_pdf_filename.format(version=self.class_version)

    def init(self, source_name):
        super(BackgroundID, self).init(source_name)

        self.larpix_gain = self.data_manager.get_attrs('/'.join(self.stopping_sel_dset_name.split('/')[:-1]))['larpix_gain']
        if self.rank == 0:
            logging.warn(f'Using larpix gain of {self.larpix_gain:0.04f}')

        # load likelihood function
        if not self.generate_likelihood_pdf:
            pdf = load_likelihood_pdf(self.likelihood_pdf_filename)
            self.bkg_likelihood_args = pdf['sig'], pdf['bkg'], pdf['bins0'], pdf['bins1']
        else:
            sig = np.zeros([len(bins)-1 for bins in self.likelihood_bins])
            bkg = np.zeros([len(bins)-1 for bins in self.likelihood_bins])
            self.bkg_likelihood_args = [sig, bkg] + [bins for bins in self.likelihood_bins]

        # save attributes
        attrs = dict(classname=self.classname, class_version=self.class_version)
        for name in self.defaults.keys():
            attrs[name] = getattr(self, name)
        self.data_manager.set_attrs(self.bkg_label_dset_name, **attrs)

        # create datasets
        self.data_manager.create_dset(self.bkg_label_dset_name, self.bkg_label_dtype)
        self.data_manager.create_dset(self.hit_label_dset_name, self.hit_label_dtype)
        self.data_manager.create_ref(self.hit_label_dset_name, self.hits_dset_name)

    def finish(self, source_name):
        if self.generate_likelihood_pdf:
            # gather likelihood histograms from all threads
            if H5FLOW_MPI:
                self.bkg_likelihood_args[0] = np.array(self.comm.gather(self.bkg_likelihood_args[0], root=0))
                self.bkg_likelihood_args[0] = np.sum(self.bkg_likelihood_args[0], axis=0)
                self.bkg_likelihood_args[1] = np.array(self.comm.gather(self.bkg_likelihood_args[1], root=0))
                self.bkg_likelihood_args[1] = np.sum(self.bkg_likelihood_args[1], axis=0)

            # save to file
            if self.rank == 0:
                if self.update_likelihood and os.path.exists(self.likelihood_pdf_filename):
                    d = np.load(self.likelihood_pdf_filename)
                    assert np.all(d['bins0'] == self.bkg_likelihood_args[2])
                    assert np.all(d['bins1'] == self.bkg_likelihood_args[3])
                    self.bkg_likelihood_args[0] += d['sig']
                    self.bkg_likelihood_args[1] += d['bkg']
                    
                np.savez_compressed(self.likelihood_pdf_filename,
                                    sig=self.bkg_likelihood_args[0],
                                    bkg=self.bkg_likelihood_args[1],
                                    bins0=self.bkg_likelihood_args[2],
                                    bins1=self.bkg_likelihood_args[3])

    def run(self, source_name, source_slice, cache):
        super(BackgroundID, self).run(source_name, source_slice, cache)
        ev = cache[source_name]
        hits = cache[self.hits_dset_name]

        if len(ev):
            # load hit xyz positions
            hit_drift = cache[self.drift_dset_name].reshape(hits.shape)
            hit_q = cache[self.charge_dset_name].reshape(hits.shape)['q']
            hit_prof = cache[self.hit_prof_dset_name].reshape(hits.shape)
            hit_prof.mask = hit_prof.mask['idx'] | (hit_prof['idx'] == -1) # ignore non-profiled hits
            hit_prof_idx = hit_prof['idx']
            hit_prof_rr = hit_prof['rr']
            hit_xyz = np.concatenate([
                np.expand_dims(hits['px'], -1), np.expand_dims(hits['py'], -1),
                np.expand_dims(hit_drift['z'], -1)], axis=-1)

            # load dQ/dx profile and stopping event selection
            stopping_sel = self.data_manager[self.stopping_sel_dset_name, source_slice].reshape(ev.shape)
            prof = self.data_manager[self.profile_dset_name, source_slice].reshape(ev.shape)
            sel_mask = stopping_sel['sel'].astype(bool)

            # find end points
            #mu_end = stopping_sel['stop_pt'] + stopping_sel['stop_pt_corr']
            hit_stop_d = np.linalg.norm(hit_xyz - (stopping_sel['stop_pt'] + stopping_sel['stop_pt_corr'])[...,np.newaxis,:], axis=-1)
            mu_end = np.take_along_axis(
                hit_xyz, np.expand_dims(np.argmax(hit_q * (hit_stop_d < 22), axis=-1), (1,2)), axis=1)
            mu_end = mu_end.reshape(ev.shape + (3,))
            no_near_hits_mask = ~((hit_stop_d < 22).any(axis=-1))
            mu_end[no_near_hits_mask] = (stopping_sel['stop_pt'] + stopping_sel['stop_pt_corr'])[no_near_hits_mask]
            #mu_end = np.take_along_axis(
            #    hit_xyz, np.expand_dims(np.argmin(np.abs(hit_prof_rr), axis=-1), (1,2)), axis=1)
            mu_end = mu_end.reshape(ev.shape + (3,))

            # find axes
            # use charge weighted average position for hits near end point
            hit_dr = hit_xyz - mu_end[...,np.newaxis,:]
            hit_d = np.linalg.norm(hit_dr, axis=-1)
            muon_flag = (hit_prof_rr >= 0)
            
            muon_range_mask = np.broadcast_to(np.expand_dims(muon_flag & (hit_d < 200), axis=-1), hit_xyz.shape)            
            mu_dir = ma.average(ma.array(hit_dr, mask=~muon_range_mask), weights=np.broadcast_to(hit_q[...,np.newaxis], hit_dr.shape), axis=1)
            mu_dir /= np.clip(np.linalg.norm(mu_dir, axis=-1, keepdims=True), 1e-15, None)

            # calculate likelihood parameters
            hit_cos_mu = np.sum((hit_xyz - mu_end[:, np.newaxis, :])
                                * mu_dir[:, np.newaxis, :], axis=-1)
            hit_cos_mu /= np.clip(np.linalg.norm(hit_xyz - mu_end[:, np.newaxis, :], axis=-1), 1e-15, None)
            hit_d = np.sqrt(np.sum((hit_xyz - mu_end[:, np.newaxis, :])**2, axis=-1))

            # score hits
            hit_score = bkg_likelihood_score(hit_cos_mu, hit_d, *self.bkg_likelihood_args)

            if self.generate_likelihood_pdf and resources['RunData'].is_mc:
                # FIXME: paths are hardcoded and loaded on each execution, might impact performance or break if datatypes change
                event_truth = np.expand_dims(self.data_manager['analysis/muon_capture/truth_labels', source_slice], axis=-1)
                hit_traj = self.data_manager[source_name, 'charge/hits', 'charge/raw_hits',
                                             'charge/packets', 'mc_truth/tracks',
                                             'mc_truth/trajectories', source_slice] # ev, hit, rawhit, packet, true track, traj
                hit_frac = self.data_manager[source_name, 'charge/hits', 'charge/raw_hits',
                                             'charge/packets',
                                             'mc_truth/packet_fraction',
                                             source_slice] # ev, hit, rawhit, packet, true track
                hit_traj = np.take_along_axis(
                    hit_traj, np.expand_dims(np.argmax(hit_frac, axis=-1), axis=(-2,-1)),
                    axis=-2)
                hit_traj = hit_traj[:,:,0].reshape(hits.shape)
                hit_label_truth = (
                    (hit_traj['trackID'] != event_truth['stopping_track_id'])
                    # | (hit_traj['parentID'] == event_truth['michel_track_id']))
                    & event_truth['michel'].astype(bool))

                michel_mask = sel_mask # event mask
                hist_mask = (~hit_label_truth.mask & ~muon_flag) # hit mask
                sig, bkg = fill_likelihood_pdf(
                    hit_cos_mu[sel_mask][hist_mask[sel_mask]],
                    hit_d[sel_mask][hist_mask[sel_mask]],
                    hit_label_truth[sel_mask][hist_mask[sel_mask]].astype(bool),
                    *self.bkg_likelihood_args)
                self.bkg_likelihood_args[0] = sig
                self.bkg_likelihood_args[1] = bkg

        # create output arrays
        bkg_label = np.empty(ev.shape, dtype=self.bkg_label_dtype)
        if len(ev):
            bkg_label['bkg_nhit'] = np.sum((hit_score < self.score_cut) & ~muon_flag, axis=-1)
            bkg_label['bkg_q'] = np.sum(hit_q * (hit_score < self.score_cut) * ~muon_flag, axis=-1)            
            bkg_label['muon_dir'] = mu_dir
            bkg_label['stop_pt'] = mu_end
        logging.info(f'total background hits (per event): {bkg_label["bkg_nhit"].tolist()}')

        hit_mask = ~hits['id'].mask
        hit_label = np.empty(hit_mask.sum(), dtype=self.hit_label_dtype)
        if np.any(hit_mask):
            hit_label['score'] = hit_score[hit_mask]
            hit_label['bkg_flag'] = ((hit_score < self.score_cut) & ~muon_flag)[hit_mask]
            hit_label['muon_flag'] = muon_flag[hit_mask]
        logging.info(f'total background hits (per batch): {hit_label["bkg_flag"].sum()}')

        # write to file
        self.data_manager.reserve_data(self.bkg_label_dset_name, source_slice)
        self.data_manager.write_data(self.bkg_label_dset_name, source_slice, bkg_label)

        hit_label_slice = self.data_manager.reserve_data(self.hit_label_dset_name, int(hit_mask.sum()))
        self.data_manager.write_data(self.hit_label_dset_name, hit_label_slice, hit_label)
        self.data_manager.write_ref(self.hit_label_dset_name, self.hits_dset_name,
                                    np.c_[hit_label_slice, hits['id'].compressed()])
