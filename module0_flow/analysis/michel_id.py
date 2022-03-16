import numpy as np
import numpy.ma as ma
import scipy.ndimage as ndimage

from h5flow import H5FLOW_MPI
from h5flow.core import H5FlowStage, resources

from module0_flow.util import units


def load_likelihood_pdf(filename):
    d = np.load(filename)
    pdf = dict()
    for key in d.keys():
        pdf[key] = d[key]
        pdf[key] = ndimage.gaussian_filter(pdf[key], sigma=0.5)
    pdf['sig'] /= pdf['sig'].sum()
    pdf['bkg'] /= pdf['bkg'].sum()
    return pdf


def fill_likelihood_pdf(cos_mu, cos_e, d, sig_label, pdf_sig, pdf_bkg,
                        cos_mu_bins, cos_e_bins, d_bins):
    v = np.concatenate(
        [np.expand_dims(cos_mu.ravel(), -1), np.expand_dims(cos_e.ravel(), -1),
         np.expand_dims(d.ravel(), -1)], axis=-1)

    sig = pdf_sig + np.histogramdd(v[sig_label], (cos_mu_bins, cos_e_bins, d_bins))[0]
    bkg = pdf_bkg + np.histogramdd(v[~sig_label], (cos_mu_bins, cos_e_bins, d_bins))[0]

    return sig, bkg


def michel_likelihood_score(cos_mu, cos_e, d, pdf_sig, pdf_bkg,
                            cos_mu_bins, cos_e_bins, d_bins):
    cos_mu_i = np.clip(np.digitize(cos_mu, cos_mu_bins)-1, 0, len(cos_mu_bins)-2)
    cos_e_i = np.clip(np.digitize(cos_e, cos_e_bins)-1, 0, len(cos_e_bins)-2)
    d_i = np.clip(np.digitize(d, d_bins)-1, 0, len(d_bins)-2)

    p_sig = pdf_sig[cos_mu_i, cos_e_i, d_i]
    p_bkg = pdf_bkg[cos_mu_i, cos_e_i, d_i]

    with np.errstate(divide='ignore', invalid='ignore'):
        score = np.log(p_sig) - np.log(p_bkg)
        if np.any(p_bkg > 0):
            score[p_bkg == 0] = (np.log(p_sig)-np.log(p_bkg[p_bkg>0].min()))[p_bkg == 0]
        if np.any(p_sig > 0):
            score[p_sig == 0] = (np.log(p_sig[p_sig > 0].min())-np.log(p_bkg))[p_sig == 0]
        score[(p_bkg == 0) & (p_sig == 0)] = 0

    return score


class MichelID(H5FlowStage):
    class_version = '0.1.0'

    defaults = dict(
        hits_dset_name='charge/hits',
        drift_dset_name='combined/hit_drift',
        profile_dset_name='analysis/stopping_muons/event_profile',
        stopping_sel_dset_name='analysis/stopping_muons/event_sel_reco',
        hit_prof_idx_dset_name='analysis/stopping_muons/hit_profile_id',

        hit_label_dset_name='analysis/michel_id/hit_michel_label',
        michel_label_dset_name='analysis/michel_id/michel_label',

        michel_e_cut=5 * units.MeV,
        michel_nhit_cut=5,

        likelihood_pdf_filename='michel_pdf-{version}.npz',
        generate_likelihood_pdf=False,
        )

    michel_label_dtype = np.dtype([
        ('michel_flag', 'u1'),
        ('michel_nhit', 'i4'),
        ('michel_e', 'f8'),
        ('michel_tagged_e', 'f8'),
        ('michel_dir', 'f8', (3,)),
        ('muon_dir', 'f8', (3,)),
        ('stop_pt', 'f8', (3,)),
        ('michel_end', 'f8', (3,))
    ])

    hit_label_dtype = np.dtype([
        ('score', 'f8'),
        ('michel_flag', 'u1'),
        ('muon_flag', 'u1')
    ])

    likelihood_bins = (np.linspace(-1, 1, 50), np.linspace(-1, 1, 50), np.geomspace(0.1, 1000, 50))

    def __init__(self, **params):
        super(MichelID, self).__init__(**params)
        for name, default in self.defaults.items():
            setattr(self, name, params.get(name, default))

        self.likelihood_pdf_filename = self.likelihood_pdf_filename.format(version=self.class_version)

    def init(self, source_name):
        super(MichelID, self).init(source_name)

        self.larpix_gain = self.data_manager.get_attrs('/'.join(self.stopping_sel_dset_name.split('/')[:-1]))['larpix_gain']
        michel_dedx = resources['ParticleData'].landau_peak(50 * units.MeV, resources['ParticleData'].e_mass, resources['Geometry'].pixel_pitch)
        self.recomb_factor = 1 / resources['LArData'].ionization_recombination(michel_dedx)

        # load likelihood function
        if not self.generate_likelihood_pdf:
            pdf = load_likelihood_pdf(self.likelihood_pdf_filename)
            self.michel_likelihood_args = pdf['sig'], pdf['bkg'], pdf['bins0'], pdf['bins1'], pdf['bins2']
        else:
            sig = np.zeros([len(bins)-1 for bins in self.likelihood_bins])
            bkg = np.zeros([len(bins)-1 for bins in self.likelihood_bins])
            self.michel_likelihood_args = [sig, bkg] + [bins for bins in self.likelihood_bins]

        # save attributes
        attrs = dict(classname=self.classname, class_version=self.class_version)
        for name in self.defaults.keys():
            attrs[name] = getattr(self, name)
        self.data_manager.set_attrs(self.michel_label_dset_name, **attrs)

        # create datasets
        self.data_manager.create_dset(self.michel_label_dset_name, self.michel_label_dtype)
        self.data_manager.create_dset(self.hit_label_dset_name, self.hit_label_dtype)
        self.data_manager.create_ref(self.hit_label_dset_name, self.hits_dset_name)

    def finish(self, source_name):
        if self.generate_likelihood_pdf:
            # gather likelihood histograms from all threads
            if H5FLOW_MPI:
                self.michel_likelihood_args[0] = np.array(self.comm.gather(self.michel_likelihood_args[0], root=0))
                self.michel_likelihood_args[0] = np.sum(self.michel_likelihood_args[0], axis=0)
                self.michel_likelihood_args[1] = np.array(self.comm.gather(self.michel_likelihood_args[1], root=0))
                self.michel_likelihood_args[1] = np.sum(self.michel_likelihood_args[1], axis=0)

            # save to file
            if self.rank == 0:
                np.savez_compressed(self.likelihood_pdf_filename,
                                    sig=self.michel_likelihood_args[0],
                                    bkg=self.michel_likelihood_args[1],
                                    bins0=self.michel_likelihood_args[2],
                                    bins1=self.michel_likelihood_args[3],
                                    bins2=self.michel_likelihood_args[4])

    def run(self, source_name, source_slice, cache):
        super(MichelID, self).run(source_name, source_slice, cache)
        ev = cache[source_name]
        hits = cache[self.hits_dset_name]

        if len(ev):
            # load hit xyz positions
            hit_drift = cache[self.drift_dset_name].reshape(hits.shape)
            hit_prof_idx = cache[self.hit_prof_idx_dset_name].reshape(hits.shape)
            hit_xyz = np.concatenate([
                np.expand_dims(hits['px'], -1), np.expand_dims(hits['py'], -1),
                np.expand_dims(hit_drift['z'], -1)], axis=-1)

            # load dQ/dx profile and stopping event selection
            stopping_sel = self.data_manager[self.stopping_sel_dset_name, source_slice].reshape(ev.shape)
            prof = self.data_manager[self.profile_dset_name, source_slice].reshape(ev.shape)
            sel_mask = stopping_sel['sel'].astype(bool)

            # find start and end points
            mu_start = np.take_along_axis(
                prof['profile_pos'], np.expand_dims(np.argmax(prof['profile_n'] > 0, axis=-1), (1, 2)), axis=1)
            mu_end = stopping_sel['stop_pt'] + stopping_sel['stop_pt_corr']
            last_profile_rr = ma.array(prof['profile_rr'], mask=(prof['profile_n'] <= 0) | (prof['profile_rr'] < 0))
            last_profile_rr = last_profile_rr.min(axis=-1, keepdims=True)
            last_profile_rr = last_profile_rr.filled(0.)
            gap_mask = (prof['profile_rr'] <= last_profile_rr) & (prof['profile_n'] > 0)
            gap_mask[...,:-1] = gap_mask[...,:-1] & (prof['profile_n'][...,1:] == 0)

            michel_end = np.take_along_axis(
                prof['profile_pos'], np.expand_dims(np.argmax(gap_mask, axis=-1), (1,2)), axis=1)

            mu_start = mu_start.reshape(ev.shape + (-1,))
            mu_end = mu_end.reshape(ev.shape + (-1,))
            michel_end = michel_end.reshape(ev.shape + (-1,))

            # find axes
            mu_dir = mu_start - mu_end
            mu_dir /= np.clip(np.linalg.norm(mu_dir, axis=-1, keepdims=True), 1e-15, None)

            michel_dir = michel_end - mu_end
            # if no track-able michel, use charge weighted average position not included in parent muon for michel axis
            no_michel_mask = np.all(michel_end == stopping_sel['stop_pt'], axis=-1)
            hit_profile_mask = np.broadcast_to(np.expand_dims(
                (hit_prof_idx == -1) | ((hit_prof_idx >= np.argmax(prof['profile_rr'] <= 0, axis=-1)[...,np.newaxis]))
                , -1), hit_xyz.shape)
            muon_flag = ~hit_profile_mask.any(axis=-1)
            michel_dir[no_michel_mask] = (ma.average(ma.array(hit_xyz, mask=~hit_profile_mask),
                                                     weights=np.broadcast_to(hits['q'][...,np.newaxis], hit_xyz.shape), axis=1)
                          - mu_end)[no_michel_mask]
            michel_dir /= np.clip(np.linalg.norm(michel_dir, axis=-1, keepdims=True), 1e-15, None)
            # special case: michel start and end are the same, and no non-profiled hits (will call all of these non-michel hits)
            no_hit_mask = ~hit_profile_mask.any(axis=-1).any(axis=-1)

            # calculate likelihood parameters
            hit_cos_mu = np.sum((hit_xyz - mu_end[:, np.newaxis, :])
                                * mu_dir[:, np.newaxis, :], axis=-1)
            hit_cos_mu /= np.clip(np.linalg.norm(hit_xyz - mu_end[:, np.newaxis, :], axis=-1), 1e-15, None)
            hit_cos_e = np.sum((hit_xyz - mu_end[:, np.newaxis, :])
                               * michel_dir[:, np.newaxis, :], axis=-1)
            hit_cos_e /= np.clip(np.linalg.norm(hit_xyz - mu_end[:, np.newaxis, :], axis=-1), 1e-15, None)
            hit_d = np.sqrt(np.sum((hit_xyz - mu_end[:, np.newaxis, :])**2, axis=-1))

            # score hits
            hit_score = michel_likelihood_score(hit_cos_mu, hit_cos_e, hit_d, *self.michel_likelihood_args)
            hit_score = hit_score * (~no_hit_mask[...,np.newaxis])
            if self.generate_likelihood_pdf and resources['RunData'].is_mc:
                # FIXME: paths are hardcoded and loaded on each execution
                event_truth = np.expand_dims(self.data_manager['analysis/muon_capture/truth_labels', source_slice], axis=-1)
                hit_traj = self.data_manager[source_name, 'charge/hits',
                                             'charge/packets', 'mc_truth/tracks',
                                             'mc_truth/trajectories', source_slice]
                hit_frac = self.data_manager[source_name, 'charge/hits',
                                             'charge/packets',
                                             'mc_truth/packet_fraction',
                                             source_slice]
                hit_traj = np.take_along_axis(
                    hit_traj, np.expand_dims(np.argmax(hit_frac, axis=-1), axis=(-2,-1)),
                    axis=-2)
                hit_traj = hit_traj.reshape(hits.shape)
                hit_label_truth = (
                    ((hit_traj['trackID'] != event_truth['stopping_track_id']))
                    # | (hit_traj['parentID'] == event_truth['michel_track_id']))
                    & event_truth['michel'].astype(bool))

                michel_mask = sel_mask & ~no_hit_mask
                hist_mask = (~hit_label_truth.mask & ~muon_flag)
                sig, bkg = fill_likelihood_pdf(
                    hit_cos_mu[michel_mask][hist_mask[michel_mask]],
                    hit_cos_e[michel_mask][hist_mask[michel_mask]],
                    hit_d[michel_mask][hist_mask[michel_mask]],
                    hit_label_truth[michel_mask][hist_mask[michel_mask]].astype(bool),
                    *self.michel_likelihood_args)
                self.michel_likelihood_args[0] = sig
                self.michel_likelihood_args[1] = bkg

            # grab michel energy (from stopping muon selection)
            michel_e = stopping_sel['remaining_e']

            # use michel tag to reconstruct energy
            lifetime = resources['LArData'].electron_lifetime(ev['unix_ts'].astype(float))[0]
            lifetime = lifetime[..., np.newaxis]
            hit_q = self.larpix_gain * hits['q'] / np.exp(-hit_drift['t_drift'] * resources['RunData'].crs_ticks / lifetime)
            hit_e = hit_q * resources['LArData'].ionization_w * self.recomb_factor
            michel_tagged_e = (((hit_score > 0) & ~muon_flag) * hit_e).sum(axis=-1)

            # cut on variables
            michel_flag = ((np.sum((hit_score > 0) & ~muon_flag, axis=-1) > self.michel_nhit_cut)
                           | (michel_tagged_e > self.michel_e_cut))

        # create output arrays
        michel_label = np.empty(ev.shape, dtype=self.michel_label_dtype)
        if len(ev):
            michel_label['michel_flag'] = michel_flag
            michel_label['michel_nhit'] = np.sum((hit_score > 0) & ~muon_flag, axis=-1)
            michel_label['michel_e'] = michel_e
            michel_label['michel_tagged_e'] = michel_tagged_e
            michel_label['michel_dir'] = michel_dir
            michel_label['muon_dir'] = mu_dir
            michel_label['stop_pt'] = mu_end
            michel_label['michel_end'] = michel_end

        hit_mask = ~hits['id'].mask
        hit_label = np.empty(hit_mask.sum(), dtype=self.hit_label_dtype)
        if np.any(hit_mask):
            hit_label['score'] = hit_score[hit_mask]
            hit_label['michel_flag'] = ((hit_score > 0) & ~muon_flag)[hit_mask]
            hit_label['muon_flag'] = muon_flag[hit_mask]

        # write to file
        self.data_manager.reserve_data(self.michel_label_dset_name, source_slice)
        self.data_manager.write_data(self.michel_label_dset_name, source_slice, michel_label)

        hit_label_slice = self.data_manager.reserve_data(self.hit_label_dset_name, int(hit_mask.sum()))
        self.data_manager.write_data(self.hit_label_dset_name, hit_label_slice, hit_label)
        self.data_manager.write_ref(self.hit_label_dset_name, self.hits_dset_name,
                                    np.c_[hit_label_slice, hits['id'].compressed()])
