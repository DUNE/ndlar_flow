import numpy as np
import numpy.ma as ma
import logging
from scipy.interpolate import interp1d
import scipy.stats as stats
from copy import deepcopy

from h5flow.core import H5FlowStage, resources
from h5flow.data import dereference_chain

from module0_flow.util.func import mode, condense_array


class StoppingMuonSelection(H5FlowStage):
    '''
        Perform a selection for stopping muons. A stopping event
        is defined by no more than one merged track segment that enters the
        detector fiducial volume and does not leave it. Creates a boolean array
        of 1:1 with events indicating stopping events, and creates a boolean
        array 1:1 with merged track segments if they individually meet the
        stopping criteria.

        A dQ/dx profile is generated per event and used to discriminate
        stopping protons and muons, as well as through-going muons.

        If the file is a MC file, also generates boolean arrays with the true
        value.


    '''
    class_version = '0.0.0'

    default_fid_cut = 20  # mm
    default_cathode_fid_cut = 20  # mm
    default_length_cut = 100  # mm
    default_profile_dx = 20  # mm
    default_profile_max_range = 1600  # mm
    default_larpix_gain = 250  # e/mV
    default_hits_dset_name = 'charge/hits'
    default_merged_dset_name = 'combined/tracklets/merged'
    default_t0_dset_name = 'combined/t0'
    default_mc_trajectory_path = ['combined/tracklets/merged',
                                  'charge/hits', 'charge/packets',
                                  'mc_truth/tracks',
                                  'mc_truth/trajectories']
    default_path = 'analysis/stopping_muons'

    event_sel_dset_name = 'event_sel_reco'
    event_profile_dset_name = 'event_profile'
    event_sel_truth_dset_name = 'event_sel_truth'

    event_sel_dtype = np.dtype([('sel', 'u1'),
                                ('stop_pt', 'f8', (3,))])

    @staticmethod
    def event_profile_dtype(dx, max_range):
        profile_bins = int(max_range / dx)
        return np.dtype([
            ('seed_pt', 'f8', (3,)),
            ('profile_rr', 'f8', (profile_bins,)),
            ('profile_dqdx', 'f8', (profile_bins,)),
            ('profile_pos', 'f8', (profile_bins, 3)),
            ('muon_likelihood', 'f8', (profile_bins,)),
            ('proton_likelihood', 'f8', (profile_bins,)),
            ('mip_likelihood', 'f8', (profile_bins,)),
        ])

    def __init__(self, **params):
        super(StoppingMuonSelection, self).__init__(**params)

        self.path = params.get('path', self.default_path)
        self.fid_cut = params.get('fid_cut', self.default_fid_cut)
        self.cathode_fid_cut = params.get('cathode_fid_cut', self.default_cathode_fid_cut)
        self.length_cut = params.get('length_cut', self.default_length_cut)
        self.larpix_gain = params.get('larpix_gain', self.default_larpix_gain)
        self.profile_dx = params.get('profile_dx', self.default_profile_dx)
        self.profile_max_range = params.get('profile_max_range',
                                            self.default_profile_max_range)
        self.hits_dset_name = params.get('hits_dset_name',
                                         self.default_hits_dset_name)
        self.t0_dset_name = params.get('t0_dset_name',
                                       self.default_t0_dset_name)
        self.hits_dset_name = params.get('hits_dset_name',
                                         self.default_hits_dset_name)
        self.merged_dset_name = params.get('merged_dset_name',
                                           self.default_merged_dset_name)
        self.mc_trajectory_path = params.get('mc_trajectory_path',
                                             self.default_mc_trajectory_path)

        self.event_profile_dtype = self.event_profile_dtype(self.profile_dx,
                                                            self.profile_max_range)

        self.regions = []

    def init(self, source_name):
        self.is_mc = resources['RunData'].is_mc

        self.data_manager.set_attrs(self.path,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    fid_cut=self.fid_cut,
                                    length_cut=self.length_cut,
                                    profile_dx=self.profile_dx,
                                    profile_max_range=self.profile_max_range,
                                    hits_dset_name=self.hits_dset_name,
                                    t0_dset_name=self.t0_dset_name,
                                    merged_dset_name=self.merged_dset_name
                                    )
        self.data_manager.create_dset(f'{self.path}/{self.event_sel_dset_name}',
                                      self.event_sel_dtype)
        self.data_manager.create_dset(f'{self.path}/{self.event_profile_dset_name}',
                                      self.event_profile_dtype)
        if self.is_mc:
            self.data_manager.create_dset(f'{self.path}/{self.event_sel_truth_dset_name}',
                                          self.event_sel_dtype)

        self.create_regions()
        self.data_manager.set_attrs(self.path, regions=np.array(self.regions))

        # create range tables used for dQ/dx profile discrimination
        self.muon_range_table = deepcopy(resources['ParticleData'].muon_range_table)
        self.proton_range_table = deepcopy(resources['ParticleData'].proton_range_table)

        # convert mean dE/dx entries to MPV dE/dx
        self.muon_range_table['dedx_mpv'] = resources['ParticleData'].landau_peak(
            self.muon_range_table['t'], resources['ParticleData'].mu_mass)
        self.proton_range_table['dedx_mpv'] = resources['ParticleData'].landau_peak(
            self.proton_range_table['t'], resources['ParticleData'].p_mass)

        # calculate recombination correction
        muon_r = resources['LArData'].ionization_recombination(
            self.muon_range_table['dedx_mpv'])
        proton_r = resources['LArData'].ionization_recombination(
            self.proton_range_table['dedx_mpv'])
        w = resources['LArData'].ionization_w

        self.muon_range_table['dqdx'] = (muon_r * self.muon_range_table['dedx_mpv']
                                         / w)
        self.proton_range_table['dqdx'] = (proton_r * self.proton_range_table['dedx_mpv']
                                           / w)
        self.muon_range_table['dqdx_width'] = (
            muon_r / w * resources['ParticleData'].landau_width(self.muon_range_table['t'],
                                                                resources['ParticleData'].mu_mass))
        self.proton_range_table['dqdx_width'] = (
            proton_r / w * resources['ParticleData'].landau_width(self.proton_range_table['t'],
                                                                  resources['ParticleData'].p_mass))

    def create_regions(self):
        x = resources['Geometry'].pixel_xy.compress((0,))
        y = resources['Geometry'].pixel_xy.compress((1,))
        z = resources['Geometry'].anode_z.compress()

        min_x, max_x = np.min(x) + self.fid_cut, np.max(x) - self.fid_cut
        min_y, max_y = np.min(y) + self.fid_cut, np.max(y) - self.fid_cut
        min_z, max_z = np.min(z) + self.fid_cut, np.max(z) - self.fid_cut

        self.regions.append((np.array([min_x, min_y, min_z]),
                             np.array([max_x, max_y, -self.cathode_fid_cut])))
        self.regions.append((np.array([min_x, min_y, self.cathode_fid_cut]),
                             np.array([max_x, max_y, max_z])))

    def in_fid(self, xyz):
        '''
            :param xyz: array ``shape: (N,3)``

            :returns: array ``shape: (N,)``

        '''
        coord_in_fid = ma.concatenate([np.expand_dims((xyz < np.expand_dims(boundary[1], 0))
                                                      & (xyz > np.expand_dims(boundary[0], 0)), axis=-1)
                                       for boundary in self.regions], axis=-1)
        in_fid = ma.all(coord_in_fid, axis=1)
        in_any_fid = ma.any(in_fid, axis=-1)
        return in_any_fid

    def contained(self, start_xyz, end_xyz):
        '''
            :param start_xyz: array ``shape: (N,3)``

            :param end_xyz: array ``shape: (N,3)``

            :returns: array ``shape: (N,)``

        '''
        start_in_fid = self.in_fid(start_xyz)
        end_in_fid = self.in_fid(end_xyz)
        track_contained = (~start_in_fid & end_in_fid) | (start_in_fid & ~end_in_fid)
        return track_contained

    def through_going(self, start_xyz, end_xyz):
        start_in_fid = self.in_fid(start_xyz)
        end_in_fid = self.in_fid(end_xyz)
        track_through_going = ~start_in_fid & ~end_in_fid
        return track_through_going

    def downward(self, start_xyz, end_xyz):
        end_in_fid = self.in_fid(end_xyz)
        dy = end_xyz[:, 1] - start_xyz[:, 1]
        return ((end_in_fid & (dy < 0)) | (~end_in_fid & (dy > 0)))

    @staticmethod
    def profile_likelihood(profile_rr, profile_dqdx, range_table):
        '''
            Calculates the likelihood of a given dqdx v. residual range profile
            using a Moyal-distribution approximation.

            Likelihood data is passed via the ``range_table`` parameter which is
            a ``dict`` with the following arrays:

                - ``range``: residual range values used in interpolation ``shape: (n_interp_pts,)``
                - ``dqdx``: dQ/dx values used in interpolation ``shape: (n_interp_pts,)``
                - ``dqdx_width``: dQ/dx sigma values ``shape: (n_interp_pts,)``

            :param profile_rr: residual range ``shape: (..., n)``

            :param profile_dqdx: dqdx ``shape: (..., n)``

            :param range_table: ``dict``, see above.

            :returns: likelihood ``shape: (..., n)``

        '''
        interp = interp1d(range_table['range'], range_table['dqdx'])
        interp_width = interp1d(range_table['range'], range_table['dqdx_width'])
        min_range = np.min(range_table['range'])
        max_range = np.max(range_table['range'])

        interp_dqdx = interp(np.clip(profile_rr, min_range, max_range))
        interp_dqdx_width = interp_width(np.clip(profile_rr, min_range, max_range))

        return stats.moyal.pdf(profile_dqdx, loc=interp_dqdx, scale=interp_dqdx_width)

    @staticmethod
    def profiled_dqdx(tracks, seed_pt, hit_xyz, hit_q, dx, max_range, mask=None):
        '''
            Generate dQ/dx profile. Algorithm is the following:

             1. Find track nearest to seed point

             2. Orient track in direction away from seed point

             3. Collect hits that lie within ``dx`` mm along the track trajectory

             4. Assign hits an overall position along event according to the projection onto the track trajectory

             5. Fill dQ/dx bins with hit charge according to hit position

             6. Set next iteration seed point as current track end point

             7. Repeat, removing previously used tracks from each iteration

            The output array contains ``int(max_range / dx)`` bins with the
            first and last bins being overflow bins.

            This is an expensive operation so a ``mask`` can be provided to skip
            the analysis of certain events. The returned array will not contain
            meaningful values for these entries.

            :param tracks: masked array, shape: (..., M)

            :param seed_pt: masked array, shape: (..., 3)

            :param hit_xyz: masked array, shape: (..., n, 3)

            :param hit_q: masked array, shape: (..., n)

            :returns: masked array, shape: (..., m)
        '''
        orig_len = len(tracks)
        if mask is not None:
            tracks = tracks[mask]
            seed_pt = seed_pt[mask]
            hit_xyz = hit_xyz[mask]
            hit_q = hit_q[mask]

        seed_d = np.minimum(np.linalg.norm(tracks['trajectory'][..., 0, :] - seed_pt, axis=-1),
                            np.linalg.norm(tracks['trajectory'][..., -1, :] - seed_pt, axis=-1))
        seed_track = np.expand_dims(np.argmin(seed_d, axis=-1), axis=-1)

        start_pt = seed_pt.copy()

        dq = np.zeros((len(tracks), int(max_range / dx)))
        dn = np.zeros((len(tracks), int(max_range / dx)), dtype=int)
        pos = np.zeros((len(tracks), int(max_range / dx), 3))

        track_mask = ~tracks['id'].mask.copy()  # True == include
        hit_mask = ~(hit_q.mask | np.any(hit_xyz.mask, axis=-1))  # True == include

        s = 0
        bins = np.linspace(0, max_range, dq.shape[-1])
        for _ in range(tracks.shape[-1]):
            logging.debug(_, 'remaining trajectories:', np.sum(track_mask))
            # find seed trajectory
            traj = np.take_along_axis(tracks, seed_track, axis=-1)['trajectory'].copy()

            # orient trajectory according to previous seed point
            traj_start_d = np.linalg.norm(traj[..., 0, :] - seed_pt, axis=-1, keepdims=True)
            traj_end_d = np.linalg.norm(traj[..., -1, :] - seed_pt, axis=-1, keepdims=True)
            is_reversed_traj = (traj_end_d < traj_start_d)
            is_reversed_traj = np.expand_dims(is_reversed_traj, axis=-1)
            traj = np.where(is_reversed_traj, traj[..., ::-1, :], traj)
            # (N, 1, npt, 3)

            # get trajectory displacement vectors
            traj_start = traj[..., :-1, :]
            traj_end = traj[..., 1:, :]
            traj_dx = traj[..., 1:, :] - traj_start
            traj_length = np.linalg.norm(traj_dx, axis=-1, keepdims=True)
            traj_length = np.clip(traj_length, 1e-15, None)
            traj_n = traj_dx / traj_length
            # (N, 1, npt-1, 3)

            # collect hits within dx of trajectory
            hit_dx = np.expand_dims(hit_xyz, axis=-2) - traj_start
            hit_td = np.linalg.norm(hit_dx - np.sum(hit_dx * traj_n, axis=-1, keepdims=True) * traj_n, axis=-1)
            hit_alpha = np.sum(hit_dx * traj_n, axis=-1) / traj_length[..., 0]
            hit_on_traj = (hit_alpha < 1) & (hit_alpha > 0)
            hit_near_traj_pt = ((np.linalg.norm(np.expand_dims(hit_xyz, axis=-2) - traj_start, axis=-1) < dx)
                                | (np.linalg.norm(np.expand_dims(hit_xyz, axis=-2) - traj_end, axis=-1) < dx))
            itraj_min_td = np.expand_dims(np.argmin(
                ma.array(hit_td, mask=~(hit_on_traj | hit_near_traj_pt)),
                axis=-1), axis=-1)
            hit_min_td = np.take_along_axis(hit_td, itraj_min_td, axis=-1)
            traj_hit_mask = (hit_min_td < dx)[..., 0] & hit_mask
            # (N, nhit)

            # project hits onto track
            traj_s = np.cumsum(traj_length[..., 0], axis=-1) + s - traj_length[..., 0]  # (N, 1, npts-1)
            hit_s = hit_alpha * traj_length[..., 0] + traj_s  # (N, nhit, npts-1) * (N, 1, npt-1, 1)
            hit_s = np.take_along_axis(hit_s, itraj_min_td, axis=-1)[..., 0]
            # (N, nhit)

            # and create a histogram
            i_bin = np.expand_dims(np.clip(np.digitize(hit_s, bins=bins) - 1, 0, len(bins) - 1), axis=-1)
            # (N, nhit, nbin)
            q_mask = ((i_bin == np.expand_dims(np.indices(dq.shape)[-1], axis=-2))
                      & np.expand_dims(traj_hit_mask, axis=-1)
                      & np.expand_dims(np.take_along_axis(track_mask, seed_track, axis=-1), axis=-1))
            masked_q = ma.array(np.broadcast_to(hit_q[..., np.newaxis], q_mask.shape), mask=~q_mask)
            xyz_mask = np.broadcast_to(q_mask[..., np.newaxis], q_mask.shape + (3,))
            masked_xyz = ma.array(np.broadcast_to(hit_xyz[..., np.newaxis, :], xyz_mask.shape), mask=~xyz_mask)
            dq += ma.sum(masked_q, axis=-2)
            dn += ma.sum(~masked_q.mask, axis=-2)
            mean_xyz = ma.mean(masked_xyz, axis=-3)
            pos += ma.mean(masked_xyz, axis=-3).filled(0) * (pos == 0)

            # termination conditions
            np.put_along_axis(track_mask, seed_track, False, axis=-1)
            if (np.all([s > max_range]) or not np.any(track_mask) or not np.any(hit_mask)):
                break

            # update variables
            s = s + np.sum(traj_length[..., 0], axis=-1)[..., np.newaxis]
            seed_pt = traj[..., -1, :].copy()
            traj_distance = np.linalg.norm(tracks['trajectory'] - np.expand_dims(seed_pt, axis=-2), axis=-1)
            traj_mask = np.any(traj_distance < dx, axis=-1) & track_mask
            seed_track = np.expand_dims(np.argmax(traj_mask, axis=-1), axis=-1)

            if not np.any(traj_mask):
                break

        end_pt = np.take_along_axis(pos, np.argmax(dq, axis=-1)[..., np.newaxis, np.newaxis], axis=-2)

        r_dq = np.empty((orig_len,) + dq.shape[1:])
        r_dn = np.empty((orig_len,) + dn.shape[1:])
        r_start_pt = np.empty((orig_len,) + start_pt.shape[1:])
        r_end_pt = np.empty((orig_len,) + end_pt.shape[1:])
        r_pos = np.empty((orig_len,) + pos.shape[1:])

        np.place(r_dq, np.broadcast_to(mask[..., np.newaxis], r_dq.shape), dq)
        np.place(r_dn, np.broadcast_to(mask[..., np.newaxis], r_dn.shape), dn)
        np.place(r_start_pt, np.broadcast_to(mask[..., np.newaxis, np.newaxis], r_start_pt.shape), start_pt)
        np.place(r_end_pt, np.broadcast_to(mask[..., np.newaxis, np.newaxis], r_end_pt.shape), end_pt)
        np.place(r_pos, np.broadcast_to(mask[..., np.newaxis, np.newaxis], r_pos.shape), pos)

        return r_dq, r_dn, r_start_pt, r_end_pt, r_pos

    def load_track_trajectories(self, source_name, source_slice, path):
        chain = list(zip([source_name] + path[:-1], path))

        data = self.data_manager.get_dset(path[-1])
        ref, ref_dir = list(zip(*[self.data_manager.get_ref(p, c) for p, c in chain]))
        regions = [self.data_manager.get_ref_region(p, c) for p, c in chain]

        return dereference_chain(source_slice, ref, data=data, regions=regions, ref_directions=ref_dir)

    def run(self, source_name, source_slice, cache):
        hits = cache[self.hits_dset_name]
        tracks = cache[self.merged_dset_name]
        t0 = cache[self.t0_dset_name].reshape(cache[source_name].shape)

        # calculate hit positions
        v_drift = resources['LArData'].v_drift
        crs_ticks = resources['RunData'].crs_ticks
        z = resources['Geometry'].get_z_coordinate(hits['iogroup'], hits['iochannel'],
                                                   (hits['ts'] - np.expand_dims(t0['ts'], axis=-1)) * v_drift
                                                   * crs_ticks)
        hit_q = self.larpix_gain * hits['q']  # convert mV -> ke
        hit_xyz = np.concatenate([
            hits['px'][..., np.newaxis], hits['py'][..., np.newaxis],
            z[..., np.newaxis]], axis=-1)

        # find all tracks that end in the fiducial volume
        track_start = tracks.ravel()['trajectory'][..., 0, :]
        track_stop = tracks.ravel()['trajectory'][..., -1, :]
        track_length = tracks.ravel()['length']
        is_stopping = (self.contained(track_start, track_stop)
                       & (track_length > self.length_cut))
        is_throughgoing = self.through_going(track_start, track_stop)
        is_downward = self.downward(track_start, track_stop)
        is_stopping = is_stopping.reshape(tracks.shape)
        is_throughgoing = is_throughgoing.reshape(tracks.shape)
        is_downward = is_downward.reshape(tracks.shape)

        if self.is_mc:
            # lookup the track's true trajectory
            track_traj = self.load_track_trajectories(source_name, source_slice,
                                                      self.mc_trajectory_path)
            track_traj = track_traj.reshape(tracks.shape + (-1,))
            track_traj = condense_array(track_traj, track_traj['trackID'].mask)

            track_true_id = mode(track_traj['trackID'])
            track_true_traj = np.take_along_axis(track_traj,
                                                 np.expand_dims(np.argmax(track_traj['trackID'] == track_true_id, axis=-1), axis=-1),
                                                 axis=-1)
            track_true_traj = track_true_traj.reshape(tracks.shape)

            # find if trajectory ends in the fiducial volume
            true_xyz_start = track_true_traj.ravel()['xyz_start'].copy()
            true_xyz_end = track_true_traj.ravel()['xyz_end'].copy()
            # FIXME: coordinates are weird between sim and data
            true_xyz_start[:, 1] += 218.236
            true_xyz_end[:, 1] += 218.236
            new_x_start = true_xyz_start[:, 2].copy()
            new_x_end = true_xyz_end[:, 2].copy()
            new_z_start = true_xyz_start[:, 0].copy()
            new_z_end = true_xyz_end[:, 0].copy()
            true_xyz_start[:, 0] = new_x_start
            true_xyz_start[:, 2] = new_z_start
            true_xyz_end[:, 0] = new_x_end
            true_xyz_end[:, 2] = new_z_end
            is_muon = ma.abs(track_true_traj['pdgId']) == 13
            is_true_stopping = self.contained(true_xyz_start, true_xyz_end)
            is_true_stopping = is_true_stopping.reshape(tracks.shape)

        # define a stopping event as one with exclusively 1 track that ends in fid.
        event_is_stopping = ((ma.sum(is_stopping, axis=-1) == 1)
                             & (t0['type'] != 0)
                             & (ma.sum(is_throughgoing, axis=-1) == 0)
                             & (ma.sum(is_stopping & is_downward, axis=-1) == 1))
        seed_pt = np.where(np.expand_dims(self.in_fid(track_start), axis=-1),
                           track_stop, track_start).reshape(tracks.shape + (3,))
        seed_pt = np.take_along_axis(seed_pt, np.argmax(is_stopping, axis=-1)[..., np.newaxis, np.newaxis],
                                     axis=-2)

        # now check the likelihood of a stopping muon
        dq, dn, start_pt, end_pt, pos = self.profiled_dqdx(tracks, seed_pt, hit_xyz,
                                                           hit_q,
                                                           mask=event_is_stopping,
                                                           dx=self.profile_dx,
                                                           max_range=self.profile_max_range)
        profile_dqdx = dq / self.profile_dx
        profile_rr = ((np.expand_dims(np.argmax(profile_dqdx, axis=-1), axis=-1)
                       - np.indices(profile_dqdx.shape)[-1] + 0.5) * self.profile_dx)

        muon_likelihood = self.profile_likelihood(profile_rr, profile_dqdx,
                                                  self.muon_range_table)
        proton_likelihood = self.profile_likelihood(profile_rr, profile_dqdx,
                                                    self.proton_range_table)
        mip_likelihood = self.profile_likelihood(np.clip(profile_rr, 1500, 1500),
                                                 profile_dqdx,
                                                 self.muon_range_table)
        # mask invalid values
        muon_likelihood[dn == 0] = -1
        proton_likelihood[dn == 0] = -1
        mip_likelihood[dn == 0] = -1

        muon_likelihood = ma.masked_values(muon_likelihood, -1)
        proton_likelihood = ma.masked_values(proton_likelihood, -1)
        mip_likelihood = ma.masked_values(mip_likelihood, -1)

        # select stopping muons
        event_is_stopping_muon = (event_is_stopping
                                  & (np.sum(np.log(muon_likelihood) - np.log(proton_likelihood), axis=-1) > 0)
                                  & (np.sum(np.log(muon_likelihood) - np.log(mip_likelihood), axis=-1) > 0))

        if self.is_mc:
            # define true stopping events as events with at least 1 muon that ends in fid.
            event_is_true_stopping = ma.sum(is_muon & is_true_stopping, axis=-1) >= 1
            true_stop_pt = np.where(np.expand_dims(self.in_fid(true_xyz_start), axis=-1),
                                    true_xyz_start, true_xyz_end).reshape(tracks.shape + (3,))
            true_stop_pt = np.take_along_axis(true_stop_pt,
                                              np.argmax(is_true_stopping, axis=-1)[..., np.newaxis, np.newaxis],
                                              axis=-2)

        # prep arrays to write to file
        event_sel = np.zeros(len(tracks), dtype=self.event_sel_dtype)
        event_sel['sel'] = event_is_stopping_muon
        event_sel['stop_pt'] = end_pt.reshape(event_sel['stop_pt'].shape)

        event_profile = np.zeros(len(tracks), dtype=self.event_profile_dtype)
        event_profile['seed_pt'] = start_pt.reshape(event_profile['seed_pt'].shape)
        event_profile['profile_rr'] = profile_rr
        event_profile['profile_dqdx'] = profile_dqdx
        event_profile['profile_pos'] = pos
        event_profile['muon_likelihood'] = muon_likelihood
        event_profile['proton_likelihood'] = proton_likelihood
        event_profile['mip_likelihood'] = mip_likelihood

        if self.is_mc:
            event_true_sel = np.zeros(len(tracks), dtype=self.event_sel_dtype)
            event_true_sel['sel'] = event_is_true_stopping
            event_true_sel['stop_pt'] = true_stop_pt.reshape(event_true_sel['stop_pt'].shape)

        # reserve data space
        event_sel_slice = self.data_manager.reserve_data(
            f'{self.path}/{self.event_sel_dset_name}', len(event_sel))
        event_profile_slice = self.data_manager.reserve_data(
            f'{self.path}/{self.event_profile_dset_name}', len(event_profile))
        if self.is_mc:
            event_sel_truth_slice = self.data_manager.reserve_data(
                f'{self.path}/{self.event_sel_truth_dset_name}',
                len(event_true_sel))

        # write
        self.data_manager.write_data(f'{self.path}/{self.event_sel_dset_name}',
                                     event_sel_slice, event_sel)
        self.data_manager.write_data(f'{self.path}/{self.event_profile_dset_name}',
                                     event_profile_slice, event_profile)
        if self.is_mc:
            self.data_manager.write_data(
                f'{self.path}/{self.event_sel_truth_dset_name}',
                event_sel_truth_slice, event_true_sel)
