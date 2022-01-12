import numpy as np
import numpy.ma as ma
import logging
from scipy.interpolate import interp1d
import scipy.stats as stats
import scipy.ndimage as ndimage
from copy import deepcopy

from h5flow.core import H5FlowStage, resources
from h5flow.data import dereference_chain

from module0_flow.util.func import mode, condense_array


class MichelSelection(H5FlowStage):
    michel_sel_dtype = np.dtype([
        ('is_michel', 'u1'),
        ('is_nomichel', 'u1'),
        ('muon_axis', 'f8', (3,)),
        ('michel_axis', 'f8', (3,)),
    ])

    michel_hit_dtype = np.dtype([])

    def __init__(self, **params):
        pass

    def init(self, source_name):
        pass

    def run(self, source_name, source_slice, cache):
        muon_selection = cache[self.event_sel_dset_name]
        profile = cache[self.event_profile_dset_name]
        t0 = cache[self.t0_dset_name]
        hits = cache[self.t0_dset_name]

        selected_events = muon_selection['sel'].astype(bool)
        # only do the analysis on selected events
        if np.any(selected_events):
            hits = hits[selected_events]
            muon_selection = muon_selection[selected_events]
            profile = profile[selected_events]
            t0 = t0[selected_events]

            hit_z = resources['Geometry'].get_z_coordinate(
                hits['iogroup'], hits['iochannel'],
                (hits['ts'] - t0['ts']) * v_drift * crs_ticks)
            hit_xyz = np.concatenate([
                hits['px'][..., np.newaxis], hits['py'][..., np.newaxis],
                hit_z[..., np.newaxis]], axis=-1)
            start_pt = profile['seed_pt']
            stop_pt = muon_selection['stop_pt']

            michel_profile_mask = (profile['profile_n'].astype(bool)
                                   & (profile['profile_rr'] < 0))
            muon_profile_mask = (profile['profile_n'].astype(bool)
                                 & (profile['profile_rr'] >= 0))

            michel_axis = self.calc_profile_axis(
                profile, stop_pt, michel_profile_mask)
            muon_axis = self.calc_profile_axis(
                profile, stop_pt, muon_profile_mask)

            hit_michel_d = self.calc_profile_distance(
                hit_xyz, profile, michel_profile_mask)
            hit_muon_d = self.calc_profile_distance(
                hit_xyz, profile, muon_profile_mask)

            hit_michel_costheta = self.calc_costheta(
                hit_xyz, stop_pt, michel_axis)
            hit_muon_costheta = self.calc_costheta(
                hit_xyz, stop_pt, muon_axis)

            hit_michel_likelihood = self.calc_michel_likelihood(
                hit_michel_d, hit_muon_d, hit_michel_costheta,
                hit_muon_costheta)

    def finish(self, source_name):
        pass

    def michel_likelihood(self, *observables):
        bins = self.likelihood_bins
        i_bin = [np.digitize(np.clip(o, bins[i][0], bins[i][-1]), bins[i]) - 2
                 for i, o in enumerate(observables)]
        likelihood = np.prod([self.likelihood_vals[i_bin[i]]
                              for i in range(len(bins))],
                             axis=0)
        return likelihood

    def load_likelihood(self, pathname):
        data = np.load(pathname)

        self.likelihood_bins = [data]

    @staticmethod
    def calc_profile_axis(profile, origin, mask):
        '''
            :param profile: stopping muon event profile (shape: (..., N))

            :param origin: origin to calculate average axis (shape: (..., N, 3))

            :param mask: profile mask to apply (shape: (..., N, profile_pts))
        '''
        profile_pos = ma.array(profile['profile_pos'],
                               mask=~np.broadcast_to(mask[..., np.newaxis],
                                                     profile['profile_pos'].shape))
        axis = profile_pos - origin[..., np.newaxis, :]
        axis = ma.mean(axis, axis=-2)
        axis /= ma.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-15)
        return axis

    @staticmethod
    def calc_profile_distance(xyz, profile, profile_mask):
        '''
            :param xyz: positions (shape: (..., N, n, 3))

            :param profile: stopping muon event profile (shape: (..., N))

            :param profile_mask: profile mask to apply (shape: (..., N, profile_pts))
        '''
        npts = np.sum(mask, axis=-1)
        profile_pos = np.empty(profile_mask.shape[:-1] + (np.max(npts), 3))
        place_mask = np.indices(profile_pos.shape)[-2] < np.expand_dims(npts, axis=(-2, -1))
        np.place(profile_pos, place_mask, profile[profile_mask].ravel())
        profile_pos = ma.array(profile_pos, mask=~place_mask)

        profile_start = profile_pos[..., :-1, :]
        profile_end = profile_pos[..., 1:, :]
        r_start = np.expand_dims(xyz, axis=-2) - np.expand_dims(profile_start, axis=-3)
        r_end = np.expand_dims(xyz, axis=-2) - np.expand_dims(profile_end, axis=-3)
        d = ma.min(ma.minimum(ma.sum(r_start**2, axis=-1), ma.sum(r_end**2, axis=-1)), axis=-1)

        n = profile_end - profile_start
        n /= ma.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-15)

        alpha = ma.sum(r_start * n, axis=-1) / np.expand_dims(
            ma.maximum(np.linalg.norm(r_end - r_start, axis=-1), 1e-15),
            axis=-2)
        td = np.sqrt(ma.sum((r_start - ma.sum(r_start * n, axis=-1, keepdims=True) * n)**2, axis=-1))

        mask = (alpha < 1) | (alpha > 0)

        distance = ma.min(ma.where(mask, td, d), axis=-1)
        return distance

    @staticmethod
    def calc_costheta(xyz, origin, axis):
        '''
            :param xyz: positions (shape: (..., N, n, 3))

            :param origin: origin (shape: (..., N, 3))

            :param axis: origin (shape: (..., N, 3))
        '''
        r = xyz - origin[..., np.newaxis, :]
        d = ma.maximum(np.linalg.norm(r, axis=-1), 1e-15)
        return ma.sum(xyz * axis[..., np.newaxis, :] / d, axis=-1)
