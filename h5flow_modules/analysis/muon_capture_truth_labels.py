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


class MuonCaptureTruthLabels(H5FlowStage):
    class_version = '0.0.0'
    default_truth_trajectories_dset_name: 'mc_truth/trajectories'
    default_truth_tracks_dset_name: 'mc_truth/tracks'
    default_truth_labels_dset_name: 'analysis/muon_capture/truth_labels'

    truth_label_dtype = np.dtype([
        ('stopping_event', 'u1'),
        ('stopping_pdg_id', 'i4'),
        ('xyz_stop', 'f8', (3,)),
        ('xyz_start', 'f8', (3,)),
        ('stopping_track_id', 'i4'),
        ('stopping_end_process', 'i4'),
        ('stopping_end_subprocess', 'i4'),
        ('michel', 'u1'),
        ('michel_e', 'f8'),
        ('michel_start_xyz', 'f8', (3,)),
        ('michel_end_xyz', 'f8', (3,)),
        ('michel_start_dxyz', 'f8', (3,)),
        ('michel_plus', 'u1'),
        ('michel_track_id', 'i4'),
        ('michel_contained', 'u1'),
        ('gamma_dir', 'f8', (2, 3)),
        ('gamma_scatter_xyz', 'f8', (2, 3)),
        ('gamma_scatter_p', 'f8', (2,)),
        ('gamma_scatter_process', 'i4', (2,)),
        ('gamma_scatter_subprocess', 'i4', (2,)),
        ('gamma_track_id', 'i4', (2,)),
        ('scatter_track_id', 'i4', (2,)),
        ('scatter_contained', 'u1'),
    ])

    def __init__(self, **params):
        super(MuonCaptureTruthLabels, self).__init__(**params)
        self.is_mc = False
        self.truth_labels_dset_name = params.get(
            'truth_labels_dset_name', self.default_truth_labels_dset_name)
        self.truth_trajectories_dset_name = params.get(
            'truth_trajectories_dset_name', self.default_truth_trajectories_dset_name)
        self.truth_tracks_dset_name = params.get(
            'truth_tracks_dset_name', self.default_truth_tracks_dset_name)

    def init(self, source_name):
        super(MuonCaptureTruthLabels, self).init(source_name)
        self.is_mc = resources['RunData'].is_mc
        if not self.is_mc:
            return
        self.data_manager.create_dset(self.truth_labels_dset_name, dtype=self.truth_label_dtype)

    def run(self, source_name, source_slice, cache):
        super(MuonCaptureTruthLabels, self).run(source_name, source_slice, cache)
        if not self.is_mc:
            return

        # get trajectory/track segment info
        traj = cache[self.truth_trajectories_dset_name]
        traj = traj.reshape(traj.shape[0], -1)  # events, trajectories
        tracks = cache[self.truth_tracks_dset_name]
        tracks = traj.reshape(traj.shape + (-1,))  # events, trajectories, segments

        # create truth labels
        truth_label = np.zeros(traj.shape[0], dtype=self.truth_label_dtype)

        # check if a trajectory stops in detector
        is_traj_stop = (~resources['Geometry'].in_fid(traj['xyz_start'].reshape(-1, 3))
                        & resources['Geometry'].in_fid(traj['xyz_end'].reshape(-1, 3))).reshape(traj.shape)
        i_traj_stop = ma.argmax(ma.array(np.linalg.norm(traj['pxyz_start'], axis=-1), mask=~is_traj_stop), axis=-1)
        traj_stop = ma.take_along_axis(traj, i_traj_stop[..., np.newaxis], axis=-1)
        track_stop = ma.take_along_axis(tracks, i_traj_stop[..., np.newaxis, np.newaxis], axis=-2)
        track_d_from_start = np.sqrt(
            (track_stop['x_start'] - traj_stop['xyz_start'][..., 0])**2
            + (track_stop['y_start'] - traj_stop['xyz_start'][..., 1])**2
            + (track_stop['z_start'] - traj_stop['xyz_start'][..., 2])**2)
        truth_label['stopping_event'] = np.any(is_traj_stop, axis=-1)
        truth_label['stopping_pdg_id'] = traj_stop['pdgId'] * (truth_label['stopping_event'])
        truth_label['xyz_stop'] = traj_stop['xyz_end'] * (truth_label['stopping_event'])
        truth_label['xyz_start'] = ma.take_along_axis(
            track_stop, ma.argmin(track_d_from_start, axis=-1)[..., np.newaxis],
            axis=-1) * (truth_label['stopping_event'])
        truth_label['stopping_end_process'] = traj_stop['end_process'] * (truth_label['stopping_event'])
        truth_label['stopping_end_subprocess'] = traj_stop['end_subprocess'] * (truth_label['stopping_event'])
        truth_label['stopping_track_id'] = traj_stop['trackID'] * (truth_label['stopping_event'])

        # now check for the presence of a Michel decay
        is_traj_michel = (
            (np.abs(traj['pdgId']) == 11) & (traj['start_process'] != 2)
            & (traj['parent_id'] == traj_stop['id']))
        michel = ma.take_along_axis(
            traj, ma.argmax(is_traj_michel, axis=-1)[..., np.newaxis], axis=-1)
        truth_label['michel'] = np.any(is_traj_michel, axis=-1)
        truth_label['michel_e'] = michel['start_pxyz'][..., 0] * (truth_label['michel'])
        truth_label['michel_start_xyz'] = michel['start_xyz'] * (truth_label['michel'])
        truth_label['michel_end_xyz'] = michel['end_xyz'] * (truth_label['michel'])
        truth_label['michel_start_pxyz'] = michel['start_pxyz'][..., 1:] * (truth_label['michel'])
        truth_label['michel_track_id'] = michel['trackID'] * (truth_label['michel'])
        truth_label['michel_contained'] = resources['Geometry'].in_fid(michel['xyz_end'].reshape(-1, 3)) * (truth_label['michel'])

        # and now get potential positron info
        is_traj_michel_plus = is_traj_michel & (traj['pdgId'] == -11)
        michel_plus = ma.take_along_axis(
            traj, ma.argmax(is_traj_michel_plus, axis=-1)[..., np.newaxis], axis=-1)
        truth_label['michel_plus'] = np.any(is_traj_michel_plus, axis=-1)

        # find positron annihilation gammas
        is_gamma = (
            (traj['pdgId'] == 22)
            & (traj['parentID'] == traj_michel_plus['trackID'])
            & (traj['start_process'] == 2)
            & (traj['start_subprocess'] == 5))  # annihilation
        gamma0 = ma.take_along_axis(
            traj, ma.argmax(is_gamma, axis=-1)[..., np.newaxis], axis=-1)
        ma.put_along_axis(is_gamma, ma.argmax(is_gamma, axis=-1), False)
        gamma1 = ma.take_along_axis(
            traj, ma.argmax(is_gamma, axis=-1)[..., np.newaxis], axis=-1)
        # find first gamma scatter (either compton or PE)
        scatters = (
            (np.abs(traj['pdgId']) == 11)
            & (traj['start_process'] == 2)
            & ((traj['start_subprocess'] == 13)
                | (traj['start_subprocess'] == 12)))
        scat0 = ma.argmin(ma.array(traj['trackID'],
                                   mask=~scatters | (traj['parent_id'] != gamma0['id'])), axis=-1)
        scat1 = ma.argmin(ma.array(traj['trackID'],
                                   mask=~scatters | (traj['parent_id'] != gamma1['id'])), axis=-1)
        scat0 = ma.take_along_axis(traj, scat0[..., np.newaxis], axis=-1)
        scat1 = ma.take_along_axis(traj, scat1[..., np.newaxis], axis=-1)
        for i, gamma, scat in enumerate([(gamma0, scat0), (gamma1, scat1)]):
            truth_label['gamma_dir'][..., i, :] = gamma['start_pxyz'][..., 1:] * (truth_label['michel_plus'])
            truth_label['gamma_scatter_xyz'][..., i, :] = scat['start_xyz'] * (truth_label['michel_plus'])
            truth_label['gamma_scatter_p'][..., i, :] = np.linalg.norm(scat['start_pxyz'], axis=-1) * (truth_label['michel_plus'])
            truth_label['gamma_scatter_process'][..., i, :] = scat['start_process'] * (truth_label['michel_plus'])
            truth_label['gamma_scatter_subprocess'][..., i, :] = scat['start_subprocess'] * (truth_label['michel_plus'])
            truth_label['gamma_track_id'][..., i] = gamma['trackID'] * (truth_label['michel_plus'])
            truth_label['scatter_track_id'][..., i] = scat['trackID'] * (truth_label['michel_plus'])
            truth_label['scatter_contained'][..., i] = resources['Geometry'].in_fid(scat['xyz_end'].reshape(-1, 3)) * (truth_label['michel_plus'])

        # save truth info
        data_slice = self.data_manager.reserve_data(self.truth_labels_dset_name, len(truth_label))
        self.data_manager.write_data(self.truth_labels_dset_name, data_slice, truth_label)
