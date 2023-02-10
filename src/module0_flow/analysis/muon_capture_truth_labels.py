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
    default_truth_trajectories_dset_name = 'mc_truth/trajectories'
    default_truth_tracks_dset_name = 'mc_truth/tracks'
    default_truth_labels_dset_name = 'analysis/muon_capture/truth_labels'

    truth_label_dtype = np.dtype([
        ('stopping_event', 'u1'),
        ('pileup_flag', 'u1'),
        ('stopping_pdg_id', 'i4'),
        ('xyz_stop', 'f4', (3,)),
        ('xyz_start', 'f4', (3,)),
        ('stopping_track_id', 'i4'),
        ('stopping_end_process', 'i4'),
        ('stopping_end_subprocess', 'i4'),
        ('michel', 'u1'),
        ('michel_start_xyz', 'f4', (3,)),
        ('michel_end_xyz', 'f4', (3,)),
        ('michel_start_pxyz', 'f4', (3,)),
        ('michel_plus', 'u1'),
        ('michel_track_id', 'i4'),
        ('michel_contained', 'u1'),
        ('michel_start_process', 'i4'),
        ('michel_start_subprocess', 'i4'),
        ('gamma_valid', 'u1', (2,)),
        ('gamma_pxyz', 'f4', (2, 3)),
        ('gamma_scatter_xyz', 'f4', (2, 3)),
        ('gamma_scatter_dt', 'f4', (2,)),
        ('gamma_scatter_de', 'f4', (2,)),        
        ('gamma_scatter_process', 'i4', (2,)),
        ('gamma_scatter_subprocess', 'i4', (2,)),
        ('gamma_track_id', 'i4', (2,)),
        ('scatter_track_id', 'i4', (2,)),
        ('scatter_valid', 'u1', (2,)),
        ('scatter_contained', 'u1', (2,)),
    ])
    ref_dtype = bool

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
        self.data_manager.create_dset(self.truth_labels_dset_name + '/stopping_track', dtype=self.ref_dtype)
        self.data_manager.create_dset(self.truth_labels_dset_name + '/michel_track', dtype=self.ref_dtype)
        self.data_manager.create_dset(self.truth_labels_dset_name + '/scatter_track', dtype=self.ref_dtype)        
        self.data_manager.create_ref(self.truth_labels_dset_name + '/stopping_track', self.truth_tracks_dset_name)
        self.data_manager.create_ref(self.truth_labels_dset_name + '/michel_track', self.truth_tracks_dset_name)
        self.data_manager.create_ref(self.truth_labels_dset_name + '/scatter_track', self.truth_tracks_dset_name)

    def run(self, source_name, source_slice, cache):
        super(MuonCaptureTruthLabels, self).run(source_name, source_slice, cache)
        if not self.is_mc:
            return

        # get trajectory/track segment info
        traj = cache[self.truth_trajectories_dset_name]
        
        # create truth labels
        truth_label = np.zeros(traj.shape[0], dtype=self.truth_label_dtype)
        data_slice = self.data_manager.reserve_data(self.truth_labels_dset_name, source_slice)
        
        if len(traj):
            traj = traj.reshape(traj.shape[0], -1)  # events, trajectories
            tracks = cache[self.truth_tracks_dset_name]
            tracks = tracks.reshape(traj.shape + (-1,))  # events, trajectories, segments
            tracks_idx = cache[self.truth_tracks_dset_name + '_idx']
            tracks_idx = tracks_idx.reshape(traj.shape + (-1,))
            
            # check for pile-up
            pileup = np.any(traj['eventID'] != traj['eventID'][...,0:1], axis=-1)

            # check if a trajectory stops in detector
            is_traj_stop = (~resources['Geometry'].in_fid(traj['xyz_start'].reshape(-1, 3))
                            & resources['Geometry'].in_fid(traj['xyz_end'].reshape(-1, 3))).reshape(traj.shape)
            stopping_event = np.any(is_traj_stop, axis=-1)
            # get largest momentum trajectory that stops in detector (mask if none stopping)
            i_traj_stop = ma.argmax(ma.array(np.linalg.norm(traj['pxyz_start'], axis=-1), mask=~is_traj_stop), axis=-1)
            traj_stop = np.take_along_axis(traj, i_traj_stop[..., np.newaxis], axis=-1).ravel()
            traj_stop = ma.array(traj_stop, mask=traj_stop.mask['trackID'] | ~stopping_event)
            track_stop = np.take_along_axis(tracks, i_traj_stop[..., np.newaxis, np.newaxis], axis=-2).reshape(traj.shape[0],-1)
            track_stop = ma.array(track_stop, mask=track_stop.mask['trackID'] | ~stopping_event[..., np.newaxis])
            track_idx_stop = np.take_along_axis(tracks_idx, i_traj_stop[..., np.newaxis, np.newaxis], axis=-2).reshape(traj.shape[0],-1)
            track_idx_stop.mask = track_idx_stop.mask | ~stopping_event[..., np.newaxis]
            # get first segment from track that stops in detector
            track_d_from_start = ma.sqrt(
                (track_stop['x_start'] - traj_stop['xyz_start'][...,0:1])**2
                + (track_stop['y_start'] - traj_stop['xyz_start'][...,1:2])**2
                + (track_stop['z_start'] - traj_stop['xyz_start'][...,2:3])**2)
            track_stop_first = np.take_along_axis(
                track_stop, ma.argmin(track_d_from_start, axis=-1)[..., np.newaxis],
                axis=-1).ravel()
            track_stop_first_xyz = np.c_[track_stop_first['x_start'], track_stop_first['y_start'], track_stop_first['z_start']]
            truth_label['stopping_event'] = stopping_event
            truth_label['pileup_flag'] = pileup
            truth_label['stopping_pdg_id'] = traj_stop['pdgId'] * (truth_label['stopping_event'])
            truth_label['xyz_stop'] = traj_stop['xyz_end'] * np.expand_dims(truth_label['stopping_event'],-1)
            truth_label['xyz_start'] = track_stop_first_xyz * np.expand_dims(truth_label['stopping_event'],-1)
            truth_label['stopping_end_process'] = traj_stop['end_process'] * (truth_label['stopping_event'])
            truth_label['stopping_end_subprocess'] = traj_stop['end_subprocess'] * (truth_label['stopping_event'])
            truth_label['stopping_track_id'] = traj_stop['trackID'] * (truth_label['stopping_event'])

            # now check for the presence of a Michel decay
            is_traj_michel = (
                (np.abs(traj['pdgId']) == 11) & (traj['start_process'] != 2)
                & (traj['eventID'] == np.expand_dims(traj_stop['eventID'],-1))
                & (traj['parentID'] == np.expand_dims(traj_stop['trackID'],-1))
                & ((np.linalg.norm(traj['pxyz_start'], axis=-1) < 1015) # FIXME: geant4 does not distinguish between Auger electrons and DIO electrons, use just the energy for now
                   | (np.linalg.norm(traj['pxyz_start'], axis=-1) > 1016)))
            michel = np.take_along_axis(
                traj, ma.argmax(is_traj_michel, axis=-1)[...,np.newaxis], axis=-1).ravel()
            michel_track_idx = np.take_along_axis(
                tracks_idx, ma.argmax(is_traj_michel, axis=-1)[...,np.newaxis,np.newaxis], axis=-2).reshape(len(michel),-1)
            michel_track_idx.mask = michel_track_idx.mask | ~np.any(is_traj_michel, axis=-1)[...,np.newaxis]
            truth_label['michel'] = np.any(is_traj_michel, axis=-1)
            truth_label['michel_start_xyz'] = michel['xyz_start'] * np.expand_dims(truth_label['michel'],-1)
            truth_label['michel_end_xyz'] = michel['xyz_end'] * np.expand_dims(truth_label['michel'],-1)
            truth_label['michel_start_pxyz'] = michel['pxyz_start'] * np.expand_dims(truth_label['michel'],-1)
            truth_label['michel_track_id'] = michel['trackID'] * (truth_label['michel'])
            truth_label['michel_start_process'] = michel['start_process'] * (truth_label['michel'])
            truth_label['michel_start_subprocess'] = michel['start_subprocess'] * (truth_label['michel'])
            truth_label['michel_contained'] = resources['Geometry'].in_fid(michel['xyz_end']) * (truth_label['michel'])

            # and now get potential positron info
            is_traj_michel_plus = is_traj_michel & (traj['pdgId'] == -11)
            traj_michel_plus = np.take_along_axis(
                traj, ma.argmax(is_traj_michel_plus, axis=-1)[...,np.newaxis], axis=-1).ravel()
            truth_label['michel_plus'] = np.any(is_traj_michel_plus, axis=-1)

            # find positron annihilation gammas
            is_gamma = (
                (traj['pdgId'] == 22)
                & (traj['eventID'] == np.expand_dims(traj_michel_plus['eventID'],-1))                
                & (traj['parentID'] == np.expand_dims(traj_michel_plus['trackID'],-1))
                & (traj['start_process'] == 2)
                & (traj['start_subprocess'] == 5))  # annihilation
            i_gamma0 = ma.argmax(is_gamma, axis=-1)[...,np.newaxis]
            gamma0 = np.take_along_axis(traj, i_gamma0, axis=-1).ravel()
            gamma0_valid = np.take_along_axis(is_gamma, i_gamma0, axis=-1).ravel()
            
            np.put_along_axis(is_gamma, i_gamma0, False, axis=-1)
            i_gamma1 = ma.argmax(is_gamma, axis=-1)[...,np.newaxis]
            gamma1 = np.take_along_axis(traj, i_gamma1, axis=-1).ravel()
            gamma1_valid = np.take_along_axis(is_gamma, i_gamma1, axis=-1).ravel()            

            # find first gamma scatter (use track segment due to funny edepsim truth for low energy)
            scat_idx = [None]*2
            for i, (valid, gamma) in enumerate([(gamma0_valid, gamma0), (gamma1_valid, gamma1)]):
                truth_label['gamma_valid'][...,i] = valid
                truth_label['gamma_pxyz'][...,i,:] = gamma['pxyz_start'] * np.expand_dims(valid,-1)
                truth_label['gamma_track_id'][...,i] = gamma['trackID'] * valid                

                # find closest segment to photon axis
                seg_xyz = np.c_[tracks['x_start'].ravel(),
                                tracks['y_start'].ravel(),
                                tracks['z_start'].ravel()].reshape(tracks.shape+(3,))
                seg_mask = ((tracks['trackID'] == gamma['trackID'][...,np.newaxis,np.newaxis])
                            | ((traj['parentID'] == gamma['trackID'][...,np.newaxis])
                               & (traj['pdgId'] == 11)
                               & (traj['start_process'] == 2)
                               & ((traj['start_subprocess'] == 13)
                                  | (traj['start_subprocess'] == 12)))[...,np.newaxis])
                seg_xyz = ma.array(
                    seg_xyz, mask=np.broadcast_to(np.expand_dims(~seg_mask,-1), seg_xyz.shape))
                n = gamma['pxyz_start'].copy()[:,np.newaxis,np.newaxis,:]
                n /= np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-15, None)
                d = (seg_xyz - truth_label['michel_end_xyz'][:,np.newaxis,np.newaxis,:])
                dp = ma.sum(n * d, axis=-1, keepdims=True)
                dt = np.sqrt(ma.sum((d - dp * n)**2, axis=-1))
                dt.mask = dt.mask | ((dp < 0)[...,0])
                
                i_min_seg = ma.argmin(dt, axis=-1)[...,np.newaxis]
                i_min_traj = ma.argmin(
                    np.take_along_axis(dt, i_min_seg, axis=-1)[...,0],
                    axis=-1)[...,np.newaxis]
                scat = np.take_along_axis(traj, i_min_traj, axis=-1).ravel()
                scat_idx[i] = np.take_along_axis(
                    np.take_along_axis(tracks_idx, i_min_seg, axis=-1)[...,0], i_min_traj, axis=-1)
                min_seg_de = np.take_along_axis(
                    np.take_along_axis(tracks['dE'], i_min_seg, axis=-1).reshape(traj.shape),
                    i_min_traj, axis=-1).ravel()
                min_seg_dt = np.take_along_axis(
                    np.take_along_axis(dt, i_min_seg, axis=-1).reshape(traj.shape),
                    i_min_traj, axis=-1).ravel()
                min_seg_xyz = np.take_along_axis(
                    np.take_along_axis(seg_xyz, i_min_seg[...,np.newaxis], axis=-2).reshape(traj.shape+(3,)),
                    i_min_traj[...,np.newaxis], axis=-2).reshape(-1,3)
                valid = valid & (min_seg_dt < 0.0002) & np.take_along_axis( # insures you only use the first scatter parallel to the photon direction
                    np.take_along_axis(seg_mask, i_min_seg, axis=-1).reshape(traj.shape),
                    i_min_traj, axis=-1).ravel()
                scat_idx[i].mask = scat_idx[i].mask | ~valid
                
                truth_label['gamma_scatter_dt'][...,i] = min_seg_dt * valid
                truth_label['gamma_scatter_xyz'][...,i,:] = min_seg_xyz * np.expand_dims(valid,-1)
                truth_label['gamma_scatter_de'][...,i] = min_seg_de * valid
                truth_label['gamma_scatter_process'][...,i] = scat['start_process'] * (gamma['trackID'] != scat['trackID']) * valid
                truth_label['gamma_scatter_subprocess'][...,i] = scat['start_subprocess'] * (gamma['trackID'] != scat['trackID']) * valid
                truth_label['scatter_track_id'][...,i] = scat['trackID'] * valid
                truth_label['scatter_valid'][...,i] = valid
                truth_label['scatter_contained'][...,i] = resources['Geometry'].in_fid(min_seg_xyz) * valid
        else:
            track_idx_stop = ma.array(np.empty((len(traj),0)), mask=True, shrink=False)
            michel_track_idx = ma.array(np.empty((len(traj),0)), mask=True, shrink=False)
            scat_idx = [ma.array(np.empty((len(traj),0)), mask=True, shrink=False)]*2
                
        # save truth info
        self.data_manager.write_data(self.truth_labels_dset_name, data_slice, truth_label)

        if len(traj):
            ev_idx = np.broadcast_to(np.r_[source_slice][...,np.newaxis], track_idx_stop.shape)
            ref = np.c_[ev_idx.ravel(), track_idx_stop.ravel()]
            ref = ref[~track_idx_stop.mask.ravel()]
        else:
            ref = np.empty((0,2))
        self.data_manager.reserve_data(self.truth_labels_dset_name + '/stopping_track', source_slice)
        self.data_manager.write_data(self.truth_labels_dset_name + '/stopping_track', source_slice, np.any(~track_idx_stop.mask, axis=-1))
        self.data_manager.write_ref(self.truth_labels_dset_name + '/stopping_track', self.truth_tracks_dset_name, ref)

        if len(traj):
            ev_idx = np.broadcast_to(np.r_[source_slice][...,np.newaxis], michel_track_idx.shape)
            ref = np.c_[ev_idx.ravel(), michel_track_idx.ravel()]
            ref = ref[~michel_track_idx.mask.ravel()]
        else:
            ref = np.empty((0,2))
        self.data_manager.reserve_data(self.truth_labels_dset_name + '/michel_track', source_slice)
        self.data_manager.write_data(self.truth_labels_dset_name + '/michel_track', source_slice, np.any(~michel_track_idx.mask, axis=-1))        
        self.data_manager.write_ref(self.truth_labels_dset_name + '/michel_track', self.truth_tracks_dset_name, ref)

        scat_idx = ma.concatenate(scat_idx, axis=-1)
        if len(traj):
            ev_idx = np.broadcast_to(np.r_[source_slice][...,np.newaxis], scat_idx.shape)
            ref = np.c_[ev_idx.ravel(), scat_idx.ravel()]
            ref = ref[~np.broadcast_to(scat_idx.mask.ravel(), ref.shape[0])]
        else:
            ref = np.empty((0,2))
        self.data_manager.reserve_data(self.truth_labels_dset_name + '/scatter_track', source_slice)
        self.data_manager.write_data(self.truth_labels_dset_name + '/scatter_track', source_slice, np.any(~scat_idx.mask, axis=-1))        
        self.data_manager.write_ref(self.truth_labels_dset_name + '/scatter_track', self.truth_tracks_dset_name, ref)

        
