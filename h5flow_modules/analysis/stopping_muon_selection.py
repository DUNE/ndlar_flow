import numpy as np
import numpy.ma as ma

from h5flow.core import H5FlowStage, resources
from h5flow.data import dereference_chain

from module0_flow.util.func import mode, condense_array


class StoppingMuonSelection(H5FlowStage):
    '''
        Perform a basic selection for stopping muons. A stopping muon event
        is defined by no more than one merged track segment that enters the
        detector fiducial volume and does not leave it. Creates a boolean array
        of 1:1 with events indicating stopping events, and creates a boolean
        array 1:1 with merged track segments if they individually meet the
        stopping criteria.

        If the file is a MC file, also generates boolean arrays with the true
        value.


    '''
    class_version = '0.0.0'

    default_fid_cut = 20 # mm
    default_cathode_fid_cut = 20 # mm
    default_length_cut = 100 # mm
    default_merged_dset_name = 'combined/tracklets/merged'
    default_t0_dset_name = 'combined/t0'
    default_mc_trajectory_path = ['combined/tracklets/merged',
                                  'charge/hits', 'charge/packets',
                                  'mc_truth/tracks',
                                  'mc_truth/trajectories']
    default_path = 'analysis/stopping_muons'

    event_sel_dset_name = 'event_sel_reco'
    track_sel_dset_name = 'track_sel_reco'
    event_sel_truth_dset_name = 'event_sel_truth'
    track_sel_truth_dset_name = 'track_sel_truth'

    event_sel_dtype = np.dtype([('sel', 'u1')])
    track_sel_dtype = np.dtype([('sel', 'u1')])

    def __init__(self, **params):
        super(StoppingMuonSelection, self).__init__(**params)

        self.path = params.get('path', self.default_path)
        self.fid_cut = params.get('fid_cut', self.default_fid_cut)
        self.cathode_fid_cut = params.get('cathode_fid_cut', self.default_cathode_fid_cut)        
        self.length_cut = params.get('fid_cut', self.default_length_cut)        
        self.t0_dset_name = params.get('t0_dset_name',
                                       self.default_t0_dset_name)
        self.merged_dset_name = params.get('merged_dset_name',
                                           self.default_merged_dset_name)
        self.mc_trajectory_path = params.get('mc_trajectory_path',
                                             self.default_mc_trajectory_path)

        self.regions = []

    def init(self, source_name):
        self.is_mc = resources['RunData'].is_mc

        self.data_manager.set_attrs(self.path,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    fid_cut=self.fid_cut,
                                    length_cut=self.length_cut,
                                    merged_dset_name=self.merged_dset_name
                                    )
        self.data_manager.create_dset(f'{self.path}/{self.event_sel_dset_name}',
                                      self.event_sel_dtype)
        self.data_manager.create_dset(f'{self.path}/{self.track_sel_dset_name}',
                                      self.track_sel_dtype)
        if self.is_mc:
            self.data_manager.create_dset(f'{self.path}/{self.event_sel_truth_dset_name}',
                                          self.event_sel_dtype)
            self.data_manager.create_dset(f'{self.path}/{self.track_sel_truth_dset_name}',
                                          self.track_sel_dtype)

        self.create_regions()
        self.data_manager.set_attrs(self.path, regions=np.array(self.regions))

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
        dy = end_xyz[:,1] - start_xyz[:,1]
        return ((end_in_fid & (dy < 0)) | (~end_in_fid & (dy > 0)))

    def load_track_trajectories(self, source_name, source_slice, path):
        chain = list(zip([source_name] + path[:-1], path))

        data = self.data_manager.get_dset(path[-1])
        ref, ref_dir = list(zip(*[self.data_manager.get_ref(p, c) for p, c in chain]))
        regions = [self.data_manager.get_ref_region(p, c) for p, c in chain]

        return dereference_chain(source_slice, ref, data=data, regions=regions, ref_directions=ref_dir)

    def run(self, source_name, source_slice, cache):
        tracks = cache[self.merged_dset_name]
        t0 = cache[self.t0_dset_name].reshape(cache[source_name].shape)

        # find all tracks that end in the fiducial volume
        is_stopping = (self.contained(tracks.ravel()['start'],
                                      tracks.ravel()['end'])
                       & (tracks.ravel()['length'] > self.length_cut))
        is_throughgoing = (self.through_going(tracks.ravel()['start'],
                                              tracks.ravel()['end']))
        is_downward = self.downward(tracks.ravel()['start'],
                                    tracks.ravel()['end'])
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
            xyz_start = track_true_traj.ravel()['xyz_start'].copy()
            xyz_end = track_true_traj.ravel()['xyz_end'].copy()
            # FIXME: coordinates are weird between sim and data
            xyz_start[:,1] += 218.236
            xyz_end[:,1] += 218.236
            new_x_start = xyz_start[:,2].copy()
            new_x_end = xyz_end[:,2].copy()
            new_z_start = xyz_start[:,0].copy()
            new_z_end = xyz_end[:,0].copy()
            xyz_start[:,0] = new_x_start
            xyz_start[:,2] = new_z_start
            xyz_end[:,0] = new_x_end
            xyz_end[:,2] = new_z_end
            is_muon = ma.abs(track_true_traj['pdgId']) == 13
            is_true_stopping = self.contained(xyz_start, xyz_end)
            is_true_stopping = is_true_stopping.reshape(tracks.shape)

        # define a stopping event as one with exclusively 1 track that ends in fid.
        event_is_stopping = ((ma.sum(is_stopping, axis=-1) == 1)
                             & (t0['type'] != 0)
                             & (ma.sum(is_throughgoing, axis=-1) == 0)
                             & (ma.sum(is_stopping & is_downward, axis=-1) == 1))

        if self.is_mc:
            # define true stopping events as events with at least 1 muon that ends in fid.
            event_is_true_stopping = ma.sum(is_muon & is_true_stopping, axis=-1) >= 1

        # prep arrays to write to file
        event_sel = np.zeros(len(tracks), dtype=self.event_sel_dtype)
        track_sel = np.zeros(len(tracks['id'].compressed()), dtype=self.track_sel_dtype)
        event_sel['sel'] = event_is_stopping
        track_sel['sel'] = is_stopping.compressed()
        if self.is_mc:
            event_true_sel = np.zeros(len(tracks), dtype=self.event_sel_dtype)
            track_true_sel = np.zeros(len(tracks['id'].compressed()), dtype=self.track_sel_dtype)
            event_true_sel['sel'] = event_is_true_stopping
            track_true_sel['sel'] = (is_true_stopping[~tracks['id'].mask].ravel()
                                     & is_muon[~tracks['id'].mask].ravel())

        # reserve data space
        event_sel_slice = self.data_manager.reserve_data(
            f'{self.path}/{self.event_sel_dset_name}', len(event_sel))
        track_sel_slice = self.data_manager.reserve_data(
            f'{self.path}/{self.track_sel_dset_name}', len(track_sel))
        if self.is_mc:
            event_sel_truth_slice = self.data_manager.reserve_data(
                f'{self.path}/{self.event_sel_truth_dset_name}',
                len(event_true_sel))
            track_sel_truth_slice = self.data_manager.reserve_data(
                f'{self.path}/{self.track_sel_truth_dset_name}',
                len(track_true_sel))

        # write
        self.data_manager.write_data(f'{self.path}/{self.event_sel_dset_name}',
                                     event_sel_slice, event_sel)
        self.data_manager.write_data(f'{self.path}/{self.track_sel_dset_name}',
                                     track_sel_slice, track_sel)
        if self.is_mc:
            self.data_manager.write_data(
                f'{self.path}/{self.event_sel_truth_dset_name}',
                event_sel_truth_slice, event_true_sel)
            self.data_manager.write_data(
                f'{self.path}/{self.track_sel_truth_dset_name}',
                track_sel_truth_slice, track_true_sel)
