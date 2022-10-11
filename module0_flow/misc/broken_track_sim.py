import numpy as np
import numpy.ma as ma

import module0_flow.reco.combined.tracklet_reco as tracklet_reco
from module0_flow.reco.combined.tracklet_merging import TrackletMerger

from h5flow.core import H5FlowStage, resources, H5FLOW_MPI


class JointPDF(object):
    def __init__(self, *bins):
        self.hist, self.bins = np.histogramdd(np.zeros((0, len(bins))), bins=bins)
        self.n = 0

    def fill(self, *val):
        self.n += len(val[0].ravel()) if not isinstance(val, ma.MaskedArray) else len(val[0].compressed())
        _sample = [np.expand_dims(np.clip(v.ravel(), self.bins[i][0], self.bins[i][-1])
                                  if not isinstance(v, ma.MaskedArray)
                                  else np.clip(v.compressed(), self.bins[i][0], self.bins[i][-1]), axis=-1)
                   for i, v in enumerate(val)]
        _sample = np.concatenate(_sample, axis=-1)
        hist, _ = np.histogramdd(_sample, bins=self.bins)
        self.hist = hist + self.hist


class BrokenTrackSim(H5FlowStage):
    '''
        Generates a realistic broken track distribution by randomly translating
        reconstructed track hits and removing hits that cross disabled sections
        of the anode plane.

        The algorithm is:

         1. select a random "source" track within the event passing a length selection cut
         2. translate the random track in x,y such that the track is still contained
         3. mask off hits that fall on disabled channels
         4. re-run track reconstruction on new hit distribution
         5. label new tracks as broken according the overlap of the new track hits with the old source track and the distance of their endpoints

        Parameters:
         - ``path``: ``str``, path to output datasets within HDF5 file
         - ``generate_2track_joint_pdf``: ``bool``, flag to generate an output .npz file that can be used by the the track merging reconstruction
         - ``joint_pdf_filename``: ``str``, path of output .npz file (if generated)
         - ``pdf_bins``: ``list`` of ``list``, bin description for each parameter in output pdf, each formatted as ``(log10(min), log10(max), nbins)``
         - ``rand_track_length_cut``: ``float``, track length cut for source track [mm]
         - ``broken_track_distance_cut``: ``float``, cut on the distance of the 2nd-closest new track endpoint from the closest source endpoint to label a track as broken

         - ``tracks_dset_name``: ``str``, path to input tracks dataset
         - ``hit_drift_dset_name``: ``str``, path to charge hit drift data
         - ``hits_dset_name``: ``str``, path to input charge hits dataset

        All of ``tracks_dset_name``, ``hits_dset_name``, and ``hit_drift_dset_name``
        are required in the cache.

        Requires Geometry and DisabledChannels resources in workflow.

        ``offset`` datatype (1:1 with event)::

            id          u4,     unique identifier
            dx          f8,     x translation applied to event
            dy          f8,     y translation applied to event
            i_track     i8,     index of track within event used as source

        ``label`` datatype (1:1 with new track dataset)::

            id                              u4,     unique identifier
            match                           u1,     1 if new track is matched to the source track
            broken                          u1,     1 if new track is broken
            neighbor                        i4,     index of neighboring track
            hit_frac                        f4,     fraction of hits that came from source track
            true_endpoint_d                 f4(2,), minimum distance endpoints to source track endpoints
            neighbor_deflection_angle       f4,     deflection angle of track and its neighbor
            neighbor_transverse_sin2theta   f4,     transverse endpoint angle of track to its neighbor
            neighbor_missing_length         f4,     missing length of track to its neighbor
            neighbor_overlap                f4,     overlap of track and its neighbor
            neighbor_sin2theta              f4,     angle of track and its neighbor

        The new ``tracklets`` dataset datatype is the same as
        ``TrackletReconstruction.tracklet_dtype``.

    '''
    class_version = '3.0.0'

    offset_dtype = np.dtype([
        ('id', 'u4'),
        ('dx', 'f4'),
        ('dy', 'f4'),
        ('i_track', 'i8')
    ])

    new_track_dtype = tracklet_reco.TrackletReconstruction.tracklet_dtype

    new_track_label_dtype = np.dtype([
        ('id', 'u4'),
        ('match', 'u1'),
        ('broken', 'u1'),
        ('neighbor', 'i4'),
        ('hit_frac', 'f4'),
        ('true_endpoint_d', 'f4', (2,)),
        ('neighbor_deflection_angle', 'f4'),
        ('neighbor_transverse_sin2theta', 'f4'),
        ('neighbor_missing_length', 'f4'),
        ('neighbor_overlap', 'f4'),
        ('neighbor_sin2theta', 'f4'),
    ])

    default_pdf_bins = [
        (-4, 0, 30),
        (-5, 0, 30),
        (0, 3, 30),
        (-2, 3, 30),
        (-5, 0, 30)
    ]
    missing_track_segments = 200
    truth_hit_frac_cut = 0.8

    def __init__(self, **params):
        super(BrokenTrackSim, self).__init__(**params)

        self.path = params.get('path', 'misc/broken_track_sim')

        self.generate_2track_joint_pdf = params.get('generate_2track_joint_pdf', True)
        self.joint_pdf_filename = params.get('joint_pdf_filename',
                                             f'joint_pdf-{self.class_version.replace(".","_")}.npz')
        self.pdf_bins = [np.logspace(*bins) for bins in params.get('pdf_bins',
                                                                   self.default_pdf_bins)]
        self.pdf = dict()
        if self.generate_2track_joint_pdf:
            self.pdf['rereco'] = JointPDF(*self.pdf_bins)
            self.pdf['origin'] = JointPDF(*self.pdf_bins)

        self.rand_track_length_cut = params.get('rand_track_length_cut', 100)
        self.broken_track_distance_cut = params.get('broken_track_distance_cut', 7.7)

        self.tracks_dset_name = params.get('tracks_dset_name', 'combined/tracklets')
        self.hits_dset_name = params.get('hits_dset_name', 'charge/hits')
        self.hit_drift_dset_name = params.get('hit_drift_dset_name', 'combined/hit_drift')

    def init(self, source_name):
        super(BrokenTrackSim, self).init(source_name)
        self.trajectory_pts = self.data_manager.get_attrs(self.tracks_dset_name)['trajectory_pts']
        self.trajectory_dx = self.data_manager.get_attrs(self.tracks_dset_name)['trajectory_dx']
        self.new_track_dtype = BrokenTrackSim.new_track_dtype(self.trajectory_pts)

        self.data_manager.create_dset(f'{self.path}/offset',
                                      dtype=self.offset_dtype)
        self.data_manager.create_dset(f'{self.path}/label',
                                      dtype=self.new_track_label_dtype)
        self.data_manager.create_dset(f'{self.path}/tracklets',
                                      dtype=self.new_track_dtype)

        self.data_manager.create_ref(source_name, f'{self.path}/tracklets')
        self.data_manager.create_ref(source_name, f'{self.path}/offset')

        d = self.setup_reco()
        self.reco = d['reco']

        self.pixel_x = np.unique(resources['Geometry'].pixel_xy.compress((0,)))
        self.pixel_y = np.unique(resources['Geometry'].pixel_xy.compress((1,)))

    def finish(self, source_name):
        super(BrokenTrackSim, self).finish(source_name)
        # gather from all processes
        if self.generate_2track_joint_pdf:
            if self.rank == 0:
                # merge
                d = dict()

                for key in self.pdf:
                    for i in range(self.size):
                        hist = (self.comm.recv(source=i)
                                if H5FLOW_MPI and i != 0
                                else self.pdf[key].hist)
                        d[key] = hist + (d[key] if i != 0 else 0)
                        d[key + '_bins'] = self.pdf[key].bins
                        n = (self.comm.recv(source=i)
                             if H5FLOW_MPI and i != 0
                             else self.pdf[key].n)
                        d[key + '_n'] = n + (d[key + '_n'] if i != 0 else 0)

                # save to file
                np.savez_compressed(self.joint_pdf_filename, **d)

            else:
                for key in self.pdf:
                    self.comm.send(self.pdf[key].hist, dest=0)
                    self.comm.send(self.pdf[key].n, dest=0)

    def run(self, source_name, source_slice, cache):
        super(BrokenTrackSim, self).run(source_name, source_slice, cache)
        tracks = cache[self.tracks_dset_name]
        hits = cache[self.hits_dset_name]
        hits_track_idx = cache[f'{self.hits_dset_name}_track_idx']
        hit_drift = cache[self.hit_drift_dset_name]
        hit_drift = hit_drift.reshape(hits.shape)
        events = np.expand_dims(cache[source_name], axis=-1)

        if len(np.r_[source_slice]) != 0:

            d = self.select_random_track(tracks)
            rand_tracks = d['rand_tracks']
            i_track = d['i_track']

            d = self.generate_random_translation(rand_tracks)
            rand_x = d['rand_x']
            rand_y = d['rand_y']

            d = self.apply_translation(hits, rand_x, rand_y)
            trans_hits = d['trans_hits']

            track_ids = self.reco.find_tracks(trans_hits, hit_drift['z'])
            new_tracks = self.reco.calc_tracks(trans_hits, hit_drift['z'], track_ids,
                                               self.trajectory_pts,
                                               self.trajectory_dx)

            d = self.find_matching_tracks(new_tracks, rand_tracks, rand_x, rand_y,
                                          track_ids, hits_track_idx)
            trans_rand_tracks = d['trans_rand_tracks']
            match = d['match']
            broken = d['broken']
            endpoint_distance_1 = d['endpoint_distance_1']
            endpoint_distance_2 = d['endpoint_distance_2']
            hit_frac = d['hit_frac']

            d = TrackletMerger.find_k_neighbor(new_tracks, broken.reshape(new_tracks.shape + (1,)) & broken.reshape(new_tracks.shape[:-1] + (1, -1)))
            neighbor = d['neighbor']
            d = TrackletMerger.find_k_neighbor(tracks)
            orig_neighbor = d['neighbor']

            if self.generate_2track_joint_pdf:
                track2_deflection_angle = TrackletMerger.calc_2track_deflection_angle(new_tracks, neighbor)
                track2_deflection_angle_orig = TrackletMerger.calc_2track_deflection_angle(tracks, orig_neighbor)
                track2_transverse_sin2theta = TrackletMerger.calc_2track_transverse_sin2theta(new_tracks, neighbor)
                track2_transverse_sin2theta_orig = TrackletMerger.calc_2track_transverse_sin2theta(tracks, orig_neighbor)
                track2_missing_length = TrackletMerger.calc_2track_missing_length(new_tracks, neighbor, TrackletMerger.missing_track_segments, self.pixel_x, self.pixel_y, resources['DisabledChannels'].disabled_channel_lut, TrackletMerger.cathode_region)
                track2_missing_length_orig = TrackletMerger.calc_2track_missing_length(tracks, orig_neighbor, TrackletMerger.missing_track_segments, self.pixel_x, self.pixel_y, resources['DisabledChannels'].disabled_channel_lut, TrackletMerger.cathode_region)
                track2_overlap = TrackletMerger.calc_2track_overlap(new_tracks, neighbor)
                track2_overlap_orig = TrackletMerger.calc_2track_overlap(tracks, orig_neighbor)
                track2_sin2theta = TrackletMerger.calc_2track_sin2theta(new_tracks, neighbor)
                track2_sin2theta_orig = TrackletMerger.calc_2track_sin2theta(tracks, orig_neighbor)

                rereco_mask = (broken[..., 0] & np.take_along_axis(broken[..., 0], neighbor, axis=1))

                self.pdf['rereco'].fill(track2_deflection_angle[rereco_mask], track2_transverse_sin2theta[rereco_mask],
                                        track2_missing_length[rereco_mask], track2_overlap[rereco_mask],
                                        track2_sin2theta[rereco_mask])
                self.pdf['origin'].fill(track2_deflection_angle_orig, track2_transverse_sin2theta_orig,
                                        track2_missing_length_orig, track2_overlap_orig,
                                        track2_sin2theta_orig)

        else:
            rand_x = ma.array(np.empty((0,)))
            new_tracks = ma.array(np.empty((0,), dtype=self.new_track_dtype))
            match = ma.array(np.empty((0,)))

        # save offsets
        offset_array = np.empty((len(rand_x.compressed()),), dtype=self.offset_dtype)
        offset_slice = self.data_manager.reserve_data(f'{self.path}/offset', len(offset_array))
        if len(offset_array):
            offset_array['id'] = np.r_[offset_slice]
            offset_array['dx'] = rand_x.compressed()
            offset_array['dy'] = rand_y.compressed()
            offset_array['i_track'] = i_track.compressed()
            self.data_manager.write_data(f'{self.path}/offset', offset_slice, offset_array)

            ref = np.c_[np.r_[source_slice][~rand_x.mask.ravel()], offset_array['id']]
        else:
            ref = np.empty((0, 2))
        self.data_manager.write_ref(source_name, f'{self.path}/offset', ref)

        # save new tracklets
        tracks_slice = self.data_manager.reserve_data(f'{self.path}/tracklets',
                                                      len(new_tracks['id'].compressed()))
        label_slice = self.data_manager.reserve_data(f'{self.path}/label', len(match.compressed()))
        if tracks_slice.stop - tracks_slice.start:
            np.place(new_tracks['id'], ~new_tracks['id'].mask, np.r_[tracks_slice].astype('u4'))
            self.data_manager.write_data(f'{self.path}/tracklets', tracks_slice, new_tracks[~new_tracks['id'].mask])

            label_array = np.empty((len(match.compressed()),), dtype=self.new_track_label_dtype)
            label_array['id'] = np.r_[label_slice]
            label_array['match'] = match.compressed()
            label_array['broken'] = broken.compressed()
            label_array['neighbor'] = neighbor.flat[~broken.mask.ravel()]
            label_array['hit_frac'] = hit_frac.compressed()
            label_array['true_endpoint_d'] = np.c_[endpoint_distance_1.compressed(), endpoint_distance_2.compressed()]
            if self.generate_2track_joint_pdf:
                label_array['neighbor_deflection_angle'] = track2_deflection_angle.flat[~broken.mask.ravel()]
                label_array['neighbor_transverse_sin2theta'] = track2_transverse_sin2theta.flat[~broken.mask.ravel()]
                label_array['neighbor_missing_length'] = track2_missing_length.flat[~broken.mask.ravel()]
                label_array['neighbor_overlap'] = track2_overlap.flat[~broken.mask.ravel()]
                label_array['neighbor_sin2theta'] = track2_sin2theta.flat[~broken.mask.ravel()]
            self.data_manager.write_data(f'{self.path}/label', label_slice, label_array)

            ref = np.c_[np.indices(new_tracks.shape)[0][~new_tracks['id'].mask] + source_slice.start, new_tracks['id'].compressed()]
        else:
            ref = np.empty((0, 2))
        self.data_manager.write_ref(source_name, f'{self.path}/tracklets', ref)

    def setup_reco(self):
        # set upt tracklet reconstruction
        config = dict(self.data_manager.get_attrs(self.tracks_dset_name))
        reco = tracklet_reco.TrackletReconstruction(
            name=None, data_manager=self.data_manager,
            **config
        )

        return dict(
            reco=reco
        )

    def select_random_track(self, tracks):
        # generate random index into tracks
        i_track = np.indices(tracks.shape)[-1]
        i_track = np.random.permutation(i_track.T).T  # transpose b/c permutation works on first dimension

        # use only indices that point to valid tracks
        _track_l = np.linalg.norm(tracks['start'] - tracks['end'], axis=-1)
        i_track_mask = (
            np.take_along_axis(tracks['id'].mask, i_track, axis=-1) |
            np.take_along_axis(_track_l < self.rand_track_length_cut, i_track, axis=-1)
        )

        # use first valid index for each event
        i_track_sel = np.argmax(~i_track_mask, axis=-1).reshape(-1, 1)
        i_track = np.take_along_axis(i_track, i_track_sel, axis=-1)
        rand_tracks = np.take_along_axis(tracks, i_track, axis=-1).copy()

        return dict(
            rand_tracks=rand_tracks,
            i_track=ma.array(i_track, mask=rand_tracks['id'].mask)
        )

    def generate_random_translation(self, rand_tracks):
        pixel_pitch = resources['Geometry'].pixel_pitch
        _x_extent = (self.pixel_x.min(),
                     self.pixel_x.max())
        _y_extent = (self.pixel_y.min(),
                     self.pixel_y.max())

        # find bounding box between track and pixel plane
        _track_x_extent = (np.minimum(rand_tracks['start'][..., 0], rand_tracks['end'][..., 0]),
                           np.maximum(rand_tracks['start'][..., 0], rand_tracks['end'][..., 0]))
        _track_y_extent = (np.minimum(rand_tracks['start'][..., 1], rand_tracks['end'][..., 1]),
                           np.maximum(rand_tracks['start'][..., 1], rand_tracks['end'][..., 1]))
        _rand_x_extent = _x_extent[0] - _track_x_extent[0], _x_extent[1] - _track_x_extent[1]
        _rand_y_extent = _y_extent[0] - _track_y_extent[0], _y_extent[1] - _track_y_extent[1]

        # generate a random translation that is contained in pixel plane and aligned with pixel pitch
        rand_i_x = np.round((np.random.uniform(size=rand_tracks.shape)
                             * (_rand_x_extent[1] - _rand_x_extent[0])
                             + _rand_x_extent[0]) / pixel_pitch).astype(int)
        rand_i_y = np.round((np.random.uniform(size=rand_tracks.shape)
                             * (_rand_y_extent[1] - _rand_y_extent[0])
                             + _rand_y_extent[0]) / pixel_pitch).astype(int)
        rand_x = rand_i_x * pixel_pitch
        rand_y = rand_i_y * pixel_pitch

        return dict(
            rand_x=rand_x,
            rand_y=rand_y
        )

    def apply_translation(self, hits, rand_x, rand_y):
        trans_hits = hits.copy()
        trans_hits['px'] = rand_x + trans_hits['px']
        trans_hits['py'] = rand_y + trans_hits['py']

        # remove hits outside of TPC
        _mask = (trans_hits['id'].mask |
                 (trans_hits['px'] > self.pixel_x.max()) |
                 (trans_hits['px'] < self.pixel_x.min()) |
                 (trans_hits['py'] > self.pixel_y.max()) |
                 (trans_hits['py'] < self.pixel_y.min()))

        # remove hits on disabled channels
        _disabled_mask = resources['DisabledChannels'].disabled_channel_lut[trans_hits['iogroup'],
                                                                            trans_hits['px'].astype(int),
                                                                            trans_hits['py'].astype(int)]
        _disabled_mask = _disabled_mask.reshape(_mask.shape)
        _mask = _mask | _disabled_mask

        trans_hits = ma.array(trans_hits, mask=_mask)

        return dict(
            trans_hits=trans_hits
        )

    def find_matching_tracks(self, new_tracks, rand_tracks, rand_x, rand_y,
                             track_ids, hits_track_idx):
        # (events, hits, 1)
        rand_track_hits = hits_track_idx == np.expand_dims(rand_tracks['id'], axis=-1)

        # (events, 1, new_tracks)
        new_id = np.expand_dims(np.indices(new_tracks.shape)[-1], axis=1)

        # (events, hits, new_tracks)
        new_track_hits = (new_id == np.expand_dims(track_ids, axis=-1))

        # (events, new_tracks)
        frac_new_hits = np.expand_dims(np.sum(rand_track_hits & new_track_hits, axis=1)
                                       / np.sum(new_track_hits, axis=1), axis=-1)
        match = frac_new_hits > self.truth_hit_frac_cut

        _dxyz = np.concatenate((
            np.expand_dims(rand_x, axis=-1),
            np.expand_dims(rand_y, axis=-1),
            np.expand_dims(np.zeros_like(rand_y), axis=-1)
        ), axis=-1)

        trans_rand_tracks = rand_tracks.copy()
        trans_rand_tracks['start'] = trans_rand_tracks['start'] + _dxyz
        trans_rand_tracks['end'] = trans_rand_tracks['end'] + _dxyz

        endpoint_distance = ma.concatenate((
            ma.sum((new_tracks['start'] - trans_rand_tracks['end'])**2, axis=-1,
                   keepdims=True),
            ma.sum((new_tracks['end'] - trans_rand_tracks['end'])**2, axis=-1,
                   keepdims=True),
            ma.sum((new_tracks['start'] - trans_rand_tracks['start'])**2, axis=-1,
                   keepdims=True),
            ma.sum((new_tracks['end'] - trans_rand_tracks['start'])**2, axis=-1,
                   keepdims=True),
        ), axis=-1)
        endpoint_distance = ma.sqrt(endpoint_distance)
        i_endpoint_distance = ma.array(ma.argsort(endpoint_distance, axis=-1),
                                       mask=endpoint_distance.mask)

        # match if closest endpoint within some radius
        endpoint_distance_1 = np.take_along_axis(endpoint_distance,
                                                 i_endpoint_distance[..., 0:1],
                                                 axis=-1)

        # id broken tracks
        endpoint_distance_2 = np.take_along_axis(endpoint_distance,
                                                 i_endpoint_distance[..., 1:2],
                                                 axis=-1)
        broken = (endpoint_distance_2 > self.broken_track_distance_cut) & match

        mask = match.mask | broken.mask | frac_new_hits.mask

        return dict(
            trans_rand_tracks=trans_rand_tracks,
            match=ma.array(match, mask=mask),
            broken=ma.array(broken, mask=mask),
            hit_frac=ma.array(frac_new_hits, mask=mask),
            endpoint_distance_1=endpoint_distance_1,
            endpoint_distance_2=endpoint_distance_2
        )
