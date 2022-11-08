import numpy as np
import numpy.ma as ma
from scipy import ndimage

from h5flow.core import H5FlowStage, resources

from module0_flow.reco.combined.tracklet_reco import TrackletReconstruction
from module0_flow.util.func import condense_array


class TrackletMerger(H5FlowStage):
    '''
        Merges existing tracks with neighbors based on a multi-dimensional
        likelihood ratio metric. The observables used in the likelihood
        estimation are:

         - ``sin^2(theta)``: angle between the two track segments
         - transverse distance: maximum transverse displacement of track from the axis of the first track [mm]
         - missing length: length of line segment between closer two endpoints that crosses active pixels [mm]
         - overlap: quadrature sum of 1D overlap of tracks in x, y, and z [mm]
         - delta-dQ/dx: difference in raw dQ/dx [mV]

        Requires an input histogram .npz file consisting of 4 arrays:

         - ``'{sig}'``: an array of shape: ``(N0, N1, ... N4)`` representing the number of signal events in each bin of the 5 observables
         - ``'{sig}_bins'``: an array of 5 arrays each with shape: ``Ni+1`` representing the bin edges
         - ``'{bkg}'``: an array of shape: ``(N0, N1, ... N4)`` representing the number of background events in each bin of the 5 observables

        The selection is performed by normalizing the input histograms to a PDF,
        calculating the ``signal/background`` likelihood ratio, and rescaling
        to a normalized metric between 0 and 1. The p-value (or inefficiency)
        of this metric is calculated based on the signal histogram. The
        track merging selection cut is applied on this p-value, e.g. a
        ``pvalue_cut = 0.05`` will result in a 95% selection efficiency for
        merging neighboring tracks (at least for the sample used to generate
        the input histograms).

        Parameters:
         - ``pdf_filename``: ``str``, path to .npz file containing multi-dimensional pdf (more details above)
         - ``pdf_sig_name``: ``str``, name of array in .npz file containing the "signal" histogram
         - ``pdf_bkg_name``: ``str``, name of array in .npz file containing the "background" histogram
         - ``pvalue_cut``: ``float``, p-value/inefficiency used as cut for likelihood ratio
         - ``max_neighbors``: ``int``, number of neighbor tracks to attempt merge procedure
         - ``hits_dset_name``: ``str``, path to input charge hits dataset
         - ``hit_drift_dset_name``: ``str``, path to charge hit drift data
         - ``hits_dset_name``: ``str``, path to input charge hits dataset
         - ``track_hits_dset_name``: ``str``, path to input track-referred charge hits dataset
         - ``tracks_dset_name``: ``str``, path to input track dataset
         - ``merged_dset_name``: ``str``, path to output track dataset

        All of ``hits_dset_name``, ``hit_drift_dset_name``, ``track_hits_dset_name``,
        and ``tracks_dset_name`` are required in the cache.

        Requires both Geometry and DisabledChannels resources in workflow.

        ``merged`` datatype is the same as the
        ``TrackletReconstruction.tracklet_dtype``.

        Example config::

            track_merge:
                classname: TrackletMerger
                requires:
                 - 'combined/tracklets'
                 - name: 'combined/track_hits
                   path: ['combined/tracklets', charge/hits']
                 - name: 'combined/track_hit_drift
                   path: ['combined/tracklets', charge/hits', 'combined/hit_drift']
                params:
                    merged_dset_name: 'combined/tracklets/merged'
                    hit_drift_dset_name: 'combined/hit_drift'
                    hits_dset_name: 'charge/hits'
                    tracks_dset_name: 'combined/tracklets'
                    pdf_filename: 'joint_pdf.npz'
                    pvalue_cut: 0.10
                    max_neighbors: 5

    '''
    class_version = '3.0.0'

    default_pdf_filename = 'joint_pdf-2_0_1.npz'
    default_pdf_sig_name = 'rereco'
    default_pdf_bkg_name = 'origin'
    default_pvalue_cut = 0.10
    default_max_neighbors = 5

    default_hit_drift_dset_name = 'combined/track_hit_drift'
    default_hits_dset_name = 'charge/hits'
    default_tracks_dset_name = 'combined/tracklets'
    default_track_hits_dset_name = 'combined/track_hits'
    default_merged_dset_name = 'combined/tracklets/merged'

    merged_dtype = TrackletReconstruction.tracklet_dtype

    missing_track_segments = 150
    cathode_region = 15

    def __init__(self, **params):
        super(TrackletMerger, self).__init__(**params)

        self.pdf_filename = params.get('pdf_filename', self.default_pdf_filename)
        self.pdf_sig_name = params.get('pdf_sig_name', self.default_pdf_sig_name)
        self.pdf_bkg_name = params.get('pdf_bkg_name', self.default_pdf_bkg_name)
        self.pvalue_cut = params.get('pvalue_cut', self.default_pvalue_cut)
        self.max_neighbors = params.get('max_neighbors', self.default_max_neighbors)

        self.hit_drift_dset_name = params.get('hit_drift_dset_name', self.default_hit_drift_dset_name)
        self.hits_dset_name = params.get('hits_dset_name', self.default_hits_dset_name)
        self.track_hits_dset_name = params.get('track_hits_dset_name', self.default_track_hits_dset_name)
        self.tracks_dset_name = params.get('tracks_dset_name', self.default_tracks_dset_name)
        self.merged_dset_name = params.get('merged_dset_name', self.default_merged_dset_name)

    def init(self, source_name):
        super(TrackletMerger, self).init(source_name)

        self.r, self.r_bins, self.statistic_bins, self.p_bins = (
            self.load_r_values(self.pdf_filename, self.pdf_sig_name,
                               self.pdf_bkg_name))

        self.data_manager.set_attrs(self.merged_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    hits_dset=self.hits_dset_name,
                                    hit_drift_dset=self.hit_drift_dset_name,
                                    tracks_dset=self.tracks_dset_name,
                                    max_neighbors=self.max_neighbors,
                                    pvalue_cut=self.pvalue_cut,
                                    pdf_filename=self.pdf_filename,
                                    pdf_sig_name=self.pdf_sig_name,
                                    pdf_bkg_name=self.pdf_bkg_name
                                    )

        self.trajectory_pts = self.data_manager.get_attrs(self.tracks_dset_name)['trajectory_pts']
        self.trajectory_dx = self.data_manager.get_attrs(self.tracks_dset_name)['trajectory_dx']

        self.merged_dtype = TrackletMerger.merged_dtype(self.trajectory_pts)
        self.data_manager.create_dset(self.merged_dset_name, self.merged_dtype)
        self.data_manager.create_ref(self.merged_dset_name, self.hits_dset_name)
        self.data_manager.create_ref(self.merged_dset_name, self.tracks_dset_name)
        self.data_manager.create_ref(source_name, self.merged_dset_name)

        self.pixel_x = np.unique(resources['Geometry'].pixel_xy.compress((0,)))
        self.pixel_y = np.unique(resources['Geometry'].pixel_xy.compress((1,)))

    def run(self, source_name, source_slice, cache):
        super(TrackletMerger, self).run(source_name, source_slice, cache)

        track_hit_drift = cache[self.hit_drift_dset_name]
        track_hits = cache[self.track_hits_dset_name]
        tracks = cache[self.tracks_dset_name]
        track_hit_drift = track_hit_drift.reshape(track_hits.shape)

        # ajacency matrix to represent if tracks should be merged or not (True == to merge)
        track_merged = np.expand_dims(np.diagflat(np.ones(tracks.shape[-1], dtype=bool)), axis=0)
        track_checked = (track_merged.copy()
                         | np.expand_dims(tracks['id'].mask, axis=1)
                         | np.expand_dims(tracks['id'].mask, axis=2))
        track_merged = np.broadcast_to(track_merged, tracks.shape + tracks.shape[-1:]).copy()
        track_checked = np.broadcast_to(track_checked, tracks.shape + tracks.shape[-1:]).copy()

        if len(np.r_[source_slice]):

            # iterative approach
            for _ in range(self.max_neighbors):
                # find neighboring tracks that have not been checked
                neighbor = self.find_k_neighbor(tracks, mask=~track_checked)['neighbor']

                # calculate the p-value for neighbor pair
                params = [
                    self.calc_2track_deflection_angle(tracks, neighbor),
                    self.calc_2track_transverse_sin2theta(tracks, neighbor),
                    self.calc_2track_missing_length(tracks, neighbor,
                                                    self.missing_track_segments,
                                                    self.pixel_x, self.pixel_y,
                                                    resources['DisabledChannels'].disabled_channel_lut,
                                                    self.cathode_region),
                    self.calc_2track_overlap(tracks, neighbor),
                    self.calc_2track_sin2theta(tracks, neighbor)
                ]
                pvalue = np.expand_dims(self.score_neighbor(self.r, self.r_bins, self.statistic_bins, self.p_bins, *params), -1)
                neighbor = np.expand_dims(neighbor, -1)

                # merge tracks that have large p-values
                should_merge = (((pvalue >= self.pvalue_cut)
                                 | np.take_along_axis(track_merged, neighbor, -1))
                                & ~neighbor.mask
                                & ~tracks['id'][..., np.newaxis])
                np.put_along_axis(track_merged, neighbor, should_merge, axis=-1)
                np.put_along_axis(track_checked, neighbor, True, axis=-1)

                if np.all(track_checked):
                    break

            # collect valid associations into track groups
            axes = np.arange(track_merged.ndim).astype(int)
            new_axes = axes.copy()
            new_axes[-1] = axes[-2]
            new_axes[-2] = axes[-1]
            track_merged = track_merged | np.transpose(track_merged, axes=new_axes)
            track_merged = self.create_groups(track_merged)

            # now, collect the hits from the original tracks into the track groups
            # get unique track groups, shape: (n_ev, n_grp, n_track)
            track_merged = np.unique(track_merged, axis=1)
            track_merged_mask = np.ones(track_merged.shape, dtype=bool)
            for ev in range(track_merged.shape[0]):
                _, index = np.unique(track_merged[ev], axis=0, return_index=True)
                track_merged_mask[ev, index] = False
            track_grp = ma.array(track_merged, mask=track_merged_mask | ~track_merged, shrink=False)
            track_grp_nhit = np.sum(np.expand_dims(tracks['nhit'], axis=1) * track_grp, axis=-1).filled(0)

            track_grp_hits_shape = track_grp.shape[:-1] + (np.max(track_grp_nhit),)
            # (n_ev, n_grp, n_hit')
            track_grp_hits = np.zeros(track_grp_hits_shape, dtype=track_hits.dtype)
            track_grp_hit_drift = np.zeros(track_grp_hits_shape, dtype=track_hit_drift.dtype)
            track_grp_id = np.zeros(track_grp_hits_shape, dtype=int)
            track_grp_hits_mask = np.ones(track_grp_hits_shape, dtype=bool)
            for grp_idx in range(track_grp_hits_shape[-2]):
                mask = np.indices(track_grp_hits[:, grp_idx].shape)[-1] < track_grp_nhit[:, grp_idx, np.newaxis]

                hit_mask = ~track_hits[track_grp[:, grp_idx].filled(False)]['id'].mask
                np.place(track_grp_hits[:, grp_idx], mask, track_hits[track_grp[:, grp_idx].filled(0)][hit_mask])
                np.place(track_grp_hit_drift[:, grp_idx], mask, track_hit_drift[track_grp[:, grp_idx].filled(0)][hit_mask])
                np.place(track_grp_id[:, grp_idx], mask, grp_idx)
                np.place(track_grp_hits_mask[:, grp_idx], mask, False)

            track_grp_hits = ma.array(track_grp_hits, mask=track_grp_hits_mask, shrink=False)
            track_grp_hit_drift = ma.array(track_grp_hit_drift, mask=track_grp_hits_mask, shrink=False)
            track_grp_id = ma.array(track_grp_id, mask=track_grp_hits_mask, shrink=False)

            new_shape = track_grp.shape[0:1] + (-1,)
            track_grp_hits = track_grp_hits.reshape(new_shape)
            track_grp_hit_drift = track_grp_hit_drift.reshape(new_shape)
            track_grp_id = track_grp_id.reshape(new_shape)

            # recalculate track parameters
            calc_shape = (track_grp_id.shape[0], -1)
            merged_tracks = TrackletReconstruction.calc_tracks(
                track_grp_hits.reshape(calc_shape), track_grp_hit_drift['z'].reshape(calc_shape),
                track_grp_id.reshape(calc_shape), self.trajectory_pts,
                self.trajectory_dx)
        else:
            merged_tracks = ma.masked_all((0, 1), dtype=self.merged_dtype)
            track_grp = ma.masked_all((0, 1, 1), dtype=bool)
            track_grp_id = ma.masked_all((0, 1), dtype=int)
            track_grp_hits = ma.masked_all((0, 1), dtype=track_hits.dtype)
            track_grp_hit_drift = ma.masked_all((0, 1), dtype=track_hit_drift.dtype)

        # save to merged track dataset
        n_tracks = np.count_nonzero(~merged_tracks['id'].mask)
        merged_tracks_mask = ~merged_tracks['id'].mask

        merged_tracks_slice = self.data_manager.reserve_data(self.merged_dset_name, n_tracks)
        np.place(merged_tracks['id'], merged_tracks_mask, np.r_[merged_tracks_slice].astype('u4'))
        self.data_manager.write_data(self.merged_dset_name, merged_tracks_slice, merged_tracks[merged_tracks_mask])

        # merged -> tracklet ref
        i_ev, i_grp, i_track = np.where(track_grp & np.expand_dims(~tracks['id'].mask, 1) & ~track_grp.mask)
        ref = np.c_[merged_tracks['id'][i_ev, i_grp].compressed(), tracks['id'][i_ev, i_track].compressed()]
        self.data_manager.write_ref(self.merged_dset_name, self.tracks_dset_name, ref)

        # merged -> hit ref
        hit_mask = (np.expand_dims(track_grp_id, 1)
                    == np.expand_dims(np.indices(merged_tracks.shape)[-1], -1))
        i_ev, i_grp, i_hit = np.where(hit_mask)
        ref = np.c_[merged_tracks['id'][i_ev, i_grp].compressed(),
                    track_grp_hits['id'][i_ev, i_hit].compressed()]
        self.data_manager.write_ref(self.merged_dset_name, self.hits_dset_name, ref)

        # event -> merged ref
        ev_id = np.broadcast_to(np.expand_dims(np.r_[source_slice], axis=-1), merged_tracks.shape)
        ref = np.c_[ev_id[merged_tracks_mask], merged_tracks['id'][merged_tracks_mask]]
        self.data_manager.write_ref(source_name, self.merged_dset_name, ref)

    @staticmethod
    def create_groups(mask):
        '''
            Combine masks of ``n x n`` ajacency matrix such that the mask of
            row i is equal to the ``OR`` of the rows that can be reached from
            ``i`` and the rows that can reach ``i``. E.g.::

                arr = [[1,0,1],
                       [0,1,0],
                       [0,0,1]]
                new_arr = create_groups(arr)
                new_arr # [[1,0,1],
                           [0,1,0],
                           [1,0,1]]

            and::

                arr = [[0,1,0],
                       [0,0,1],
                       [1,1,0]]
                new_arr = create_groups(arr)
                new_arr # [[1,1,1],
                           [1,1,1],
                           [0,1,1]]

            :param mask: ajacency matrix (``shape: (..., n, n)``)

            :returns: updated ajacency matrix (``shape: (..., n, n)``)
        '''
        new_mask = np.zeros_like(mask)

        # get index of masks (starting with True values)
        i_mask = np.indices(mask.shape)[-1]
        j_mask = np.indices(mask.shape)[-2]
        step = 0
        while (step < i_mask.shape[-1]):
            # step through indices
            # get other index (shape: (..., n, 1))
            ii_mask = np.expand_dims(i_mask[..., step], axis=-1)
            jj_mask = np.expand_dims(j_mask[..., step], axis=-1)
            # get other mask (shape: (..., n, n))
            other_mask = np.take_along_axis(mask, ii_mask, -2)
            # get other matched to current (shape: (..., n, 1))
            other_matched = np.take_along_axis(mask, ii_mask, -1)
            # get self matched to current (shape: (..., n, 1))
            self_matched = np.take_along_axis(other_mask, jj_mask, -1)

            # combine with current track(s)
            new_mask[:] = (new_mask | (other_mask & other_matched) | (other_mask & self_matched))
            step += 1

        if np.all(new_mask == mask):
            return new_mask
        return TrackletMerger.create_groups(new_mask)

    @staticmethod
    def find_k_neighbor(tracks, mask=None, k=1):
        '''
            Find ``k``-th neighbor based on endpoint distance and require no overlap:

             - ``tracks`` is an (N,M) array of tracks
             - ``mask`` is boolean of same shape as ``tracks``
             - ``mask`` true indicates a valid track to search for neighbors

        '''
        ntracks = tracks.shape[-1]
        if mask is None:
            mask = np.ones(tracks.shape + tracks.shape[-1:], dtype=bool)
        mask = (mask
                & ~np.diagflat(np.ones(ntracks, dtype=bool)).reshape(1, ntracks, ntracks)
                & np.expand_dims(~tracks['id'].mask, axis=1)
                & np.expand_dims(~tracks['id'].mask, axis=2))

        start1 = np.expand_dims(tracks['start'], axis=1)
        start2 = np.expand_dims(tracks['start'], axis=2)
        end1 = np.expand_dims(tracks['end'], axis=1)
        end2 = np.expand_dims(tracks['end'], axis=2)

        endpoint_distance = ma.concatenate((
            ma.sum((start1 - end2)**2, axis=-1, keepdims=True),
            ma.sum((end1 - end2)**2, axis=-1, keepdims=True),
            ma.sum((start1 - start2)**2, axis=-1, keepdims=True),
            ma.sum((end1 - start2)**2, axis=-1, keepdims=True),
        ), axis=-1)
        endpoint_distance = ma.sqrt(endpoint_distance)
        endpoint_distance = ma.array(endpoint_distance.min(axis=-1), mask=~mask, shrink=False)

        neighbor = ma.argsort(endpoint_distance, axis=-1)[..., k - 1].reshape(tracks.shape)
        neighbor = ma.array(neighbor, mask=tracks['id'].mask | np.all(~mask, axis=-1), shrink=False)
        neighbor.fill_value = -1
        neighbor = ma.array(neighbor.filled(), mask=neighbor.mask, shrink=False)
        neighbor.fill_value = -1
        return dict(neighbor=neighbor)

    @staticmethod
    def poca(start_xyz0, end_xyz0, start_xyz1, end_xyz1):
        '''
            Finds the scale factor to point of closest approach of two lines
            each defined by 2 3D points. The scale factor is a number between 0
            and 1 representing the position along the line. To extract the
            3D point of closest approach on each line::

                s0, s1 = poca(start0, end0, start1, end1) # shape: (N, 1)
                poca0 = (1 - s0) * start0 + s0 * end0 # shape: (N, 3)
                poca1 = (1 - s1) * start1 + s1 * end1

            :param {start, end}_xyz(i): start/end point of line i, ``shape: (..., N, 3)``

            :returns: ``tuple`` of line segment 0 and 1, ``shape: (..., N, 1)``
        '''
        orig_mask0 = start_xyz0.mask | end_xyz0.mask
        orig_mask1 = start_xyz1.mask | end_xyz1.mask
        orig_mask0, orig_mask1 = np.broadcast_arrays(orig_mask0, orig_mask1)
        start_xyz0, end_xyz0, start_xyz1, end_xyz1 = np.broadcast_arrays(
            start_xyz0, end_xyz0, start_xyz1, end_xyz1)

        d = start_xyz0 - start_xyz1
        v0, v1 = (end_xyz0 - start_xyz0, end_xyz1 - start_xyz1)
        l0, l1 = (np.linalg.norm(v0, axis=-1, keepdims=True),
                  np.linalg.norm(v1, axis=-1, keepdims=True))
        with np.errstate(divide='ignore', invalid='ignore'):
            v0 /= l0
            v1 /= l1
        v0[(l0 == 0)[..., 0]] = 0
        v1[(l1 == 0)[..., 0]] = 0
        v_dp = np.sum(v0 * v1, axis=-1, keepdims=True)

        with np.errstate(divide='ignore', invalid='ignore'):
            s0 = (-np.sum(d * v0, axis=-1, keepdims=True)
                  + np.sum(d * v1, axis=-1, keepdims=True) * v_dp) / (1 - v_dp**2)
            s1 = (np.sum(d * v1, axis=-1, keepdims=True)
                  - np.sum(d * v0, axis=-1, keepdims=True) * v_dp) / (1 - v_dp**2)

            s0 /= l0
            s1 /= l1

            # handle 0 length line segment
            s0[l0 == 0] = 0.5
            s1[l1 == 0] = 0.5

            # handle parallel segments
            parallel_mask = (1 - v_dp**2 == 0)[..., 0]
            if np.any(parallel_mask):
                # grab mean position
                p = (start_xyz0 + end_xyz0 + start_xyz1 + end_xyz1) / 4
                # calculate perpendicular points on other segments
                d0 = (start_xyz0 - p) - v0 * np.sum((start_xyz0 - p) * v0,
                                                    axis=-1, keepdims=True)
                s0[parallel_mask] = np.sum((p + d0) * v0 / l0, axis=-1,
                                           keepdims=True)[parallel_mask]
                d1 = (start_xyz1 - p) - v1 * np.sum((start_xyz1 - p) * v1,
                                                    axis=-1, keepdims=True)
                s1[parallel_mask] = np.sum((p + d1) * v1 / l1, axis=-1,
                                           keepdims=True)[parallel_mask]

        mask0 = np.any(orig_mask0, axis=-1, keepdims=True)
        mask1 = np.any(orig_mask1, axis=-1, keepdims=True)
        s0 = ma.array(s0, mask=np.broadcast_to(mask0, s0.shape), shrink=False)
        s1 = ma.array(s1, mask=np.broadcast_to(mask1, s1.shape), shrink=False)
        return s0, s1

    @staticmethod
    def closest_trajectories(tracks0, tracks1):
        '''
            :param tracks0: track dtype of shape: ``(..., M,)``

            :param tracks1: track dtype of shape: ``(..., M,)``

            :returns: start and end points of closest trajectory segments and points of closest approach, shape: ``(..., M, 3)``

        '''
        start0 = tracks0['trajectory'][..., :-1, :]  # (N, M, n0-1, 3)
        end0 = tracks0['trajectory'][..., 1:, :]  # (N, M, n0-1, 3)
        start1 = tracks1['trajectory'][..., :-1, :]  # (N, M, n1-1, 3)
        end1 = tracks1['trajectory'][..., 1:, :]  # (N, M, n1-1, 3)

        # reshape -> (N, M, n0-1, 1, 3) and (N, M, 1, n1-1, 3)
        start0 = np.expand_dims(start0, -2)
        end0 = np.expand_dims(end0, -2)
        start1 = np.expand_dims(start1, -3)
        end1 = np.expand_dims(end1, -3)

        # find point of closest approach
        s0, s1 = TrackletMerger.poca(start0, end0, start1, end1)
        s0 = ma.clip(s0, 0, 1)
        s1 = ma.clip(s1, 0, 1)

        poca0 = (1 - s0) * start0 + s0 * end0
        poca1 = (1 - s1) * start1 + s1 * end1
        poca_d = np.linalg.norm(poca0 - poca1, axis=-1)
        poca_d = ma.array(poca_d, mask=(s0.mask | s1.mask), shrink=False)

        # remove segments with 0 length
        mask = ((np.linalg.norm(end0 - start0, axis=-1) == 0)
                | (np.linalg.norm(end1 - start1, axis=-1) == 0))
        poca_d[mask] = poca_d.max()

        # minimize point of closest approach
        min_poca_d0 = np.expand_dims(ma.argmin(poca_d, axis=-1), -1)  # (n, M, n0-1, 1)
        poca0 = np.take_along_axis(poca0, np.expand_dims(min_poca_d0, -1), -2)  # (n, M, n0-1, 1, 3)
        poca1 = np.take_along_axis(poca1, np.expand_dims(min_poca_d0, -1), -2)  # (n, M, n0-1, 1, 3)
        poca_d = np.take_along_axis(poca_d, min_poca_d0, -1)  # (n, M, n0-1, 1)

        min_poca_d1 = np.expand_dims(ma.argmin(poca_d, axis=-2), -2)  # (n, M, 1, 1)
        poca0 = np.take_along_axis(poca0, np.expand_dims(min_poca_d1, -1), -3)  # (n, M, 1, 1, 3)
        poca1 = np.take_along_axis(poca1, np.expand_dims(min_poca_d1, -1), -3)  # (n, M, 1, 1, 3)
        poca_d = np.take_along_axis(poca_d, min_poca_d1, -2)  # (n, M, 1, 1)
        min_poca_d0 = np.take_along_axis(min_poca_d0, min_poca_d1, -2)  # (n, M, 1, 1)

        start0 = np.take_along_axis(start0, np.expand_dims(min_poca_d1, -1), -3)  # (n, M, 1, 1, 3)
        end0 = np.take_along_axis(end0, np.expand_dims(min_poca_d1, -1), -3)  # (n, M, 1, 1, 3)
        start1 = np.take_along_axis(start1, np.expand_dims(min_poca_d0, -1), -2)  # (n, M, 1, 1, 3)
        end1 = np.take_along_axis(end1, np.expand_dims(min_poca_d0, -1), -2)  # (n, M, 1, 1, 3)

        start0 = start0.reshape(tracks0.shape + (3,))
        end0 = end0.reshape(tracks0.shape + (3,))
        start1 = start1.reshape(tracks1.shape + (3,))
        end1 = end1.reshape(tracks1.shape + (3,))
        poca0 = poca0.reshape(tracks0.shape + (3,))
        poca1 = poca1.reshape(tracks1.shape + (3,))

        mask = start0.mask | end0.mask | start1.mask | end1.mask | poca0.mask | poca1.mask
        start0.mask[mask] = True
        end0.mask[mask] = True
        start1.mask[mask] = True
        end1.mask[mask] = True
        poca0.mask[mask] = True
        poca1.mask[mask] = True

        return (start0, end0, start1, end1, poca0, poca1)

    @staticmethod
    def calc_2track_deflection_angle(tracks, neighbor):
        ntracks = tracks.shape[1]
        neighbor_tracks = np.take_along_axis(tracks, neighbor, axis=1)

        start, end, neighbor_start, neighbor_end, poca, neighbor_poca = (
            TrackletMerger.closest_trajectories(tracks, neighbor_tracks))

        orig_mask = poca.mask.copy() | neighbor_poca.mask.copy()
        poca = (poca + neighbor_poca) / 2

        # calculate deflection angle to farthest point on neighboring segment
        neighbor_far = np.where(
            np.linalg.norm(poca - neighbor_start, axis=-1, keepdims=True)
            > np.linalg.norm(poca - neighbor_end, axis=-1, keepdims=True),
            neighbor_start, neighbor_end)
        ang1 = np.sum((neighbor_far - poca) * (poca - start), axis=-1)
        ang1 /= np.linalg.norm((neighbor_far - poca), axis=-1) + 1e-15
        ang1 /= np.linalg.norm((poca - start), axis=-1) + 1e-15
        ang1 = np.arccos(np.clip(ang1, -1, 1))

        mask = (tracks['id'].mask | neighbor.mask.reshape(ang1.shape)
                | (neighbor == -1).reshape(ang1.shape))
        return ma.array(ang1 / np.pi, mask=mask, shrink=False)

    @staticmethod
    def calc_2track_transverse_sin2theta(tracks, neighbor):
        ntracks = tracks.shape[1]
        neighbor_tracks = np.take_along_axis(tracks, neighbor, axis=-1)

        start1, end1, start2, end2, _, _ = TrackletMerger.closest_trajectories(
            tracks, neighbor_tracks)

        d = ma.concatenate((
            np.expand_dims(start1 - end2, axis=-1),
            np.expand_dims(end1 - end2, axis=-1),
            np.expand_dims(start1 - start2, axis=-1),
            np.expand_dims(end1 - start2, axis=-1)
        ), axis=-1)
        i_max = np.expand_dims(ma.argmax(np.sqrt(ma.sum(d * d, axis=-2, keepdims=True)), axis=-1), axis=-1)
        d = np.take_along_axis(d, i_max, axis=-1)[..., 0]
        d_norm = ma.sqrt(ma.sum(d**2, axis=-1, keepdims=True))
        d_norm[d_norm == 0] = 1
        d /= d_norm

        # transverse d
        track_d = end1 - start1
        track_d_mask = np.all(track_d == 0, axis=-1)
        track_d[track_d_mask] = (tracks['end'] - tracks['start'])[track_d_mask]
        track_d /= ma.sqrt(ma.sum(track_d**2, axis=-1, keepdims=True))
        l_d = np.abs(ma.sum(d * track_d, axis=-1))
        l = np.sqrt(ma.sum(d * d, axis=-1))
        t_d = np.clip(l**2 - l_d**2, 0, 1)

        mask = (tracks['id'].mask |
                neighbor.mask.reshape(t_d.shape)
                | (neighbor == -1).reshape(t_d.shape))
        return ma.array(t_d, mask=mask, shrink=False)

    @staticmethod
    def make_missing_segment(start1, end1, start2, end2):
        track_d = np.concatenate((
            np.sum((start1 - end2)**2, axis=-1, keepdims=True),
            np.sum((end1 - end2)**2, axis=-1, keepdims=True),
            np.sum((start1 - start2)**2, axis=-1, keepdims=True),
            np.sum((end1 - start2)**2, axis=-1, keepdims=True),
        ), axis=-1)
        i_min = np.expand_dims(np.argmin(track_d, axis=-1), axis=-1)
        missing_track_start = np.select(
            (i_min == 0,
             i_min == 1,
             i_min == 2,
             i_min == 3),
            (start1, end1, start1, end1))
        missing_track_end = np.select(
            (i_min == 0,
             i_min == 1,
             i_min == 2,
             i_min == 3),
            (end2, end2, start2, start2))
        return missing_track_start, missing_track_end

    @staticmethod
    def calc_2track_missing_length(tracks, neighbor, missing_track_segments,
                                   pixel_x, pixel_y, disabled_channel_lut,
                                   cathode_region, pixel_pitch=None):
        # create missing track segment
        _n_steps = missing_track_segments
        neighbor_tracks = np.take_along_axis(tracks, neighbor, axis=-1)
        start1, end1, start2, end2, poca1, poca2 = TrackletMerger.closest_trajectories(
            tracks, neighbor_tracks)

        # _missing_start, _missing_end = TrackletMerger.make_missing_segment(
        #     start1, end1, start2, end2)
        _missing_start, _missing_end = poca1, poca2

        # interpolate
        _missing_x, _dx = np.linspace(_missing_start[..., 0], _missing_end[..., 0],
                                      _n_steps, axis=-1, retstep=True)
        _missing_y, _dy = np.linspace(_missing_start[..., 1], _missing_end[..., 1],
                                      _n_steps, axis=-1, retstep=True)
        _missing_z, _dz = np.linspace(_missing_start[..., 2], _missing_end[..., 2],
                                      _n_steps, axis=-1, retstep=True)
        _ds = np.sqrt(_dx**2 + _dy**2 + _dz**2)
        _missing_length = _ds * _n_steps

        pixel_pitch = pixel_pitch if pixel_pitch is not None else resources['Geometry'].pixel_pitch
        _ix = np.clip(np.digitize(_missing_x, pixel_x + pixel_pitch / 2) - 1,
                      0, len(pixel_x) - 1)
        _iy = np.clip(np.digitize(_missing_y, pixel_y + pixel_pitch / 2) - 1,
                      0, len(pixel_x) - 1)

        _missing_pixel_x = pixel_x[_ix]
        _missing_pixel_y = pixel_y[_iy]
        _missing_iogroup = (np.sign(_missing_z) / 2 + 1.5).astype(int)

        _hidden_length = _ds * (
            (disabled_channel_lut[_missing_iogroup,
                                  _missing_pixel_x.astype(int),
                                  _missing_pixel_y.astype(int)].reshape(_missing_iogroup.shape)
             | (np.abs(_missing_z) < cathode_region)).sum(axis=-1))
        missing_length = _missing_length - _hidden_length

        mask = (tracks['id'].mask
                | neighbor.mask.reshape(missing_length.shape)
                | (neighbor == -1).reshape(missing_length.shape))
        return ma.array(missing_length, mask=mask, shrink=False)

    @staticmethod
    def calc_2track_overlap(tracks, neighbor):
        _ntracks = tracks.shape[1]
        neighbor = neighbor.reshape(tracks.shape + (1,))
        _track1_min = np.minimum(tracks['start'], tracks['end'])
        _track1_max = np.maximum(tracks['start'], tracks['end'])
        _track2_min = np.take_along_axis(np.minimum(tracks['start'], tracks['end']),
                                         neighbor, axis=1)
        _track2_max = np.take_along_axis(np.maximum(tracks['start'], tracks['end']),
                                         neighbor, axis=1)

        overlap = (np.minimum(_track2_max, _track1_max)
                   - np.maximum(_track2_min, _track1_min))
        overlap = np.clip(overlap, 0, None)
        overlap = np.sqrt(np.sum(overlap**2, axis=-1))
        mask = (tracks['id'].mask
                | neighbor.mask.reshape(overlap.shape)
                | (neighbor == -1).reshape(overlap.shape))
        return ma.array(overlap, mask=mask, shrink=False)

    @staticmethod
    def calc_2track_sin2theta(tracks, neighbor):
        ntracks = tracks.shape[1]
        neighbor_tracks = np.take_along_axis(tracks, neighbor, axis=-1)
        start1, end1, start2, end2, _, _ = TrackletMerger.closest_trajectories(
            tracks, neighbor_tracks)
        dxyz = end1 - start1
        mask = np.all(dxyz == 0, axis=-1)
        dxyz[mask] = (tracks['end'] - tracks['start'])[mask]
        dxyz /= np.sqrt(np.sum((dxyz)**2, axis=-1, keepdims=True))

        dxyz_neighbor = end2 - start2
        mask = np.all(dxyz_neighbor == 0, axis=-1)
        dxyz_neighbor[mask] = (neighbor_tracks['end'] - neighbor_tracks['start'])[mask]
        dxyz_neighbor /= np.sqrt(np.sum((dxyz_neighbor)**2, axis=-1, keepdims=True))
        sin2theta = 1 - np.sum(dxyz * dxyz_neighbor, axis=-1)**2
        mask = (tracks['id'].mask | neighbor.mask.reshape(sin2theta.shape)
                | (neighbor == -1).reshape(sin2theta.shape))
        return ma.array(sin2theta, mask=mask, shrink=False)

    @staticmethod
    def load_r_values(filename, sig_key, bkg_key):
        '''
            Load the N-D pdf histogram from an .npz file. Loads and normalizes
            the histograms stored under ``{sig_key}`` and ``{bkg_key}`` with
            bins stored under ``{key}_bins`` to create a PDF. The likelihood
            ratio (``R``) is then calculated and converted to a normalized
            value between 0-1 (``r``) with the following transformation::

                r = 1 - e^(-R)

            Bins with 0 entries are assigned an ``R``-value of 0.

            :param filename: path to .npz file with arrays

            :param sig_key: name of "signal" histogram in .npz file

            :param bkg_key: name of "background" histogram in .npz file

            :returns: ``tuple`` of r histogram (``shape: (N0, N1, ...)``), r bins in each dimension (``shape: (D, Ni)``), an array possible r values (``shape: (1001,)``, and corresponding p-values (``shape: (1001,)``)

        '''
        pdf = dict(np.load(filename, allow_pickle=True))

        ndimage.gaussian_filter(pdf[sig_key], 1.5, output=pdf[sig_key], mode='nearest')
        ndimage.gaussian_filter(pdf[bkg_key], 1.5, output=pdf[bkg_key], mode='nearest')

        sig_norm = np.sum(pdf[sig_key])
        bkg_norm = np.sum(pdf[bkg_key])
        with np.errstate(divide='ignore', invalid='ignore'):
            r = 1 - np.exp(-(pdf[sig_key] / sig_norm) / (pdf[bkg_key] / bkg_norm))
        r_inf_mask = (pdf[bkg_key] == 0) & (pdf[sig_key] > 0)
        r[r_inf_mask] = 1
        r_zero_mask = (pdf[sig_key] == 0) & (pdf[bkg_key] > 0)
        r[r_zero_mask] = 0
        r_undef_mask = (pdf[sig_key] == 0) & (pdf[bkg_key] == 0)
        r[r_undef_mask] = 0.5
        r_bins = pdf[sig_key + '_bins']

        idx = np.where(pdf[sig_key])
        weights = pdf[sig_key][idx].flatten()

        statistic_bins = np.r_[0, np.geomspace(np.min(r[r > 0]), 1, 1000)]
        statistic, statistic_bins = np.histogram(r[idx].flatten(),
                                                 bins=statistic_bins, weights=weights)
        p_bins = 1 - np.cumsum(statistic[::-1])[::-1] / np.sum(statistic)

        return r, r_bins, statistic_bins, p_bins

    @staticmethod
    def score_neighbor(r, r_bins, statistic_bins, p_bins, *params):
        '''
            Calculates a p-value based on a binned, multi-dimensional PDF

            :param r: likelihood ratio, ``shape: (N,)*D``

            :param r_bins: bin edge for each parameter, ``shape: (D, N+1)``

            :param statistic_bins: bins for statistic, range 0-1, ``shape: (n,)``

            :param p_bins: bins for p value range 0-1, ``shape: (n,)``

            :param *params: array of parameters to use to calculate p-value, requires ``D`` parameters in the same sequence as listed in the bins, each with the same shape

            :returns: array of same shape as the ``params`` arrays with a p-value between 0-1

        '''
        i_bin = [np.clip(np.digitize(np.clip(p, b[0], b[-1]), b) - 1,
                         0, len(b) - 2) for b, p in zip(r_bins, params)]
        statistic = r[tuple(i_bin)]
        pvalue = p_bins[np.clip(np.digitize(statistic, statistic_bins), 0, len(statistic_bins) - 2)]
        return pvalue
