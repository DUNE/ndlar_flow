import numpy as np
import numpy.ma as ma
import logging

from h5flow.core import H5FlowStage, resources

from module0_flow.combined.tracklet_reco import TrackletReconstruction


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
         - ``t0_dset_name``: ``str``, path to input t0 dataset
         - ``hits_dset_name``: ``str``, path to input charge hits dataset
         - ``track_hits_dset_name``: ``str``, path to input track-referred charge hits dataset
         - ``tracks_dset_name``: ``str``, path to input track dataset
         - ``merged_dset_name``: ``str``, path to output track dataset

        All of ``hits_dset_name``, ``t0_dset_name``, ``track_hits_dset_name``,
        and ``tracks_dset_name`` are required in the cache.

        Requires both Geometry and DisabledChannels resources in workflow.

        ``merged`` datatype is the same as the
        ``TrackletReconstruction.tracklet_dtype``.

        Example config::

            track_merge:
                classname: TrackletMerger
                requires:
                 - 'combined/tracklets'
                 - 'charge/hits'
                 - 'combined/t0'
                params:
                    merged_dset_name: 'combined/tracklets/merged'
                    t0_dset_name: 'combined/t0'
                    hits_dset_name: 'charge/hits'
                    tracks_dset_name: 'combined/tracklets'
                    pdf_filename: 'joint_pdf.npz'
                    pvalue_cut: 0.10
                    max_neighbors: 5

    '''
    class_version = '0.0.0'

    default_pdf_filename = 'joint_pdf.npz'
    default_pdf_sig_name = 'rereco'
    default_pdf_bkg_name = 'origin'
    default_pvalue_cut = 0.10
    default_max_neighbors = 5

    default_t0_dset_name = 'combined/t0'
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

        self.t0_dset_name = params.get('t0_dset_name', self.default_t0_dset_name)
        self.hits_dset_name = params.get('hits_dset_name', self.default_hits_dset_name)
        self.track_hits_dset_name = params.get('track_hits_dset_name', self.default_track_hits_dset_name)
        self.tracks_dset_name = params.get('tracks_dset_name', self.default_tracks_dset_name)
        self.merged_dset_name = params.get('merged_dset_name', self.default_merged_dset_name)

    def init(self, source_name):
        self.r, self.r_bins, self.statistic_bins, self.p_bins = (
            self.load_r_values(self.pdf_filename, self.pdf_sig_name,
                               self.pdf_bkg_name))

        self.data_manager.set_attrs(self.merged_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    hits_dset=self.hits_dset_name,
                                    t0_dset=self.t0_dset_name,
                                    tracks_dset=self.tracks_dset_name,
                                    max_neighbors=self.max_neighbors,
                                    pvalue_cut=self.pvalue_cut,
                                    pdf_filename=self.pdf_filename,
                                    pdf_sig_name=self.pdf_sig_name,
                                    pdf_bkg_name=self.pdf_bkg_name
                                    )

        self.data_manager.create_dset(self.merged_dset_name, self.merged_dtype)
        self.data_manager.create_ref(self.merged_dset_name, self.hits_dset_name)
        self.data_manager.create_ref(self.merged_dset_name, self.tracks_dset_name)
        self.data_manager.create_ref(source_name, self.merged_dset_name)

        self.pixel_x = np.unique(resources['Geometry'].pixel_xy.compress((0,)))
        self.pixel_y = np.unique(resources['Geometry'].pixel_xy.compress((1,)))

    def run(self, source_name, source_slice, cache):
        t0 = cache[self.t0_dset_name]
        track_hits = cache[self.track_hits_dset_name]
        tracks = cache[self.tracks_dset_name]

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
                    self.calc_2track_sin2theta(tracks, neighbor),
                    self.calc_2track_transverse_endpoint_d(tracks, neighbor),
                    self.calc_2track_missing_length(tracks, neighbor,
                                                    self.missing_track_segments,
                                                    self.pixel_x, self.pixel_y,
                                                    resources['DisabledChannels'].disabled_channel_lut,
                                                    self.cathode_region),
                    self.calc_2track_overlap(tracks, neighbor),
                    self.calc_2track_ddqdx(tracks, neighbor)
                ]
                pvalue = np.expand_dims(self.score_neighbor(self.r, self.r_bins, self.statistic_bins, self.p_bins, *params), -1)
                neighbor = np.expand_dims(neighbor, -1)

                # merge tracks that have large p-values
                should_merge = (((pvalue >= self.pvalue_cut)
                                 | np.take_along_axis(track_merged, neighbor, -1))
                                & ~neighbor.mask)
                np.put_along_axis(track_merged, neighbor, should_merge, axis=-1)
                np.put_along_axis(track_checked, neighbor, True, axis=-1)

                if np.all(track_checked):
                    break

            # collect valid associations into track groups
            track_merged = self.create_groups(track_merged)

            # now, collect the hits from the original tracks into the track groups
            # get unique track groups, shape: (n_ev, n_grp, n_track)
            track_merged = np.unique(track_merged, axis=1)
            track_grp = ma.array(track_merged, mask=(np.diff(track_merged.astype(int), axis=1, prepend=False) == 0) | ~track_merged)
            # index by track groups, shape: (n_ev, n_grp, n_track)
            track_grp_id = ma.array(np.indices(track_grp.shape)[1], mask=track_grp.mask)
            track_grp_id = np.expand_dims(track_grp_id, axis=-1)
            # cast track hits into a broadcastable shape: (n_ev, 1, n_track, n_hit)
            track_grp_hits = np.expand_dims(track_hits, axis=1)

            # broadcast into same shape: (n_ev, n_grp, n_track, n_hit)
            hit_shape = np.maximum(track_grp_id.shape, track_grp_hits.shape)
            track_grp_hits_mask = np.broadcast_to(track_grp_hits['id'].mask, hit_shape) | track_grp_id.mask
            track_grp_id = np.broadcast_to(track_grp_id, hit_shape)
            track_grp_hits = np.broadcast_to(track_grp_hits, hit_shape)
            # mask and condense: (n_ev, n_hit') [used by track reco calculation]
            new_shape = track_grp.shape[0:1] + (-1,)
            track_grp_hits = self.condense_array(track_grp_hits.reshape(new_shape),
                                                 track_grp_hits_mask.reshape(new_shape))
            track_grp_id = self.condense_array(track_grp_id.reshape(new_shape),
                                               track_grp_hits_mask.reshape(new_shape))
            logging.warning(f'track_hits: {track_grp_hits.shape} ({track_grp_hits.nbytes/1024/1024:0.02f}MB)')

            # recalculate track parameters
            calc_shape = (track_grp_id.shape[0], -1)
            merged_tracks = TrackletReconstruction.calc_tracks(track_grp_hits.reshape(calc_shape), t0, track_grp_id.reshape(calc_shape))
        else:
            merged_tracks = ma.masked_all((0, 1), dtype=TrackletReconstruction.tracklet_dtype)
            track_grp = ma.masked_all((0, 1, 1), dtype=bool)
            track_grp_id = ma.masked_all((0, 1), dtype=int)
            track_grp_hits = ma.masked_all((0, 1), dtype=track_hits.dtype)

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
    def condense_array(arr, mask):
        '''
            Densify a masked array on last axis, throwing out invalid values
            (up to the size needed to keep the array regular). E.g.::

                mask = [[False, True, True],
                        [False, False, True],
                        [True, False, True]]

            will condense a 3x3 array to shape: ``(3, 2)`` and produce a final
            mask of::

                new_mask = [[False, True],
                            [False, False],
                            [False, True]]

        '''
        axis = -1
        n_valid = np.expand_dims(np.count_nonzero(~mask, axis=axis), axis=axis)

        new_shape = list(arr.shape)
        new_shape[axis] = n_valid.max()
        condensed = np.empty(new_shape, dtype=arr.dtype)
        idx = np.indices(condensed.shape)[-1]
        np.place(condensed, idx < n_valid, arr[~mask])

        return ma.array(condensed, mask=idx >= n_valid)

    @staticmethod
    def create_groups(mask):
        '''
            Combine masks of ``n x n`` ajacency matrix such that the mask of
            row i is equal to the ``OR`` of the rows that can be reached from
            ``i``. E.g.::

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
                new_arr # [[0,0,1],
                           [0,1,0],
                           [0,1,1]]

            :param mask: ajacency matrix (``shape: (..., n, n)``)

            :returns: updated ajacency matrix (``shape: (..., n, n)``)
        '''

        # get index of masks (starting with True values)
        i_mask = np.argsort(mask, axis=-1)[..., ::-1]
        step = 0
        while (step < i_mask.shape[-1]):
            # step through indices
            # get other index (shape: (..., n, 1))
            j_mask = np.expand_dims(i_mask[..., step], axis=-1)
            # get other mask (shape: (..., n, n))
            other_mask = np.take_along_axis(mask, j_mask, -2)
            # get other matched to current (shape: (..., n, 1))
            other_matched = np.take_along_axis(mask, j_mask, -1)

            if not np.any(other_matched):
                break

            # combine with current track
            mask = (mask | (other_mask & other_matched))
            step += 1

        return mask

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
        endpoint_distance = ma.array(endpoint_distance.min(axis=-1), mask=~mask)

        neighbor = ma.argsort(endpoint_distance, axis=-1)[..., k - 1].reshape(tracks.shape)
        neighbor = ma.array(neighbor, mask=tracks['id'].mask | np.all(~mask, axis=-1))
        neighbor.fill_value = -1
        neighbor = ma.array(neighbor.filled(), mask=neighbor.mask)
        neighbor.fill_value = -1
        return dict(neighbor=neighbor)

    @staticmethod
    def calc_2track_sin2theta(tracks, neighbor):
        ntracks = tracks.shape[1]
        neighbor = neighbor.reshape(tracks.shape + (1,))
        dxyz = tracks['start'] - tracks['end']
        dxyz /= np.sqrt(np.sum((dxyz)**2, axis=-1, keepdims=True))
        dxyz_neighbor = np.take_along_axis(dxyz, neighbor, axis=1)
        sin2theta = 1 - np.sum(dxyz * dxyz_neighbor, axis=-1)**2
        mask = (tracks['id'].mask | neighbor.mask.reshape(sin2theta.shape)
                | (neighbor == -1).reshape(sin2theta.shape))
        return ma.array(sin2theta, mask=mask)

    @staticmethod
    def calc_2track_transverse_endpoint_d(tracks, neighbor):
        ntracks = tracks.shape[1]
        neighbor = neighbor.reshape(tracks.shape + (1,))
        start1 = tracks['start']
        start2 = np.take_along_axis(tracks['start'], neighbor, axis=1)
        end1 = tracks['end']
        end2 = np.take_along_axis(tracks['end'], neighbor, axis=1)
        d = ma.concatenate((
            np.expand_dims(start1 - end2, axis=-1),
            np.expand_dims(end1 - end2, axis=-1),
            np.expand_dims(start1 - start2, axis=-1),
            np.expand_dims(end1 - start2, axis=-1)
        ), axis=-1)
        i_max = np.expand_dims(ma.argmin(np.sqrt(ma.sum(d * d, axis=-2, keepdims=True)), axis=-1), axis=-1)
        d = np.squeeze(np.take_along_axis(d, i_max, axis=-1))

        # transverse d
        track_d = tracks['end'] - tracks['start']
        track_d /= ma.sqrt(ma.sum(track_d**2, axis=-1, keepdims=True))
        l_d = np.abs(ma.sum(d * track_d, axis=-1))
        l = np.sqrt(ma.sum(d * d, axis=-1))
        t_d = np.sqrt(l**2 - l_d**2)

        mask = (tracks['id'].mask |
                neighbor.mask.reshape(t_d.shape)
                | (neighbor == -1).reshape(t_d.shape))
        return ma.array(t_d, mask=mask)

    @staticmethod
    def make_missing_segment(start, end, neighbor):
        start1 = start
        start2 = np.take_along_axis(start, neighbor, axis=1)
        end1 = end
        end2 = np.take_along_axis(end, neighbor, axis=1)

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
        neighbor = neighbor.reshape(tracks.shape + (1,))
        _missing_start, _missing_end = TrackletMerger.make_missing_segment(tracks['start'],
                                                                           tracks['end'], neighbor)
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

        _hidden_length = (
            (disabled_channel_lut[_missing_iogroup,
                                  _missing_pixel_x.astype(int),
                                  _missing_pixel_y.astype(int)].reshape(_missing_iogroup.shape)
             | (np.abs(_missing_z) < cathode_region)).sum(axis=-1))
        missing_length = _missing_length - _hidden_length
        missing_length = np.clip(missing_length, 0, None)

        mask = (tracks['id'].mask
                | neighbor.mask.reshape(missing_length.shape)
                | (neighbor == -1).reshape(missing_length.shape))
        return ma.array(missing_length, mask=mask)

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
        return ma.array(overlap, mask=mask)

    @staticmethod
    def calc_2track_ddqdx(tracks, neighbor):
        _ntracks = tracks.shape[1]
        neighbor = neighbor.reshape(tracks.shape)
        _track1_dqdx = tracks['q'] / tracks['length']
        _track2_dqdx = np.take_along_axis(tracks['q'] / tracks['length'], neighbor, axis=1)
        ddqdx = np.abs(_track1_dqdx - _track2_dqdx)
        mask = (tracks['id'].mask
                | neighbor.mask.reshape(ddqdx.shape)
                | (neighbor == -1).reshape(ddqdx.shape))
        return ma.array(ddqdx, mask=mask)

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

        norm = np.sum(pdf[sig_key])
        r = 1 - np.exp(-(pdf[sig_key] / norm) / (pdf[bkg_key] / np.sum(pdf[bkg_key]) + 1e-15))
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
            Calculates a p-value based on a binned, multi-dimensional CDF

            :param r: normalized CDF, ``shape: (N,)*D``

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
