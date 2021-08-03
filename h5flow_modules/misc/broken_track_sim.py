import numpy as np
import numpy.ma as ma
from collections import defaultdict
import logging

import module0_flow.combined.tracklet_reco as tracklet_reco
from module0_flow.resources.geometry import LUT

from h5flow.core import H5FlowStage, resources


class JointPDF(object):
    def __init__(self, *bins):
        self.hist, self.bins = np.histogramdd(np.empty((0, len(bins))), bins=bins)
        self.n = 0

    def fill(self, *val):
        self.n += len(val[0])
        _sample = np.concatenate([np.expand_dims(v.ravel()
                                                 if not isinstance(v, ma.MaskedArray)
                                                 else v.compressed(), axis=-1)
                                  for v in val], axis=-1)
        self.hist, _ = np.histogramdd(_sample, bins=self.bins)


class BrokenTrackSim(H5FlowStage):
    '''

    Example config::

        broken_track_sim:
            classname: BrokenTrackSim
            requires:
             - 'combined/trackets'
             - 'charge/hits'
             - 'combined/t0'
            params:
                path: 'misc/broken_track_sim'
                rand_track_length_cut: 100 # mm


    '''
    offset_dtype = np.dtype([
        ('id', 'u4'),
        ('dx', 'f4'),
        ('dy', 'f4'),
        ('i_track', 'i8')
    ])

    new_track_dtype = tracklet_reco.tracklet_dtype

    new_track_label_dtype = np.dtype([
        ('id', 'u4'),
        ('match', 'u1'),
        ('broken', 'u1')
    ])

    default_pdf_bins = [
        (0, 1, 100),
        (0, 600, 100),
        (0, 600, 100),
        (0, 5, 100),
    ]
    missing_track_segments = 100

    def __init__(self, **params):
        super(BrokenTrackSim, self).__init__(**params)

        path = params.get('path', 'misc/broken_track_sim')

        self.generate_2track_joint_pdf = params.get('generate_2track_joint_pdf', True)
        self.pdf_bins = [np.linspace(*bins) for bins in params.get('pdf_bins',
                                                                   self.default_pdf_bins)]
        self.pdf = dict()
        if self.generate_2track_joint_pdf:
            self.pdf['rereco'] = JointPDF(*self.pdf_bins)
            self.pdf['origin'] = JointPDF(*self.pdf_bins)

        self.rand_track_length_cut = params.get('rand_track_length_cut', 100)
        self.endpoint_distance_cut = params.get('endpoint_distance_cut', 7.7)
        self.broken_track_distance_cut = params.get('broken_track_distance_cut', 7.7)
        self.cathode_region = params.get('cathode_region', 15)

        self.disabled_channels_list = params.get('disabled_channels_list',
                                                 'module0-run1-selftrigger-disabled-list.json')
        self.missing_asic_list = params.get('missing_asic_list',
                                            'module0-network-absent-ASICs.json')

        self.tracks_dset_name = params.get('tracks_dset_name', 'combined/tracklets')
        self.hits_dset_name = params.get('hits_dset_name', 'charge/hits')
        self.t0_dset_name = params.get('t0_dset_name', 'combined/t0')

    def init(self, source_name):
        # save lookup table for disabled channels
        self.disabled_channels_lut, self.disabled_xy = self.load_disabled_channels_lut()
        disabled_channels_lut_meta, disabled_channels_lut_arr = self.disabled_channels_lut.to_array()
        self.data_manager.create_dset(f'{self.path}/disabled_channels',
                                      dtype=self.disabled_channels_lut_arr.dtype)
        self.data_manager.reserve_data(f'{self.path}/disabled_channels',
                                       slice(0, len(disabled_channels_lut_arr)))
        self.data_manager.write_data(f'{self.path}/disabled_channels',
                                     slice(0, len(disabled_channels_lut_arr)), disabled_channels_lut_arr)
        self.data_manager.set_attrs(f'{self.path}/disabled_channels',
                                    meta=disabled_channels_lut_meta)

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
        self.pixel_y = np.unique(resources['Geometry'].pixel_xy.compress((0,)))

    def finish(self, source_name):
        # gather from all processes
        if self.generate_2track_joint_pdf:
            pdf = self.comm.gather(self.pdf, root=0) if H5FLOW_MPI else [self.pdf]

            if self.rank == 0:
                # merge
                d = dict()

                for key in self.pdf:
                    d[key] = np.sum([p[key].hist for p in pdf], axis=0)
                    d[key + '_bins'] = self.pdf[key].bins
                    d[key + '_n'] = np.sum([p[key].n for p in pdf], axis=0)

                # save to file
                np.savez_compressed('joint_pdf.npz', **d)

    def run(self, source_name, source_slice, cache):
        tracks = cache[self.tracks_dset_name]
        hits = cache[self.hits_dset_name]
        hits_track_idx = cache[f'{self.hits_dset_name}_track_idx']
        t0 = cache[self.t0_dset_name]
        events = np.expand_dims(cache[source_name], axis=-1)

        d = self.select_random_track(tracks)
        rand_tracks = d['rand_tracks']
        i_track = d['i_track']

        d = self.generate_random_translation(rand_tracks)
        rand_x = d['rand_x']
        rand_y = d['rand_y']

        d = self.apply_translation(hits, rand_x, rand_y)
        trans_hits = d['trans_hits']

        track_ids = self.reco.find_tracks(trans_hits, t0)
        new_tracks = self.reco.calc_tracks(trans_hits, t0, track_ids)

        d = self.find_matching_tracks(new_tracks, rand_tracks, rand_x, rand_y,
                                      track_ids, hits_track_idx)
        trans_rand_tracks = d['trans_rand_tracks']
        match = d['match']
        broken = d['broken']
        endpoint_distance_1 = d['endpoint_distance_1']
        endpoint_distance_2 = d['endpoint_distance_2']
        broken_track_mask = match & broken

        if self.generate_2track_joint_pdf:
            track2_costheta = calc_2track_costheta(new_tracks,
                                                   broken_track_mask)
            track2_costheta_orig = calc_2track_costheta(tracks)
            track2_endpoint_distance = calc_2track_endpoint_distance(new_tracks,
                                                                     broken_track_mask)
            track2_endpoint_distance_orig = calc_2track_endpoint_distance(tracks)
            track2_missing_length = calc_2track_missing_length(new_tracks,
                                                               broken_track_mask)
            track2_missing_length_orig = calc_2track_missing_length(tracks)
            track2_overlap = calc_2track_overlap(new_tracks, broken_track_mask)
            track2_overlap_orig = calc_2track_overlap(tracks)

            self.pdf['rereco'].fill(track2_costheta, track2_endpoint_distance,
                                    track2_missing_length, track2_overlap)
            self.pdf['origin'].fill(track2_costheta_orig, track2_endpoint_distance_orig,
                                    track2_missing_length_orig, track2_overlap_orig)

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

        return dict(
            rand_tracks=np.take_along_axis(tracks, i_track, axis=-1),
            i_track=i_track
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

    def apply_translation(hits, rand_x, rand_y):
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
        _disabled_mask = self.disabled_channels_lut[trans_hits['iogroup'],
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
        # # (events, hits, 1)
        # rand_track_hits = hits_track_idx == np.expand_dims(rand_tracks['id'], axis=-1)

        # # (events, new_tracks)
        # new_id = np.indices(new_tracks.shape)[-1]

        # # (events, hits, new_tracks)
        # new_track_hits = new_id == track_ids

        # # (events, new_tracks)
        # frac_new_hits = (np.sum(rand_track_hits & new_track_hits, axis=1)
        #                  / np.sum(new_track_hits, axis=1))

        # old method (using start and end points)
        _dxyz = np.concatenate((
            np.expand_dims(rand_x, axis=-1),
            np.expand_dims(rand_y, axis=-1),
            np.expand_dims(np.zeros_like(rand_y), axis=-1)
        ), axis=-1)

        trans_rand_tracks = rand_tracks.copy()
        trans_rand_tracks['start'] = trans_rand_tracks['start'] + _dxyz
        trans_rand_tracks['end'] = trans_rand_tracks['end'] + _dxyz

        endpoint_distance = ma.concatenate((
            ma.sum((new_tracks['start'] - trans_rand_tracks['end'])**2, axis=-1, keepdims=True),
            ma.sum((new_tracks['end'] - trans_rand_tracks['end'])**2, axis=-1, keepdims=True),
            ma.sum((new_tracks['start'] - trans_rand_tracks['start'])**2, axis=-1, keepdims=True),
            ma.sum((new_tracks['end'] - trans_rand_tracks['start'])**2, axis=-1, keepdims=True),
        ), axis=-1)
        endpoint_distance = ma.sqrt(endpoint_distance)
        i_endpoint_distance = ma.array(ma.argsort(endpoint_distance, axis=-1), mask=endpoint_distance.mask)

        # match if closest endpoint within some radius
        endpoint_distance_1 = np.take_along_axis(endpoint_distance, i_endpoint_distance[..., 0:1], axis=-1)
        match = (endpoint_distance_1 < ENDPOINT_DISTANCE_CUT)

        # id broken tracks
        endpoint_distance_2 = np.take_along_axis(endpoint_distance, i_endpoint_distance[..., 1:2], axis=-1)
        broken = (endpoint_distance_2 > BROKEN_TRACK_DISTANCE_CUT) & match

        return dict(
            trans_rand_tracks=trans_rand_tracks,
            match=match,
            broken=broken,
            endpoint_distance_1=endpoint_distance_1,
            endpoint_distance_2=endpoint_distance_2
        )

    def calc_2track_costheta(self, tracks, mask=None):
        if mask:
            _xyz_mask = np.broadcast_to(mask, tracks['start'].shape)
        else:
            _xyz_mask = np.ones_like(tracks['start'].shape)
        _ntracks = tracks.shape[1]
        _dxyz = ma.array(tracks['start'] - tracks['end'],
                         mask=~_xyz_mask | tracks['start'].mask)
        _dxyz /= np.sqrt(np.sum((_dxyz)**2, axis=-1, keepdims=True))
        costheta = np.abs(np.sum(np.expand_dims(_dxyz, axis=1) * np.expand_dims(_dxyz, axis=2), axis=-1))
        # remove i==j entries
        _mask = costheta.mask | np.diagflat(np.ones(_ntracks, dtype=bool)).reshape(1, _ntracks, _ntracks)
        return ma.array(costheta, mask=_mask)

    def calc_2track_endpoint_distance(self, tracks, mask):
        if mask:
            _xyz_mask = np.broadcast_to(mask, tracks['start'].shape)
        else:
            _xyz_mask = np.ones_like(tracks['start'].shape)
        _ntracks = new_tracks.shape[1]
        _start1 = ma.array(np.expand_dims(new_tracks['start'], axis=1),
                           mask=~_xyz_mask | tracks['start'].mask)
        _start2 = ma.array(np.expand_dims(new_tracks['start'], axis=2),
                           mask=~_xyz_mask | tracks['start'].mask)
        _end1 = ma.array(np.expand_dims(new_tracks['end'], axis=1),
                         mask=~_xyz_mask | tracks['start'].mask)
        _end2 = ma.array(np.expand_dims(new_tracks['end'], axis=2),
                         mask=~_xyz_mask | tracks['start'].mask)
        _d = ma.concatenate((
            ma.sum((_start1 - _end2)**2, axis=-1, keepdims=True),
            ma.sum((_end1 - _end2)**2, axis=-1, keepdims=True),
            ma.sum((_start1 - _start2)**2, axis=-1, keepdims=True),
            ma.sum((_end1 - _start2)**2, axis=-1, keepdims=True),
        ), axis=-1)
        _d = np.sqrt(_d)
        distance = ma.amin(_d, axis=-1)
        # remove i==j entries
        _mask = distance.mask | np.diagflat(np.ones(_ntracks, dtype=bool)).reshape(1, _ntracks, _ntracks)
        return ma.array(distance, mask=_mask)

    def make_missing_segment(self, starts, ends):
        start1 = np.expand_dims(starts, axis=1)
        start2 = np.expand_dims(starts, axis=2)
        end1 = np.expand_dims(ends, axis=1)
        end2 = np.expand_dims(ends, axis=2)

        track_d = np.concatenate((
            np.sum((start1 - end2)**2, axis=-1, keepdims=True),
            np.sum((end1 - end2)**2, axis=-1, keepdims=True),
            np.sum((start1 - start2)**2, axis=-1, keepdims=True),
            np.sum((end1 - start2)**2, axis=-1, keepdims=True),
        ), axis=-1)
        i_min = np.argsort(track_d, axis=-1)[..., 0:1]
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

    def calc_2track_missing_length(self, tracks, mask):
        # create missing track segment
        _n_steps = self.missing_track_segments
        if mask:
            _xyz_mask = np.broadcast_to(mask, tracks['start'].shape)
        else:
            _xyz_mask = np.ones_like(tracks['start'].shape)
        _ntracks = new_tracks.shape[1]
        _start = ma.array(tracks['start'], mask=~_xyz_mask | tracks['start'].mask)
        _end = ma.array(tracks['end'], mask=~_xyz_mask | tracks['start'].mask)
        _missing_start, _missing_end = make_missing_segment(_start, _end)
        # interpolate
        _missing_x, _dx = np.linspace(_missing_start[..., 0], _missing_end[..., 0], _n_steps, axis=-1, retstep=True)
        _missing_y, _dy = np.linspace(_missing_start[..., 1], _missing_end[..., 1], _n_steps, axis=-1, retstep=True)
        _missing_z, _dz = np.linspace(_missing_start[..., 2], _missing_end[..., 2], _n_steps, axis=-1, retstep=True)
        _ds = np.sqrt(_dx**2 + _dy**2 + _dz**2)
        _missing_length = _ds * _n_steps

        pixel_pitch = resources['Geometry'].pixel_pitch
        _ix = np.digitize(_missing_x, self.pixel_x + pixel_pitch / 2) - 1
        _iy = np.digitize(_missing_y, self.pixel_y + pixel_pitch / 2) - 1
        _ix[_ix >= len(self.pixel_x)] = 0
        _iy[_iy >= len(self.pixel_y)] = 0

        _missing_pixel_x = self.pixel_x[_ix]
        _missing_pixel_y = self.pixel_y[_iy]
        _missing_iogroup = (np.sign(_missing_z) / 2 + 1.5).astype(int)

        _hidden_length = (
            (self.disabled_channels_lut[_missing_iogroup,
                                        _missing_pixel_x.astype(int),
                                        _missing_pixel_y.astype(int)].reshape(_missing_iogroup.shape)
             | (np.abs(_missing_z) < self.cathode_region)).sum(axis=-1))
        _mask = (np.expand_dims(~mask, axis=1) | np.expand_dims(~mask, axis=2)
                 | np.diagflat(np.ones(_ntracks, dtype=bool)).reshape(1, _ntracks, _ntracks, 1))
        missing_length = ma.array(_missing_length - _hidden_length, mask=_mask)
        missing_length = np.clip(missing_length, 0, None)
        return missing_length

    def calc_2track_overlap(self, tracks, mask):
        _ntracks = new_tracks.shape[1]
        if mask:
            _xyz_mask = np.broadcast_to(mask, tracks['start'].shape)
        else:
            _xyz_mask = np.ones_like(tracks['start'].shape)
        _track1_min = np.expand_dims(ma.array(np.minimum(tracks['start'],
                                                         tracks['end']),
                                              mask=~_xyz_mask), axis=1)
        _track1_max = np.expand_dims(ma.array(np.maximum(tracks['start'],
                                                         tracks['end']),
                                              mask=~_xyz_mask), axis=1)
        _track2_min = np.expand_dims(ma.array(np.minimum(tracks['start'],
                                                         tracks['end']),
                                              mask=~_xyz_mask), axis=2)
        _track2_max = np.expand_dims(ma.array(np.maximum(tracks['start'],
                                                         tracks['end']),
                                              mask=~_xyz_mask), axis=2)
        overlap = np.minimum(_track2_max - _track1_min, _track1_max
                             - _track2_min)
        overlap = np.clip(overlap, 0, None)
        _mask = (np.broadcast_to(np.diagflat(np.ones(_ntracks, dtype=bool)).reshape(1, _ntracks, _ntracks, 1),
                                 overlap.shape) | _track1_min.mask | _track2_min.mask)
        return ma.array(np.sqrt(np.sum(overlap**2, axis=-1)),
                        mask=np.any(_mask, axis=-1))

    def calc_2track_ddqdx(tracks, mask):
        return

    def load_disabled_channels_lut(self):
        '''
        Loads a disabled channels lookup-table from the parameters::

            disabled_channels_list
            missing_asic_list

        :returns: a boolean ``module0_flow.resources.geometry.LUT`` instance, with keys of ``(io_group, pixel_x.astyp(int), pixel_y.astyp(int))`` and a list of xy coordinates for each disabled channel

        '''

        # first load disabled channels list
        with open(self.disabled_channels_list, 'r') as fi:
            data = json.load(fi)

        # get disabled channels from file
        io_group = list()
        io_channel = list()
        chip_id = list()
        channel_id = list()
        x = list()
        y = list()
        tpc = list()
        for key in data:
            if key == 'All':
                continue
            io_group_, io_channel_, chip_id_ = key.split('-')
            for ch in data[key]:
                io_group.append(int(io_group_))
                io_channel.append(int(io_channel_))
                chip_id.append(int(chip_id_))
                channel_id.append(int(ch))

        pixel_xy = resources['Geometry'].pixel_xy
        chip_key = (np.array(io_group), np.array(io_channel), np.array(chip_id), np.array(channel_id))
        xy = pixel_xy[chip_key]

        # then load missing asic pixels
        with open(self.missing_asic_list, 'r') as fi:
            data = json.load(fi)

        # add to lists
        for io_group_ in data:
            for asic in data[io_group_]:
                io_group.append(int(io_group_))
                xy = np.append(xy, np.array([asic]), axis=0)

        disable_channels_lut = LUT(bool,
                                   (min(io_group), max(io_group)),
                                   (min(xy[:, 0].astype(int)) - 1, max(xy[:, 0].astype(int)) + 1),
                                   (min(xy[:, 1].astype(int)) - 1, max(xy[:, 1].astype(int)) + 1),
                                   default=False)
        # apply a fudge factor to account for any rounding errors
        for dx in (+1, 0, -1):
            for dy in (+1, 0, -1):
                disable_channels_lut[(io_group, xy[:, 0].astype(int) + dx, xy[:, 1].astype(int) + dy)] = True

        return disable_channels_lut, xy
