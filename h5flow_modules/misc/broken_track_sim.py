import numpy as np
import numpy.ma as ma
from collections import defaultdict
import logging
import json

import module0_flow.combined.tracklet_reco as tracklet_reco
from module0_flow.resources.geometry import LUT

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
                   for i,v in enumerate(val)]
        _sample = np.concatenate(_sample, axis=-1)
        hist, _ = np.histogramdd(_sample, bins=self.bins)
        self.hist = hist + self.hist


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

    new_track_dtype = tracklet_reco.TrackletReconstruction.tracklet_dtype

    new_track_label_dtype = np.dtype([
        ('id', 'u4'),
        ('match', 'u1'),
        ('broken', 'u1'),
        ('neighbor', 'i4'),
        ('hit_frac', 'f4'),
        ('true_endpoint_d', 'f4', (2,)),
        ('neighbor_sin2theta', 'f4'),
        ('neighbor_endpoint_d', 'f4'),
        ('neighbor_missing_length', 'f4'),
        ('neighbor_overlap', 'f4'),
        ('neighbor_ddqdx', 'f4')
    ])

    default_pdf_bins = [
        (-5, 0, 30),
        (0, 3, 30),
        (0, 3, 30),
        (-2, 3, 30),
        (-3, 2, 30)
    ]
    missing_track_segments = 200
    truth_hit_frac_cut = 0.8

    def __init__(self, **params):
        super(BrokenTrackSim, self).__init__(**params)

        self.path = params.get('path', 'misc/broken_track_sim')

        self.generate_2track_joint_pdf = params.get('generate_2track_joint_pdf', True)
        self.pdf_bins = [np.logspace(*bins) for bins in params.get('pdf_bins',
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
                                      dtype=disabled_channels_lut_arr.dtype)
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
        self.pixel_y = np.unique(resources['Geometry'].pixel_xy.compress((1,)))

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

        if len(np.r_[source_slice]) != 0:

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
            hit_frac = d['hit_frac']

            d = self.find_neighbor(new_tracks, broken.reshape(new_tracks.shape))
            neighbor = d['neighbor']
            d = self.find_neighbor(tracks)
            orig_neighbor = d['neighbor']

            if self.generate_2track_joint_pdf:
                track2_sin2theta = self.calc_2track_sin2theta(new_tracks, neighbor)
                track2_sin2theta_orig = self.calc_2track_sin2theta(tracks, orig_neighbor)
                track2_endpoint_distance = self.calc_2track_endpoint_distance(new_tracks, neighbor)
                track2_endpoint_distance_orig = self.calc_2track_endpoint_distance(tracks, orig_neighbor)
                track2_missing_length = self.calc_2track_missing_length(new_tracks, neighbor)
                track2_missing_length_orig = self.calc_2track_missing_length(tracks, orig_neighbor)
                track2_overlap = self.calc_2track_overlap(new_tracks, neighbor)
                track2_overlap_orig = self.calc_2track_overlap(tracks, orig_neighbor)
                track2_ddqdx = self.calc_2track_ddqdx(new_tracks, neighbor)
                track2_ddqdx_orig = self.calc_2track_ddqdx(tracks, orig_neighbor)                

                self.pdf['rereco'].fill(track2_sin2theta, track2_endpoint_distance,
                                        track2_missing_length, track2_overlap,
                                        track2_ddqdx)
                self.pdf['origin'].fill(track2_sin2theta_orig, track2_endpoint_distance_orig,
                                        track2_missing_length_orig, track2_overlap_orig,
                                        track2_ddqdx_orig)
                
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
            ref = np.empty((0,2))
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
            label_array['neighbor_sin2theta'] = track2_sin2theta.flat[~broken.mask.ravel()]
            label_array['neighbor_endpoint_d'] = track2_endpoint_distance.flat[~broken.mask.ravel()]
            label_array['neighbor_missing_length'] = track2_missing_length.flat[~broken.mask.ravel()]
            label_array['neighbor_overlap'] = track2_overlap.flat[~broken.mask.ravel()]
            label_array['neighbor_ddqdx'] = track2_ddqdx.flat[~broken.mask.ravel()]
            self.data_manager.write_data(f'{self.path}/label', label_slice, label_array)
            
            ref = np.c_[np.indices(new_tracks.shape)[0][~new_tracks['id'].mask]+source_slice.start, new_tracks['id'].compressed()]
        else:
            ref = np.empty((0,2))
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
        # (events, hits, 1)
        rand_track_hits = hits_track_idx ==  np.expand_dims(rand_tracks['id'], axis=-1)

        # (events, 1, new_tracks)
        new_id = np.expand_dims(np.indices(new_tracks.shape)[-1], axis=1)

        # (events, hits, new_tracks)
        new_track_hits = (new_id == np.expand_dims(track_ids, axis=-1))

        # (events, new_tracks)
        frac_new_hits = (np.sum(rand_track_hits & new_track_hits, axis=1)
                          / np.sum(new_track_hits, axis=1))
        match = np.expand_dims(frac_new_hits > self.truth_hit_frac_cut, axis=-1)

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

        # id broken tracks
        endpoint_distance_2 = np.take_along_axis(endpoint_distance, i_endpoint_distance[..., 1:2], axis=-1)
        broken = (endpoint_distance_2 > self.broken_track_distance_cut) & match
        
        return dict(
            trans_rand_tracks=trans_rand_tracks,
            match=match,
            broken=broken,
            hit_frac=frac_new_hits,
            endpoint_distance_1=endpoint_distance_1,
            endpoint_distance_2=endpoint_distance_2
        )

    def find_neighbor(self, tracks, mask=None):
        '''
        Find neighbor based on endpoint distance:
        
         - ``tracks`` is an (N,M) array of tracks
         - ``mask`` is boolean of same shape as ``tracks``
         - ``mask`` true indicates a valid track to search for neighbors

        '''
        ntracks = tracks.shape[-1]
        if mask is None:
            mask = np.ones(tracks.shape, dtype=bool)
        mask = (np.expand_dims(mask, axis=1) & np.expand_dims(mask, axis=2)
                & ~np.diagflat(np.ones(ntracks, dtype=bool)).reshape(1, ntracks, ntracks)
                & np.expand_dims(~tracks['id'].mask, axis=1) & np.expand_dims(~tracks['id'].mask, axis=2))

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
        endpoint_distance = endpoint_distance.min(axis=-1)
        endpoint_distance = ma.array(endpoint_distance, mask=~mask)
        
        neighbor = ma.argmin(endpoint_distance, axis=-1).reshape(tracks.shape)
        neighbor = ma.array(neighbor, mask=tracks['id'].mask | np.all(~mask, axis=-1))
        neighbor.fill_value = -1
        neighbor = ma.array(neighbor.filled(), mask=neighbor.mask)
        return dict(neighbor=neighbor)

    def calc_2track_sin2theta(self, tracks, neighbor):
        ntracks = tracks.shape[1]
        neighbor = neighbor.reshape(tracks.shape + (1,))
        dxyz = tracks['start'] - tracks['end']
        dxyz /= np.sqrt(np.sum((dxyz)**2, axis=-1, keepdims=True))
        dxyz_neighbor = np.take_along_axis(dxyz, neighbor, axis=1)
        sin2theta = 1 - np.sum(dxyz * dxyz_neighbor, axis=-1)**2
        mask = (tracks['id'].mask | neighbor.mask.reshape(sin2theta.shape)
                | (neighbor == -1).reshape(sin2theta.shape))
        return ma.array(sin2theta, mask=mask)

    def calc_2track_endpoint_distance(self, tracks, neighbor):
        ntracks = tracks.shape[1]
        neighbor = neighbor.reshape(tracks.shape + (1,))
        start1 = tracks['start']
        start2 = np.take_along_axis(tracks['start'], neighbor, axis=1)
        end1 = tracks['end']
        end2 = np.take_along_axis(tracks['end'], neighbor, axis=1)
        d = ma.concatenate((
            ma.sum((start1 - end2)**2, axis=-1, keepdims=True),
            ma.sum((end1 - end2)**2, axis=-1, keepdims=True),
            ma.sum((start1 - start2)**2, axis=-1, keepdims=True),
            ma.sum((end1 - start2)**2, axis=-1, keepdims=True),
        ), axis=-1)
        d = np.sqrt(d)
        distance = ma.amin(d, axis=-1)
        mask = (tracks['id'].mask |
                neighbor.mask.reshape(distance.shape)
                | (neighbor == -1).reshape(distance.shape))
        return ma.array(distance, mask=mask)

    def make_missing_segment(self, start, end, neighbor):
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

    def calc_2track_missing_length(self, tracks, neighbor):
        # create missing track segment
        _n_steps = self.missing_track_segments
        neighbor = neighbor.reshape(tracks.shape + (1,))
        _missing_start, _missing_end = self.make_missing_segment(tracks['start'], tracks['end'], neighbor)
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
        missing_length = _missing_length - _hidden_length
        missing_length = np.clip(missing_length, 0, None)

        mask = (tracks['id'].mask
                | neighbor.mask.reshape(missing_length.shape)
                | (neighbor == -1).reshape(missing_length.shape))
        return ma.array(missing_length, mask=mask)

    def calc_2track_overlap(self, tracks, neighbor):
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

    def calc_2track_ddqdx(self, tracks, neighbor):
        _ntracks = tracks.shape[1]
        neighbor = neighbor.reshape(tracks.shape)
        _track1_dqdx = tracks['q']/tracks['length']
        _track2_dqdx = np.take_along_axis(tracks['q']/tracks['length'], neighbor, axis=1)
        ddqdx = np.abs(_track1_dqdx - _track2_dqdx)
        mask = (tracks['id'].mask
                | neighbor.mask.reshape(ddqdx.shape)
                | (neighbor == -1).reshape(ddqdx.shape))
        return ma.array(ddqdx, mask=mask)

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
