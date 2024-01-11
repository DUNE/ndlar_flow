import numpy as np
import logging
from module0_flow.util.functions import get_pixel_bins, HoughOnArray, d_pnt2line, trackLengthhoughcenter
import numpy.lib.recfunctions as rfn

from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
from skimage import data

from h5flow.core import H5FlowStage, resources

import matplotlib.pyplot as plt
from matplotlib import cm


class LineFinder(H5FlowStage):

    defaults = dict(
        hough_i_dset_name='charge/hough_i',
        hough_o_dset_name='charge/hough_o',
        hough_hits_o_dset_name='charge/hough_hits_o',
        events_dset_name='charge/events',
        max_peaks=1,
        min_pixels=10,
        mode = 'external',
        events_start = 0,
        radius = 8
        )

    class_version = '0.0.0'

    hough_o_dtype = np.dtype([('hi_x', 'f8'), ('hi_y', 'f8'), ('hi_z', 'f8'), ('hf_x', 'f8'), ('hf_y', 'f8'), ('hf_z', 'f8'), ('h_ax', 'f8'), ('h_ay', 'f8'), ('h_az', 'f8'), ('h_bx', 'f8'), ('h_by', 'f8'), ('h_bz', 'f8'), ('evid', 'u2'), ('votes', 'u2'), ('length', 'f8'), ('useful', 'u2')])
#    hough_o_dtype = np.dtype([('hi_x', 'f8'), ('hi_y', 'f8'), ('hf_x', 'f8'), ('hf_y', 'f8'), ('angle', 'f8'), ('evid', 'u2'), ('votes', 'u2')])
    hough_hits_o_dtype = np.dtype([('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('Q', 'f8'), ('E', 'f8'), ('dist', 'f8'), ('id','u2'), ('keep', 'bool')])


    def __init__(self, **params):
        super(LineFinder, self).__init__(**params)
        for key in self.defaults:
            setattr(self, key, params.get(key, self.defaults[key]))

    def init(self, source_name):

        super(LineFinder, self).init(source_name)

        self.data_manager.create_dset(self.hough_o_dset_name, dtype=self.hough_o_dtype)
        self.data_manager.create_dset(self.hough_hits_o_dset_name, dtype=self.hough_hits_o_dtype)

        self.data_manager.create_ref(self.events_dset_name, self.hough_hits_o_dset_name)
        self.data_manager.create_ref(self.hough_o_dset_name, self.events_dset_name)

    def run(self, source_name, source_slice, cache):
        super(LineFinder, self).run(source_name, source_slice, cache)

        if self.mode == 'external':

            # hits array should at least have 'x', 'y', 'z', 'Q', 'E' and 'id'

            hits = cache[self.hough_i_dset_name]
            events = cache[self.events_dset_name]
            hit_pass = 1

            if len(hits[hits['next']>0]) < 50:
                hit_pass = 0

            hough_o_array = np.empty(events['id'].shape, dtype=self.hough_o_dtype)

            id = events['id'][0]

            aX, aY, aZ, bX, bY, bZ, npoints, hi_x, hf_x, hi_y, hf_y, hi_z, hf_z, mask = HoughOnArray(hits, id, self.min_pixels, self.max_peaks)

            if npoints == 0 and hf_z == 0:
                hit_pass = 0

            dist = d_pnt2line(hits, hi_x, hf_x, hi_y, hf_y, hi_z, hf_z)

            if np.isnan(dist).any():
                hit_pass = 0

            if len(dist[dist<self.radius]) < 25:
                hit_pass = 0

            tracklength, aX_nl, aY_nl, aZ_nl, mask_dist = trackLengthhoughcenter(hits, aX, aY, aZ, bX, bY, bZ, self.radius, dist)

            hough_o_array['hi_x'] = hi_x
            hough_o_array['hi_y'] = hi_y
            hough_o_array['hi_z'] = hi_z
            hough_o_array['hf_x'] = hf_x
            hough_o_array['hf_y'] = hf_y
            hough_o_array['hf_z'] = hf_z
            hough_o_array['h_ax'] = aX_nl
            hough_o_array['h_ay'] = aY_nl
            hough_o_array['h_az'] = aZ_nl
            hough_o_array['h_bx'] = bX
            hough_o_array['h_by'] = bY
            hough_o_array['h_bz'] = bZ
            hough_o_array['evid'] = id
            hough_o_array['votes'] = npoints
            hough_o_array['length'] = tracklength
            hough_o_array['useful'] = hit_pass

            hough_hits_o_array = np.empty(hits[0]['id'].shape, dtype=self.hough_hits_o_dtype)

            hough_hits_o_array['x'] = hits[0]['x']
            hough_hits_o_array['y'] = hits[0]['y']
            hough_hits_o_array['z'] = hits[0]['z']
            hough_hits_o_array['Q'] = hits[0]['Q']
            hough_hits_o_array['E'] = hits[0]['E']
            hough_hits_o_array['dist'] = dist
            hough_hits_o_array['id'] = hits[0]['id']
            hough_hits_o_array['keep'] = mask * mask_dist

            hough_hits_o_slice = self.data_manager.reserve_data(self.hough_hits_o_dset_name, len(hough_hits_o_array))
            self.data_manager.write_data(self.hough_hits_o_dset_name, hough_hits_o_slice, hough_hits_o_array)

            hough_o_slice = self.data_manager.reserve_data(self.hough_o_dset_name, len(hough_o_array))
            self.data_manager.write_data(self.hough_o_dset_name, hough_o_slice, hough_o_array)

            ev_id = np.arange(source_slice.start, source_slice.stop, dtype=int).reshape(-1, 1)

            hits_ev_id = np.broadcast_to(ev_id, (1, len(hough_hits_o_array)))

            ref1 = np.c_[hits_ev_id[0], hough_hits_o_array['id']]
            self.data_manager.write_ref(self.events_dset_name, self.hough_hits_o_dset_name, ref1)

            hough_line_id = np.broadcast_to(ev_id, (1, len(hough_o_array)))

            ref = np.c_[hough_line_id[0], [hough_o_array['evid']]]
            self.data_manager.write_ref(self.hough_o_dset_name, self.events_dset_name, ref)

        if self.mode == 'selfmade':

            hits = cache[self.hough_i_dset_name]

            io = 1

            for id in np.unique(hits['evid']):

                data_i = hits[hits['evid']==id]

                data_i = data_i[data_i['iogroup'] == int(io)]

                if len(data_i) < 50:
                    continue

                bins_x, bins_y = get_pixel_bins('mod0', 'edge')
                hist, x, y = np.histogram2d(data_i['x'], data_i['y'], bins = (bins_x, bins_y))
                hist[hist>1] == 1
                h, theta, d = hough_line(hist)

                for votes, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks = self.max_peaks, threshold=self.min_pixels)):

                    hough_o_array = np.empty(hits['evid'].shape, dtype=self.hough_o_dtype)

                    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
                    y1 = (dist - hist.shape[1] * np.cos(angle)) / np.sin(angle)

                    hough_o_array['iogroup'] = np.unique(data_i['iogroup'])
                    hough_o_array['hi_x'] = 0
                    hough_o_array['hi_y'] = y0
                    hough_o_array['hf_x'] = hist.shape[1]
                    hough_o_array['hf_y'] = y1
                    hough_o_array['angle'] = angle
                    hough_o_array['evid'] = id
                    hough_o_array['votes'] = votes

                    hough_o_slice = self.data_manager.reserve_data(self.hough_o_dset_name, 1)
                    self.data_manager.write_data(self.hough_o_dset_name, hough_o_slice, hough_o_array[0])
