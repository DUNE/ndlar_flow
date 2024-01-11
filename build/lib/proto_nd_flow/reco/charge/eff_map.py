import numpy as np
import logging
import pandas as pd
import random

from h5flow.core import H5FlowStage, resources
from proto_nd_flow.util.functions import remove_close_values

class EffMap(H5FlowStage):


    defaults = dict(
        hough_hits_o_dset_name='charge/hough_hits_o',
        hough_o_dset_name='charge/hough_o',
        effs_dset_name ='charge/effs',
        mode = '5by5',
        within_rad = False,
        radius = 40,
        num_pixels_plot = 5,
        )

    class_version = '0.0.0'

    effs_dtype = np.dtype([('dist_x', 'f8'), ('dist_y', 'f8'), ('hit', 'bool'), ('start_X', 'f8'), ('start_Y', 'f8'), ('start_Z', 'f8'), ('angle', 'f8')])

    def __init__(self, **params):

        super(EffMap, self).__init__(**params)

        for key in self.defaults:
            setattr(self, key, params.get(key, self.defaults[key]))

    def init(self, source_name):

        super(EffMap, self).init(source_name)
        self.data_manager.create_dset(self.effs_dset_name, dtype=self.effs_dtype)

    def run(self, source_name, source_slice, cache):

        if self.mode == 'selfmade':

            super(EffMap, self).run(source_name, source_slice, cache)

            hough_hits_o_data = cache[self.hough_hits_o_dset_name]
            hough_o_data = cache[self.hough_o_dset_name]



#        self.evids = np.r_[self.evids, hough_o_data['evid']]

#        self.lines = np.r_[self.lines, hough_o_data]

            if len(hough_o_data) > 0:
                self.lines = hough_o_data

            module = 'mod1'
            iogroup = '1'
            radius = 44
            lim = 3.5*4.434

            df_pixel_map = pd.read_pickle('../../DUNE/geometry/working_map_'+module+'_' + iogroup + '.pkl')

            if module == 'mod2':
                radius = 38
                lim = 3.5*3.8

            df_pixel_map = df_pixel_map.astype('float32')
            df_pixel_map['start_Y'] = np.round(df_pixel_map['start_Y'], 3)
            df_pixel_map['start_X'] = np.round(df_pixel_map['start_X'], 3)

            hits = hough_i_data[np.isin(hough_i_data['evid'], self.evids)]

            if len(hits) > 0:

                line = self.lines[self.lines['evid'] == np.unique(hits['evid'])]

                track_length = np.sqrt(np.square(line['hf_x'] - line['hi_x']) + np.square(line['hf_y'] - line['hi_y']))

                line_start = 0
                num_segments = int(np.floor(track_length/radius))

                random_points = [random.uniform(line_start + i * radius, line_start + (i + 1) * radius) for i in
                                 range(num_segments)]

                remove_close_values(random_points, radius)

                bX = (line['hf_x'] - line['hi_x']) / track_length
                bY = (line['hf_y'] - line['hi_y']) / track_length

                ArX = line['hi_x']+bX*random_points
                ArY = line['hi_y']+bY*random_points

                for p in np.c_[ArX, ArY]:

                    val = df_pixel_map[(np.abs(p[0] - df_pixel_map['start_X'])<lim) & (np.abs(p[1] - df_pixel_map['start_Y'])<lim)]
                    df_seg = np.empty(val['start_X'].shape, dtype=self.effs_dtype)

                    df_seg['start_X'] = val['start_X'].values
                    df_seg['start_Y'] = val['start_Y'].values
                    df_seg['dist_x'] = p[0] - df_seg['start_X']
                    df_seg['dist_y'] = p[1] - df_seg['start_Y']

                    df_seg['hit'] = ((np.isin(df_seg['start_Y'], hits['y'])) & (np.isin(df_seg['start_X'], hits['x'])))

                    df_seg['angle'] = np.ones(len(df_seg))*line['angle']

                    print(df_seg)
                    eff_slice = self.data_manager.reserve_data(self.effs_dset_name, len(df_seg))
                    self.data_manager.write_data(self.effs_dset_name, eff_slice, df_seg)

        elif self.mode == '5by5':

            super(EffMap, self).run(source_name, source_slice, cache)

            hough_hits_o_data = cache[self.hough_hits_o_dset_name]
            hough_o_data = cache[self.hough_o_dset_name]

            if hough_hits_o_data is None:
                return 0

            if hough_o_data[0]['useful']==0:
                print('event not worth studying')
                return 0

            if self.within_rad == True:
                hough_hits_o_data = hough_hits_o_data[hough_hits_o_data['keep']==True]

#            df_pixel_map = pd.read_pickle('./geometry/working_map_2x2_assembly.pkl')
            df_pixel_map = pd.read_pickle('../../DUNE/geometry/working_map_mod1_1.pkl')

            df_pixel_map = df_pixel_map.astype('float32')
            df_pixel_map['start_Y'] = np.round(df_pixel_map['start_Y'], 3)
            df_pixel_map['start_X'] = np.round(df_pixel_map['start_X'], 3)

            num_segments = int(np.floor(hough_o_data[0]['length']/self.radius))

            line_start = -hough_o_data[0]['length']/2

            random_points = [random.uniform(line_start + i * self.radius, line_start + (i + 1) * self.radius) for i in
                             range(num_segments)]

            remove_close_values(random_points, self.radius)

            ArX = hough_o_data[0]['h_ax']+hough_o_data[0]['h_bx']*random_points
            ArY = hough_o_data[0]['h_ay']+hough_o_data[0]['h_by']*random_points
            ArZ = hough_o_data[0]['h_az']+hough_o_data[0]['h_bz']*random_points

            for p in np.c_[ArX, ArY, ArZ]:

                val = df_pixel_map[(np.abs(p[0] - df_pixel_map['start_X'])<(self.num_pixels_plot/2 * self.radius)) & (np.abs(p[1] - df_pixel_map['start_Y'])<(self.num_pixels_plot/2 * self.radius))]

                seg_array = np.empty(val['start_X'].shape, dtype=self.effs_dtype)
                seg_array['start_X'] = val['start_X'].values
                seg_array['start_Y'] = val['start_Y'].values
                seg_array['start_Z'] = np.ones(len(seg_array))*p[2]

                seg_array['dist_x'] = p[0] - seg_array['start_X']
                seg_array['dist_y'] = p[1] - seg_array['start_Y']


                seg_array['hit'] = ((np.isin(np.round(seg_array['start_Y'], 3), np.round(hough_hits_o_data['y'], 3))) & (np.isin(np.round(seg_array['start_X'], 3), np.round(hough_hits_o_data['x'], 3))))

                seg_array['angle'] = np.ones(len(seg_array))*hough_o_data[0]['h_bz']

                eff_slice = self.data_manager.reserve_data(self.effs_dset_name, len(seg_array))
                self.data_manager.write_data(self.effs_dset_name, eff_slice, seg_array)





