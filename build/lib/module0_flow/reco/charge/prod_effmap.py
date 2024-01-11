import numpy as np
import logging
import pandas as pd
import os

from h5flow.core import H5FlowStage, resources
from module0_flow.util.functions import remove_close_values, get_extent_25, plot_eff_maps



class ProdEffMap(H5FlowStage):

    defaults = dict(
        effs_dset_name ='charge/effs',
        module = 'mod1',
        iogroup = '1',
        within_rad = False,
        mode = '5x5',
        axis_bounds = 20,
        output_dir = './randomsegments',
        plot = True,
        plot_dir = './plots',
        )

    class_version = '0.0.0'

    effs_dtype = np.dtype([('dist_x', 'f8'), ('dist_y', 'f8'), ('hit', 'bool'), ('start_X', 'f8'), ('start_Y', 'f8'), ('start_Z', 'f8'), ('angle', 'f8')])

    def __init__(self, **params):

        super(ProdEffMap, self).__init__(**params)

        for key in self.defaults:
            setattr(self, key, params.get(key, self.defaults[key]))

    def init(self, source_name):

        super(ProdEffMap, self).init(source_name)
        self.data_manager.create_dset(self.effs_dset_name, dtype=self.effs_dtype)

    def run(self, source_name, source_slice, cache):

        super(ProdEffMap, self).run(source_name, source_slice, cache)
        effs_data = cache[self.effs_dset_name]

        if self.mode == 'fullmap':
            print('Add later')

        elif self.mode == '5x5':

            ps = 4.434

            if self.module == 'mod2':
                ps = 3.8

            bin_2d_x = np.arange(-5 * ps / 2, 5 * ps / 2 + ps / self.axis_bounds, ps / self.axis_bounds)
            bin_2d_y = np.arange(-5 * ps / 2, 5 * ps / 2 + ps / self.axis_bounds, ps / self.axis_bounds)

            hit_hist, binsx, binsy = np.histogram2d(effs_data['dist_x'][effs_data['hit']==True], effs_data['dist_y'][effs_data['hit']==True], bins=[bin_2d_x, bin_2d_y]) 
            all_hist, binsx, binsy = np.histogram2d(effs_data['dist_x'], effs_data['dist_y'], bins=[bin_2d_x, bin_2d_y])

            file_path_all = self.output_dir + "/pixel_bounds_" + str(self.axis_bounds) + '/' + self.module + "/all_pixels/"
            file_path_hit = self.output_dir + "/pixel_bounds_" + str(self.axis_bounds) + '/' + self.module + "/hit_pixels/"

            if not os.path.exists(file_path_all):
                os.makedirs(file_path_all)

            if not os.path.exists(file_path_all + "all_pixels_" + self.data_manager.filepath.split('.')[0] + ".npy"):
                np.save(self.output_dir + "/pixel_bounds_" + str(self.axis_bounds) + '/' + self.module + "/all_pixels/all_pixels_" + self.data_manager.filepath.split('.')[0] + ".npy", all_hist)

            if not os.path.exists(file_path_hit):
                os.makedirs(file_path_hit)

            if not os.path.exists(file_path_hit + "hit_pixels_" + self.data_manager.filepath.split('.')[0] + ".npy"):
                np.save(self.output_dir + "/pixel_bounds_" + str(self.axis_bounds) + '/' + self.module + "/hit_pixels/hit_pixels_" + self.data_manager.filepath.split('/')[0] + ".npy", hit_hist)

            if self.plot:
                print('generating plots')
                extent = get_extent_25(ps)
                plot_eff_maps(all_hist, hit_hist, extent, self.module, self.iogroup, self.plot_dir)

