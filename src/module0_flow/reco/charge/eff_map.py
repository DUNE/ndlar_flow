import numpy as np
import logging

from h5flow.core import H5FlowStage, resources

class EffMap(H5FlowStage):


    defaults = dict(
        hough_i_dset_name='charge/hough_i',
        hough_o_dset_name='charge/hough_o',
        eff_dset_name ='charge/effs'
        )

    class_version = '0.0.0'

    effs_dtype = np.dtype([('iogroup', 'u1'), ('hi_x', 'f8'), ('hi_y', 'f8'), ('hf_x', 'f8'), ('hf_y', 'f8'), ('angle', 'f8'), ('evid', 'u1'), ('votes', 'u1')])


    def __init__(self, **params):
        super(EffMap, self).__init__(**params)
        for key in self.defaults:
            setattr(self, key, params.get(key, self.defaults[key]))

    def init(self, source_name):
        super(EffMap, self).init(source_name)

    def run(self, source_name, source_slice, cache):
        super(EffMap, self).run(source_name, source_slice, cache)

    def print():
        print('wooooh')

