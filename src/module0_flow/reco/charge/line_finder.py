import numpy as np
import logging

from h5flow.core import H5FlowStage, resources

class LineFinder(H5FlowStage):


    defaults = dict(
        hits_dset_name='charge/hits',
        events_dset_name='charge/events',
        ext_trigs_dset_name='charge/ext_trigs',
        hough_dset_name='charge/hough',
        max_peaks=5,
        min_pixels=100
        )

    class_version = '0.0.0'

    def __init__(self, **params):
        super(LineFinder, self).__init__(**params)
        for key in self.defaults:
            setattr(self, key, params.get(key, self.defaults[key]))

    def init(self, source_name):
        super(LineFinder, self).init(source_name)

    def run(self, source_name, source_slice, cache):
        super(LineFinder, self).run(source_name, source_slice, cache)

        hits_data = cache[self.hits_dset_name]
        events_data = cache[self.events_dset_name]
        ext_trigs_data = cache[self.ext_trigs_dset_name]


# [('id', '<u4'), ('px', '<f8'), ('py', '<f8'), ('ts', '<f8'), ('ts_raw', '<u8'), ('q', '<f8'), ('iogroup', 'u1'), ('iochannel', 'u1'), ('chipid', 'u1'), ('channelid', 'u1'), ('geom', '<i8')]



    def print():
        print('wooooh')

