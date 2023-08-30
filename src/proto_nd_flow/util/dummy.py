import numpy as np
import numpy.ma as ma
import logging
from collections import defaultdict

from h5flow.core import H5FlowStage, resources

from proto_nd_flow.reco.charge.calib_final_hits import CalibHitBuilder


class Dummy(H5FlowStage):
    '''
    this is a placeholder
 '''
    class_version = '0.0.0'
    defaults = dict(
        events_dset_name = 'charge/events',
        hits_name = 'charge/calib_final_hits',
        hit_charge_name = 'charge/calib_final_hits',
        hits_hough_name = 'charge/calib_hough_hits'
        )
    hough_dtype = CalibHitBuilder.calib_hits_dtype

    def __init__(self, **params):
        print('running dummy.py!!!!!')
        super(Dummy, self).__init__(**params)
        for key in self.defaults:
            setattr(self, key, params.get(key, self.defaults[key]))
        self.merge_mode = self.merge_mode.lower()
        assert self.merge_mode in self.valid_merge_modes, f'invalid merge mode: {self.merge_mode}'

    def init(self, source_name):
        print('running dummy.py!!!!!')
        super(Dummy, self).init(source_name)

        self.hit_frac_dtype = np.dtype([
            ('fraction', f'({self.max_contrib_segments},)f8'),
            ('segment_id', f'({self.max_contrib_segments},)u8')
        ])

        self.data_manager.create_dset(self.hits_hough_name, dtype=self.hough_dtype)
        self.data_manager.create_ref(self.hits_name, self.hits_hough_name)
        self.data_manager.create_ref(source_name, self.hits_hough_name)
        self.data_manager.create_ref(self.events_dset_name, self.hits_hough_name)
