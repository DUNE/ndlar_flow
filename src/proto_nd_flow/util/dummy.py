import numpy as np
import numpy.ma as ma
import logging
from collections import defaultdict

from h5flow.core import H5FlowStage, resources

from proto_nd_flow.reco.charge.calib_prompt_hits import CalibHitBuilder


class Dummy(H5FlowStage):
    '''
    this is a placeholder
 '''
    class_version = '0.0.0'
    defaults = dict(
        events_dset_name = 'charge/events',
        hits_name = 'charge/calib_prompt_hits',
        hit_charge_name = 'charge/calib_prompt_hits',
        merged_name = 'charge/hits/calib_merged_hits',
        max_merge_steps = 5,
        max_contrib_segments = 200,
        merge_mode = 'last-first',
        merge_cut = 50, # CRS ticks
        mc_hit_frac_dset_name = 'mc_truth/calib_final_hit_backtrack'
        )
    valid_merge_modes = ['last-first', 'pairwise']

    merged_dtype = CalibHitBuilder.calib_hits_dtype

    sum_fields = ['Q','E']
    weighted_mean_fields = ['t_drift', 'ts_pps','x']

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

