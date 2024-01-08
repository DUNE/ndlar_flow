import numpy as np
import numpy.ma as ma
import logging
from collections import defaultdict

from h5flow.core import H5FlowStage, resources

from proto_nd_flow.reco.charge.calib_prompt_hits import CalibHitBuilder


class hough(H5FlowStage):
    '''
    This module was adapted from CalibHitMerger
    The goal is to take the charge/calib_prompt_hits and perform a Hough transform
    This could also be performed on calib_merged_hits that are in the "final" stage
    The outputs are saved as a set of hits along the line that are a subset of the inital input hits
    Currently no hough transform implemented!!! Just an empty module
    ksutton 8/30/23
      '''
    class_version = '0.0.0'
    defaults = dict(
        events_dset_name = 'charge/events',
        hits_name = 'charge/calib_prompt_hits',
        hit_charge_name = 'charge/calib_prompt_hits',
        output_name = 'charge/hits/calib_hough_hits',
        #mc_hit_frac_dset_name = 'mc_truth/calib_hough_hit_backtrack'
        )

    output_dtype = CalibHitBuilder.calib_hits_dtype

    def __init__(self, **params):
        super(hough, self).__init__(**params)
        for key in self.defaults:
            setattr(self, key, params.get(key, self.defaults[key]))
     #   self.output_mode = self.output_mode.lower()
     #   assert self.output_mode in self.valid_output_modes, f'invalid output mode: {self.output_mode}'

    def init(self, source_name):
        super(hough, self).init(source_name)

       # self.hit_frac_dtype = np.dtype([
       #     ('fraction', f'({self.max_contrib_segments},)f8'),
       #     ('segment_id', f'({self.max_contrib_segments},)u8')
       # ])

        self.data_manager.create_dset(self.output_name, dtype=self.output_dtype)
       # self.data_manager.create_dset(self.mc_hit_frac_dset_name, dtype=self.hit_frac_dtype)
        self.data_manager.create_ref(self.hits_name, self.output_name)
        self.data_manager.create_ref(source_name, self.output_name)
       # self.data_manager.create_ref(self.output_name,self.mc_hit_frac_dset_name)
        self.data_manager.create_ref(self.events_dset_name, self.output_name)

    #@staticmethod
    def output_hits(self,hits, weights, seg_fracs):
        '''
        currently does nothing, need to add in Hough transform here
'''

        new_seg_bt = np.array(seg_fracs[0])
        new_frac_bt = np.array(seg_fracs[1])
        iteration_count = 0
        mask = hits.mask['id'].copy()
        new_hits = hits.data.copy()
        weights = weights.data.copy()
        old_ids = hits.data['id'].copy()[...,np.newaxis]
        old_id_mask = hits.mask['id'].copy()[...,np.newaxis]
        
        new_hit_idx = np.broadcast_to(np.cumsum(~mask.ravel(), axis=0).reshape(mask.shape + (1,)), old_ids.shape)-1
  #      back_track = np.full(shape=new_hits.shape,fill_value=0.,dtype=self.hit_frac_dtype)
       
        return (
            ma.array(new_hits, mask=mask),
            np.c_[np.extract(~(old_id_mask | mask[...,np.newaxis]), old_ids), np.extract(~(old_id_mask | mask[...,np.newaxis]), new_hit_idx)],
   #         ma.array(back_track, mask=mask)
            )

    def run(self, source_name, source_slice, cache):
        super(hough, self).run(source_name, source_slice, cache)

        #get the event id, backtracking, and hits from the input file
        event_id = np.r_[source_slice]
        packet_frac_bt = cache['packet_frac_backtrack']
        packet_seg_bt = cache['packet_seg_backtrack']
        hits = cache[self.hits_name]

        #get the new hits, references, and backtracking for the 
  #      output, ref, back_track = self.output_hits(hits, weights=hits['Q'], seg_fracs=[packet_seg_bt,packet_frac_bt])
        output, ref = self.output_hits(hits, weights=hits['Q'], seg_fracs=[packet_seg_bt,packet_frac_bt])


        output_mask = output.mask['id'] #not sure what this does yet

        # first write the new hits to the file 
        new_nhit = int((~output_mask).sum())
        output_slice = self.data_manager.reserve_data(self.output_name, new_nhit)
        output_idx = np.r_[output_slice].astype(output.dtype['id'])
        if new_nhit > 0:
            ref[:,1] += output_idx[0] # offset references based on reserved region in output file
            np.place(output['id'], ~output_mask, output_idx)

        self.data_manager.write_data(self.output_name, output_idx, output[~output_mask])
        #output_bt_slice = self.data_manager.reserve_data(self.mc_hit_frac_dset_name, new_nhit)
        #self.data_manager.write_data(self.mc_hit_frac_dset_name, output_idx, back_track[~output_mask])

        # HACK: Remove duplicate refs. Would be nice to actually understand and
        # fix the origin of these duplicates.
        ref = np.unique(ref, axis=0)
        # sort based on the ID of the prompt hit, to make analysis more convenient
        ref = ref[np.argsort(ref[:, 0])]

        # finally, write the references
        self.data_manager.write_ref(self.hits_name, self.output_name, ref)
        #self.data_manager.write_ref(self.output_name,self.mc_hit_frac_dset_name,np.c_[output_idx,output_idx])
        ev_ref = np.c_[(np.indices(output_mask.shape)[0] + source_slice.start)[~output_mask], output_idx]
        self.data_manager.write_ref(source_name, self.output_name, ev_ref)
        self.data_manager.write_ref(self.events_dset_name, self.output_name, ev_ref)

