import math
import pickle as pkl

import numpy as np
import numpy.lib.recfunctions as rfn
import numpy.ma as ma
from scipy.linalg import eigh

from h5flow.core import H5FlowStage, resources

# import numpy.ma as ma
# import logging
# from collections import defaultdict

# from proto_nd_flow.reco.charge.calib_prompt_hits import CalibHitBuilder


class Hough3D(H5FlowStage):
    '''
    Merges the specified cached hits based on their unique channel id and timestamp:
     - q -> sum(q)
     - ts -> sum(ts * q) / sum(q)

    Two algorithms for selecting pairs of hits to merge have been implemented:

     - `'pairwise'`: On each iteration, sort all hits by unique y-z position and timestamp. Then, merge every pair of hits that fall within the merge cut. If an odd number of hits fall should be merged, the earliest hit of a group is excluded from the iteration.
     - `'last-first'`: On each iteration, sort all hits by unique y-z and timestamp. Then, merge the last pair of hits that fall within the merge cut within each contiguous chunk of neighboring hits.

    Both algorithms should produce very similar results.

    Example config::

      hit_merging:
        classname: CalibHitMerger
        path: module0_flow.reco.charge.hit_merger
        requires:
          - 'charge/hits'
        params:
          events_dset_name: 'charge/events'
          hits_name: 'charge/hits'
          hit_charge_name: 'charge/hits' # dataset to grab 'q' from
          merged_name: 'charge/hits/merged'
          mc_hit_frac_dset_name: ``str``, optional, output dataset path for hit charge fraction truth (if present)
          merge_cut: 30 # merge hits with delta t < merge_cut [CRS ticks]
          merge_mode: 'last-first'
    '''
    class_version = '0.0.0'

    # Below can be removed? Unused?? #
    defaults = dict(
        events_dset_name = 'charge/events',
        hits_name = 'charge/calib_prompt_hits',
        opt_dx = 2,
        sphere_granularity = 4,
        opt_nlines = 0,
        opt_minvotes = 10,
        opt_verbose = 0,
        weights = False,
        # hit_charge_name = 'charge/calib_prompt_hits',
        # merged_name = 'charge/hits/calib_merged_hits',
        # max_merge_steps = 5,
        # max_contrib_segments = 200,
        # merge_mode = 'last-first',
        # merge_cut = 50, # CRS ticks
        mc_hit_frac_dset_name = 'mc_truth/calib_final_hit_backtrack'
        )
    # (End of "Below") #

    # valid_merge_modes = ['last-first', 'pairwise']

    
    hough3d_out_dtype = np.dtype([
        ('clusters', 'u8')
    ])


    # merged_dtype = CalibHitBuilder.calib_hits_dtype

    

    # sum_fields = ['Q','E']
    # weighted_mean_fields = ['t_drift', 'ts_pps','x']

    def __init__(self, **params):
        super(Hough3D, self).__init__(**params)
        # self.hits_name = params.get('hits_name')
        self.hough3d_dset_name = params.get('hough3d_dset_name','charge/hough3d')
        self.events_dset_name = params.get('events_dset_name','charge/events')
        self.hits_name = params.get('hits_name','charge/calib_prompt_hits')
        # implement params.get(VAR_STR,DEFAULT) for all below
        self.opt_dx = 2
        self.sphere_granularity = 4
        self.opt_nlines = 0
        self.opt_minvotes = 10
        self.opt_verbose = 0
        self.weights = False
        self.max_contrib_segments = 200
        # self.hit_charge_name = 'charge/calib_prompt_hits'
        # self.merged_name = 'charge/hits/calib_merged_hits'
        # self.max_merge_steps = 5
        # self.max_contrib_segments = 200
        # self.merge_mode = 'last-first'
        # self.merge_cut = 50 # CRS ticks
        self.mc_hit_frac_dset_name = params.get('mc_hit_frac_dset_name','mc_truth/hough3d_backtrack')
        # self.merge_mode = self.merge_mode.lower()
        # assert self.merge_mode in self.valid_merge_modes, f'invalid merge mode: {self.merge_mode}'
        # self.max_contrib_segments = params.get('max_contrib_segments','')
        with open('/global/homes/s/seschwar/sphere.pkl','rb') as f:
            sphere_dict = pkl.load(f)
        # try:
        self.sphere = np.array(sphere_dict['unique_vertices'][self.sphere_granularity])
        # except:print('')
    def init(self, source_name):
        super(Hough3D, self).init(source_name)
        # Remove below? #
        self.hit_frac_dtype = np.dtype([
            ('fraction', f'({self.max_contrib_segments},)f8'),
            ('segment_id', f'({self.max_contrib_segments},)u8')
        ])
        # End #

        # save all config info?? #  calib_prompt_hits.py L::112
        ## CODE HERE ##

        # set up new datasets
        self.data_manager.create_dset(self.hough3d_dset_name, dtype=self.hough3d_out_dtype)
        self.data_manager.create_dset(self.mc_hit_frac_dset_name, dtype=self.hit_frac_dtype) # Keep or no? Seems fishy...

        self.data_manager.create_ref(source_name,self.hough3d_dset_name)
        self.data_manager.create_ref(self.events_dset_name,self.hough3d_dset_name)
        # self.data_manager.create_ref(self.calib_hits_dset_name, self.packets_dset_name) # Seems good to remove...
        self.data_manager.create_ref(self.hits_name, self.hough3d_dset_name) # Need to keep
        # self.data_manager.create_ref(self.calib_hits_dset_name, self.mc_hit_frac_dset_name) # Keep or no? (connected to above)

        # self.create_ref()

        # self.data_manager.create_dset(self.merged_name, dtype=self.merged_dtype)
        # self.data_manager.create_dset(self.mc_hit_frac_dset_name, dtype=self.hit_frac_dtype)
        # self.data_manager.create_ref(self.hits_name, self.merged_name)
        # self.data_manager.create_ref(source_name, self.merged_name)
        # self.data_manager.create_ref(self.merged_name,self.mc_hit_frac_dset_name)
        # self.data_manager.create_ref(self.events_dset_name, self.merged_name)

    def run(self, source_name, source_slice, cache):
        super(Hough3D, self).run(source_name, source_slice, cache)

        # event_id = np.r_[source_slice]
        packet_frac_bt = cache['packet_frac_backtrack']
        packet_seg_bt = cache['packet_seg_backtrack']
        # hits = cache[self.hits_name]
        hits = cache[self.hits_name]
        # print(np.sum(hits.mask))
        # x,y,z=hits['x'],hits['y'],hits['z']
        # xm,ym,zm=x.mask,y.mask,z.mask
        # print(x[xm])
        # print(xm,ym,zm,sep='\n')
        
        # print(hits_mask.shape)
        # print(hits.mask.shape)
        # print(np.sum(hits_mask))
        # print(np.sum(x.mask))


        # hits = ma.array(hits,mask=hits.mask)
        # print(hits.shape)
        
        # # print(hits)
        # print(np.sum(hits_mask,axis=-1))
        # print(sum(1 for i in hits[0]))
        # print(sum(1 for i in hits[1]))
        # -print(sum(1 for i in hits[2]))
        # print(hits_mask.shape)
        # print(hits[hits_mask].shape)
        # -print(np.sum(hits_mask,axis=-1))
        # # np.newaxis()
        # print(*map(len,hits))
        # print(*map(sum,hits_mask))
        # print(hits_mask[:,:10])
        # print(hits_mask[:,-10:])
        
        # x,y,z=hits['x'],hits['y'],hits['z']
        # print(x.shape)
        # print(x[:,-10:])
        # print(y[:,:10])
        # print(z[:,:10])
        # print(y.shape)
        # print(z.shape)
        # print(hits.size)
        # print()
        # merged, ref, back_track = self.merge_hits(hits, weights=hits['Q'], seg_fracs=[packet_seg_bt,packet_frac_bt],dt_cut=self.merge_cut, sum_fields=self.sum_fields, weighted_mean_fields=self.weighted_mean_fields, max_steps=self.max_merge_steps, mode=self.merge_mode)
        
        hits_mask = ~rfn.structured_to_unstructured(hits.mask).any(axis=-1)
        print(np.sum(np.sum(hits_mask,-1)>2500))
        
        hough3d_out = np.concatenate([*[self.hough3d(np.c_[event_hits['x'],event_hits['y'],event_hits['z']][~event_hits['x'].mask])for event_hits in hits]],-1)
        # hough3d_out = [self.hough3d(np.c_[event_hits['x'],event_hits['y'],event_hits['z']])for event_hits in ma.array(hits,mask=hits_mask)]
        
        # print(*map(len,hough3d_out))
        # print(hough3d_out.dtype)
        # merged_mask = merged.mask['id']

        # first write the new merged hits to the file
        # print(len(hough3d_out))
        
        # print(hough3d_out[-1])

        hough3d_slice = self.data_manager.reserve_data(self.hough3d_dset_name, len(hough3d_out))
        self.data_manager.write_data(self.hough3d_dset_name, hough3d_slice, hough3d_out)

        # save references

        parent_idcs = np.arange(source_slice.start, source_slice.stop).reshape(-1, 1, 1)
        child_idcs = np.clip(np.arange(hough3d_slice.start, hough3d_slice.start + 10).reshape(1, -1, 1) + parent_idcs - source_slice.start, 0, hough3d_slice.stop - 1)
        parent_idcs, child_idcs = np.broadcast_arrays(parent_idcs, child_idcs)
        ref = np.unique(np.concatenate((parent_idcs, child_idcs), axis=-1).reshape(-1, 2), axis=0)  # reshape to (parent, child), and only use unique references (repeats can be used)
        #  2. then write them into the file (no space reservation needed)
        self.data_manager.write_ref(source_name, self.hough3d_dset_name, ref)

        # print(source_slice)
        # raw_ev_id = np.broadcast_to(np.expand_dims(np.r_[source_slice], axis=-1), hits.size)
        # ref = np.c_[raw_ev_id[hits_mask], hough3d_slice.start+np.arange(hits.size,dtype=int)]
        # # raw_event -> clusterer
        # self.data_manager.write_ref(source_name, self.hough3d_dset_name, ref)
        # event -> clusterer
        self.data_manager.write_ref(self.events_dset_name, self.hough3d_dset_name, ref)

        # hit -> clusterer
        ref = np.c_[hits['id'], hits['id']]
        self.data_manager.write_ref(self.hits_name, self.hough3d_dset_name, ref)

        # # hit -> packet # Almost definitely gets removed?
        # ref = np.c_[calib_hits_arr['id'], index_arr]
        # self.data_manager.write_ref(self.calib_hits_dset_name, self.packets_dset_name, ref)

        # # hit -> backtracking
        # if has_mc_truth:
        #     self.data_manager.write_ref(self.calib_hits_dset_name,self.mc_hit_frac_dset_name,np.c_[calib_hits_arr['id'],calib_hits_arr['id']])


        # hough_out
        # new_nhit = int((~merged_mask).sum())
        # merge_slice = self.data_manager.reserve_data(self.merged_name, new_nhit)
        # merge_idx = np.r_[merge_slice].astype(merged.dtype['id'])
        # if new_nhit > 0:
        #     ref[:,1] += merge_idx[0] # offset references based on reserved region in output file
        #     np.place(merged['id'], ~merged_mask, merge_idx)

        # self.data_manager.write_data(self.hough3d_dset_name, hough3d_slice.start + np.arange(hits.size), hough3d_out)

        # HACK: Remove duplicate refs. Would be nice to actually understand and
        # fix the origin of these duplicates.
        # ref = np.unique(ref, axis=0)
        # sort based on the ID of the prompt hit, to make analysis more convenient
        # ref = ref[np.argsort(ref[:, 0])]
        # self.data_manager.write_ref(self.hits_name, self.hough3d_dset_name, ref)

        # if back tracking information was available, write the merged back tracking
        # dataset to file 
        # if back_track is not None:
        #     merge_bt_slice = self.data_manager.reserve_data(self.mc_hit_frac_dset_name, new_nhit)
        #     self.data_manager.write_data(self.mc_hit_frac_dset_name, merge_idx, back_track[~merged_mask])
        #     self.data_manager.write_ref(self.merged_name,self.mc_hit_frac_dset_name,np.c_[merge_idx,merge_idx])

        # RESOLVE ME #
        # self.sphere = [] # MOVE ME # ***
        
        # Need to understand below lines... #
        # ev_ref = np.c_[(np.indices(merged_mask.shape)[0] + source_slice.start)[~merged_mask], merge_idx]
        # self.data_manager.write_ref(source_name, self.hough3d_dset_name, ev_ref)
        # self.data_manager.write_ref(self.events_dset_name, self.hough3d_dset_name, ev_ref)
        # ... #
    
    # def load_sphere(sphere_granularity=4):
    #     return []
        
    def hough3d(self, hits, max_hits = -1, weights=[], dx=2, max_lines=3, min_votes=10, verbose=0):
        # refer to hough3dlines.py (for s schwartz only)
        # print(hits.shape)
        # print(hits.size)
        nhits = len(hits)
        if nhits > 0:
            clusters = []
            pw_clusters = np.full(nhits,-1)
            for cluster_ind, cluster_hit_inds in enumerate(clusters):
                pw_clusters[cluster_hit_inds] = cluster_ind
            # print(f'{nhits} > 2500')
            return pw_clusters
        # else:print(f'{nhits} <= 2500')
        # print(nhits)
        if verbose >= 1:print(f'nhits = {nhits}')
        if -1 < max_hits < nhits:
            np.full(nhits,-1)
        # not using weights for running time optimization
        if nhits == 0:
            if verbose >= 1:print('No hits in event')
            return{}
        if max_lines < 0:
            print('max_lines must be 0 or positive')
            return{}
        if min_votes < 2:
            print('min_votes must be >= 2')
            return{}

        # hit_indices = np.arange(nhits)

        # Can make everything a index reference list to the input hit array
        point_cloud_inds = np.arange(nhits)

        dir_vecs = self.sphere
        n_dirs = len(dir_vecs)
        if verbose >= 2:print(f'Successfully loadaed {n_dirs} direction vertices')

        def orthogonal_lsq(point_cloud):
            if len(point_cloud):
                a = np.mean(point_cloud)
            else:
                a = np.zeros(3)
            centered = np.matrix(point_cloud - a)
            scatter = centered.getH() * centered
            eigen_val, eigen_vec = eigh(scatter)
            b = eigen_vec[:,2]
            rc = eigen_val[2]
            return rc, a, b
        
        max_bound_point = np.max(hits,0) # OR [max_x, max_y, max_z]
        min_bound_point = np.min(hits,0) # OR [min_x, min_y, min_z]

        # shift = (max_bound_point + min_bound_point)/2 # OR DON'T DO THIS - Will need to center coordinate system properly in either case

        # DO NOT Want to apply shift to hits...
        # Instead will use "Origin"

        origin = (max_bound_point + min_bound_point)/2

        extent_bound_box = np.linalg.norm(max_bound_point - min_bound_point) # Redundant computation? Does it matter?


        # Lines below should be removable with fixed bounds...
        if dx == 0:
            dx = extent_bound_box / 64
        elif dx > extent_bound_box:
            print('dx too large')
            return{}
        # else:
        #     dx = dx

        index_shift_n = int(np.ceil((extent_bound_box/dx-1)/2)) # N = 1+2n, n>=(d/dx-1)/2 [from (2n+1)dx >= d] # Side length of lattice in number of cells # Half of extent[rounded up to odd multiple of dx] over dx
        lattice_side_len_anchor_points = 1+2*index_shift_n # Number
        # anchor_point_shift = index_shift_n * dx # End to origin (Anchor Points)
        # lattice_bounding_box_side_length = (lattice_side_len_anchor_points+1) * dx # End to end length of x' space (= previous + dx)

        def xy_from_index(x_ind,y_ind):
            return dx*(x_ind - index_shift_n), dx*(y_ind - index_shift_n)
        def points_close_to_line_inds(point_cloud,a,b): # Return indices of points close to line
            return point_cloud_inds[np.where(np.linalg.norm(point_cloud - (a + np.array([i*b for i in np.dot(point_cloud-a,b)])),axis=1) <= dx)]
        def point_vote(point_cloud,add=True):
            c = [-1,1][add]
            for b_ind, b in enumerate(dir_vecs):
                beta = 1/(1+b[2])

                # x = (1-b[0]**2*beta)*point_cloud[:,0] - b[0]*b[1]*beta*point_cloud[:,1] - b[0]*point_cloud[:,2]
                # y = -b[0]*b[1]*point_cloud[:,0] + (1-b[1]**2*beta)*point_cloud[:,1] - b[1]*point_cloud[:,2]
                # x_ind, y_ind = np.array(np.rint(x/dx+index_shift_n),dtype=np.int16), np.array(np.rint(y/dx+index_shift_n),dtype=np.int16) # these are indices, of course they can be small ints
                
                # Below code could likely be changed from a for loop into a magical np line...
                for x_ind,y_ind in np.array(np.c_[(1-b[0]**2*beta)*point_cloud[:,0] - b[0]*b[1]*beta*point_cloud[:,1] - b[0]*point_cloud[:,2],-b[0]*b[1]*point_cloud[:,0] + (1-b[1]**2*beta)*point_cloud[:,1] - b[1]*point_cloud[:,2]]/dx+index_shift_n,dtype=int):
                    if 0 <= x_ind < lattice_side_len_anchor_points > y_ind >= 0:
                        accumulator_arr[b_ind,x_ind,y_ind] += c
                        # For flattened accumulator_arr use np.ravel_multi_index

        accumulator_arr = np.zeros((n_dirs,lattice_side_len_anchor_points,lattice_side_len_anchor_points)) # Accumulator Array A, |B| X |X'| X |Y'| 
        # For flattened accumulator_arr use np.zeros (n_dirs * X * Y) i.e. product - (this should be obvious)

        clusters = []

        # print(len(point_cloud_inds),len(hits))
        point_vote(hits[point_cloud_inds],True) # fill accumulator array

        nlines = 0
        point_cluster_inds = []

        while len(point_cloud_inds) >= min_votes and (max_lines == 0 or max_lines > nlines):
            point_vote(hits[point_cluster_inds],False) # Subtract votes for clustered points (First time does nothing)
            b_ind, x_ind, y_ind = np.unravel_index(accumulator_arr.argmax(),accumulator_arr.shape) # idrk why this is necessary...
            # For flattened accumulator use np.unravel_index(A.argmax(),(num_b,num_x,num_y))

            x,y = xy_from_index(x_ind,y_ind)
            # Get direction vector b
            b = dir_vecs[b_ind]
            # Anchor point a
            a = x*np.array([1-b[0]**2/(1+b[2]),-b[0]*b[1]/(1+b[2]),-b[0]]) + y*np.array([-b[0]*b[1]/(1+b[2]),1-b[1]**2/(1+b[2]),-b[1]]) # Anchor point
            if verbose >= 2:
                print(f'Info: Highest number of Hough votes is {accumulator_arr[b_ind,x_ind,y_ind]} for the following line: a=({a[0]},{a[1]},{a[2]}), b=({b[0]},{b[1]},{b[2]})')
            
            point_cluster_inds = points_close_to_line_inds(hits[point_cloud_inds],a,b)
            rc,a,b = orthogonal_lsq(hits[point_cluster_inds])
            if rc == 0:
                if verbose >= 1:
                    print('rc == 0')
                break

            point_cluster_inds = points_close_to_line_inds(hits[point_cloud_inds],a,b)
            nvotes = point_cluster_inds.size
            if nvotes < min_votes:
                if verbose >= 1:print(f'nvotes = {nvotes} < min_votes = {min_votes}')
                break

            # rc,a,b = orthogonal_lsq(hits[point_cluster_inds]) # Only necessary if wanting to return line parameters that describe final found lin

            # TODO - OPTIMIZE ME
            point_cloud_inds = np.setdiff1d(point_cloud_inds,point_cluster_inds)

            nlines += 1

            clusters += [point_cluster_inds]
        # End of while loop

        if verbose >= 2:print(*clusters,sep='\n')

        # pair-wise cluster indices
        pw_clusters = np.full(nhits,-1)
        for cluster_ind, cluster_hit_inds in enumerate(clusters):
            pw_clusters[cluster_hit_inds] = cluster_ind
        return pw_clusters