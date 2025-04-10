import numpy as np
import numpy.ma as ma
import logging
from collections import defaultdict

from h5flow.core import H5FlowStage, resources

from proto_nd_flow.reco.charge.calib_prompt_hits import CalibHitBuilder


class CalibHitMerger(H5FlowStage):
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
        super(CalibHitMerger, self).__init__(**params)
        for key in self.defaults:
            setattr(self, key, params.get(key, self.defaults[key]))
        self.merge_mode = self.merge_mode.lower()
        assert self.merge_mode in self.valid_merge_modes, f'invalid merge mode: {self.merge_mode}'

    def init(self, source_name):
        super(CalibHitMerger, self).init(source_name)

        self.hit_frac_dtype = np.dtype([
            ('fraction', f'({self.max_contrib_segments},)f8'),
            ('segment_ids', f'({self.max_contrib_segments},)i8')
        ])

        self.data_manager.create_dset(self.merged_name, dtype=self.merged_dtype)
        if resources['RunData'].is_mc:
            self.data_manager.create_dset(self.mc_hit_frac_dset_name, dtype=self.hit_frac_dtype)
        self.data_manager.create_ref(self.hits_name, self.merged_name)
        self.data_manager.create_ref(source_name, self.merged_name)
        if resources['RunData'].is_mc:
            self.data_manager.create_ref(self.merged_name,self.mc_hit_frac_dset_name)
        self.data_manager.create_ref(self.events_dset_name, self.merged_name)

    #@staticmethod
    def merge_hits(self,hits, weights, seg_fracs, dt_cut, sum_fields=None, weighted_mean_fields=None, max_steps=-1, mode='last-first'):
        '''
        Combines hits along the second axis on unique channels with a delta t less than dt_cut. Continues
        until no hits (or merged hits) are within dt_cut of each other

        :param hits: original hits array, shape: (N,M)

        :param weights: values used for weighted mean, shape: (N,M)

        :param fracs: fractional contributions of true segments per packet

        :param dt_cut: delta t cut to merge hits (float) [CRS ticks]

        :sum_fields: list of fields in ``hits`` and that should be *summed* when combined, must not be in ``weighted_mean_fields``

        :weighted_mean_fields: list of fields in ``hits`` and that should be averaged using the weights when combined, must not be in ``sum_fields``

        :param max_steps: optional, maximum number of merges to apply to pairs of neighboring hits (<0 == no limit, 0 == skip merging, >0 == limit steps)

        :param mode: optional, merging strategy, either `'last-first'` (on each iteration merges the last hit pair) or `'pairwise'` (on each iteration merges each unique hit pair)

        :returns: new hit array, shape: (N,m), new hit charge array, shape: (N,m), and an index array with shape (L,2), [:,0] being the index into the original hit array and [:,1] being the flattened index into the compressed new array

        '''

        has_mc_truth = seg_fracs is not None
        iteration_count = 0
        mask = hits.mask['id'].copy()
        new_hits = hits.data.copy()
        weights = weights.data.copy()
        old_ids = hits.data['id'].copy()[...,np.newaxis]
        old_id_mask = hits.mask['id'].copy()[...,np.newaxis]
        if has_mc_truth:
            hit_contributions = np.full(shape=weights.shape+(3,self.max_contrib_segments),fill_value=0.)
            # initialize hit contribution lists with unmerged data.
            # [there is probably a more pythonic way of doing this...]
            prompt_hit_max_seg_contrib = seg_fracs['segment_ids'].shape[-1]
            hit_contributions[:,:,0,:] = np.pad(np.expand_dims(weights,axis=2),((0,0),(0,0),(0,self.max_contrib_segments-1)),'mean')
            hit_contributions[:,:,1,:] = np.pad(np.squeeze(seg_fracs['fraction']),((0,0),(0,0),(0, self.max_contrib_segments-prompt_hit_max_seg_contrib)),'constant',constant_values=0)
            hit_contributions[:,:,2,:] = np.pad(np.squeeze(seg_fracs['segment_ids']),((0,0),(0,0),(0, self.max_contrib_segments-prompt_hit_max_seg_contrib)),'constant',constant_values=-1)
        while new_hits.size > 0 and iteration_count != max_steps:
            iteration_count += 1
            # algorithm is iterative, but typically only needs to loop a few (~2-3) times
            # so we'll spit a warning if we reach the maximum number of steps
            if iteration_count == max_steps:
                logging.info(f'Hit merging algorithm reached max step limit {max_steps}')

            # sort array along last axis to find groups of hits on the same channel, use a stable sort with the aim of improving performance on later iterations
            isort = np.argsort(ma.array(new_hits, mask=mask), axis=-1, order=['z','y','ts_pps','t_drift'], kind='stable')
            mask = np.take_along_axis(mask, isort, axis=-1)
            new_hits = np.take_along_axis(new_hits, isort, axis=-1)
            weights = np.take_along_axis(weights, isort, axis=-1)
            if has_mc_truth: hit_contributions = np.take_along_axis(hit_contributions,isort[...,np.newaxis,np.newaxis],axis=-3)
            old_ids = np.take_along_axis(old_ids, isort[...,np.newaxis], axis=-2)
            old_id_mask = np.take_along_axis(old_id_mask, isort[...,np.newaxis], axis=-2)
            N_new_hits = new_hits.shape[0]*new_hits.shape[1]-np.count_nonzero(mask)
            print('current number of merged hits =',N_new_hits)
            
            # identify neighboring hits on the same channel
            dt = np.abs(np.diff(new_hits['ts_pps'].astype(int), axis=-1))
            same_channel = (
                (new_hits['z'][..., :-1] == new_hits['z'][..., 1:])
                & (new_hits['y'][..., :-1] == new_hits['y'][..., 1:])
                & (new_hits['io_group'][..., :-1] == new_hits['io_group'][..., 1:])
            )

            # flag valid hits if they are on the same channel and are close in time
            to_merge = (dt < dt_cut) & same_channel & ~mask[...,:-1] & ~mask[...,1:]

            if mode == 'last-first':
                # only combine unambiguous pairs of hits on a channel on each iteration
                to_merge[...,:-1] = ~to_merge[...,1:] & to_merge[...,:-1]
            elif mode == 'pairwise':
                # combine every available pair of hits on each iteration
                to_merge[...,:-1] = to_merge[...,:-1] & (np.cumsum(to_merge, axis=-1) % 2 == 0)[...,:-1]
            else:
                raise RuntimeError(f'invalid merge mode: {mode}')

            print("merging:",np.count_nonzero(to_merge))

            # exits loop if no remaining hits to combine
            if np.any(to_merge):
                # move 2nd hit into position of first hit, combining attributes along the way
                hit0 = np.extract(to_merge, new_hits[...,:-1])
                hit1 = np.extract(to_merge, new_hits[...,1:])

                # these fields will be summed hit[i][field] -> hit[i+1][field] + hit[i][field]
                for field in sum_fields:
                    if field in new_hits.dtype.names:
                        np.place(new_hits[...,:-1][field], to_merge, hit0[field] + hit1[field])

                # these fields will use the charge-weighted average hit[i][field] -> (hit[i+1][field] * q[i+1] + hit[i][field] * q[i]) / (q[i+1] + q[i])
                q0 = np.extract(to_merge, weights[...,:-1])
                q1 = np.extract(to_merge, weights[...,1:])
                qsum = np.abs(q0) + np.abs(q1)
                # regularize so there are no nans
                qsum = np.where(qsum == 0, 1e-300, qsum)
                # it is not obvious how to treat the possibility of negative charge values (e.g. noise)
                # this should(?) be rare, so we'll just spit out a warning
                if np.any((q0 < 0) | (q1 < 0)):
                    logging.info(f'Hit merging encountered negative value(s) (count={((q0 < 0) | (q1 < 0)).sum()}) in charge weighting, results may be unreliable')
                w0 = np.abs(q0)/qsum
                w1 = np.abs(q1)/qsum
                for field in weighted_mean_fields:
                    if field in new_hits.dtype.names:
                        base = np.minimum(hit0[field], hit1[field]) # improves precision of weighted sum if values are large (e.g. timestamps)
                        np.place(new_hits[...,:-1][field], to_merge, ((hit0[field]-base) * w0 + (hit1[field]-base) * w1).astype(new_hits.dtype[field]) + base)
                # combine weights for next iteration
                np.place(weights[...,:-1], to_merge, weights[...,:-1] + weights[...,1:])
                if has_mc_truth:
                    for hit_it, hit_cont in np.ndenumerate(weights[...,:-1]):
                        if (not to_merge[hit_it]) | mask[hit_it]:
                            #print('skipping')
                            continue
                        e = np.argwhere(hit_contributions[...,:-1][hit_it][1]==0)[0][0]
                        f = np.argwhere(hit_contributions[...,:][hit_it[0],hit_it[1]+1][1]==0)[0][0]
                        # merge the hit contributions:
                        for comb_it in range(f):
                            hit_contributions[...,:-1][hit_it][1][e+comb_it] = hit_contributions[...,:][hit_it[0],hit_it[1]+1][1][comb_it]
                            hit_contributions[...,:-1][hit_it][0][e+comb_it] = hit_contributions[...,:][hit_it[0],hit_it[1]+1][0][comb_it]
                            hit_contributions[...,:-1][hit_it][2][e+comb_it] = hit_contributions[...,:][hit_it[0],hit_it[1]+1][2][comb_it]
                            # and remove them from the hit that was merged in
                            hit_contributions[hit_it[0],hit_it[1]+1][0][comb_it] = 0.
                            hit_contributions[hit_it[0],hit_it[1]+1][1][comb_it] = 0.
                            hit_contributions[hit_it[0],hit_it[1]+1][2][comb_it] = -1

                # now we mask off hits that have already been merged
                mask[...,1:] = mask[...,1:] | to_merge

                # and track the hit ids of the hits that were merged by propogating the indices forward
                if mode == 'last-first':
                    old_id_mask = np.concatenate([old_id_mask[...,0:1], old_id_mask], axis=-1)
                    old_ids = np.concatenate([old_ids[...,0:1], old_ids], axis=-1)
                    id_merge = np.broadcast_to(to_merge[...,np.newaxis], to_merge.shape + old_ids.shape[-1:])
                    divider = 1
                elif mode == 'pairwise':
                    old_id_mask = np.concatenate([old_id_mask, old_id_mask], axis=-1)
                    old_ids = np.concatenate([old_ids, old_ids], axis=-1)
                    id_merge = np.broadcast_to(to_merge[...,np.newaxis], to_merge.shape + old_ids.shape[-1:])
                    divider = old_ids.shape[-1]//2
                else:
                    raise RuntimeError(f'invalid mode {mode}')
                # move ids from hit[i+1] to hit[i] (while keeping the ids for hit[i])
                np.place(old_ids[...,:-1,divider:], id_merge[...,divider:], np.extract(id_merge[...,divider:], old_ids[...,1:,divider:]))
                # copy the id mask for hit[i+1] into hit[i] (while keeping the id mask for hit[i])
                np.place(old_id_mask[...,:-1,divider:], id_merge[...,divider:], np.extract(id_merge[...,divider:], old_id_mask[...,1:,divider:]))
                # and clear the id mask for hit[i+1]
                np.place(old_id_mask[...,1:,:], id_merge, True)
            else:
                break

        # calculate segment contributions for each merged hit
        if has_mc_truth:
            back_track = np.full(shape=new_hits.shape,fill_value=0.,dtype=self.hit_frac_dtype)
            # loop over hits
            for hit_it, hit in np.ndenumerate(new_hits):
                if mask[hit_it]: continue
                hit_contr = hit_contributions[hit_it]

                # renormalize the fractional contributions given the charge weighted average
                # YC, 2024-06-03: I think we should check this norm is consistent with the sum of Q from all contributed prompt hits
                norm = np.sum(np.multiply(hit_contr[0],hit_contr[1]))
                if norm == 0.: norm = 1.
                tmp_bt_0 = np.multiply(hit_contr[0],hit_contr[1])/norm # fractional contributions
                tmp_bt_1 = hit_contr[2] # segment_ids

                # merge unique track contributions
                track_dict = defaultdict(lambda:0)
                for track in zip(tmp_bt_0,tmp_bt_1):
                    track_dict[track[1]] += track[0]
                track_dict = dict(track_dict)
                bt_unique_segs = np.array(list(track_dict.keys()))
                bt_unique_frac = np.array(list(track_dict.values()))
                n_conts = bt_unique_frac.shape[0]
                isort = np.flip(np.argsort(np.abs(bt_unique_frac), axis=-1, kind='stable'))
                bt_unique_segs = np.take_along_axis(bt_unique_segs, isort, axis=-1)
                bt_unique_frac = np.take_along_axis(bt_unique_frac, isort, axis=-1)
                back_track[hit_it]['fraction'] = [0.]*self.max_contrib_segments
                back_track[hit_it]['segment_ids'] = [-1]*self.max_contrib_segments
                back_track[hit_it]['fraction'][:bt_unique_frac.shape[0]] = bt_unique_frac
                back_track[hit_it]['segment_ids'][:bt_unique_segs.shape[0]] = bt_unique_segs
        else: back_track = None

        new_hit_idx = np.broadcast_to(np.cumsum(~mask.ravel(), axis=0).reshape(mask.shape + (1,)), old_ids.shape)-1

        return (
            ma.array(new_hits, mask=mask),
            np.c_[np.extract(~(old_id_mask | mask[...,np.newaxis]), old_ids), np.extract(~(old_id_mask | mask[...,np.newaxis]), new_hit_idx)],
            ma.array(back_track, mask=mask) if back_track is not None else None
            )

    def run(self, source_name, source_slice, cache):
        super(CalibHitMerger, self).run(source_name, source_slice, cache)

        event_id = np.r_[source_slice]
        packet_frac_bt = cache['packet_frac_backtrack']
        hits = cache[self.hits_name]

        merged, ref, back_track = self.merge_hits(hits, weights=hits['Q'], seg_fracs=packet_frac_bt,dt_cut=self.merge_cut, sum_fields=self.sum_fields, weighted_mean_fields=self.weighted_mean_fields, max_steps=self.max_merge_steps, mode=self.merge_mode)

        merged_mask = merged.mask['id']

        # first write the new merged hits to the file
        new_nhit = int((~merged_mask).sum())
        merge_slice = self.data_manager.reserve_data(self.merged_name, new_nhit)
        merge_idx = np.r_[merge_slice].astype(merged.dtype['id'])
        if new_nhit > 0:
            ref[:,1] += merge_idx[0] # offset references based on reserved region in output file
            np.place(merged['id'], ~merged_mask, merge_idx)

        self.data_manager.write_data(self.merged_name, merge_idx, merged[~merged_mask])

        # HACK: Remove duplicate refs. Would be nice to actually understand and
        # fix the origin of these duplicates.
        ref = np.unique(ref, axis=0)
        # sort based on the ID of the prompt hit, to make analysis more convenient
        ref = ref[np.argsort(ref[:, 0])]
        self.data_manager.write_ref(self.hits_name, self.merged_name, ref)

        # if back tracking information was available, write the merged back tracking
        # dataset to file 
        if back_track is not None:
            merge_bt_slice = self.data_manager.reserve_data(self.mc_hit_frac_dset_name, new_nhit)
            self.data_manager.write_data(self.mc_hit_frac_dset_name, merge_idx, back_track[~merged_mask])
            self.data_manager.write_ref(self.merged_name,self.mc_hit_frac_dset_name,np.c_[merge_idx,merge_idx])

        ev_ref = np.c_[(np.indices(merged_mask.shape)[0] + source_slice.start)[~merged_mask], merge_idx]
        self.data_manager.write_ref(source_name, self.merged_name, ev_ref)
        self.data_manager.write_ref(self.events_dset_name, self.merged_name, ev_ref)
