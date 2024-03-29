import numpy as np
import numpy.ma as ma
import logging

from h5flow.core import H5FlowStage, resources

from module0_flow.reco.charge.hit_builder import HitBuilder


class HitMerger(H5FlowStage):
    '''
    Merges the specified cached hits based on their unique channel id and timestamp:
     - q -> sum(q)
     - ts -> sum(ts * q) / sum(q)

    Two algorithms for selecting pairs of hits to merge have been implemented:

     - `'pairwise'`: On each iteration, sort all packets by unique channel and timestamp. Then, merge every pair of hits that fall within the merge cut. If an odd number of hits fall should be merged, the earliest hit of a group is excluded from the iteration.
     - `'last-first'`: On each iteration, sort all packets by unique channel and timestamp. Then, merge the last pair of hits that fall within the merge cut within each contiguous chunk of neighboring hits.

    Both algorithms should produce very similar results.

    Example config::

      hit_merging:
        classname: HitMerger
        path: module0_flow.reco.charge.hit_merger
        requires:
          - 'charge/hits'
          - name: 'charge/hits_idx'
            path: ['charge/hits']
            index_only: True
        params:
          hits_name: 'charge/hits'
          hit_charge_name: 'charge/hits' # dataset to grab 'q' from
          hits_idx_name: 'charge/hits_idx'
          merged_name: 'charge/hits/merged'
          merged_q_name: 'charge/hits/merged_q' # optional, for when a separate hit charge dataset is used for 'q'
          merge_cut: 30 # merge hits with delta t < merge_cut [CRS ticks]
          max_merge_steps: 5 # maximum number of iterations to use when merging
          merge_mode: 'last-first'
    '''
    class_version = '0.0.0'
    defaults = dict(
        hits_name = 'charge/hits',
        hit_charge_name = 'charge/hits',
        hits_idx_name = 'charge/hits_idx',
        merged_name = 'charge/hits/merged',
        merged_q_name = None,
        max_merge_steps = 5,
        merge_mode = 'last-first',
        merge_cut = 30 # CRS ticks
        )
    valid_merge_modes = ['last-first', 'pairwise']

    merged_dtype = HitBuilder.hits_dtype

    sum_fields = ['q']
    weighted_mean_fields = ['ts', 'ts_raw', 'f']

    def __init__(self, **params):
        super(HitMerger, self).__init__(**params)
        for key in self.defaults:
            setattr(self, key, params.get(key, self.defaults[key]))
        self.merge_mode = self.merge_mode.lower()
        assert self.merge_mode in self.valid_merge_modes, f'invalid merge mode: {self.merge_mode}'

    def init(self, source_name):
        super(HitMerger, self).init(source_name)

        self.data_manager.create_dset(self.merged_name, dtype=self.merged_dtype)
        self.data_manager.create_ref(self.hits_name, self.merged_name)
        self.data_manager.create_ref(source_name, self.merged_name)

    @staticmethod
    def merge_hits(hits, weights, dt_cut, sum_fields=None, weighted_mean_fields=None, hit_q=None, max_steps=-1, mode='last-first'):
        '''
        Combines hits along the second axis on unique channels with a delta t less than dt_cut. Continues
        until no hits (or merged hits) are within dt_cut of each other

        :param hits: original hits array, shape: (N,M)

        :param weights: values used for weighted mean, shape: (N,M)

        :param dt_cut: delta t cut to merge hits (float) [CRS ticks]

        :sum_fields: list of fields in ``hits`` and ``hit_q`` that should be *summed* when combined, must not be in ``weighted_mean_fields``

        :weighted_mean_fields: list of fields in ``hits`` and ``hit_q`` that should be averaged using the weights when combined, must not be in ``sum_fields``

        :param hit_q: optional, hit charge array, shape: (N,M)

        :param max_steps: optional, maximum number of merges to apply to pairs of neighboring hits (<0 == no limit, 0 == skip merging, >0 == limit steps)

        :param mode: optional, merging strategy, either `'last-first'` (on each iteration merges the last hit pair) or `'pairwise'` (on each iteration merges each unique hit pair)

        :returns: new hit array, shape: (N,m), new hit charge array, shape: (N,m), and an index array with shape (L,2), [:,0] being the index into the original hit array and [:,1] being the flattened index into the compressed new array

        '''
        iteration_count = 0
        mask = hits.mask['id'].copy()
        new_hits = hits.data.copy()
        weights = weights.data.copy()
        old_ids = hits.mask['id'].copy()[...,np.newaxis]
        old_id_mask = hits.mask['id'].copy()[...,np.newaxis]
        if hit_q is not None:
            new_hit_q = hit_q.copy()
        while new_hits.size > 0 and iteration_count != max_steps:
            iteration_count += 1
            # algorithm is iterative, but typically only needs to loop a few (~2-3) times
            # so we'll spit a warning if we reach the maximum number of steps
            if iteration_count == max_steps:
                logging.info(f'Hit merging algorithm reached max step limit {max_steps}')
            
            # sort array along last axis to find groups of hits on the same channel, use a stable sort with the aim of improving performance on later iterations
            isort = np.argsort(ma.array(new_hits, mask=mask), axis=-1, order=['iogroup','iochannel','chipid','channelid','ts_raw'], kind='stable')
            mask = np.take_along_axis(mask, isort, axis=-1)
            new_hits = np.take_along_axis(new_hits, isort, axis=-1)
            weights = np.take_along_axis(weights, isort, axis=-1)
            old_ids = np.take_along_axis(old_ids, isort[...,np.newaxis], axis=-2)
            old_id_mask = np.take_along_axis(old_id_mask, isort[...,np.newaxis], axis=-2)
            if hit_q is not None:
                new_hit_q = np.take_along_axis(new_hit_q, isort, axis=-1)
            
            # identify neighboring hits on the same channel
            dt = np.abs(np.diff(new_hits['ts_raw'].astype(int), axis=-1))
            same_channel = (
                (new_hits['iogroup'][..., :-1] == new_hits['iogroup'][..., 1:])
                & (new_hits['iochannel'][..., :-1] == new_hits['iochannel'][..., 1:])
                & (new_hits['chipid'][..., :-1] == new_hits['chipid'][..., 1:])
                & (new_hits['channelid'][..., :-1] == new_hits['channelid'][..., 1:])
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

            # exits loop if no remaining hits to combine
            if np.any(to_merge):
                # move 2nd hit into position of first hit, combining attributes along the way
                hit0 = np.extract(to_merge, new_hits[...,:-1])
                hit1 = np.extract(to_merge, new_hits[...,1:])
                if hit_q is not None:
                    hit_q0 = np.extract(to_merge, new_hit_q[...,:-1])
                    hit_q1 = np.extract(to_merge, new_hit_q[...,1:])

                # these fields will be summed hit[i][field] -> hit[i+1][field] + hit[i][field]
                for field in sum_fields:
                    if field in new_hits.dtype.names:
                        np.place(new_hits[...,:-1][field], to_merge, hit0[field] + hit1[field])
                    if hit_q is not None and field in new_hit_q.dtype.names:
                        np.place(new_hit_q[...,:-1][field], to_merge, hit_q0[field] + hit_q1[field])

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
                    if hit_q is not None and field in new_hit_q.dtype.names:
                        base = np.minimum(hit_q0[field], hit_q1[field]) # improves precision of weighted sum if values are large (e.g. timestamps)
                        np.place(new_hit_q[...,:-1][field], to_merge, ((hit_q0[field]-base) * w0 + (hit_q1[field]-base) * w1).astype(hit_q.dtype[field]) + base)
                # combine weights for next iteration
                np.place(weights[...,:-1], to_merge, weights[...,:-1] + weights[...,1:])

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

        new_hit_idx = np.broadcast_to(np.cumsum(~mask.ravel(), axis=0).reshape(mask.shape + (1,)), old_ids.shape)-1

        return (
            ma.array(new_hits, mask=mask),
            np.c_[np.extract(~(old_id_mask | mask[...,np.newaxis]), old_ids), np.extract(~(old_id_mask | mask[...,np.newaxis]), new_hit_idx)],
            ma.array(new_hit_q, mask=mask) if hit_q is not None else None
            )
    

    def run(self, source_name, source_slice, cache):
        super(HitMerger, self).run(source_name, source_slice, cache)

        event_id = np.r_[source_slice]
        hits = cache[self.hits_name]
        hit_q = cache[self.hit_charge_name].reshape(hits.shape)
        hits_idx = cache[self.hits_idx_name].reshape(hits.shape)

        merged, ref, merged_q = self.merge_hits(hits, weights=hit_q['q'], dt_cut=self.merge_cut, sum_fields=self.sum_fields, weighted_mean_fields=self.weighted_mean_fields,
                                                hit_q=hit_q if self.merged_q_name else None, max_steps=self.max_merge_steps, mode=self.merge_mode)
        merged_mask = merged.mask['id']

        # first write the new merged hits to the file
        new_nhit = int((~merged_mask).sum())
        merge_slice = self.data_manager.reserve_data(self.merged_name, new_nhit)
        merge_idx = np.r_[merge_slice].astype(merged.dtype['id'])
        if new_nhit > 0:
            ref[:,1] += merge_idx[0] # offset references based on reserved region in output file
            np.place(merged['id'], ~merged_mask, merge_idx)
        self.data_manager.write_data(self.merged_name, merge_idx, merged[~merged_mask])

        # then if we need a separate charge dataset, write that
        if self.merged_q_name:
            self.data_manager.reserve_data(self.merged_q_name, new_nhit)
            if new_nhit > 0:
                np.place(merged_q['id'], ~merged_mask, merge_idx)
            self.data_manager.write_data(self.merged_q_name, merge_idx, merged_q[~merged_mask])
            self.data_manager.write_ref(self.merged_name, self.merged_q_name, np.c_[merge_idx, merge_idx])
        
        # finally, write the event -> hit references
        self.data_manager.write_ref(self.hits_name, self.merged_name, ref)
        ev_ref = np.c_[(np.indices(merged_mask.shape)[0] + source_slice.start)[~merged_mask], merge_idx]
        self.data_manager.write_ref(source_name, self.merged_name, ev_ref)
