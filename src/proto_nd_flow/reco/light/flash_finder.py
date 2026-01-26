import numpy as np
import numpy.ma as ma
from collections import defaultdict

from h5flow.core import H5FlowStage, resources
from h5flow.data import dereference

import sklearn.cluster as cluster

import proto_nd_flow.util.units as units

VERBOSE = False

class FlashFinder(H5FlowStage):
    class_version = '1.0.0'

    defaults = dict(
        cwvfm_dset_name = 'light/cwvfm',
        sipm_hits_dset_name = 'light/sipm_hits',
        sum_hits_dset_name = 'light/sum_hits',
        flash_dset_name = 'light/flash',
        flash_sipm_dset_name = 'light/flash_sipm',
        eps = 5,
        min_samples = 1,
        nchantpc = 48
    )

    def flash_dtype(self, nchantpc):
        return np.dtype([
            ('id', 'u4'),
            ('tpc', 'u1'),
            ('n_sum_hits', 'u4'),
            ('sample_range', 'u2', (2,)),
            ('hit_time_range', 'f4', (2,)),
            ('rising_spline_range', 'f4', (2,)),
            ('tot_sum', 'f4'),
            ('tot_max', 'f4'),
            ('tot_sum_spline', 'f4'),
            ('tot_max_spline', 'f4'),
            ('sum_pe_ch', 'f4', (2, nchantpc//2)),
            ('max_pe_ch', 'f4', (2, nchantpc//2))
        ])

    def flash_sipm_dtype(self, nchantpc):
        return np.dtype([
            ('id', 'u4'),
            ('tpc', 'u1'),
            ('n_sipm_hits', 'u4'),
            ('sample_range', 'u2', (2,)),
            ('hit_time_range', 'f4', (2,)),
            ('rising_spline_range', 'f4', (2,)),
            ('tot_sum', 'f4'),
            ('tot_max', 'f4'),
            ('tot_sum_spline', 'f4'),
            ('tot_max_spline', 'f4'),
            ('sum_pe_ch', 'f4', (2, nchantpc//2)),
            ('max_pe_ch', 'f4', (2, nchantpc//2))
        ])

    def __init__(self, **params):
        super(FlashFinder, self).__init__(**params)
        
        # set up parameters
        for key, val in self.defaults.items():
            setattr(self, key, params.get(key, val))

        self.flash_dtype = self.flash_dtype(self.nchantpc)
        self.flash_sipm_dtype = self.flash_sipm_dtype(self.nchantpc)

    def init(self, source_name):
        super(FlashFinder, self).init(source_name)

        cwvfm_dset = self.data_manager.get_dset(self.cwvfm_dset_name)
        self.sum_hits_dset = self.data_manager.get_dset(self.sum_hits_dset_name)
        self.sipm_hits_dset = self.data_manager.get_dset(self.sipm_hits_dset_name)

        self.dbs = cluster.DBSCAN(eps=self.eps, min_samples=self.min_samples)
        
        # get waveform shape information
        self.nadc = cwvfm_dset.dtype['samples'].shape[0]
        self.nchan = cwvfm_dset.dtype['samples'].shape[1]
        self.ntpc = self.nadc
        self.nsamples = cwvfm_dset.dtype['samples'].shape[2]
        
        # Load channel map
        self.rel_pos_map = np.zeros((self.nadc, self.nchan, 3))
        for adc in range(self.nadc):
            self.rel_pos_map[adc, :, :] = resources['Geometry'].sipm_rel_pos[(adc, range(self.nchan))]
        
        # create datasets and references
        self.data_manager.create_dset(self.flash_dset_name, dtype=self.flash_dtype)
        self.data_manager.create_dset(self.flash_sipm_dset_name, dtype=self.flash_sipm_dtype)

        self.data_manager.create_ref(source_name, self.flash_dset_name)
        self.data_manager.create_ref(source_name, self.flash_sipm_dset_name)
        self.data_manager.create_ref(self.sum_hits_dset_name, self.flash_dset_name)
        self.data_manager.create_ref(self.sipm_hits_dset_name, self.flash_sipm_dset_name)
        self.data_manager.set_attrs(self.flash_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    sum_hits_dset=self.sum_hits_dset_name,
                                    sipm_hits_dset=self.sipm_hits_dset_name
                                   )

    def get_tpc_channels(self,itpc):
        '''
        Returns array with (adc,channel) indices for given TPC
        The returnd array has size (2,nchantpc//2,2) where
        the first index is the side of the tpc (looking from cathode to anode, 0:left,1:right)
        and the second is the vertical position from bottom to top.
        The third axis is the index for adc (0) or channel (1).
        '''
        return_arr = np.zeros((2,self.nchantpc//2,2),dtype='i2')
        for iside in range(2):
            indices = np.where((self.rel_pos_map[..., 0] == itpc) & (self.rel_pos_map[..., 1] == iside))
            indexed_values = list(zip(indices[0], indices[1]))
            indexed_values.sort(key=lambda x: self.rel_pos_map[x[0], x[1], 2])
            return_arr[iside,:,:] = np.array(indexed_values)

        return(return_arr)
                                        
    def get_tpc_mask(self, adc, chan, itpc):
        return (self.rel_pos_map[adc, chan, 0] == itpc)

    def get_extrema(self, input_array):
        return np.array([input_array.min(), input_array.max()])
        
    def run(self, source_name, source_slice, cache):
        super(FlashFinder, self).run(source_name, source_slice, cache)
        events = cache[source_name]
        cwvfms = cache[self.cwvfm_dset_name].reshape(events.shape)['samples']
     
        #Get assosciate hits for events slice
        sum_hits, sum_hits_idx = self._load_hit_data(source_name, source_slice, self.sum_hits_dset_name, self.sum_hits_dset)
        sipm_hits, sipm_hits_idx = self._load_hit_data(source_name, source_slice, self.sipm_hits_dset_name, self.sipm_hits_dset)

        sum_flash_list, sum_ev_ref_list, sum_hit_ref_list = [], [], []
        sipm_flash_list, sipm_ev_ref_list, sipm_hit_ref_list = [], [], []
      
        if VERBOSE: print("# events in slice: ",len(events))
        for i, ev in enumerate(events):
            for itpc in range(self.ntpc):
                # Sum Hit flashes
                mask = (sum_hits[i,:]['tpc'] == itpc)
                self._process_sum_flash(i, itpc, mask, sum_hits, sum_hits_idx, cwvfms, source_slice, sum_flash_list, sum_ev_ref_list, sum_hit_ref_list)

                # SiPM Hit flashes
                sipm_hit_block = sipm_hits[i,:]
                adc_arr = sipm_hit_block['adc'].astype(int)
                chan_arr = sipm_hit_block['chan'].astype(int)
                sipm_mask = (self.rel_pos_map[adc_arr, chan_arr, 0] == itpc)
                self._process_sipm_flash(i, itpc, sipm_mask, sipm_hits, sipm_hits_idx, cwvfms, source_slice, sipm_flash_list, sipm_ev_ref_list, sipm_hit_ref_list)

        self._finalize_flash(self.flash_dset_name, sum_flash_list, sum_ev_ref_list, sum_hit_ref_list, source_name, self.sum_hits_dset_name)
        self._finalize_flash(self.flash_sipm_dset_name, sipm_flash_list, sipm_ev_ref_list, sipm_hit_ref_list, source_name, self.sipm_hits_dset_name)

    def _load_hit_data(self, source_name, source_slice, dset_name, dset):
        ref_dset, ref_dir = self.data_manager.get_ref(source_name, dset_name)
        ref_region = self.data_manager.get_ref_region(source_name, dset_name)
        idx = dereference(source_slice, ref_dset, region=ref_region, ref_direction=ref_dir, indices_only=True)
        data = dereference(source_slice, ref_dset, data=dset, region=ref_region, ref_direction=ref_dir)
        return data, idx

    def _process_sum_flash(self, i, itpc, mask, sum_hits, sum_hits_idx, cwvfms, source_slice, flash_list, ev_ref_list, hit_ref_list):
        tpc_hits = sum_hits[i][mask]
        tpc_hits_idx = sum_hits_idx[i][mask]
        if not len(tpc_hits): return
        labels = self.dbs.fit_predict(tpc_hits['sample_idx'].reshape(-1, 1))
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # skip if there's no light clusters
        if n_clusters == 0: return

        flashes = np.empty(n_clusters, dtype=self.flash_dtype)
        ev_ref = np.full(n_clusters, np.r_[source_slice][i], dtype='u4')
        sum_ref = np.empty((len(tpc_hits_idx), 2), dtype='u4')

        if VERBOSE:
            print("    TPC #",itpc," #Clusters ",n_clusters)
            print("       #Sum Hits    ",sum_hits[i,mask].shape)
            print("       Sum Hits:    ",sum_hits[i,mask]["sample_idx"])
            print("       Sum Hits IDs:    ",sum_hits[i,mask]["id"])
            print("       Labels:  ",labels)
        #Handle clusters
        # Note: Clusters pre-sorted in time by DBSCAN
        for cl in range(n_clusters):
            cl_hits = tpc_hits[labels == cl]
            flashes[cl]['id'] = 0
            flashes[cl]['tpc'] = itpc
            flashes[cl]['n_sum_hits'] = len(cl_hits)
            flashes[cl]['sample_range'] = self.get_extrema(cl_hits['sample_idx'])
            flashes[cl]['hit_time_range'] = self.get_extrema(cl_hits['busy_ns'])
            flashes[cl]['rising_spline_range'] = self.get_extrema(cl_hits['busy_ns'] + cl_hits['rising_spline'])
            flashes[cl]['tot_sum'] = cl_hits['sum'].sum()
            flashes[cl]['tot_max'] = cl_hits['max'].sum()
            flashes[cl]['tot_sum_spline'] = cl_hits['sum_spline'].sum()
            flashes[cl]['tot_max_spline'] = cl_hits['max_spline'].sum()

            ch_idx = self.get_tpc_channels(itpc)
            flash_slice = slice(flashes[cl]['sample_range'][0], flashes[cl]['sample_range'][1] + 1)
            flashes[cl]['sum_pe_ch'] = np.sum(cwvfms[i, ch_idx[..., 0], ch_idx[..., 1], flash_slice], axis=-1)
            flashes[cl]['max_pe_ch'] = np.max(cwvfms[i, ch_idx[..., 0], ch_idx[..., 1], flash_slice], axis=-1)

        sum_ref[:, 0] = tpc_hits_idx[labels >= 0]
        sum_ref[:, 1] = labels[labels >= 0]

        flash_list.append(flashes)
        ev_ref_list.append(ev_ref)
        hit_ref_list.append(sum_ref)

    def _process_sipm_flash(self, i, itpc, mask, sipm_hits, sipm_hits_idx, cwvfms, source_slice, flash_list, ev_ref_list, hit_ref_list):
        tpc_hits = sipm_hits[i][mask]
        tpc_hits_idx = sipm_hits_idx[i][mask]
        if not len(tpc_hits): return

        labels = self.dbs.fit_predict(tpc_hits['sample_idx'].reshape(-1, 1))
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # skip if there's no light hits
        if n_clusters == 0: return

        flashes = np.empty(n_clusters, dtype=self.flash_sipm_dtype)
        ev_ref = np.full(n_clusters, np.r_[source_slice][i], dtype='u4')
        hit_ref = np.empty((len(tpc_hits_idx), 2), dtype='u4')
        self.tpc_chan_to_local = dict()
        chan_counter = defaultdict(int)  # counts channels per (tpc, side)
        
        # for adc in range(self.rel_pos_map.shape[0]):
        #     for chan in range(self.rel_pos_map.shape[1]):
        #         tpc = int(self.rel_pos_map[adc, chan, 0])
        #         side = int(self.rel_pos_map[adc, chan, 1])
        #         pos =  int(self.rel_pos_map[adc, chan, 2])
        #         y = resources['Geometry'].sipm_abs_pos[(adc, chan)][0][1]
        #         if tpc < 0:
        #             continue
        #         local_idx = chan_counter[(tpc, side)]  # integer count
        #         print(tpc, side, pos, resources['Geometry'].sipm_abs_pos[(adc, chan)], adc, chan, local_idx)
        #         self.tpc_chan_to_local[(tpc, side, y, adc, chan)] = (side, local_idx)
        
        #         chan_counter[(tpc, side)] += 1
      
        if VERBOSE:
            print("    TPC #",itpc," #Clusters ",n_clusters)
            print("       #SiPM Hits    ",sipm_hits[i,mask].shape)
            print("       SiPM Hits:    ",sipm_hits[i,mask]["sample_idx"])
            print("       SiPM Hits IDs:    ",sipm_hits[i,mask]["id"])
            print("       Labels:  ",labels)
        
         #Handle clusters
        # Note: Clusters pre-sorted in time by DBSCAN
        for cl in range(n_clusters):
            cl_hits = tpc_hits[labels == cl]
            flashes[cl]['id'] = 0
            flashes[cl]['tpc'] = itpc
            flashes[cl]['n_sipm_hits'] = len(cl_hits)
            flashes[cl]['sample_range'] = self.get_extrema(cl_hits['sample_idx'])
            flashes[cl]['hit_time_range'] = self.get_extrema(cl_hits['busy_ns'])
            flashes[cl]['rising_spline_range'] = self.get_extrema(cl_hits['busy_ns'] + cl_hits['rising_spline'])
            flashes[cl]['tot_sum'] = cl_hits['sum'].sum()
            flashes[cl]['tot_max'] = cl_hits['max'].sum()
            flashes[cl]['tot_sum_spline'] = cl_hits['sum_spline'].sum()
            flashes[cl]['tot_max_spline'] = cl_hits['max_spline'].sum()
            
            #get pe per sipm information
            sum_pe = np.zeros((2, self.nchantpc // 2))
            max_pe = np.zeros((2, self.nchantpc // 2))
            
            # flash_slice = slice(flashes[cl]['sample_range'][0], flashes[cl]['sample_range'][1] + 1)
            
            # for h in cl_hits:
            #     adc = h['adc']
            #     chan = h['chan']
            #     tpc = int(self.rel_pos_map[adc, chan, 0])
            #     side = int(self.rel_pos_map[adc, chan, 1])
            
            #     side_idx, local_idx = self.tpc_chan_to_local[(tpc, side, adc, chan)]
            #     wf = cwvfms[i, adc, chan, flash_slice]
            
            #     sum_pe[side_idx, local_idx] = np.sum(wf)
            #     max_pe[side_idx, local_idx] = np.max(wf)
            
            # flashes[cl]['sum_pe_ch'] = sum_pe
            # flashes[cl]['max_pe_ch'] = max_pe
           # Step 1: get normal channel index array
            # ch_idx = self.get_tpc_channels(itpc)  # shape: (2, nchan_per_side, 2)
            
            # flash_slice = slice(flashes[cl]['sample_range'][0], flashes[cl]['sample_range'][1] + 1)
            # adc_arr = sipm_hits[i]['adc']
            # chan_arr = sipm_hits[i]['chan']
            
            # # Masked arrays: filter only valid entries (non-masked)
            # valid_mask = ~(ma.getmaskarray(adc_arr) | ma.getmaskarray(chan_arr))
            
            # valid_adc = adc_arr[valid_mask]
            # valid_chan = chan_arr[valid_mask]
            
            # valid_pairs = set(zip(valid_adc, valid_chan))
           
            # # Step 3: make a boolean mask with same shape as first two dims of ch_idx
            # ch_mask = np.array(
            #     [[(adc, chan) in valid_pairs for adc, chan in row] for row in ch_idx],
            #     dtype=bool
            # )
            
            # # Step 4: get sums/max as usual
            # sum_pe_ch = np.sum(cwvfms[i, ch_idx[..., 0], ch_idx[..., 1], flash_slice], axis=-1)
            # max_pe_ch = np.max(cwvfms[i, ch_idx[..., 0], ch_idx[..., 1], flash_slice], axis=-1)
            
            # # Step 5: zero out channels not in sipm_hits
            # sum_pe_ch[~ch_mask] = 0
            # max_pe_ch[~ch_mask] = 0



            # Step 6: store
 
            # sum_pe = np.zeros((2, self.nchantpc // 2))
            # max_pe = np.zeros((2, self.nchantpc // 2))
            
            # for h in cl_hits:
            #     adc = h['adc']
            #     chan = h['chan']
            #     tpc = int(self.rel_pos_map[adc, chan, 0])
            #     side = int(self.rel_pos_map[adc, chan, 1])
                
            #     key = (tpc, side, adc, chan)
            #     if key not in self.tpc_chan_to_local:
            #         continue
                
            #     side_idx, local_idx = self.tpc_chan_to_local[key]

            #     sum_pe[side_idx, local_idx] += h['sum']
            #     max_pe[side_idx, local_idx] += h['max']
            
            # # flashes[cl]['sum_pe_ch'] = sum_pe
            # # flashes[cl]['max_pe_ch'] = max_pe


            ch_idx = self.get_tpc_channels(itpc)
            flash_slice = slice(flashes[cl]['sample_range'][0], flashes[cl]['sample_range'][1] + 1)
            
            # Shape: (2, nchans_per_side)
            sum_pe = np.sum(cwvfms[i, ch_idx[..., 0], ch_idx[..., 1], flash_slice], axis=-1)
            max_pe = np.max(cwvfms[i, ch_idx[..., 0], ch_idx[..., 1], flash_slice], axis=-1)
            
        
            # Get ordering for each side based on Y position
            for side in (0, 1):
                # Extract adc and chan for this side
                adc_side = ch_idx[side, :, 0]
                chan_side = ch_idx[side, :, 1]
            
                # Get Y positions
                y_pos = np.array([resources['Geometry'].sipm_abs_pos[(adc, chan)][0][1]
                                  for adc, chan in zip(adc_side, chan_side)])
                if itpc==0: print(y_pos)
                # Sort by Y (bottom to top)
                order = np.argsort(y_pos)
            
                # Apply ordering
                sum_pe[side] = sum_pe[side][order]
                max_pe[side] = max_pe[side][order]

            
            flashes[cl]['sum_pe_ch'] = sum_pe
            flashes[cl]['max_pe_ch'] = max_pe
            # print(sum_pe, flashes[cl]['sum_pe_ch'])
        hit_ref[:, 0] = tpc_hits_idx[labels >= 0]
        hit_ref[:, 1] = labels[labels >= 0]

        flash_list.append(flashes)
        ev_ref_list.append(ev_ref)
        hit_ref_list.append(hit_ref)

    def _finalize_flash(self, dset_name, flash_list, ev_list, hit_list, source_name, hit_name):
        if not flash_list: return
        flash_data = np.concatenate(flash_list)
        flash_slice = self.data_manager.reserve_data(dset_name, len(flash_data))
        flash_data['id'] = np.r_[flash_slice]
        self.data_manager.write_data(dset_name, flash_slice, flash_data)

        ev_data = np.concatenate(ev_list)
        ref_ev = np.array([(ev, flash_data[i]['id']) for i, ev in enumerate(ev_data)])
        self.data_manager.write_ref(source_name, dset_name, ref_ev)

        split_data = np.split(flash_data, np.cumsum([arr.shape[0] for arr in flash_list])[:-1])
        for j, flash in enumerate(split_data):
            hit_list[j] = np.c_[hit_list[j][:, 0], flash['id'][hit_list[j][:, 1]]]
        ref_hits = np.concatenate(hit_list)
        self.data_manager.write_ref(hit_name, dset_name, ref_hits)


