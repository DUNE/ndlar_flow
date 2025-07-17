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
            ('deconv_sum', 'f4', (2, nchantpc//2)),
            ('deconv_max', 'f4', (2, nchantpc//2))
        ])

    def flash_sipm_dtype(self, nchan):
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
            ('sum_pe_ch', 'f4', nchan),
            ('max_pe_ch', 'f4', nchan)
        ])

    def __init__(self, **params):
        super().__init__(**params)
        for key, val in self.defaults.items():
            setattr(self, key, params.get(key, val))

        self.flash_dtype = self.flash_dtype(self.nchantpc)
        self.flash_sipm_dtype = self.flash_sipm_dtype(self.nchantpc)

    def init(self, source_name):
        super().init(source_name)

        cwvfm_dset = self.data_manager.get_dset(self.cwvfm_dset_name)
        self.sum_hits_dset = self.data_manager.get_dset(self.sum_hits_dset_name)
        self.sipm_hits_dset = self.data_manager.get_dset(self.sipm_hits_dset_name)

        self.dbs = cluster.DBSCAN(eps=self.eps, min_samples=self.min_samples)

        self.nadc = cwvfm_dset.dtype['samples'].shape[0]
        self.nchan = cwvfm_dset.dtype['samples'].shape[1]
        self.ntpc = self.nadc
        self.nsamples = cwvfm_dset.dtype['samples'].shape[2]

        self.rel_pos_map = np.zeros((self.nadc, self.nchan, 3))
        for adc in range(self.nadc):
            self.rel_pos_map[adc, :, :] = resources['Geometry'].sipm_rel_pos[(adc, range(64))]

        self.data_manager.create_dset(self.flash_dset_name, dtype=self.flash_dtype)
        self.data_manager.create_dset(self.flash_sipm_dset_name, dtype=self.flash_sipm_dtype)

        self.data_manager.create_ref(source_name, self.flash_dset_name)
        self.data_manager.create_ref(source_name, self.flash_sipm_dset_name)
        self.data_manager.create_ref(self.sum_hits_dset_name, self.flash_dset_name)
        self.data_manager.create_ref(self.sipm_hits_dset_name, self.flash_sipm_dset_name)

    def get_tpc_mask(self, adc, chan, itpc):
        return (self.rel_pos_map[adc, chan, 0] == itpc)

    def get_extrema(self, input_array):
        return np.array([input_array.min(), input_array.max()])
    
    def get_tpc_channels(self, itpc):
        return_arr = np.zeros((2, self.nchantpc//2, 2), dtype='i2')
        for iside in range(2):
            indices = np.where((self.rel_pos_map[..., 0] == itpc) & (self.rel_pos_map[..., 1] == iside))
            indexed = list(zip(indices[0], indices[1]))
            indexed.sort(key=lambda x: self.rel_pos_map[x[0], x[1], 2])
            return_arr[iside, :, :] = np.array(indexed)
        return return_arr
        
    def run(self, source_name, source_slice, cache):
        super().run(source_name, source_slice, cache)
        events = cache[source_name]
        cwvfms = cache[self.cwvfm_dset_name].reshape(events.shape)['samples']

        sum_hits, sum_hits_idx = self._load_hit_data(source_name, source_slice, self.sum_hits_dset_name, self.sum_hits_dset)
        sipm_hits, sipm_hits_idx = self._load_hit_data(source_name, source_slice, self.sipm_hits_dset_name, self.sipm_hits_dset)

        sum_flash_list, sum_ev_ref_list, sum_hit_ref_list = [], [], []
        sipm_flash_list, sipm_ev_ref_list, sipm_hit_ref_list = [], [], []

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
                self._process_sipm_flash(i, itpc, sipm_mask, sipm_hits, sipm_hits_idx, source_slice, sipm_flash_list, sipm_ev_ref_list, sipm_hit_ref_list)

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
        if n_clusters == 0: return

        flashes = np.empty(n_clusters, dtype=self.flash_dtype)
        ev_ref = np.full(n_clusters, np.r_[source_slice][i], dtype='u4')
        sum_ref = np.empty((len(tpc_hits_idx), 2), dtype='u4')

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
            flashes[cl]['deconv_sum'] = np.sum(cwvfms[i, ch_idx[..., 0], ch_idx[..., 1], flash_slice], axis=-1)
            flashes[cl]['deconv_max'] = np.max(cwvfms[i, ch_idx[..., 0], ch_idx[..., 1], flash_slice], axis=-1)

        sum_ref[:, 0] = tpc_hits_idx[labels >= 0]
        sum_ref[:, 1] = labels[labels >= 0]

        flash_list.append(flashes)
        ev_ref_list.append(ev_ref)
        hit_ref_list.append(sum_ref)

    def _process_sipm_flash(self, i, itpc, mask, sipm_hits, sipm_hits_idx, source_slice, flash_list, ev_ref_list, hit_ref_list):
        tpc_hits = sipm_hits[i][mask]
        tpc_hits_idx = sipm_hits_idx[i][mask]
        if not len(tpc_hits): return

        labels = self.dbs.fit_predict(tpc_hits['sample_idx'].reshape(-1, 1))
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters == 0: return

        flashes = np.empty(n_clusters, dtype=self.flash_sipm_dtype)
        ev_ref = np.full(n_clusters, np.r_[source_slice][i], dtype='u4')
        hit_ref = np.empty((len(tpc_hits_idx), 2), dtype='u4')
        
        self.tpc_chan_to_local = dict()
        chan_counter = defaultdict(int)
        
        for adc in range(self.rel_pos_map.shape[0]):
            for chan in range(self.rel_pos_map.shape[1]):
                tpc = int(self.rel_pos_map[adc, chan, 0])
                if tpc >= 0:
                    local_idx = chan_counter[tpc]
                    self.tpc_chan_to_local[(tpc, adc, chan)] = local_idx
                    chan_counter[tpc] += 1
                    
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

            sum_pe = np.zeros(self.nchantpc)
            max_pe = np.zeros(self.nchantpc)

            for h in cl_hits:
                adc = h['adc']
                chan = h['chan']
                tpc = int(self.rel_pos_map[adc, chan, 0])
                
                local_idx = self.tpc_chan_to_local.get((tpc, adc, chan), -1)
                if local_idx == -1:
                    continue
                sum_pe[local_idx] += h['sum']
                max_pe[local_idx] += h['max']
            flashes[cl]['sum_pe_ch'] = sum_pe
            flashes[cl]['max_pe_ch'] = max_pe

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


