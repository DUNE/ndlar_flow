import numpy as np
import numpy.lib.recfunctions as rfn
from sklearn.cluster import DBSCAN

from h5flow.core import H5FlowStage, resources
import proto_nd_flow.util.units as units

class ClusterBuilder(H5FlowStage):
    '''
        Module to cluster hits using DBSCAN.
    
        Example config:
            classname: ClusterBuilder # reco/charge/cluster_finder.py
            path: proto_nd_flow.reco.charge.cluster_finder
            requires:
              - 'charge/events'
              - 'charge/calib_prompt_hits'

            params:
              # inputs
              events_dset_name: 'charge/events'
              hits_name: 'charge/calib_prompt_hits'
              eps: 2 # cm, DBSCAN clustering distance parameter
              min_samples: 1 # minimum samples per cluster for DBSCAN
              # outputs
              clusters_dset_name: 'charge/clusters'
        
        ``clusters`` datatype::

            x              f8, pixel x location (min, mid, max) [cm]
            y              f8, pixel y location (min, mid, max) [cm]
            z              f8, pixel z location (min, mid, max) [cm]
            t_drift        f8, drift time (min, mid, max)
            ts_pps         u8, PPS packet timestamp (min, mid, max) [ns]
            unix_ts        u8, unix timestamp [s]
            Q              f8, hit charge [ke-] 
            E              f8, hit energy [MeV]
            io_group       u8, io group ID (PACMAN number) [only when isolate_tpcs is True]

    '''
    class_version = '0.0.0'
    defaults = dict(
        events_dset_name = 'charge/events',
        hits_name = 'charge/calib_prompt_hits',
        clusters_dset_name = 'charge/clusters',
        eps = 2,
        min_samples = 1,
        )
    
    def __init__(self, **params):
        super(ClusterBuilder, self).__init__(**params)
        for key in self.defaults:
            setattr(self, key, params.get(key, self.defaults[key]))
        
    def init(self, source_name):
        super(ClusterBuilder, self).init(source_name)
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.clusters_dtype = np.dtype([('id', 'u4'), ('nhit', '<u4'), ('Q', '<f8'), ('E', 'f8'), 
                                        ('io_group', '<u8'), ('unix_ts', '<u8'), ('x', 'f8', (3,)), ('y', 'f8', (3,)), \
                                        ('ts_pps', 'u8', (3,)), ('z', 'f8', (3,)), ('t_drift', 'f8', (3,))])
        #else:
        #    self.clusters_dtype = np.dtype([('id', 'u4'), ('nhit', '<u4'), ('Q', '<f8'), ('E', 'f8'), 
        #                                    ('x', 'f8', (3,)), ('y', 'f8', (3,)), ('ts_pps', 'u8', (3,)), \
        #                                    ('z', 'f8', (3,)), ('t_drift', 'f8', (3,))])
        self.data_manager.create_dset(self.clusters_dset_name, dtype=self.clusters_dtype)
        self.data_manager.create_ref(self.clusters_dset_name, self.hits_name)
        self.data_manager.create_ref(self.events_dset_name, self.clusters_dset_name)
        self.data_manager.create_ref(source_name, self.clusters_dset_name)
        
    def run(self, source_name, source_slice, cache):
        super(ClusterBuilder, self).run(source_name, source_slice, cache)
        events = cache[source_name]
        hits = cache[self.hits_name][~rfn.structured_to_unstructured(cache[self.hits_name].mask).any(axis=-1)]
        hits_ref_dset, hits_ref_dir = self.data_manager.get_ref(source_name,self.hits_name)
        
        hits_ev_id = hits_ref_dset[:,0][np.isin(hits_ref_dset[:,1], hits['id'])]
        hits_unix_ts = events['unix_ts'][np.searchsorted(events['id'], hits_ev_id)]
        
        labels = np.zeros((0,), dtype='int')
        hits_temp = np.zeros((0,), dtype=hits.dtype)
        hits_ev_id_temp = np.zeros((0,))
        hits_unix_ts_temp = np.zeros((0,))
        for io_group in np.unique(hits['io_group']):
            hits_mask = (hits['io_group'] == io_group) & (~np.isnan(hits['z']) & ~np.isnan(hits['y']))
            hit_coordinates = np.hstack((hits[hits_mask]['z'][:, np.newaxis], hits[hits_mask]['y'][:, np.newaxis], hits[hits_mask]['ts_pps'][:, np.newaxis]))
            hit_coordinates[:,2] = (hit_coordinates[:,2]*resources['RunData'].crs_ticks + hits_unix_ts[hits_mask]*1e6)*resources['LArData'].v_drift/units.cm
            
            db = self.dbscan.fit(hit_coordinates)
            if len(labels):
                max_index=np.max(labels)
            else:
                max_index=0
            labels = np.concatenate((labels, np.array(db.labels_, dtype='int')+max_index))
            hits_temp = np.concatenate((hits_temp, hits[hits_mask]))
            hits_unix_ts_temp = np.concatenate((hits_unix_ts_temp, hits_unix_ts[hits_mask]))
            hits_ev_id_temp = np.concatenate((hits_ev_id_temp, hits_ev_id[hits_mask]))
        hits = hits_temp
        hits_ev_id = hits_ev_id_temp
        hits_unix_ts = hits_unix_ts_temp
        
        labels_mask = labels != -1
        hits = hits[labels_mask]
        hits_ev_id = hits_ev_id[labels_mask]
        hits_unix_ts = hits_unix_ts[labels_mask]
        labels = labels[labels_mask]
        indices_sorted = np.argsort(labels)
        labels = labels[indices_sorted]
        hits = hits[indices_sorted]
        
        n_vals = np.bincount(labels)
        n_vals_mask = n_vals != 0
        n_vals = n_vals[n_vals_mask]
        q_clusters = np.bincount(labels, weights=hits['Q'])[n_vals_mask]
        E_clusters = np.bincount(labels, weights=hits['E'])[n_vals_mask]
        
        label_indices = np.concatenate(([0], np.flatnonzero(labels[:-1] != labels[1:])+1, [len(labels)]))[1:-1]
        label_timestamps = np.split(hits['ts_pps'], label_indices)
        label_t_drift = np.split(hits['t_drift'], label_indices)
        label_x = np.split(hits['x'], label_indices)
        label_y = np.split(hits['y'], label_indices)
        label_z = np.split(hits['z'], label_indices)
        label_t = np.split(hits['ts_pps'], label_indices)
        
        min_timestamps = np.array(list(map(np.min, label_timestamps)), dtype='u8')
        max_timestamps = np.array(list(map(np.max, label_timestamps)), dtype='u8')
        mid_timestamps = ((min_timestamps+max_timestamps)/2).astype('u8')
        min_t_drift = np.array(list(map(np.min, label_t_drift)), dtype='f8')
        max_t_drift = np.array(list(map(np.max, label_t_drift)), dtype='f8')
        mid_t_drift = ((min_t_drift+max_t_drift)/2).astype('f8')
        x_min = np.array(list(map(min, label_x)))
        x_max = np.array(list(map(max, label_x)))
        x_mid = (x_min+x_max)/2
        y_min = np.array(list(map(min, label_y)))
        y_max = np.array(list(map(max, label_y)))
        y_mid = (y_min+y_max)/2
        z_min = np.array(list(map(min, label_z)))
        z_max = np.array(list(map(max, label_z)))
        z_mid = (z_min+z_max)/2
        
        clusters_data = np.zeros((len(n_vals),), dtype=self.clusters_dtype)
        clusters_data['nhit'] = n_vals
        clusters_data['Q'] = q_clusters
        clusters_data['E'] = E_clusters
        clusters_data['ts_pps'][:,0] = min_timestamps
        clusters_data['ts_pps'][:,1] = mid_timestamps
        clusters_data['ts_pps'][:,2] = max_timestamps
        clusters_data['x'][:,0] = x_min
        clusters_data['x'][:,1] = x_mid
        clusters_data['x'][:,2] = x_max
        clusters_data['y'][:,0] = y_min
        clusters_data['y'][:,1] = y_mid
        clusters_data['y'][:,2] = y_max
        clusters_data['z'][:,0] = z_min
        clusters_data['z'][:,1] = z_mid
        clusters_data['z'][:,2] = z_max
        clusters_data['t_drift'][:,0] = min_t_drift
        clusters_data['t_drift'][:,1] = mid_t_drift
        clusters_data['t_drift'][:,2] = max_t_drift
        label_io_group = np.split(hits['io_group'], label_indices)
        io_group_clusters = np.array(list(map(np.min, label_io_group)))
        clusters_data['io_group'] = io_group_clusters
        label_unix_ts = np.split(hits_unix_ts, label_indices)
        unix_ts_clusters = np.array(list(map(np.min, label_unix_ts)))
        clusters_data['unix_ts'] = unix_ts_clusters
        label_ev_id = np.split(hits_ev_id, label_indices)
        ev_id_clusters = np.array(list(map(np.min, label_ev_id)))
        
        # save clusters
        clusters_slice = self.data_manager.reserve_data(self.clusters_dset_name, len(clusters_data))
        clusters_data['id'] = clusters_slice.start + np.arange(len(clusters_data), dtype=int)
        self.data_manager.write_data(self.clusters_dset_name, clusters_slice, clusters_data)
        
        # setup references
        
        # cluster -> calib hit
        ref = np.c_[np.repeat(clusters_data['id'], n_vals), hits['id']]
        self.data_manager.write_ref(self.clusters_dset_name, self.hits_name, ref)
        #print(f'source slice = {source_slice}')
        
        # raw_event -> cluster
        ref = np.c_[ev_id_clusters, clusters_data['id']]
        self.data_manager.write_ref(source_name, self.clusters_dset_name, ref)
        # event -> cluster
        self.data_manager.write_ref(self.events_dset_name, self.clusters_dset_name, ref)
        