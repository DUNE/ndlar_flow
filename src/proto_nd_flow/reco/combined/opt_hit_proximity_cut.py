import numpy as np

from h5flow.core import H5FlowStage, resources
from tqdm import tqdm 
class OptHitProximityCut(H5FlowStage):
    '''
        Module to require charge clusters to be near optical hits
    
        Example config:
            classname: OptHitProximityCut # reco/combined/opt_hit_proximity_cut.py
            path: proto_nd_flow.reco.combined.opt_hit_proximity_cut.py
            requires:
                  - 'charge/clusters'
                  - 'light/sum_hits'

            params:
              # inputs
              events_dset_name: 'charge/events'
              clusters_dset_name: 'charge/clusters'
              ext_trigs_dset_name: 'charge/ext_trigs'
              light_events_dset_name: 'light/events'
              sum_hits_dset_name: 'light/sum_hits'
              y_cut: 15 # cm, distance from photon detector center in y
              z_cut: 30 # cm, distance from photon detector in z
              use_x_drift: True # bool, set to expect x drift in sum hits or not (maybe temporary?)
              # outputs
              new_clusters_dset_name: 'charge/clusters_optProx'

    '''
    class_version = '0.0.0'
    defaults = dict(
        events_dset_name = 'charge/events',
        clusters_dset_name = 'charge/clusters',
        sum_hits_dset_name = 'light/sum_hits',
        ext_trigs_dset_name = 'charge/ext_trigs',
        swvfms_dset_name = 'light/swvfm',
        light_events_dset_name = 'light/events',
        new_clusters_dset_name = 'charge/clusters_optProx',
        y_cut = 15,
        z_cut = 30,
        use_x_drift = True
        )
    
    def __init__(self, **params):
        super(OptHitProximityCut, self).__init__(**params)
        for key in self.defaults:
            setattr(self, key, params.get(key, self.defaults[key]))
        
    def init(self, source_name):
        super(OptHitProximityCut, self).init(source_name)
        self.clusters_dset = np.array(self.data_manager.get_dset(self.clusters_dset_name))
        self.clusters_dtype = self.clusters_dset.dtype
        
        self.clusters_ref_dset, self.clusters_ref_dir = self.data_manager.get_ref(self.events_dset_name, self.clusters_dset_name)
        self.light_events_ref_dset, self.light_events_ref_dir = self.data_manager.get_ref(self.events_dset_name,self.light_events_dset_name)
        self.sum_hits_ref_dset, self.sum_hits_ref_dir = self.data_manager.get_ref(self.light_events_dset_name,self.sum_hits_dset_name)
        
        clusters_to_light_event_id = self.light_events_ref_dset[:,1][np.isin(self.light_events_ref_dset[:,0], self.clusters_ref_dset[:,0])]
        sorted_indices = np.argsort(clusters_to_light_event_id)
        clusters_to_light_event_id = clusters_to_light_event_id[sorted_indices]
        clusters_sorted = self.clusters_dset[sorted_indices]
        unique_light_id, start_indices = np.unique(clusters_to_light_event_id, return_index=True)
        end_indices = np.roll(start_indices, shift=-1)
        end_indices[-1] = len(clusters_sorted) - 1
        
        self.light_id_chunk_indices = {}
        for val, start_idx, end_idx in zip(unique_light_id, start_indices, end_indices):
            self.light_id_chunk_indices[val] = (start_idx, end_idx)

        self.clusters_dset = clusters_sorted
        self.data_manager.create_dset(self.new_clusters_dset_name, dtype=self.clusters_dtype)
        self.data_manager.create_ref(self.new_clusters_dset_name, self.clusters_dset_name)
        self.data_manager.create_ref(self.new_clusters_dset_name, source_name)
        self.data_manager.create_ref(self.new_clusters_dset_name, self.swvfms_dset_name)
        
    def run(self, source_name, source_slice, cache):
        super(OptHitProximityCut, self).run(source_name, source_slice, cache)
        sum_hits = cache[source_name]
        
        sum_hit_ids_ref, cluster_ids_ref, swvfm_ids_ref = [], [], []
        new_clusters_data = np.zeros((0,), dtype=self.clusters_dtype)
        #clusters_keep_mask = np.zeros(len(self.clusters_dset), dtype=bool)
        #cluster_to_hitID = np.zeros(len(self.clusters_dset))
        
        #io_groups = np.unique(clusters['io_group']).astype('int')
        #io_group_masks = np.zeros((len(io_groups), len(clusters)), dtype=bool)
        #for io_group in io_groups:
        #    io_group_masks[io_group-1] = io_group == clusters['io_group']
        sum_hit_ids_ref = []
        sum_hits_ref_dset_0 = self.sum_hits_ref_dset[:,0]
        for sum_hit in sum_hits:
            #charge_event_index = light_events_ref_dset[:,0][sum_hits_ref_dset[:,0][sum_hit['id']]]
            #clusters_ref_dset[:,1][clusters_ref_dset[:,0] == charge_event_index]
            light_event_id = sum_hits_ref_dset_0[sum_hit['id']]
            try:
                cluster_indices = self.light_id_chunk_indices[light_event_id]
            except:
                continue
            clusters = self.clusters_dset[cluster_indices[0]:cluster_indices[1]]
            midpoint = (sum_hit['boundary'][1]/10 + sum_hit['boundary'][0]/10)/2
            if self.use_x_drift:
                z_center, y_center = midpoint[2], midpoint[1]
            else:
                z_center, y_center = midpoint[0], midpoint[1]
                
            if z_center > 0:
                z_mask = z_center - self.z_cut < clusters['z'][:,1]
            else:
                z_mask = z_center + self.z_cut > clusters['z'][:,1]
            
            if not np.any(z_mask):
                continue
                
            if y_center > 0:
                y_mask = (y_center - self.y_cut < clusters['y'][:,1]) & \
                         (y_center + self.y_cut > clusters['y'][:,1])
            else:
                y_mask = (y_center + self.y_cut > clusters['y'][:,1]) & \
                         (y_center - self.y_cut < clusters['y'][:,1])
        
            if not np.any(y_mask):
                continue
                
            #mask = y_mask & z_mask & io_group_masks[sum_hit['tpc']] #(sum_hit['tpc'] == clusters['io_group']+1)
            io_mask = sum_hit['tpc'] == clusters['io_group']-1
            if not np.any(io_mask):
                continue
            mask = y_mask & z_mask & io_mask

            #clusters_keep_mask[mask] = True
            #cluster_to_hitID[mask] = sum_hit['id']
            #cluster_keep_ids = clusters[mask]['id']
            clusters = clusters[mask]
            new_clusters_data = np.concatenate((new_clusters_data, clusters))
            #cluster_ids_ref += list(cluster_keep_ids)
            sum_hit_ids_ref += [sum_hit['id']]*len(clusters)
        
        #new_clusters_data = clusters[clusters_keep_mask]
        cluster_ids_ref = new_clusters_data['id']
        sum_hit_ids_ref = np.array(sum_hit_ids_ref)
        #sum_hit_ids_ref = cluster_to_hitID[clusters_keep_mask]
        
        # save new clusters dataset
        new_clusters_slice = self.data_manager.reserve_data(self.new_clusters_dset_name, len(new_clusters_data))
        new_cluster_ids_ref = np.arange(0, len(cluster_ids_ref), 1)
        new_clusters_data['id'] = new_clusters_slice.start + new_cluster_ids_ref
        print(len(new_clusters_data))
        self.data_manager.write_data(self.new_clusters_dset_name, new_clusters_slice, new_clusters_data)
        ### setup references
        # new clusters to clusters
        ref = np.hstack((new_cluster_ids_ref[:, np.newaxis], cluster_ids_ref[:, np.newaxis]))
        self.data_manager.write_ref(self.new_clusters_dset_name, self.clusters_dset_name, ref)
        
        # new clusters to sum hits
        ref = np.hstack((new_cluster_ids_ref[:, np.newaxis], sum_hit_ids_ref[:, np.newaxis]))
        self.data_manager.write_ref(self.new_clusters_dset_name, self.sum_hits_dset_name, ref)
        
        # new clusters to swvfm
        #ref = np.hstack((new_cluster_ids_ref, swvfm_ids_ref))
        #self.data_manager.write_ref(self.new_clusters_dset_name, self.swvfms_dset_name, ref)