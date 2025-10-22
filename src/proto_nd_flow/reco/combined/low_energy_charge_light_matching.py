import numpy as np
import h5py
import logging
import os
from tqdm import tqdm
import time

from h5flow.core import H5FlowStage, resources
from h5flow import H5FLOW_MPI

import proto_nd_flow.reco.charge.raw_event_generator as r

class LowEnergyChargeLightMatching(H5FlowStage):
    '''
    Match charge clusters with optical hits, requiring clusters to be in close proximity to at least 1 light hit. 
    This script is meant to run on a file created by light event reconstruction and contains the `simple_hits`
    dataset. It will search a directory for charge files that overlap in time, where the charge files
    contain clusters created by the LowEnergyEventBuilder. 

    The output is a dataset with matched clusters that satisfy optical proximity and a dataset 
    with all matched clusters only if at least one satisfies optical proximity in a given event.
    References are created for associating which clusters go to which light events and optical hits. 

    Parameters:
        - ``clusters_dset_name``: ``str``, path to input clusters
        - ``clusters_hits_dset_name``: ``str`` path to input cluster hits
        - ``clusters_matched_dset_name``: ``str`` path to output clusters that satisfy optical proximity
        - ``clusters_matched_all_dset_name``: ``str`` path to output clusters matched in time and tpc 
                                                      where at least one satisfies optical proximity
        - ``clusters_hits_matched_dset_name``: ``str`` path to output cluster hits
        - ``sum_hits_dset_name``: ``str`` path to input sum hits
        - ``pps_matching_window``: ``float`` time window in microseconds for matching light events and charge clusters
        - ``proximity_distance_z_ACL``: ``float`` minimum Z distance from simple hit location allowed for a cluster (ACL)
        - ``proximity_distance_y_ACL``: ``float`` minimum Y distance from simple hit location allowed for a cluster (ACL)
        - ``proximity_distance_z_LCM``: ``float`` minimum Z distance from simple hit location allowed for a cluster (LCM)
        - ``proximity_distance_y_LCM``: ``float`` minimum Y distance from simple hit location allowed for a cluster (LCM)
        - ``charge_data_dir``: ``str`` path to data dir
    '''
    print('hello 1')
    class_version = '0.0.0'
    default_pps_matching_window = 20
    default_proximity_distance_z_ACL = 20
    default_proximity_distance_y_ACL = 20
    default_proximity_distance_z_LCM = 20
    default_proximity_distance_y_LCM = 20
    default_clusters_matched_dset_name = 'charge/clusters_matched'
    default_clusters_matched_all_dset_name = 'charge/clusters_matched_all'
    default_clusters_hits_matched_dset_name = 'charge/clusters_hits_matched'
    default_clusters_dset_name = 'charge/clusters'
    default_clusters_hits_dset_name = 'charge/clusters_hits'
    default_light_events_dset_name = 'light/events'
    default_sum_hits_dset_name = 'light/simple_hits'
    default_swvfm_dset_name = 'light/schan_wvfm'
    default_charge_data_dir = ''
    default_is_FSD = False
    default_is_mc = False
    default_single_module = -1
    def __init__(self, **params):
        super(LowEnergyChargeLightMatching, self).__init__(**params)
        self.clusters_dset_name = params.get('clusters_dset_name', self.default_clusters_dset_name)
        self.clusters_hits_dset_name = params.get('clusters_hits_dset_name', self.default_clusters_hits_dset_name)
        self.clusters_hits_matched_dset_name = params.get('clusters_hits_matched_dset_name', self.default_clusters_hits_matched_dset_name)
        self.clusters_matched_dset_name = params.get('clusters_matched_dset_name', self.default_clusters_matched_dset_name)
        self.clusters_matched_all_dset_name = params.get('clusters_matched_all_dset_name', self.default_clusters_matched_all_dset_name)
        self.sum_hits_dset_name = params.get('sum_hits_dset_name', self.default_sum_hits_dset_name)
        self.light_events_dset_name = params.get('light_events_dset_name', self.default_light_events_dset_name)
        self.pps_matching_window = params.get('pps_matching_window', self.default_pps_matching_window)
        self.upper_pps_matching_window = 300
        self.lower_pps_matching_window = 100
        self.proximity_distance_z_ACL = params.get('proximity_distance_z_ACL', self.default_proximity_distance_z_ACL)
        self.proximity_distance_y_ACL = params.get('proximity_distance_y_ACL', self.default_proximity_distance_y_ACL)
        self.proximity_distance_z_LCM = params.get('proximity_distance_z_LCM', self.default_proximity_distance_z_LCM)
        self.proximity_distance_y_LCM = params.get('proximity_distance_y_LCM', self.default_proximity_distance_y_LCM)
        self.is_mc = params.get('is_mc', self.default_is_mc)
        self.charge_data_dir = params.get('charge_data_dir', self.default_charge_data_dir)
        self.is_FSD = params.get('is_FSD', self.default_is_FSD)
        self.single_module = params.get('single_module', self.default_single_module)
        
    def init(self, source_name):
        super(LowEnergyChargeLightMatching, self).init(source_name)
        self.clusters_dtype = r.RawEventGenerator.clusters_dtype
        self.clusters_hits_dtype = r.RawEventGenerator.clusters_hits_dtype

        # get light event pps and unix timestamps for matching
        light_events = self.data_manager.get_dset(self.light_events_dset_name)
        utime_ms = light_events['utime_ms']
        if not self.is_mc:
            self.event_unix_time = (utime_ms[utime_ms != 0]*1e-3).astype('int')
        else:
            self.event_unix_time = (utime_ms[:,0]).astype('int')
        tai_ns = light_events['tai_ns']
        tai_ns = (tai_ns[tai_ns != 0]).astype('int')
        if self.is_mc:
            self.event_pps_time = (tai_ns[:,0].astype('int'))*1e-3
        else:
            self.event_pps_time = (tai_ns*1e-9 - (tai_ns*1e-9).astype('int'))*1e9*1e-3
        light_unix_span = (min(self.event_unix_time), max(self.event_unix_time))
        
        # find charge files that overlap in time to light file
        self.matched_charge_files = []
        charge_files = os.listdir(self.charge_data_dir)
        total_tries = 5 # retry reading... in case multiple jobs reading the same file at the same time
        for charge_file in tqdm(charge_files, desc='Finding charge file matches: '):
            charge_file = os.path.join(self.charge_data_dir, charge_file)
            if not charge_file.split('.')[-1] in ['hdf5', 'h5']:
                continue
            loops = 0
            skip_current_file = False
            f = 0
            while True:
                if loops > total_tries:
                    print('Could not read charge file, skipping')
                    skip_current_file=True
                    break
                try:
                    loops+=1
                    f = h5py.File(charge_file, 'r')
                    #print(list(f.keys()))
                    if 'light' in f.keys():
                        print('skipping')
                        skip_current_file=True
                        break
                except:
                    print(f'Could not read file, trying again ({loops}/{total_tries})')
                    time.sleep(1)
                    continue
                break # if successfully read file, continue to rest of script
            if skip_current_file:
                continue
            with f:
                charge_unix_span = [f[self.clusters_dset_name+'/data'][0]['unix_ts'], f[self.clusters_dset_name+'/data'][-1]['unix_ts']]
                if charge_unix_span[0] <= light_unix_span[1] and light_unix_span[0] <= charge_unix_span[1]:
                    self.matched_charge_files.append(charge_file)
                    
        if not len(self.matched_charge_files):
            raise ValueError(f'Could not match input light file to any charge files in {self.charge_data_dir}')
        else:
            print(f"Matched the following charge files to the input light file: {self.matched_charge_files}")
            
        self.clusters_dset_list, self.unix_chunk_indices_list, self.unix_masks_list = {}, [], []
        self.cluster_pps_list, self.cluster_unix_list = {}, {}
        self.cluster_x_list, self.cluster_y_list, self.cluster_z_list, self.cluster_io_list  = {},{},{},{}
        self.clusters_hits_dset_list, self.clusters_hits_ref_region_list = {}, {}
        
        for ifile, matched_file in enumerate(self.matched_charge_files):
            self.cluster_pps_list[ifile] = {}
            self.cluster_unix_list[ifile] = {}
            self.cluster_x_list[ifile] = {}
            self.cluster_y_list[ifile] = {}
            self.cluster_z_list[ifile] = {}
            self.cluster_io_list[ifile] = {}
            self.clusters_dset_list[ifile] = {}
            
        # loop through charge files to pre-make various masks to speed up event loop
        for ifile, charge_file in enumerate(self.matched_charge_files):
            with h5py.File(charge_file, 'r') as f:
                clusters_dset = np.array(f[self.clusters_dset_name+'/data'])
                is_matched_mask = clusters_dset['is_matched'].astype('bool')
                clusters_dset = clusters_dset[is_matched_mask]

                clusters_hits_ref_region = np.array(f[self.clusters_dset_name+'/ref/'+self.clusters_hits_dset_name+'/ref_region'])
                clusters_hits_ref_region = clusters_hits_ref_region[is_matched_mask]
                clusters_hits_dset = np.array(f[self.clusters_hits_dset_name+'/data'])
                
                # sorted by unix, find start and stop indices of each unix value
                #sorted_indices = np.argsort(clusters_dset['unix_ts'])
                #clusters_dset[:] = clusters_dset[sorted_indices]

                #unique_unix, start_indices = np.unique(clusters_dset['unix_ts'], return_index=True)
                #end_indices = np.roll(start_indices, shift=-1)
                #end_indices[-1] = len(clusters_dset) - 1
                
                #unix_chunk_indices = {}
                #for unix_val, start_idx, end_idx in zip(unique_unix, start_indices, end_indices):
                #    unix_chunk_indices[int(unix_val)] = (start_idx, end_idx)
                #self.clusters_dset_list.append(clusters_dset)
                #self.unix_chunk_indices_list.append(unix_chunk_indices)
                self.cluster_pps_list[ifile] = clusters_dset['ts'][:,1]
                self.cluster_unix_list[ifile] = clusters_dset['unix_ts']
                self.cluster_x_list[ifile] = clusters_dset['x_pix'][:,1]
                self.cluster_z_list[ifile] = clusters_dset['z_pix'][:,1]
                self.cluster_y_list[ifile] = clusters_dset['y_pix'][:,1]
                self.cluster_io_list[ifile] = clusters_dset['io_group']
                self.clusters_dset_list[ifile] = clusters_dset
                self.clusters_hits_dset_list[ifile] = clusters_hits_dset
                self.clusters_hits_ref_region_list[ifile] = clusters_hits_ref_region
                
                """
                unix_masks = {}
                for unix in np.unique(clusters_dset['unix_ts']):
                    unix_mask = clusters_dset['unix_ts'] == int(unix)

                    #self.unix_masks_list.append(unix_masks)
                    self.cluster_pps_list[ifile][int(unix)] = clusters_dset['ts'][:,1][unix_mask]
                    self.cluster_unix_list[ifile][int(unix)] = clusters_dset['unix_ts'][unix_mask]
                    self.cluster_x_list[ifile][int(unix)] = clusters_dset['x_pix'][:,1][unix_mask]
                    self.cluster_z_list[ifile][int(unix)] = clusters_dset['z_pix'][:,1][unix_mask]
                    self.cluster_y_list[ifile][int(unix)] = clusters_dset['y_pix'][:,1][unix_mask]
                    self.cluster_io_list[ifile][int(unix)] = clusters_dset['io_group'][unix_mask]
                    self.clusters_dset_list[ifile][int(unix)] = clusters_dset[unix_mask]
                """
                #self.clusters_hits_dset = self.data_manager.get_dset(self.clusters_hits_dset_name)
            
                #self.clusters_hits_ref = self.data_manager.get_ref(self.clusters_dset_name, self.clusters_hits_dset_name)
                self.light_events_sum_hits_ref = self.data_manager.get_ref(self.light_events_dset_name, self.sum_hits_dset_name)

        self.data_manager.set_attrs(self.clusters_matched_dset_name,
                                    clusters_matched_all_dset_name=self.clusters_matched_all_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    source_dset=source_name,
                                    sum_hits_dset=self.sum_hits_dset_name,
                                    clusters_dset=self.clusters_dset_name)
        # create the datasets and references
        self.sum_hits_dset = self.data_manager.get_dset(self.sum_hits_dset_name)
        self.data_manager.create_dset(self.clusters_matched_dset_name, dtype=self.clusters_dtype)
        self.data_manager.create_dset(self.clusters_matched_all_dset_name, dtype=self.clusters_dtype)
        self.data_manager.create_dset(self.clusters_hits_matched_dset_name, self.clusters_hits_dtype)
        self.data_manager.create_ref(self.clusters_matched_dset_name, self.sum_hits_dset_name)
        self.data_manager.create_ref(self.clusters_matched_dset_name, source_name)
        self.data_manager.create_ref(self.clusters_matched_all_dset_name, self.sum_hits_dset_name)
        self.data_manager.create_ref(self.clusters_matched_all_dset_name, source_name)
        self.data_manager.create_ref(self.clusters_matched_dset_name, self.clusters_hits_matched_dset_name)

        # make dict of event -> sum_hits indices
        change_indices = np.where(np.diff(self.light_events_sum_hits_ref[0][:,0]) != 0)[0] + 1
        start_indices = np.r_[0, change_indices]
        stop_indices = np.r_[change_indices, len(self.light_events_sum_hits_ref[0][:,0])]
        self.sum_hits_range_dict = {}
        eventids=self.light_events_sum_hits_ref[0][:,0]
        for start, stop in tqdm(zip(start_indices, stop_indices)):
            key = eventids[start]
            self.sum_hits_range_dict[int(key)] = (int(start), int(stop))
        self.total_matched_clusters = 0
        self.total_clusters = 0
        self.prox_masks_all = {}
        for ifile, charge_file in enumerate(self.matched_charge_files):
            prox_masks = {} #{int(unix):{} for unix in np.unique(list(self.cluster_unix_list[ifile].keys()))}
            #for i, boundary in enumerate(self.sum_hits_dset['boundary']):
            #for i, boundary in enumerate(np.unique(self.sum_hits_dset['boundary'], axis=0)):
            for i, sum_chan in enumerate(np.unique(self.sum_hits_dset['sum_chan'])):
                sum_chan_mask = self.sum_hits_dset['sum_chan'] == sum_chan
                boundary = self.sum_hits_dset['boundary'][sum_chan_mask][0]
                trap_type = self.sum_hits_dset['trap_type'][sum_chan_mask][0]
                tpc = self.sum_hits_dset['tpc'][sum_chan_mask][0]
                #if self.single_module == 2 and sum_chan in [1, 3, 5, 7]:
                #    tpc = 1
                #if self.single_module == 2 and sum_chan in [9, 11, 13, 15]:
                #    tpc = 0
                if trap_type == 0:
                    prox_distance_z = self.proximity_distance_z_ACL
                    prox_distance_y = self.proximity_distance_y_ACL
                else:
                    prox_distance_z = self.proximity_distance_z_LCM
                    prox_distance_y = self.proximity_distance_y_LCM
                det_position = (boundary[1]+boundary[0])/2
                min_x_boundary = min(boundary[0][0], boundary[1][0])
                max_x_boundary = max(boundary[0][0], boundary[1][0])
                #if not tuple(det_position) in det_pos_so_far:
                #for unix in np.unique(list(self.cluster_unix_list[ifile].keys())):
                if self.is_FSD:
                    tpc_mask = (boundary[0][0] - 5 < self.cluster_x_list[ifile][int(unix)]) & \
                               (boundary[1][0] + 5 > self.cluster_x_list[ifile][int(unix)])
                else:
                    #if tpc == 0:
                    #    tpc = 1
                    #elif tpc == 1:
                    #    tpc = 0
                    #tpc_mask = self.cluster_io_list[ifile][int(unix)]-1 == tpc
                    tpc_mask = self.cluster_io_list[ifile]-1 == tpc
                    #tpc_mask = self.cluster_io_list[ifile][int(unix)]-1 == self.sum_hits_dset[i]['tpc']
                
                #prox_masks[int(unix)][sum_chan] = (self.cluster_z_list[ifile][int(unix)] > det_position[2] - prox_distance_z) & \
                #                    (self.cluster_z_list[ifile][int(unix)] < det_position[2] + prox_distance_z) & \
                #                    (self.cluster_y_list[ifile][int(unix)] > det_position[1] - prox_distance_y) & \
                #                    (self.cluster_y_list[ifile][int(unix)] < det_position[1] + prox_distance_y) & \
                #                    tpc_mask 
                if sum_chan in [0, 2, 4, 6, 8, 10, 12, 14, 9, 11, 13, 15]:
                    det_position[2] = det_position[2]*-1
                #if self.single_module == 2 and not ((sum_chan in [1,5]) or (sum_chan in [3, 7])):
                #    det_position[2] = det_position[2]*-1  
                if self.single_module == 2 and (sum_chan == 0):
                    det_position[1] = 15.50975
                    det_position[2] = abs(det_position[2])
                elif self.single_module == 2 and (sum_chan == 5):
                    det_position[1] = 46.52925
                    det_position[2] = abs(det_position[2])
                prox_masks[sum_chan] = (self.cluster_z_list[ifile] > det_position[2] - prox_distance_z) & \
                                    (self.cluster_z_list[ifile] < det_position[2] + prox_distance_z) & \
                                    (self.cluster_y_list[ifile] > det_position[1] - prox_distance_y) & \
                                    (self.cluster_y_list[ifile] < det_position[1] + prox_distance_y) & \
                                    tpc_mask 
            self.prox_masks_all[ifile] = prox_masks
        #for i in range(len(self.cluster_pps_list)):
        #    self.total_clusters += len(self.cluster_pps_list[i])
        
    def run(self, source_name, source_slice, cache):
        super(LowEnergyChargeLightMatching, self).run(source_name, source_slice, cache)
        event_data = cache[source_name]
        #clusters_hits_matched = []
        rel_cluster_id = []
        
        for ifile in range(len(self.clusters_dset_list)):
                    
            matched_cluster_indices = []
            
            for ievent in range(len(event_data)):
                total_matches = 0
                clusters_matched = []
                hits_matched = []
                #hits_matched_cluster_id = []
                
                clusters_matched_all = []
                light_indices, event_indices = [], []
                light_indices_all, event_indices_all = [], []
                try:
                    sum_hits_range = self.sum_hits_range_dict[event_data[ievent]['id']]
                except:
                    continue
                sum_hits_in_event = self.sum_hits_dset[sum_hits_range[0]:sum_hits_range[1]]
                utime_ms = event_data[ievent]['utime_ms']
                light_unix = ((utime_ms[utime_ms != 0]*1e-3).astype('int'))[0]
                tai_ns = event_data[ievent]['tai_ns']
                tai_ns = (tai_ns[tai_ns != 0]).astype('int')
                light_pps = ((tai_ns*1e-9 - (tai_ns*1e-9).astype('int'))*1e9*1e-3)[0]
                #try:
                #    self.cluster_pps_list[ifile][light_unix]
                #except:
                #    continue
                #time_mask = (self.cluster_pps_list[ifile][light_unix].astype('float') < light_pps + self.upper_pps_matching_window) \
                #& (self.cluster_pps_list[ifile][light_unix].astype('float') > light_pps - self.lower_pps_matching_window) #\
                #& (self.cluster_unix_list[ifile][light_unix] == int(light_unix))
                time_mask = (self.cluster_pps_list[ifile].astype('float') < light_pps + self.upper_pps_matching_window) \
                & (self.cluster_pps_list[ifile].astype('float') > light_pps - self.lower_pps_matching_window) \
                & (self.cluster_unix_list[ifile] == int(light_unix))
                if np.any(self.clusters_dset_list[ifile][time_mask]['nhit'] > 10):
                    continue

                if not np.any(time_mask):
                    continue
                if not len(sum_hits_in_event):
                    print('no sum hits in event')
                
                #matching_mask = time_mask #[unix_indices[0]:unix_indices[1]]
                
                for sum_hit in sum_hits_in_event:
                    boundary = sum_hit['boundary']
                    det_position = (boundary[1]+boundary[0])/2
                    light_tpc = sum_hit['tpc']
                    sum_chan = sum_hit['sum_chan']
                    
                    if self.single_module == 1 and sum_hit['amplitude'] < 3000:
                        continue
                    #matching_mask = time_mask & self.prox_masks_all[ifile][light_unix][sum_chan]#[unix_indices[0]:unix_indices[1]]
                    matching_mask = time_mask & self.prox_masks_all[ifile][sum_chan]#[unix_indices[0]:unix_indices[1]]
                    self.total_matched_clusters += np.sum(matching_mask)

                    matched_indices = np.where(matching_mask)[0]
                    for cluster_index in matched_indices:
                        cluster = self.clusters_dset_list[ifile][cluster_index]
                        hits_region = self.clusters_hits_ref_region_list[ifile][cluster_index]
                        #cluster = self.clusters_dset_list[ifile][light_unix][cluster_index]
                        cluster_matched = np.zeros((1,), dtype=self.clusters_dtype)
                        for name in self.clusters_dtype.names:
                            if name in cluster.dtype.names:
                                cluster_matched[name] = cluster[name]
                        if cluster_index in matched_indices:
                            clusters_matched.append(cluster_matched[0])
                            light_indices.append(sum_hit['id'])
                            event_indices.append(event_data[ievent]['id'])
                            for hit in self.clusters_hits_dset_list[ifile][hits_region[0]:hits_region[1]]:
                                hits_matched.append(hit)
                                #hits_matched_cluster_id.append(rel_cluster_id)
                
                # write datasets and references
                #print(f"{len(clusters_matched)=}")
                if len(clusters_matched):
                    clusters_matched_dset = np.array(clusters_matched)
                    clusters_matched_slice = self.data_manager.reserve_data(self.clusters_matched_dset_name, len(clusters_matched_dset))
                    clusters_matched_dset['id'][:] = np.arange(clusters_matched_slice.start, clusters_matched_slice.stop)
                    self.data_manager.write_data(self.clusters_matched_dset_name, clusters_matched_slice, clusters_matched_dset)

                    ref = np.c_[clusters_matched_dset['id'], np.array(light_indices)]
                    self.data_manager.write_ref(self.clusters_matched_dset_name, self.sum_hits_dset_name, ref)

                    ref = np.c_[clusters_matched_dset['id'], np.array(event_indices)]
                    self.data_manager.write_ref(self.clusters_matched_dset_name, source_name, ref)

                    clusters_hits_matched = np.array(hits_matched)
                    clusters_hits_matched_slice = self.data_manager.reserve_data(self.clusters_hits_matched_dset_name, len(clusters_hits_matched))
                    clusters_hits_matched['id'][:] = np.arange(clusters_hits_matched_slice.start, clusters_hits_matched_slice.stop)
                    self.data_manager.write_data(self.clusters_hits_matched_dset_name, clusters_hits_matched_slice, clusters_hits_matched)

                    ref = np.c_[np.repeat(clusters_matched_dset['id'], clusters_matched_dset['nhit']), clusters_hits_matched['id']]
                    self.data_manager.write_ref(self.clusters_matched_dset_name, self.clusters_hits_matched_dset_name, ref)
            print(f'total clusters matched = {self.total_matched_clusters}')
