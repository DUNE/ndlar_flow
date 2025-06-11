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
        self.upper_pps_matching_window = 400
        self.lower_pps_matching_window = 100
        self.proximity_distance_z_ACL = params.get('proximity_distance_z_ACL', self.default_proximity_distance_z_ACL)
        self.proximity_distance_y_ACL = params.get('proximity_distance_y_ACL', self.default_proximity_distance_y_ACL)
        self.proximity_distance_z_LCM = params.get('proximity_distance_z_LCM', self.default_proximity_distance_z_LCM)
        self.proximity_distance_y_LCM = params.get('proximity_distance_y_LCM', self.default_proximity_distance_y_LCM)
        self.charge_data_dir = params.get('charge_data_dir', self.default_charge_data_dir)

    def init(self, source_name):
        super(LowEnergyChargeLightMatching, self).init(source_name)
        self.clusters_dtype = r.RawEventGenerator.clusters_dtype
        self.clusters_hits_dtype = r.RawEventGenerator.clusters_hits_dtype
        
        # get light event pps and unix timestamps for matching
        light_events = self.data_manager.get_dset(self.light_events_dset_name)
        utime_ms = light_events['utime_ms']
        self.event_unix_time = (utime_ms[utime_ms != 0]*1e-3).astype('int')
        tai_ns = light_events['tai_ns']
        tai_ns = (tai_ns[tai_ns != 0]).astype('int')
        self.event_pps_time = (tai_ns*1e-9 - (tai_ns*1e-9).astype('int'))*1e9*1e-3
        light_unix_span = (min(self.event_unix_time), max(self.event_unix_time))

        # find charge files that overlap in time to light file
        self.matched_charge_files = []
        charge_files = os.listdir(self.charge_data_dir)
        total_tries = 5 # retry reading... in case multiple jobs reading the same file at the same time
        for charge_file in tqdm(charge_files, desc='Finding charge file matches: '):
            charge_file = os.path.join(self.charge_data_dir, charge_file)
            if not charge_file.split('.')[-1] in ['hdf5']:
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
                except:
                    print(f'Could not read file, trying again ({loops}/{total_tries})')
                    time.sleep(1)
                    continue
                break # if successfully read file, continue to rest of script
            if skip_current_file:
                continue
            with f:
                charge_unix_span = (f[self.clusters_dset_name+'/data'][0]['unix_ts'], f[self.clusters_dset_name+'/data'][-1]['unix_ts'])
                if charge_unix_span[0] <= light_unix_span[1] and light_unix_span[0] <= charge_unix_span[1]:
                    self.matched_charge_files.append(charge_file)
                    
        if not len(self.matched_charge_files):
            raise ValueError(f'Could not match input light file to any charge files in {self.charge_data_dir}')
        else:
            print(f"Matched the following charge files to the input light file: {self.matched_charge_files}")

        self.clusters_dset_list, self.unix_chunk_indices_list, self.unix_masks_list = [], [], []
        self.cluster_pps_list, self.cluster_unix_list = [], []
        self.cluster_x_list, self.cluster_y_list, self.cluster_z_list,  = [], [], []
        
        # loop through charge files to pre-make various masks to speed up event loop
        for charge_file in self.matched_charge_files:
            with h5py.File(charge_file, 'r') as f:
                clusters_dset = np.array(f[self.clusters_dset_name+'/data'])
                is_matched_mask = clusters_dset['is_matched'].astype('bool')
                clusters_dset = clusters_dset[is_matched_mask]

                # sorted by unix, find start and stop indices of each unix value
                sorted_indices = np.argsort(clusters_dset['unix_ts'])
                clusters_dset[:] = clusters_dset[sorted_indices]

                unique_unix, start_indices = np.unique(clusters_dset['unix_ts'], return_index=True)
                end_indices = np.roll(start_indices, shift=-1)
                end_indices[-1] = len(clusters_dset) - 1

                unix_chunk_indices = {}
                for unix_val, start_idx, end_idx in zip(unique_unix, start_indices, end_indices):
                    unix_chunk_indices[int(unix_val)] = (start_idx, end_idx)
                self.clusters_dset_list.append(clusters_dset)
                self.unix_chunk_indices_list.append(unix_chunk_indices)
                
                unix_masks = {}
                for unix in np.unique(clusters_dset['unix_ts']):
                    unix_masks[int(unix)] = clusters_dset['unix_ts'] == unix

                self.unix_masks_list.append(unix_masks)
                self.cluster_pps_list.append(clusters_dset['ts'][:,1])
                self.cluster_unix_list.append(clusters_dset['unix_ts'])
                self.cluster_x_list.append(clusters_dset['x_pix'][:,1])
                self.cluster_z_list.append(clusters_dset['z_pix'][:,1])
                self.cluster_y_list.append(clusters_dset['y_pix'][:,1])
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
        #self.data_manager.create_dset(self.clusters_hits_matched_dset_name, self.clusters_hits_dtype)
        self.data_manager.create_ref(self.clusters_matched_dset_name, self.sum_hits_dset_name)
        self.data_manager.create_ref(self.clusters_matched_dset_name, source_name)
        self.data_manager.create_ref(self.clusters_matched_all_dset_name, self.sum_hits_dset_name)
        self.data_manager.create_ref(self.clusters_matched_all_dset_name, source_name)

        # make dict of event -> sum_hits indices
        change_indices = np.where(np.diff(self.light_events_sum_hits_ref[0][:,0]) != 0)[0] + 1
        start_indices = np.r_[0, change_indices]
        stop_indices = np.r_[change_indices, len(self.light_events_sum_hits_ref[0][:,0])]
        self.sum_hits_range_dict = {}
        eventids=self.light_events_sum_hits_ref[0][:,0]
        for start, stop in tqdm(zip(start_indices, stop_indices)):
            key = eventids[start]
            self.sum_hits_range_dict[int(key)] = (int(start), int(stop))
        
    def run(self, source_name, source_slice, cache):
        super(LowEnergyChargeLightMatching, self).run(source_name, source_slice, cache)
        event_data = cache[source_name]
        clusters_hits_matched = []
        
        for ifile in range(len(self.clusters_dset_list)):
            prox_masks = {}
            for i, boundary in enumerate(self.sum_hits_dset['boundary']):
                trap_type = self.sum_hits_dset[i]['trap_type']
                if trap_type == 0:
                    prox_distance_z = self.proximity_distance_z_ACL
                    prox_distance_y = self.proximity_distance_y_ACL
                else:
                    prox_distance_z = self.proximity_distance_z_LCM
                    prox_distance_y = self.proximity_distance_y_LCM
                det_position = (boundary[1]+boundary[0])/2
                min_x_boundary = min(boundary[0][0], boundary[1][0])
                max_x_boundary = max(boundary[0][0], boundary[1][0])
                if not tuple(det_position) in prox_masks.keys():
                    prox_masks[tuple(det_position)] = (self.cluster_z_list[ifile] > det_position[2] - prox_distance_z) & \
                                        (self.cluster_z_list[ifile] < det_position[2] + prox_distance_z) & \
                                        (self.cluster_y_list[ifile] > det_position[1] - prox_distance_y) & \
                                        (self.cluster_y_list[ifile] < det_position[1] + prox_distance_y) & \
                                        (boundary[0][0] - 5 < self.cluster_x_list[ifile]) & \
                                        (boundary[1][0] + 5 > self.cluster_x_list[ifile])
                    
            matched_cluster_indices = []
            for ievent in range(len(event_data)):
                total_matches = 0
                clusters_matched = []
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
                if int(light_unix) not in self.unix_chunk_indices_list[ifile].keys():
                    continue
                unix_indices = self.unix_chunk_indices_list[ifile][int(light_unix)]
                
                time_mask = (self.cluster_pps_list[ifile][unix_indices[0]:unix_indices[1]].astype('float') < light_pps + self.upper_pps_matching_window) \
                & (self.cluster_pps_list[ifile][unix_indices[0]:unix_indices[1]].astype('float') > light_pps - self.lower_pps_matching_window) #\
                if not np.any(time_mask):
                    continue
                if not len(sum_hits_in_event):
                    print('no sum hits in event')
                
                for sum_hit in sum_hits_in_event:
                    boundary = sum_hit['boundary']
                    det_position = (boundary[1]+boundary[0])/2
                    light_tpc = sum_hit['tpc']
                    
                    matching_mask = time_mask & prox_masks[tuple(det_position)][unix_indices[0]:unix_indices[1]]
                    matching_mask_all = time_mask

                    matched_indices = unix_indices[0]+np.where(matching_mask)[0]
                    matched_indices_all = unix_indices[0]+np.where(matching_mask_all)[0]
                    for cluster_index in matched_indices:
                        cluster = self.clusters_dset_list[ifile][cluster_index]
                        cluster_matched = np.zeros((1,), dtype=self.clusters_dtype)
                        for name in self.clusters_dtype.names:
                            cluster_matched[name] = cluster[name]
                        if cluster_index in matched_indices:
                            clusters_matched.append(cluster_matched[0])
                            light_indices.append(sum_hit['id'])
                            event_indices.append(event_data[ievent]['id'])
                        if cluster_index in matched_indices_all:
                            clusters_matched_all.append(cluster_matched[0])
                            light_indices_all.append(sum_hit['id'])
                            event_indices_all.append(event_data[ievent]['id'])
                    for cluster_index in matched_indices_all:
                        cluster = self.clusters_dset_list[ifile][cluster_index]
                        
                        cluster_matched = np.zeros((1,), dtype=self.clusters_dtype)
                        for name in self.clusters_dtype.names:
                            cluster_matched[name] = cluster[name]
                        if cluster_index in matched_indices_all:
                            clusters_matched_all.append(cluster_matched[0])
                            light_indices_all.append(sum_hit['id'])
                            event_indices_all.append(event_data[ievent]['id'])
                        #for cluster_hit_index in self.clusters_hits_ref[0][:,1][self.clusters_hits_ref[0][:,0] == cluster_matched['id']]:
                        #    hit = np.zeros((1,), dtype=self.clusters_hits_dtype)
                        #    for name in self.clusters_hits_dtype.names:
                        #        hit[name] = self.clusters_hits_dset[cluster_hit_index][name]
                        #    clusters_hits_matched.append(hit)
                
                # write datasets and references
                print(f"{len(clusters_matched)=}")
                if len(clusters_matched):
                    clusters_matched_dset = np.array(clusters_matched)
                    clusters_matched_slice = self.data_manager.reserve_data(self.clusters_matched_dset_name, len(clusters_matched_dset))
                    clusters_matched_dset['id'][:] = np.arange(clusters_matched_slice.start, clusters_matched_slice.stop)
                    self.data_manager.write_data(self.clusters_matched_dset_name, clusters_matched_slice, clusters_matched_dset)

                    clusters_matched_all_dset = np.array(clusters_matched_all)
                    clusters_matched_all_slice = self.data_manager.reserve_data(self.clusters_matched_all_dset_name, len(clusters_matched_all_dset))
                    clusters_matched_all_dset['id'][:] = np.arange(clusters_matched_all_slice.start, clusters_matched_all_slice.stop)
                    self.data_manager.write_data(self.clusters_matched_all_dset_name, clusters_matched_all_slice, clusters_matched_all_dset)

                    ref = np.c_[clusters_matched_dset['id'], np.array(light_indices)]
                    self.data_manager.write_ref(self.clusters_matched_dset_name, self.sum_hits_dset_name, ref)

                    ref = np.c_[clusters_matched_all_dset['id'], np.array(light_indices_all)]
                    self.data_manager.write_ref(self.clusters_matched_all_dset_name, self.sum_hits_dset_name, ref)
        
                    ref = np.c_[clusters_matched_dset['id'], np.array(event_indices)]
                    self.data_manager.write_ref(self.clusters_matched_dset_name, source_name, ref)

                    ref = np.c_[clusters_matched_all_dset['id'], np.array(event_indices_all)]
                    self.data_manager.write_ref(self.clusters_matched_all_dset_name, source_name, ref)
            
                    #clusters_hits_matched = np.concatenate(clusters_hits_matched)
                    #clusters_hits_matched_slice = self.data_manager.reserve_data(self.clusters_hits_matched_dset_name, len(clusters_hits_matched))
                    #clusters_hits_matched['id'][:] = np.arange(clusters_hits_matched_slice.start, clusters_hits_matched_slice.stop)
                    #self.data_manager.write_data(self.clusters_hits_matched_dset_name, clusters_hits_matched_slice, clusters_hits_matched)

                    #ref = np.c_[clusters_matched['id'], np.repeat(clusters_hits_matched['id'], clusters_matched['nhit'])]
                    #self.data_manager.write_ref(self.clusters_matched_dset_name, self.clusters_hits_matched_dset_name, ref)