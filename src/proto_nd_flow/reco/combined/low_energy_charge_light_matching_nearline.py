import numpy as np
import h5py
import logging
import os
from tqdm import tqdm
import time
import re
from zoneinfo import ZoneInfo
from datetime import datetime

from h5flow.core import H5FlowStage, resources
from h5flow import H5FLOW_MPI

import proto_nd_flow.reco.charge.raw_event_generator as r

class LowEnergyChargeLightMatching(H5FlowStage):
    '''
    Match charge clusters with optical hits, requiring clusters to be in close proximity to at least 1 light hit. 
    This script is meant to run on a file created by light event reconstruction and contains the `simple_hits`
    dataset. It will search a directory for charge files that overlap in time, where the charge files
    contain clusters created by the LowEnergyEventBuilder. 

    Note: This class assumes light events and hits are present in the output/input file. 
    The light dataset can be copied to the output file with:
        `h5copy -i $input_file -o $output_file -s light -d light -f ref`

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
    default_cluster_nhit_limit = 10
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
        self.cluster_nhit_limit = params.get('cluster_nhit_limit', self.default_cluster_nhit_limit)
        self.is_mc = params.get('is_mc', self.default_is_mc)
        self.charge_data_dir = params.get('charge_data_dir', self.default_charge_data_dir)
        self.is_FSD = params.get('is_FSD', self.default_is_FSD)
        
    def init(self, source_name):
        super(LowEnergyChargeLightMatching, self).init(source_name)
        self.clusters_dtype = r.RawEventGenerator.clusters_dtype
        self.clusters_hits_dtype = r.RawEventGenerator.clusters_hits_dtype
        print(f'{self.data_manager.filepath=}')
        
        # assuming clusters data in output file
        with h5py.File(self.data_manager.filepath, 'r') as f:
            self.clusters_dset = self.data_manager.get_dset(self.clusters_dset_name)
            self.clusters_hits_ref_region = self.data_manager.get_ref_region(self.clusters_dset_name, self.clusters_hits_dset_name)
            self.clusters_hits_dset = self.data_manager.get_dset(self.clusters_hits_dset_name)
            #self.clusters_dset = np.array(f[os.path.join(self.clusters_dset_name, 'data')])
            #self.clusters_hits_ref_region = np.array(f[os.path.join(self.clusters_dset_name, 'ref', self.clusters_hits_dset_name, 'ref_region')])
            #self.clusters_hits_dset = np.array(f[os.path.join(self.clusters_hits_dset_name, 'data')])
            
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
        #light_unix_span = (min(self.event_unix_time), max(self.event_unix_time))
        #pattern = re.compile(r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})")
        
        #clusters_dset = np.array(f[self.clusters_dset_name+'/data'])
        is_matched_mask = self.clusters_dset['is_matched'].astype('bool')
        self.clusters_dset = self.clusters_dset[is_matched_mask]

        #clusters_hits_ref_region = self.data_manager.get_ref_region(self.clusters_dset_name, self.clusters_hits_dset_name)
        self.clusters_hits_ref_region = self.clusters_hits_ref_region[is_matched_mask]
        #clusters_hits_dset = np.array(f[self.clusters_hits_dset_name+'/data'])
        #clusters_hits_dset = self.data_manager.get_dset(self.clusters_hits_dset_name)
        
        self.cluster_pps_list = self.clusters_dset['ts'][:,1]
        self.cluster_unix_list = self.clusters_dset['unix_ts']
        self.cluster_x_list = self.clusters_dset['x_pix'][:,1]
        self.cluster_z_list = self.clusters_dset['z_pix'][:,1]
        self.cluster_y_list = self.clusters_dset['y_pix'][:,1]
        self.cluster_io_list = self.clusters_dset['io_group']
        #self.clusters_dset_list = clusters_dset
        #self.clusters_hits_dset_list = clusters_hits_dset
        #self.clusters_hits_ref_region_list = clusters_hits_ref_region
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
        #self.prox_masks_all = {}

        # pre-make masks for improved speed
        self.prox_masks = {} 
        
        for i, sum_chan in enumerate(np.unique(self.sum_hits_dset['sum_chan'])):
            sum_chan_mask = self.sum_hits_dset['sum_chan'] == sum_chan
            boundary = self.sum_hits_dset['boundary'][sum_chan_mask][0]
            trap_type = self.sum_hits_dset['trap_type'][sum_chan_mask][0]
            tpc = self.sum_hits_dset['tpc'][sum_chan_mask][0]
            
            if trap_type == 0:
                prox_distance_z = self.proximity_distance_z_ACL
                prox_distance_y = self.proximity_distance_y_ACL
            else:
                prox_distance_z = self.proximity_distance_z_LCM
                prox_distance_y = self.proximity_distance_y_LCM
                
            det_position = (boundary[1]+boundary[0])/2
            min_x_boundary = min(boundary[0][0], boundary[1][0])
            max_x_boundary = max(boundary[0][0], boundary[1][0])
            
            # warning: only valid for 2x2
            tpc_mask = self.cluster_io_list-1 == tpc
            
            self.prox_masks[sum_chan] = (self.cluster_z_list > det_position[2] - prox_distance_z) & \
                                (self.cluster_z_list < det_position[2] + prox_distance_z) & \
                                (self.cluster_y_list > det_position[1] - prox_distance_y) & \
                                (self.cluster_y_list < det_position[1] + prox_distance_y) & \
                                tpc_mask 
                    
    def run(self, source_name, source_slice, cache):
        super(LowEnergyChargeLightMatching, self).run(source_name, source_slice, cache)
        event_data = cache[source_name]
        #clusters_hits_matched = []
        rel_cluster_id = []
        
        matched_cluster_indices = []
        
        for ievent in range(len(event_data)):
            total_matches = 0
            clusters_matched = []
            hits_matched = []
            
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
            
            time_mask = (self.cluster_pps_list.astype('float') < light_pps + self.upper_pps_matching_window) \
                & (self.cluster_pps_list.astype('float') > light_pps - self.lower_pps_matching_window) \
                & (self.cluster_unix_list == int(light_unix))
            if np.any(self.clusters_dset[time_mask]['nhit'] > self.cluster_nhit_limit):
                continue

            if not np.any(time_mask):
                continue
            #if not len(sum_hits_in_event):
            #    print('no sum hits in event')
                            
            for sum_hit in sum_hits_in_event:
                boundary = sum_hit['boundary']
                det_position = (boundary[1]+boundary[0])/2
                light_tpc = sum_hit['tpc']
                sum_chan = sum_hit['sum_chan']
                
                matching_mask = time_mask & self.prox_masks[sum_chan]
                self.total_matched_clusters += np.sum(matching_mask)

                matched_indices = np.where(matching_mask)[0]
                for cluster_index in matched_indices:
                    cluster = self.clusters_dset[cluster_index]
                    hits_region = self.clusters_hits_ref_region[cluster_index]
                    cluster_matched = np.zeros((1,), dtype=self.clusters_dtype)
                    for name in self.clusters_dtype.names:
                        if name in cluster.dtype.names:
                            cluster_matched[name] = cluster[name]
                    if cluster_index in matched_indices:
                        clusters_matched.append(cluster_matched[0])
                        light_indices.append(sum_hit['id'])
                        event_indices.append(event_data[ievent]['id'])
                        for hit in self.clusters_hits_dset[hits_region[0]:hits_region[1]]:
                            hits_matched.append(hit)
            
            # write datasets and references
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
