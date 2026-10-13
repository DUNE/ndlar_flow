#Imports
import logging

import numpy as np
import numpy.ma as ma

from h5flow.core import H5FlowStage, resources
from h5flow.core import resources

from h5flow import H5FLOW_MPI
import h5flow
from h5flow.data import dereference

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


from scipy.spatial.distance import cdist

import statistics



class RockMuonSelection(H5FlowStage):
    '''
    This will perform a selection for rock muons. Rock muons are 
    determined by straight tracks that penetrate two faces of the 
    detector.

    Crossing tags in ``analysis/rock_muon_tracks`` (one slot per module / TPC,
    sized from the geometry stored in the file):

     - ``a2a``: bool ``(n_modules,)``. ``a2a[m]`` is True if the track reaches
       both anodes of module ``m`` (anode-to-anode).
     - ``a2c``: bool ``(n_modules, 2)``. ``a2c[m, tpc]`` is True if the track goes
       from the anode to the cathode in that TPC (anode-to-cathode), independent
       of ``a2a``, so an anode-to-anode module is normally ``[True, True]``.
       ``tpc = 0`` is the TPC whose anode is at the module's lower x, ``tpc = 1``
       the one at higher x. Anode-to-cathode only: ``a2c & ~a2a[:, None]``.

    Module ``m`` is index ``m`` of ``geometry_info`` ``module_RO_bounds``. The
    dataset attributes ``tpc_anode_x`` ``(n_modules, 2)`` and ``tpc_io_groups``
    ``(n_modules, 2, n)``, padded with -1, give the anode x and io_groups of each
    slot. E.g. on the 2x2, ``tpc_io_groups[:, :, 0]`` is ``[[2, 1], [4, 3], [6, 5], [8, 7]]``.

    All parameters in ``defaults`` can be set in the yaml; anything left out uses
    the default. ``hits_dset_names`` lists the hit datasets to use in order of
    preference: the first one present in the file whose ``id`` equals its row
    index is used (the track/segment -> hit references are written from ``id``),
    and its name is stored in the ``hits_dset_name`` attribute of the tracks
    dataset. Every dataset in the list must also be in the yaml ``requires``.
    '''
    #Empty List so tracks can be counted
    
    #Datasets and cuts, all can be set in the yaml
    defaults = dict([
    ('events_dset_name', 'charge/events'),
    ('hits_dset_names', ['charge/calib_prompt_hits']), #first one present in the file is used
    ('rock_muon_hits_dset_name', 'analysis/rock_muon_tracks'),
    ('rock_muon_segments_dset_name', 'analysis/rock_muon_segments'),

    ('length_cut', 100), #cm, minimum track length
    ('max_avg_distance', 1.5), #cm, maximum mean hit distance from the track line
    ('noise_hit_max_distance', 3.5), #cm, hits further from the track line are removed
    ('face_tolerance', 2.1), #cm, distance to a detector face, anode or cathode to count as reaching it
    ('min_cluster_hits', 11), #minimum hits in a cluster to try it as a track
    ('min_tpc_hits', 3), #minimum hits in a TPC for anode/cathode crossing tags

    ('dbscan_eps', 1), #cm
    ('dbscan_min_samples', 3),
    ('max_clusters_for_merging', 150), #events with more clusters are not merged
    ('merge_min_hits', 6), #minimum hits in each cluster to merge
    ('merge_min_centroid_dist', 2), #cm
    ('merge_max_centroid_dist', 100), #cm
    ('merge_max_distance', 3), #cm, maximum mean distance of the merged hits from the line
    ('merge_max_angle', 20), #degrees, maximum angle between the two clusters

    ('segment_length', 2), #cm
    ])

    #Datatype wanted
    
    # crossing tag lists are appended in init(), sized from the geometry
    rock_muon_track_base_fields = [
        ('event_id','i4'),
        ('rock_muon_id', 'i4'),
        ('length','f8'),
        ('x_start', 'f8'),
        ('y_start','f8'),
        ('z_start', 'f8'),
        ('x_end','f8'),
        ('y_end', 'f8'),
        ('z_end', 'f8'),
        ('exp_var', 'f8'), 
        ('theta_xz','f8'), 
        ('theta_yz', 'f8'), 
        ('theta_z','f8'),
        # new
        ('PCA_i', 'f8'),
        ('PCA_j','f8'),
        ('PCA_k', 'f8'),
    ]
   
    rock_muon_segments_dtype = np.dtype([
        ('rock_segment_id', 'i4'),
        ('x_start', 'f8'),
        ('y_start','f8'),
        ('z_start','f8'),
        ('dE', 'f8'),
        ('x_end', 'f8'),
        ('y_end','f8'),
        ('z_end', 'f8'),
        ('dQ','f8'),
        ('dx','f8'),
        ('x_mid','f8'),
        ('y_mid','f8'),
        ('z_mid','f8'),
        ('t','f8'),
        ('io_group', 'i4'),
        ('nhits', 'i4'),
    ])

    def __init__(self, **params):
        
        super(RockMuonSelection,self).__init__(**params) # needed to inherit H5FlowStage functionality
        
        for key,val in self.defaults.items():
            setattr(self, key, params.get(key, val))

        #Track and segment counters
        self.track_count = -1
        self.segment_count = -1
            
    def init(self, source_name):
        
        super(RockMuonSelection, self).init(source_name)

        self.hits_dset_name = self._choose_hits_dset()

        # Crossing tags use the geometry stored in the file, one slot per module / TPC:
        # a2a[module], a2c[module, tpc] with tpc 0 = low-x anode, 1 = high-x anode
        self._load_tpc_io_groups()
        self.rock_muon_track_dtype = np.dtype(self.rock_muon_track_base_fields + [
            ('a2a', '?', (self.n_modules,)),
            ('a2c', '?', (self.n_modules, 2)),
        ])
        
        self.data_manager.create_dset(self.rock_muon_hits_dset_name,
                                      dtype = self.rock_muon_track_dtype)

        #Record the hits used and the configuration
        config = {key: getattr(self, key) for key in self.defaults}
        config['hits_dset_names'] = np.array(self.hits_dset_names, dtype='S') # h5py attrs can't hold a list of str
        self.data_manager.set_attrs(self.rock_muon_hits_dset_name,
                                    hits_dset_name=self.hits_dset_name,
                                    **config)

        # io_groups reading each a2c slot [module, tpc], padded with -1
        n_max = max(len(ios) for module in self.tpc_io_groups for ios in module)
        tpc_io_groups = np.full((self.n_modules, 2, n_max), -1, dtype='i4')
        for m_id, module in enumerate(self.tpc_io_groups):
            for tpc, ios in enumerate(module):
                tpc_io_groups[m_id, tpc, :len(ios)] = ios
        self.data_manager.set_attrs(self.rock_muon_hits_dset_name,
            tpc_io_groups=tpc_io_groups,
            tpc_anode_x=np.asarray(self.module_bounds)[:, :, 0],
            a2a_description='a2a[module]: True if the track reaches both anodes of the module (anode-to-anode). '
                            'module = index of geometry_info module_RO_bounds',
            a2c_description='a2c[module, tpc]: True if the track goes from the anode to the cathode in that TPC '
                            '(anode-to-cathode), independent of a2a. tpc 0 = anode at the '
                            "module's lower x, tpc 1 = anode at higher x; see tpc_anode_x and tpc_io_groups",
            tpc_io_groups_description='io_groups reading each a2c slot [module, tpc, :], padded with -1')
        
        self.data_manager.create_dset(self.rock_muon_segments_dset_name,
                                      dtype = self.rock_muon_segments_dtype)

        self.data_manager.create_ref(self.events_dset_name, self.rock_muon_hits_dset_name) 
        
        self.data_manager.create_ref(self.rock_muon_hits_dset_name,self.hits_dset_name)

        self.data_manager.create_ref(self.rock_muon_hits_dset_name, self.rock_muon_segments_dset_name,)
        
        self.data_manager.create_ref(self.rock_muon_segments_dset_name, self.hits_dset_name)

    def _choose_hits_dset(self):
        '''
        First dataset in ``hits_dset_names`` that is in the file and whose ``id`` equals its row
        index, since the track/segment -> hit references are written from ``id``.
        '''
        for name in self.hits_dset_names:
            if not self.data_manager.dset_exists(name):
                continue
            #check id matches row number
            n = min(len(self.data_manager.get_dset(name)), 100000)
            if not np.array_equal(self.data_manager.get_dset(name)['id'][:n], np.arange(n)):
                if self.rank == 0:
                    logging.warning(f'{name}: id is not the row index (file made before the id fix?), not using it')
                continue
            if self.rank == 0:
                logging.warning(f'RockMuonSelection: using {name}')
                if name.endswith('calib_final_hits'):
                    logging.warning('calib_final_hits in old FSD files are merged hits, which the selection should not use')
            return name
        raise RuntimeError(f'RockMuonSelection: none of {self.hits_dset_names} is in the file with id == row index')
    
    #@staticmethod
    '''
    def cluster(self, PromptHits_ev):
        
        index_of_track_hits = []

        positions = np.column_stack((PromptHits_ev['x'], PromptHits_ev['y'], PromptHits_ev['z']))
        
        hit_cluster = DBSCAN(eps = 8, min_samples = 1).fit(positions)
        
        unique_labels = np.unique(hit_cluster.labels_)

        for unique in unique_labels:
            index = np.where(hit_cluster.labels_ == unique)[0]
            index_of_track_hits.append(index)

        return index_of_track_hits
    '''
    #@staticmethod
    def cluster(self,PromptHits_ev):
        index_of_track_hits = []
        positions = np.column_stack((PromptHits_ev['x'], PromptHits_ev['y'], PromptHits_ev['z']))
    
        # Perform DBSCAN clustering
        hit_cluster = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(positions)
    
        cluster_labels = hit_cluster.labels_

        unique_labels = np.unique(cluster_labels)

        if len(unique_labels) < self.max_clusters_for_merging:

            # Collect indices of hits for each cluster
            for unique in unique_labels:
                index = np.where(cluster_labels == unique)[0]
                index_of_track_hits.append(index)

            index = 0
            while index < len(index_of_track_hits):
                center_of_masses = [np.mean(positions[cluster], axis=0) for cluster in index_of_track_hits]
                center_of_1 = np.mean(positions[index_of_track_hits[index]], axis=0)

                # Compute distances and lengths
                distances = np.linalg.norm(center_of_masses - center_of_1, axis=1)
                lengths = [len(cluster) for cluster in index_of_track_hits]
            
                combined_dist_length = [[distances[k], lengths[k]] for k in range(len(distances))]

                # Create a list of indices
                indices = list(range(len(combined_dist_length)))

                # Sort indices based on length (descending) and distance (ascending)
                sorted_indices = sorted(indices, key=lambda i: (-combined_dist_length[i][1], combined_dist_length[i][0]))
            
                explained_var, direction, original_mean = self.PCAs(PromptHits_ev[index_of_track_hits[index]])

                # Try merging with sorted clusters
                for j in sorted_indices:
                    if (j == index) | (len(index_of_track_hits[j]) < self.merge_min_hits) | (len(index_of_track_hits[index]) < self.merge_min_hits) | (combined_dist_length[j][0] > self.merge_max_centroid_dist) | (combined_dist_length[j][0] < self.merge_min_centroid_dist):  # Skip merging with itself
                        continue
                    explained_var, direction2, original_mean = self.PCAs(PromptHits_ev[index_of_track_hits[j]])
                    hits_of_testing_merge = np.concatenate((positions[index_of_track_hits[index]], positions[index_of_track_hits[j]]))
                    center_of_merge = np.mean(hits_of_testing_merge, axis=0)

                    projections = np.dot(hits_of_testing_merge - center_of_merge, direction[:, np.newaxis]) * direction + center_of_merge
                    distances = np.linalg.norm(hits_of_testing_merge - projections, axis=1)
                    average_dist = np.mean(distances)
                    sim_direction = np.rad2deg(np.arccos(np.abs(np.dot(direction, direction2))))

                    if (average_dist <= self.merge_max_distance) & (sim_direction <= self.merge_max_angle):
                        index_of_track_hits[index] = np.concatenate([index_of_track_hits[index], index_of_track_hits[j]])
                        index_of_track_hits.pop(j)
                        center_of_masses.pop(j)
                        #print(f'Merging cluster {index} with cluster {j}')
                        break  # Recompute centers and distances after merge
                else:
                    index += 1
        else:
            for unique in unique_labels:
                index = np.where(hit_cluster.labels_ == unique)[0]
                index_of_track_hits.append(index)

        return index_of_track_hits 
    #@staticmethod
    
    def PCAs(self,hits_of_track):
        scaler = StandardScaler()
        
        positions = np.column_stack((hits_of_track['x'], hits_of_track['y'], hits_of_track['z']))
         
        X_train = positions
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)

        pca = PCA(1) # 1 component

        pca.fit(X_train)

        explained_var = pca.explained_variance_ratio_[0]
        
        scaled_vector = pca.components_[0]
        
        unscaled_vector = scaler.scale_ * scaled_vector

        normalized_direction_vector = unscaled_vector/np.linalg.norm(unscaled_vector)
        
        scaled_mean = pca.mean_

        original_mean = scaler.inverse_transform(scaled_mean.reshape(1, -1)).flatten()
        
        return  explained_var, normalized_direction_vector, original_mean
    
    #@staticmethod
    def length(self,hits):
        #Get Hit positions
        hit_positions = np.column_stack((hits['x'], hits['y'], hits['z']))
        
        hdist = cdist(hit_positions, hit_positions)
         
        max_value_index = np.argmax(hdist)
        # Convert flattened index to row and column indices
        max_value_row = max_value_index // hdist.shape[1]
        max_value_col = max_value_index % hdist.shape[1]
        
        indices = [max_value_row, max_value_col]
        
        start_hit, end_hit = hit_positions[np.min(indices)], hit_positions[np.max(indices)]
        
        return np.max(hdist), start_hit, end_hit
    
    '''
    Checks to see if start/end point of track are close to two different faces of detector. If they are this will return True. Note: >= -1 just in case if a hit is reconstructed outside of detector.
    '''
    def close_to_two_faces(self,boundaries, hits):
        # Boundaries are in the order [xmin, ymin, zmin, xmax, ymax, zmax]
        penetrated = False
        threshold = self.face_tolerance

        test_face = [False] * len(boundaries)
        for index, face in enumerate(boundaries):
            if (index == 0) or (index == 3):
                distance = np.abs(face - hits['x'])
                #print(f"Checking x boundaries at index {index}: face = {face}, distances = {distance}")

                if np.any(distance <= threshold):
                    test_face[index] = True
        
            elif (index == 1) or (index == 4):
                distance = np.abs(face - hits['y'])
                if np.any(distance <= threshold):
                    test_face[index] = True
        
            elif (index == 2) or (index == 5): 
                distance = np.abs(face - hits['z'])
                if np.any(distance <= threshold):
                    test_face[index] = True
        #print(test_face)
        if sum(test_face)>= 2:
            penetrated = True
    
        return penetrated
    
    def _load_tpc_io_groups(self):
        '''
        From the Geometry resource: module readout bounds (x min/max are the two anode planes,
        the cathode is midway) and, for each module, the io_groups reading its low-x and
        high-x anodes (from the tile_id and anode_drift_coordinate lookup tables).
        '''
        geometry = resources['Geometry']
        self.module_bounds = geometry.module_RO_bounds
        self.n_modules = len(self.module_bounds)

        io_group, io_channel = geometry.tile_id.keys()
        tile = geometry.tile_id[(io_group, io_channel)]
        io_group, tile = io_group[tile > 0], tile[tile > 0]
        io_groups = np.unique(io_group)
        self.n_io_groups = len(io_groups)
        io_groups_per_module = self.n_io_groups // self.n_modules

        self.tpc_io_groups = [([], []) for _ in range(self.n_modules)]
        for io in io_groups:
            anode = np.unique(geometry.anode_drift_coordinate[(tile[io_group == io],)])
            m_id = (io - 1) // io_groups_per_module
            side = np.flatnonzero(np.isclose(anode[0], self.module_bounds[m_id][:, 0]))
            if len(anode) != 1 or len(side) != 1:
                raise ValueError(f'io_group {io}: anode x {anode} is not one anode plane of module {m_id}')
            self.tpc_io_groups[m_id][side[0]].append(io)

    def anode_cathode_tags(self, hits):
        '''
        Anode-to-anode (A2A) and anode-to-cathode (A2C) crossings. Returns a2a[module], True if
        the track reaches both anodes of the module, and a2c[module, tpc], True if it goes from
        the anode to the cathode in that TPC (independent of a2a).
        '''
        tol, min_hits = self.face_tolerance, self.min_tpc_hits
        a2a = np.zeros(self.n_modules, dtype=bool)
        a2c = np.zeros((self.n_modules, 2), dtype=bool)

        for m_id, (lo, hi) in enumerate(self.module_bounds):
            cathode = 0.5 * (lo[0] + hi[0])

            near_anode, near_cathode = [], []
            for anode, tpc_io_groups in zip((lo[0], hi[0]), self.tpc_io_groups[m_id]):
                x = hits['x'][np.isin(hits['io_group'], tpc_io_groups)]
                enough = len(x) >= min_hits
                near_anode.append(enough and np.any(np.abs(x - anode) <= tol))
                near_cathode.append(enough and np.any(np.abs(x - cathode) <= tol))

            a2a[m_id] = all(near_anode)
            a2c[m_id] = np.logical_and(near_anode, near_cathode)

        return a2a, a2c

    #@staticmethod
    def clean_noise_hits(self, hits):
        positions = np.column_stack((hits['x'], hits['y'], hits['z']))

        # Perform PCA to find the principal component
        pca = PCA(n_components=1)
        pca.fit(positions)
        track_direction = pca.components_[0]
        hits_mean = pca.mean_

        # Project points onto the principal component (the line)
        projections = np.dot(positions - hits_mean, track_direction[:, np.newaxis]) * track_direction + hits_mean

        # Calculate the Euclidean distance between each point and its projection on the line
        distances = np.linalg.norm(positions - projections, axis=1)
        
        mask_good = distances <= self.noise_hit_max_distance

        filtered_hits = hits[mask_good]

        return filtered_hits
    #@staticmethod
    def average_distance(self, hits):
        positions = np.column_stack((hits['x'], hits['y'], hits['z']))

        # Perform PCA to find the principal component
        pca = PCA(n_components=1)
        pca.fit(positions)
        track_direction = pca.components_[0]
        hits_mean = pca.mean_



        # Project points onto the principal component (the line)
        projections = np.dot(positions - hits_mean, track_direction[:, np.newaxis]) * track_direction + hits_mean

        # Calculate the Euclidean distance between each point and its projection on the line
        distances = np.linalg.norm(positions - projections, axis=1)
        #print(np.mean(distances))
        average_distances = np.mean(distances)

        return average_distances
    
    #@staticmethod
    def select_muon_track(self,hits,Min_max_detector_bounds):
            muon_hits = []

            min_boundaries = Min_max_detector_bounds[0] #Geometry lar_detector_bounds are already x,y,z like the hits
            max_boundaries = Min_max_detector_bounds[1]
            
            faces_of_detector = np.concatenate((min_boundaries,max_boundaries))

            L_cut = self.length_cut #minimum track length requirement
            
            filtered_hits = self.clean_noise_hits(hits)
            
            explained_var, direction_vector,mean_point = self.PCAs(filtered_hits)
                
            l_track, start_point, end_point = self.length(filtered_hits)
            
            avg_distance = self.average_distance(filtered_hits)

            if (avg_distance <= self.max_avg_distance) & (l_track >= L_cut):

                penetrated = self.close_to_two_faces(faces_of_detector, filtered_hits)

                if penetrated == True:
                    #filtered_hits = self.clean_noise_hits(hits)

                    muon_hits.append(filtered_hits)

                    #Get the new hits info
                    #explained_var, direction_vector,mean_point = self.PCAs(filtered_hits)

                    #l_track, start_point, end_point = self.length(filtered_hits)

            return np.array(muon_hits), l_track, start_point, end_point, explained_var, direction_vector
    
    #@staticmethod
    def angle(self,direction_vector):
        magnitude = np.linalg.norm(direction_vector)

        # Calculate the unit vector in the xz-plane
        normal_vector_xz = np.array([0, 1, 0])
        
        # Calculate the dot product between the direction vector and the unit vector in the yz-plane
        dot_product = np.dot(direction_vector, normal_vector_xz)

        # Calculate the angle between the direction vector and the yz-plane
        theta_xz = np.arccos(dot_product / magnitude)

        # Convert the angle from radians to degrees
        theta_xz = np.degrees(theta_xz)
        
        normal_vector_yz = np.array([1, 0, 0])

        # Calculate the dot product between the direction vector and the unit vector in the yz-plane
        dot_product = np.dot(direction_vector, normal_vector_yz)

        # Calculate the angle between the direction vector and the yz-plane
        theta_yz = np.arccos(dot_product / magnitude)

        # Convert the angle from radians to degrees
        theta_yz = np.degrees(theta_yz)
        if direction_vector[2] > 0:
            theta_z = np.degrees(np.arctan(np.sqrt(direction_vector[0]**2 + direction_vector[1]**2)/direction_vector[2]))
        elif direction_vector[2] < 0:
            theta_z = 180 + np.degrees(np.arctan(np.sqrt(direction_vector[0]**2 + direction_vector[1]**2)/direction_vector[2]))
        else:
            theta_z = 90
        return theta_xz, theta_yz, theta_z

    #@staticmethod
    def TPC_separation(self, hits):
        hits_tpc = []

        io_groups = np.unique(hits['io_group'])
        
        for io_group in io_groups:
            mask = hits['io_group'] == io_group

            hits_of_tpc = hits[mask]
            if len(hits_of_tpc) != 0:
                hits_tpc.append(hits_of_tpc)

        return hits_tpc
    
    
    #@staticmethod
    def segments(self,muon_hits):
        segment_info = []

        hit_ref = []
        segment_to_track_ref = []

        track = muon_hits[0] #Makes sure hits go back to a (n,) shape instead of (1,n) shape
    
        tpc_hits = self.TPC_separation(track)

        given_scale = self.segment_length

        for hits in tpc_hits:
            if len(hits) != 0:
                io_group_of_tpc = np.unique(hits['io_group'])

                tpc_var, principal_component, tpc_mean = self.PCAs(hits)
       
                points = np.array([[hit['x'], hit['y'], hit['z']] for hit in hits])
            
                centered_points = points - tpc_mean

                projections = np.dot(centered_points, principal_component)
                projected_hits = tpc_mean + np.outer(projections, principal_component)
            
                # Step 7: Find the minimum and maximum projections
                t_min = np.min(projections)
                t_max = np.max(projections)
            
                # Step 8: Compute the endpoints of the finite line
                # Line endpoint 1: mean + t_min * principal_component
                line_point_1 = tpc_mean + t_min * principal_component

                # Line endpoint 2: mean + t_max * principal_component
                line_point_2 = tpc_mean + t_max * principal_component
            
                line_defined_points = [line_point_1,line_point_2]

                line_start = line_defined_points[np.argmax([line_point_1[2], line_point_2[2]])]
                line_end = line_defined_points[np.argmin([line_point_1[2], line_point_2[2]])]
            
                #lets make segments
                if principal_component[2] < 0:
                    principal_component = -principal_component

                initial_jump_size = given_scale
                jump_vector = initial_jump_size * principal_component
            
                for i in range(1,1000):
                    break_out = False
                
                    segment_start = line_start - (i-1)*jump_vector
                    segment_end = segment_start - jump_vector
                
                    if segment_end[2] >= line_end[2]:
                        seg_info = self.grab_segment_info(segment_end, segment_start, projected_hits, hits, hit_ref, segment_to_track_ref)

                        if seg_info is not None:
                            segment_info.append(seg_info)   
                    
                    
                    else:
                        segment_end = line_end
                        seg_info = self.grab_segment_info(segment_end, segment_start, projected_hits, hits, hit_ref, segment_to_track_ref)
                        break_out = True
                        if seg_info is not None:
                            segment_info.append(seg_info)  
                    
                    
                    if break_out == True:
                        break


        return segment_info, hit_ref, segment_to_track_ref

    def grab_segment_info(self,segment_end, segment_start, projected_hits, hits, hit_ref, segment_to_track_ref):
            min_bounds = [min([segment_end[i],segment_start[i]]) for i in range(0,3)]
            max_bounds = [max([segment_end[i],segment_start[i]]) for i in range(0,3)]
            condition = (projected_hits[:,2] >= min_bounds[2]) & (projected_hits[:,2] <= max_bounds[2])
        
            condition = (
                    (projected_hits[:, 0] >= min_bounds[0]) & (projected_hits[:, 0] <= max_bounds[0]) &
                    (projected_hits[:, 1] >= min_bounds[1]) & (projected_hits[:, 1] <= max_bounds[1]) &
                    (projected_hits[:, 2] >= min_bounds[2]) & (projected_hits[:, 2] <= max_bounds[2])
                )
        
            hits_of_segment = hits[condition]
        
            if len(hits_of_segment) != 0:
                hits_positions = np.column_stack((hits_of_segment['x'],hits_of_segment['y'],hits_of_segment['z']))
     
                x_start, y_start, z_start = segment_start[0], segment_start[1], segment_start[2]
                x_end, y_end, z_end = segment_end[0], segment_end[1], segment_end[2]
                x_mid, y_mid, z_mid = (x_start+x_end)/2, (y_start + y_end)/2, (z_start + z_end)/2

                Energy_of_segment = sum(hits_of_segment['E'])
                Q_of_segment = sum(hits_of_segment['Q'])
                drift_time = (max(hits_of_segment['t_drift'])+min(hits_of_segment['t_drift']))/2
            
                io_group_of_segment = np.unique(hits_of_segment['io_group'])[0]
                self.segment_count += 1

                
                for hit in hits_of_segment:
                    hit_ref.append([self.segment_count, hit['id']])
                segment_to_track_ref.append([self.track_count, self.segment_count])
                dx = np.linalg.norm(segment_start-segment_end)
            
                return [self.segment_count, x_start, y_start, z_start, Energy_of_segment, x_end, y_end, z_end, Q_of_segment, dx, x_mid, y_mid,z_mid, drift_time, io_group_of_segment, len(hits_of_segment)]
            else:
                #print(f'No hits found for segment: start={segment_start}, end={segment_end}')
                return None

    def run(self, source_name, source_slice, cache):
        
        super(RockMuonSelection, self).run(source_name, source_slice, cache)
                    
        event_id = np.r_[source_slice]
        
        Min_max_detector_bounds = resources['Geometry'].lar_detector_bounds 
        PromptHits_ev = cache[self.hits_dset_name][0]

        PromptHits_ev_positions = np.column_stack((PromptHits_ev['x'], PromptHits_ev['y'], PromptHits_ev['z']))
        
        nan_indices = np.unique(np.argwhere(np.isnan(PromptHits_ev_positions))[:,0]) 
        
        if len(nan_indices) >   0:
            PromptHits_ev = np.delete(PromptHits_ev,nan_indices, axis = 0)

        if len(PromptHits_ev) < self.dbscan_min_samples:
            return
        
        hit_indices = self.cluster(PromptHits_ev)
        
        for indices in hit_indices:
            if len(indices) >= self.min_cluster_hits:
                hits = PromptHits_ev[indices]
                hits = self.clean_noise_hits(hits)
                if len(hits) < 1:
                    continue
                muon_track,length_of_track, start_point, end_point, explained_var, direction_vector = self.select_muon_track(hits,Min_max_detector_bounds)
                 
                if len(muon_track) != 0:
                    #Loop through tracks and changes the DBSCAN cluster_id to a given track number
                    self.track_count += 1 
                    track_number = self.track_count

                    #PCA gives the track axis but not its sign: point along the beam (+z), as in segments()
                    if direction_vector[2] < 0:
                        direction_vector = -direction_vector

                    #Get angle of track
                    theta_xz, theta_yz,theta_z = self.angle(direction_vector)

                    #Tag crossing tracks
                    a2a, a2c = self.anode_cathode_tags(muon_track[0])
                    
                    #Fill track info
                    track_info = [event_id[0],track_number,length_of_track, start_point[0],start_point[1],start_point[2], end_point[0],end_point[1],end_point[2], explained_var, theta_xz, theta_yz, theta_z, 
                                  direction_vector[0], direction_vector[1], direction_vector[2], 
                                  a2a, a2c
                                 ]
                    
                    track_info = np.array([tuple(track_info)], dtype = self.rock_muon_track_dtype)
                    #Get segments
                    segments_list, segment_hit_ref, segment_track_ref = self.segments(muon_track)
                    
                    #  1. reserve a new data region within the output dataset
                    rock_muon_slice = self.data_manager.reserve_data(self.rock_muon_hits_dset_name, 1)


                    #  2. write the data to the new data region
                    self.data_manager.write_data(self.rock_muon_hits_dset_name, rock_muon_slice, track_info)
            
                    segments_array = np.array([tuple(sub) for sub in segments_list], dtype = self.rock_muon_segments_dtype) #Converts array of list to array of tuples
            
                    nMuon_segments = len(segments_array)
                    # 3. reserve a new data region within the rock muon segment dataset
                    rock_muon_segments_slice = self.data_manager.reserve_data(self.rock_muon_segments_dset_name, nMuon_segments)

                    # 4. Write the data into the rock muon segments data region
                    self.data_manager.write_data(self.rock_muon_segments_dset_name, rock_muon_segments_slice, segments_array)
                    
                    #Reference hits to their track

                    
                    track_ref = np.array([(track_number,x) for x in muon_track['id'][0]])
                    
                    track_event_ref = np.array([(track_number, event_id[0])])
                    
                    #print(track_ref)            
                    segment_track_ref = np.array([(x) for x in segment_track_ref])
                     
                    segment_hit_ref = np.array([(x) for x in segment_hit_ref])
                    
                    #Write References
                    self.data_manager.write_ref(self.rock_muon_hits_dset_name,self.hits_dset_name, track_ref)
                    self.data_manager.write_ref(self.rock_muon_hits_dset_name,self.events_dset_name, track_event_ref) 
                    self.data_manager.write_ref(self.rock_muon_hits_dset_name,self.rock_muon_segments_dset_name, segment_track_ref)
                    self.data_manager.write_ref(self.rock_muon_segments_dset_name, self.hits_dset_name, segment_hit_ref)
                # event -> hit
                #self.data_manager.write_ref(self.rock_muon_segments_dset_name, self.rock_muon_hits_dset_name, ref)
                
