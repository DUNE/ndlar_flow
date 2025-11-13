#Imports
import warnings
import numpy as np
from h5flow.core import H5FlowStage, resources
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

class RockMuonSelection(H5FlowStage):
    '''
    This will perform a selection for rock muons. Rock muons are 
    determined by straight tracks that penetrate two faces of the 
    detector.
    '''
    #Empty List so tracks can be counted
    
    #Detector Boundaries, Cuts
    defaults = dict([
    ('x_boundaries', np.array([-63.931, -3.069, 3.069, 63.931])), #cm
        
    ('y_boundaries', np.array([-268-42-19.8543, -268-42+103.8543])),#cm
        
    ('z_boundaries', np.array([1300-64.3163,  1300-2.6837, 1300+2.6837, 1300+64.3163])), #cm
    
    
    ('length_cut', 100), #cm
        
    ('MEVR', 0.974), #Miniumum explained variance ratio

    ('track_count', -1),

    ('segment_count', -1)
    ])
    

    #Datasets
    events_dset_name = 'charge/events'
    PromptHits_dset_name = 'charge/calib_prompt_hits'
    
    #Datatype wanted
    
    rock_muon_track_dtype = np.dtype([('event_id','i4'),('rock_muon_id', 'i4'),('length','f8'),('x_start', 'f8'),('y_start','f8'),('z_start', 'f8'),('x_end','f8'),('y_end', 'f8'),('z_end', 'f8'),('exp_var', 'f8'), ('theta_xz','f8'), ('theta_yz', 'f8'), ('theta_z','f8')])
   
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
        ('nhits', 'i4'),
        ('dx','f8'),
        ('x_mid','f8'),
        ('y_mid','f8'),
        ('z_mid','f8'),
        ('t','f8'),
        ('io_group', 'i4')
    ])

    
    rock_muon_hits_dset_name = 'analysis/rock_muon_tracks'

    rock_muon_segments_dset_name = 'analysis/rock_muon_segments'
    
    def __init__(self, **params):
        
        super(RockMuonSelection,self).__init__(**params) # needed to inherit H5FlowStage functionality
        
        for key,val in self.defaults.items():
            setattr(self, key, params.get(key, val))
            
        self.length_cut = params.get('length_cut', dict())
        

    def init(self, source_name):
        
        super(RockMuonSelection, self).init(source_name)
        
        attrs = dict()
        
        for key in self.defaults:
            
            attrs[key] = getattr(self, key)
        
        #self.data_manager.set_attrs(self.path,
        #                            classname=self.classname,
        #                            class_version=self.class_version,
        #                            **attrs)
        
        self.data_manager.create_dset(self.rock_muon_hits_dset_name,
                                      dtype = self.rock_muon_track_dtype)
        
        self.data_manager.create_dset(self.rock_muon_segments_dset_name,
                                      dtype = self.rock_muon_segments_dtype)

        self.data_manager.create_ref(self.events_dset_name, self.rock_muon_hits_dset_name) 
        
        self.data_manager.create_ref(self.rock_muon_hits_dset_name,self.PromptHits_dset_name)

        self.data_manager.create_ref(self.rock_muon_hits_dset_name, self.rock_muon_segments_dset_name,)
        
        self.data_manager.create_ref(self.rock_muon_segments_dset_name, self.PromptHits_dset_name)
    

    def merge_test(self, main_cluster_direction:np.ndarray, main_cluster_mean:np.ndarray, test_clusters:np.ndarray, average_dist:float) -> bool:
        """Merge test clusters to main cluster."""
        
        distances = [
            self.average_distance(test_cluster,
                            main_cluster_mean, 
                            main_cluster_direction) 
            for test_cluster in test_clusters]
        
        
        indices = [index for index, dist in enumerate(distances) if dist <= average_dist]

        return indices
    
    def cluster(self, PromptHits_ev:np.ndarray, average_dist:float):
        """Cluster an event of hits, does not necessarily have to be prompt hits."""
        positions = np.column_stack((
            PromptHits_ev['x'],
            PromptHits_ev['y'],
            PromptHits_ev['z']
        ))

        dbscan = DBSCAN(min_samples=6, eps=4*.4434)
        clusters = dbscan.fit(positions)
        labels = clusters.labels_

        remove_noise = (labels != -1)

        non_noise_hits = PromptHits_ev[remove_noise]
        non_noise_positions = positions[remove_noise]
        non_noise_labels = labels[remove_noise]

        indicies_of_clusters = []

        for label in np.unique(non_noise_labels):

            indicies_of_clusters.append(
                np.where(non_noise_labels==label)[0]
                )

        direction_each_cluster = np.array([
            self.PCAs(non_noise_positions[indices])[1]
            for indices in indicies_of_clusters
            ])
        
        sorted_indices_cluster_directions = sorted(
        range(len(direction_each_cluster)),
        key=lambda v: max(abs(x) for x in direction_each_cluster[v]),
        reverse=True
        )
        
        positions_per_cluster = [
        non_noise_positions[indicies_of_clusters[i]].data
        for i in sorted_indices_cluster_directions
        ]

        sorted_indices_of_cluster = [
            indicies_of_clusters[index]
            for index in sorted_indices_cluster_directions
        ]
        sorted_directions = [direction_each_cluster[index] for index in sorted_indices_cluster_directions]

        mean_per_cluster = [
            np.mean(cluster, axis=0)
            for cluster in positions_per_cluster
        ]
        
        new_cluster_indices = []

        test = list(range(len(sorted_indices_of_cluster)))

        merged_flags = [False] * len(sorted_indices_of_cluster)
        
        new_cluster_indices = []
    
        for main_idx in test:
            
            if merged_flags[main_idx]:
                continue

            main_cluster_mean = mean_per_cluster[main_idx]
            main_cluster_direction = sorted_directions[main_idx]

            test_indices = [i for i in range(len(sorted_indices_of_cluster)) if i != main_idx and not merged_flags[i]]
            test_clusters = [positions_per_cluster[i] for i in test_indices]
            
            indices_merge = self.merge_test(main_cluster_direction, main_cluster_mean, test_clusters, average_dist)

            if indices_merge:
                clusters_to_merge = [main_idx] + [test_indices[i] for i in indices_merge]
                merge_indices_flattened = np.concatenate([sorted_indices_of_cluster[index] for index in clusters_to_merge])
                
                for index in clusters_to_merge:
                    merged_flags[index] =True

                new_cluster_indices.append(merge_indices_flattened)
                
            else:
                new_cluster_indices.append(sorted_indices_of_cluster[main_idx])
                merged_flags[main_idx] = True
                continue
        if new_cluster_indices:
            for j in range(len(new_cluster_indices)):
                new_c = non_noise_hits[new_cluster_indices[j]]
                
                indices = np.where(np.isin(PromptHits_ev,new_c))[0]
                new_cluster_indices[j] = indices
        
            return new_cluster_indices
        else:
            return indicies_of_clusters
        
    #@staticmethod
    def PCAs(self, hit_positions:np.ndarray):
        """Compute the PCA for a set of hit positions reutrning the direction and mean position."""
        warnings.filterwarnings(action='ignore', category=RuntimeWarning)

        #Scale data
        mean = np.mean(hit_positions, axis=0)
        std = np.std(hit_positions, axis=0)
        std1 = np.array([s if s != 0 else 1e-9 for s in std])

        X_train = (hit_positions - mean)/std1

        pca = PCA(1) # 1 component

        pca.fit(X_train)

        explained_var = pca.explained_variance_ratio_[0]
        scaled_direction_vector = pca.components_[0]
        unscaled_vector = std * scaled_direction_vector

        normalized_direction_vector = unscaled_vector/np.linalg.norm(unscaled_vector)

        return  explained_var, normalized_direction_vector, mean
    
    #@staticmethod
    def length(self, hits:np.ndarray):
        """Get length of track."""
        hit_positions = np.column_stack((hits['x'], hits['y'], hits['z']))
        
        hdist = cdist(hit_positions, hit_positions)
         
        max_value_index = np.argmax(hdist)

        max_value_row = max_value_index // hdist.shape[1]
        max_value_col = max_value_index % hdist.shape[1]
        
        indices = [max_value_row, max_value_col]
        
        start_hit, end_hit = hit_positions[np.min(indices)], hit_positions[np.max(indices)]
        
        return np.max(hdist), start_hit, end_hit

    def close_to_two_faces(self, boundaries, hits):
        """Test if a track goes through the detector."""
        penetrated = False

        test_face = [False] * len(boundaries)
        threshold = 2.1
        for index, face in enumerate(boundaries):
            if (index == 0) or (index == 3):
                distance = np.abs(face - hits['x'])

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

        if sum(test_face)>= 2:
            penetrated = True
    
        return penetrated

    def clean_noise_hits(self, positions, track_direction, hits_mean):
        """Returns mask of positions that are more than 3.5 centimeter away from track."""
        projections = np.dot(positions - hits_mean, track_direction[:, np.newaxis]) * track_direction + hits_mean

        # Calculate the Euclidean distance between each point and its projection on the line
        distances = np.linalg.norm(positions - projections, axis=1)
        
        mask_good = distances <= 3.5
        return mask_good
    
    def average_distance(self, positions:np.ndarray, hits_mean:np.ndarray, track_direction:np.ndarray):
        """Return average distance from track."""
        projections = np.dot(positions - hits_mean, track_direction[:, np.newaxis]) * track_direction + hits_mean

        distances = np.linalg.norm(positions - projections, axis=1)

        average_distances = np.mean(distances)

        return average_distances
    
    #@staticmethod
    def select_muon_track(self, hits, Min_max_detector_bounds):
            muon_hits = []

            min_boundaries = np.flip(Min_max_detector_bounds[0]) 
            max_boundaries = np.flip(Min_max_detector_bounds[1])
            
            faces_of_detector = np.concatenate((min_boundaries,max_boundaries))

            hit_positions = np.column_stack((
                hits['x'], hits['y'], hits['z']
            ))

            L_cut = self.length_cut 

            explained_var, direction_vector, hits_mean_position = self.PCAs(hit_positions)
            
            mask = self.clean_noise_hits(hit_positions, direction_vector, hits_mean_position)
            
            filtered_hits = hits[mask]
            
            avg_distance = self.average_distance(hit_positions[mask], hits_mean_position, direction_vector)
            
            l_track, start_point, end_point = self.length(filtered_hits)
            if (avg_distance <= 1.5) & (l_track >= L_cut):

                penetrated = self.close_to_two_faces(faces_of_detector, filtered_hits)

                if penetrated:

                    muon_hits.append(filtered_hits)

            return np.array(muon_hits), l_track, start_point, end_point, explained_var, direction_vector
    
    #@staticmethod
    def angle(self, direction_vector):
        """Get angles of muon."""
        magnitude = np.linalg.norm(direction_vector)

        normal_vector_xz = np.array([0, 1, 0])
        
        dot_product = np.dot(direction_vector, normal_vector_xz)

        theta_xz = np.arccos(dot_product / magnitude)

        theta_xz = np.degrees(theta_xz)
        
        normal_vector_yz = np.array([1, 0, 0])

        dot_product = np.dot(direction_vector, normal_vector_yz)

        theta_yz = np.arccos(dot_product / magnitude)

        theta_yz = np.degrees(theta_yz)

        theta_z = np.degrees(np.arctan2(np.sqrt(direction_vector[0]**2 + direction_vector[1]**2), direction_vector[2]))
        
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
        """Create rock muon segments."""
        segment_info = []

        hit_ref = []
        segment_to_track_ref = []

        track = muon_hits[0] #Makes sure hits go back to a (n,) shape instead of (1,n) shape
    
        tpc_hits = self.TPC_separation(track)

        given_scale = 2

        for hits in tpc_hits:
            if len(hits) != 0:
                hit_positions = np.array([[hit['x'], hit['y'], hit['z']] for hit in hits])
            
                tpc_var, principal_component, tpc_mean = self.PCAs(hit_positions)
       
                centered_points = hit_positions - tpc_mean

                projections = np.dot(centered_points, principal_component)
                projected_hits = tpc_mean + np.outer(projections, principal_component)

                t_min = np.min(projections)
                t_max = np.max(projections)

                #End points
                line_point_1 = tpc_mean + t_min * principal_component
                line_point_2 = tpc_mean + t_max * principal_component
            
                line_defined_points = [line_point_1,line_point_2]

                line_start = line_defined_points[
                    np.argmax([line_point_1[2], line_point_2[2]])
                    ]
                
                line_end = line_defined_points[
                    np.argmin([line_point_1[2], line_point_2[2]])
                    ]
            
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
                    
                    
                    if break_out:
                        break


        return segment_info, hit_ref, segment_to_track_ref

    def grab_segment_info(self,segment_end, segment_start, projected_hits, hits, hit_ref, segment_to_track_ref):
            """Create wanted segment information"""
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
            
                return [self.segment_count, x_start, y_start, z_start, Energy_of_segment, x_end, y_end, z_end, Q_of_segment,len(hits_of_segment), dx, x_mid, y_mid,z_mid, drift_time, io_group_of_segment]
            else:
                return None

    def run(self, source_name, source_slice, cache):
            
            super(RockMuonSelection, self).run(source_name, source_slice, cache)
                        
            event_id = np.r_[source_slice]
            
            Min_max_detector_bounds = resources['Geometry'].lar_detector_bounds 
            PromptHits_ev = cache[self.PromptHits_dset_name][0]
            

            PromptHits_ev_positions = np.column_stack((PromptHits_ev['x'], PromptHits_ev['y'], PromptHits_ev['z']))
            
            nan_indices = np.unique(np.argwhere(np.isnan(PromptHits_ev_positions))[:,0]) 
            
            if len(nan_indices) >   0:
                PromptHits_ev = np.delete(PromptHits_ev,nan_indices, axis = 0)

            unique_points, counts = np.unique(PromptHits_ev_positions, axis=0, return_counts=True)

            for unique_point, count in zip(unique_points, counts):

                if count > 1000:
                    mask = np.all(PromptHits_ev_positions != unique_point, axis =1)
    
                    PromptHits_ev = PromptHits_ev[mask]
                    
            if len(PromptHits_ev) >= 100:
                hit_indices = self.cluster(PromptHits_ev, 2)
            
            if 'hit_indices' in locals():
                for indices in hit_indices:
                    if len(indices) > 10:
                        hits = PromptHits_ev[indices]

                        if len(hits) < 1:
                            continue
                        muon_track,length_of_track, start_point, end_point, explained_var, direction_vector = self.select_muon_track(hits,Min_max_detector_bounds)
                        
                        if len(muon_track) != 0:
                            #Loop through tracks and changes the DBSCAN cluster_id to a given track number
                            self.track_count += 1 
                            track_number = self.track_count
                            
                            #Get angle of track
                            theta_xz, theta_yz,theta_z = self.angle(direction_vector)
                            
                            #Fill track info
                            track_info = [event_id,track_number,length_of_track, start_point[0],start_point[1],start_point[2], end_point[0],end_point[1],end_point[2], explained_var, theta_xz, theta_yz, theta_z]
                            
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
                            self.data_manager.write_ref(self.rock_muon_hits_dset_name,self.PromptHits_dset_name, track_ref)
                            self.data_manager.write_ref(self.rock_muon_hits_dset_name,self.events_dset_name, track_event_ref) 
                            self.data_manager.write_ref(self.rock_muon_hits_dset_name,self.rock_muon_segments_dset_name, segment_track_ref)
                            self.data_manager.write_ref(self.rock_muon_segments_dset_name, self.PromptHits_dset_name, segment_hit_ref)
