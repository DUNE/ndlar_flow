#Imports
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
    '''
    #Empty List so tracks can be counted
    
    #Detector Boundaries, Cuts
    defaults = dict([
    ('x_boundaries', np.array([-63.931, -3.069, 3.069, 63.931])), #cm
        
    ('y_boundaries', np.array([-268-42-19.8543, -268-42+103.8543])),#cm
        
    ('z_boundaries', np.array([1300-64.3163,  1300-2.6837, 1300+2.6837, 1300+64.3163])), #cm
    
    
    ('length_cut', 55), #cm
        
    ('MEVR', 0.974), #Miniumum explained variance ratio

    ('track_count', -1),

    ('segment_count', -1)
    ])
    
    #Datasets
    events_dset_name = 'charge/events'
    PromptHits_dset_name = 'charge/calib_prompt_hits'
    FinalHits_dset_name = 'charge/calib_final_hits'
     
    #Datatype wanted
    
    rock_muon_track_dtype = np.dtype([('event_id','i4'),('rock_muon_id', 'i4'),('length','f8'),('x_start', 'f8'),('y_start','f8'),('z_start', 'f8'),('x_end','f8'),('y_end', 'f8'),('z_end', 'f8'),('exp_var', 'f8'), ('theta_xz','f8'), ('theta_yz', 'f8')])
   
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
        ('io_group', 'i4')
    ])

    
    rock_muon_hits_dset_name = 'analysis/rock_muon_tracks'

    rock_muon_segments_dset_name = 'analysis/rock_muon_segments'
    
    def __init__(self, **params):
        
        super(RockMuonSelection,self).__init__(**params) # needed to inherit H5FlowStage functionality
        
        for key,val in self.defaults.items():
            setattr(self, key, params.get(key, val))
            
        self.x_boundaries = params.get('x_boundaries',dict())
        
        self.y_boundaries = params.get('y_boundaries', dict())
        
        self.z_boundaries = params.get('z_boundaries', dict())
        
        self.length_cut = params.get('length_cut', dict())
 
        self.MEVR = params.get('MEVR', dict())
            
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
    
    #@staticmethod
    def cluster(self, PromptHits_ev):
        
        index_of_track_hits = []

        positions = np.column_stack((PromptHits_ev['x'], PromptHits_ev['y'], PromptHits_ev['z']))
        
        hit_cluster = DBSCAN(eps = 8, min_samples = 1).fit(positions)
        
        unique_labels = np.unique(hit_cluster.labels_)

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
    def close_to_two_faces(self, boundaries, start_point, end_point):
        # Boundaries are in the order [xmin, ymin, zmin, xmax, ymax, zmax]
        min_bounds = boundaries[:3]  # [xmin, ymin, zmin]
        max_bounds = boundaries[3:]  # [xmax, ymax, zmax]

        # Calculate the distances from points to min and max bounds
        end_distances_to_min_bounds = abs(min_bounds - end_point)
        end_distances_to_max_bounds = abs(max_bounds - end_point)
        start_distances_to_min_bounds = abs(min_bounds - start_point)
        start_distances_to_max_bounds = abs(max_bounds - start_point)

        # Check if any distance is within 2 cm for both min and max boundaries
        end_mask_within_2cm_min = (end_distances_to_min_bounds <= 2) 
        end_mask_within_2cm_max = (end_distances_to_max_bounds <= 2)
        start_mask_within_2cm_min = (start_distances_to_min_bounds <= 2)
        start_mask_within_2cm_max = (start_distances_to_max_bounds <= 2)
        
        end_mask_within_2cm = np.concatenate((end_mask_within_2cm_min,end_mask_within_2cm_max))
        start_mask_within_2cm = np.concatenate((start_mask_within_2cm_min,start_mask_within_2cm_max))
        

        # Identify which boundaries are within 2 cm
        end_indices_within_2cm = np.where(end_mask_within_2cm)[0]
        start_indices_within_2cm = np.where(start_mask_within_2cm)[0]

        # Determine if the track penetrates through different boundaries
        if len(end_indices_within_2cm) > 0 and len(start_indices_within_2cm) > 0:
            # Check if there are any different indices (meaning close to different boundaries)
            penetrated = np.any(start_indices_within_2cm != end_indices_within_2cm)
        else:
            penetrated = False

        return penetrated

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
        
        mask_good = distances <= 3.5

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
            
            min_boundaries = Min_max_detector_bounds[0] #bounds are z,y,x and hits x,y,z, so bounds must be flipped
            max_boundaries = Min_max_detector_bounds[1]
            
            faces_of_detector = np.concatenate((min_boundaries,max_boundaries))
            
            MEVR = self.MEVR #Minimum explained variance ratio

            L_cut = self.length_cut #minimum track length requirement
            
            filtered_hits = self.clean_noise_hits(hits)
            
            explained_var, direction_vector,mean_point = self.PCAs(filtered_hits)
                
            l_track, start_point, end_point = self.length(filtered_hits)
            
            avg_distance = self.average_distance(filtered_hits)

            if (avg_distance < 1.5) & (l_track > L_cut):

                penetrated = self.close_to_two_faces(faces_of_detector, start_point, end_point)

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
        
        return theta_xz, theta_yz

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
    def grab_segment_info(self, direction_vector, segment, mean_point, positive_values, negative_values, hit_ref, values,i):
        for hit in segment:
            hit_ref.append([self.segment_count,hit['id']])

        if (direction_vector[2] > 0 and values == negative_values) or (direction_vector[2] < 0 and values == positive_values):
            direction_vector = -direction_vector

        diff_in_z = abs(np.max(segment['z']) - np.min(segment['z']))
         
        dx = abs(diff_in_z/direction_vector[2])
        #print(dx) 
        start_of_segment = mean_point + (abs(i)-1)*direction_vector*dx
        end_of_segment = mean_point + abs(i)*direction_vector*dx

        x_start, y_start, z_start = start_of_segment[0], start_of_segment[1], start_of_segment[2]
        x_end, y_end, z_end = end_of_segment[0], end_of_segment[1], end_of_segment[2]
        x_mid, y_mid, z_mid = (x_start+x_end)/2 , (y_start+y_end)/2 , (z_start+z_end)/2

        #dEdx

        Energy_of_segment = sum(segment['E'])

        # dQdx
        Charge_of_segment = sum(segment['Q'])

        #Drift time of segment
        drift_time = np.mean(segment['t_drift'])
    
        return x_start, y_start, z_start, Energy_of_segment, x_end, y_end, z_end, Charge_of_segment, x_mid, y_mid, z_mid, drift_time, dx

    #@staticmethod
    def segments(self,muon_hits, est_length_of_track):
        segment_info = []
        
        segment_to_track_ref = []
        hit_ref = []
        
        track = muon_hits[0] #Makes sure hits go back to a (n,) shape instead of (1,n) shape
        
        #l_track, start_point_track, end_point_track = self.length(track)
        ex_var, direction, mean_point = self.PCAs(track)
        #print(sum(track['Q'])/l_track)
        number_of_hits = len(track)
        hit_density = number_of_hits/est_length_of_track
        
        tpc_hits = self.TPC_separation(track)
        
        same_pixel_pitch = [1,2,3,4,7,8]

        for hits in tpc_hits:
            io_group_of_tpc = np.unique(hits['io_group'])
            
            if io_group_of_tpc in same_pixel_pitch:
                scale = 0.4434
            else:
                scale = 0.3857
            if len(hits) != 0:
                tpc_var, tpc_direction, tpc_mean = self.PCAs(hits)

                max_z = np.max(hits['z'])
                min_z = np.min(hits['z'])

                positive_steps = np.arange(1,1000)
                negative_steps = np.arange(-1,-1000,-1)
        
                for steps in [positive_steps, negative_steps]:
                    for i in steps:
                        break_out = False
                        if (direction[2] < 0 and i > 0) or (direction[2] > 0 and i < 0):
                            direction_to_jump = - direction
                    
                        else:
                            direction_to_jump = direction

                        segment_start = tpc_mean + (abs(i)-1)*scale*direction_to_jump
                        segment_end = tpc_mean + abs(i)*scale*direction_to_jump
                        '''
                        if abs(segment_point1[2]) > abs(segment_point2[2]):
                            segment_end = segment_point1
                            segment_start = segment_point2
                        else:
                            segment_end = segment_point2
                            segment_start = segment_point1
                        '''
                        if (segment_end[2] > max_z) & (i > 0):
                            jump_size = (max_z - segment_start[2])/direction_to_jump[2]
                            segment_end = segment_start + jump_size*direction_to_jump
                            break_out = True
                             
                        if (segment_end[2] < min_z) & (i < 0):
                            jump_size = abs(min_z - segment_start[2])/abs(direction_to_jump[2])
                            #print('inital end',segment_end[2])
                            segment_end = segment_start + jump_size*direction_to_jump
                            
                            break_out = True
                            #print('final end , start', segment_end[2], segment_start[2])
                            #print('min z', min_z)
                        #Grab segment info
                        lower_bound = np.min([segment_start[2], segment_end[2]])
                        upper_bound = np.max([segment_start[2], segment_end[2]])
                
                        mask = ma.masked_where((hits['z'] >= lower_bound) & (hits['z'] <= upper_bound), hits)
                        indices_of_masked_values = np.ma.where(mask.mask)[0]
                        hits_of_segment = hits[indices_of_masked_values]
                        
                        if len(hits_of_segment) == 0:
                            continue

                        hits = np.delete(hits, indices_of_masked_values)
                
                        x_start, y_start, z_start = segment_start[0], segment_start[1], segment_start[2]
                        x_end, y_end, z_end = segment_end[0], segment_end[1], segment_end[2]
                        x_mid, y_mid, z_mid = (x_start+x_end)/2, (y_start + y_end)/2, (z_start + z_end)/2
                
                        Energy_of_segment = sum(hits_of_segment['E'])
                        Q_of_segment = sum(hits_of_segment['Q'])
                        drift_time = np.mean(hits_of_segment['t_drift'])
                        
                        io_group_of_segment = np.unique(hits_of_segment['io_group'])[0]
                        self.segment_count += 1

                        dx = np.linalg.norm(segment_start-segment_end)
                        segment_info.append([self.segment_count, x_start, y_start, z_start, Energy_of_segment, x_end, y_end, z_end, Q_of_segment, dx, x_mid, y_mid,z_mid, drift_time, io_group_of_segment])
                    
                        for hitss in hits_of_segment:
                            hit_ref.append([self.segment_count, hitss['id']])

                        segment_to_track_ref.append([self.track_count, self.segment_count])
                        if break_out:
                            #print('Now breaking', break_out)
                            break
      
        return segment_info, hit_ref, segment_to_track_ref


    def run(self, source_name, source_slice, cache):
        
        super(RockMuonSelection, self).run(source_name, source_slice, cache)
                    
        event_id = np.r_[source_slice]
        Min_max_detector_bounds = resources['Geometry'].lar_detector_bounds
         
        PromptHits_ev = cache[self.PromptHits_dset_name][0]

        PromptHits_ev_positions = np.column_stack((PromptHits_ev['x'], PromptHits_ev['y'], PromptHits_ev['z']))
        
        nan_indices = np.unique(np.argwhere(np.isnan(PromptHits_ev_positions))[:,0]) 
        
        if len(nan_indices) >   0:
            PromptHits_ev = np.delete(PromptHits_ev,nan_indices, axis = 0)
        
        hit_indices = self.cluster(PromptHits_ev)
        
        for indices in hit_indices:
            if len(indices) > 10:
                hits = PromptHits_ev[indices]
                hits = self.clean_noise_hits(hits)
                if len(hits) < 1:
                    continue
                muon_track,length_of_track, start_point, end_point, explained_var, direction_vector = self.select_muon_track(hits,Min_max_detector_bounds)
                 
                if len(muon_track) != 0:
                    #Loop through tracks and changes the DBSCAN cluster_id to a given track number
                    self.track_count += 1 
                    track_number = self.track_count
                    
                    #Get angle of track
                    theta_xz, theta_yz = self.angle(direction_vector)
                    
                    #Fill track info
                    track_info = [event_id,track_number,length_of_track, start_point[0],start_point[1],start_point[2], end_point[0],end_point[1],end_point[2], explained_var, theta_xz, theta_yz]
                    
                    track_info = np.array([tuple(track_info)], dtype = self.rock_muon_track_dtype)
                    #Get segments
                    segments_list, segment_hit_ref, segment_track_ref = self.segments(muon_track, length_of_track)
                    
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
                    
                    #print(track_ref)            
                    segment_track_ref = np.array([(x) for x in segment_track_ref])
                     
                    segment_hit_ref = np.array([(x) for x in segment_hit_ref])
                    
                    #Write References
                    self.data_manager.write_ref(self.rock_muon_hits_dset_name,self.PromptHits_dset_name, track_ref)
                
                    self.data_manager.write_ref(self.rock_muon_hits_dset_name,self.rock_muon_segments_dset_name, segment_track_ref)

                    self.data_manager.write_ref(self.rock_muon_segments_dset_name, self.PromptHits_dset_name, segment_hit_ref)
                # event -> hit
                #self.data_manager.write_ref(self.rock_muon_segments_dset_name, self.rock_muon_hits_dset_name, ref)
                
