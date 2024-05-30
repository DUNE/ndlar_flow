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
    
    rock_muon_track_dtype = np.dtype([('rock_muon_id', 'i4'),('length','f8'),('x_start', 'f8'),('y_start','f8'),('z_start', 'f8'),('x_end','f8'),('y_end', 'f8'),('z_end', 'f8'),('exp_var', 'f8'), ('theta_xz','f8'), ('theta_yz', 'f8')])
   
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
        ('t','f8')
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
        
        hit_cluster = DBSCAN(eps = 8, min_samples = 3).fit(positions)
         
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
    #@staticmethod
    def close_to_two_faces(self,boundaries, start_point, end_point):
        end_point_distance_to_min_bounds = abs(boundaries - end_point)
        start_point_distance_to_min_bounds = abs(boundaries - start_point)
    
        end_mask_within_2cm, start_mask_within_2cm = np.ravel(end_point_distance_to_min_bounds <= 2), np.ravel(start_point_distance_to_min_bounds <= 2)
    
        end_indices_within_2cm = np.where(end_mask_within_2cm == True)[0]
        start_indices_within_2cm = np.where(start_mask_within_2cm == True)[0]
    
        if (len(end_indices_within_2cm) > 0) & (len(start_indices_within_2cm) > 0):
            penetrated = np.any(start_indices_within_2cm != end_indices_within_2cm)
        else:
            penetrated = False
        
        return penetrated
    
    #@staticmethod
    def select_muon_track(self,hits):
            muon_hits = []
            tpc_bounds = np.array(resources['Geometry'].regions)/10 #Numbers are off by factor of 10

            bounds = np.concatenate(resources['Geometry'].regions)/10

            mask_upper = (bounds[:,0] == np.max(bounds[:,0])) & (bounds[:,1] == np.max(bounds[:,1])) & (bounds[:,2] == np.max(bounds[:,2]))
            mask_lower = (bounds[:,0] == np.min(bounds[:,0])) & (bounds[:,1] == np.min(bounds[:,1])) & (bounds[:,2] == np.min(bounds[:,2]))
            
            min_boundaries = np.flip(bounds[mask_lower]) #bounds are z,y,x and hits x,y,z, so bounds must be flipped
            max_boundaries = np.flip(bounds[mask_upper])
            
            faces_of_detector = np.concatenate((min_boundaries,max_boundaries))
            
            MEVR = self.MEVR #Minimum explained variance ratio

            L_cut = self.length_cut #minimum track length requirement
                
            explained_var, direction_vector,mean_point = self.PCAs(hits)
                
            l_track, start_point, end_point = self.length(hits)
                         
            if (explained_var > MEVR) & (l_track > L_cut):

                penetrated = self.close_to_two_faces(faces_of_detector, start_point, end_point)

                if penetrated == True:
                    muon_hits.append(hits)

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
        string_x_boundaries = self.x_boundaries
        string_y_boundaries = self.y_boundaries
        string_z_boundaries = self.z_boundaries

        x_boundaries = [float(string) for string in string_x_boundaries]
        y_boundaries= [eval(string) for string in string_y_boundaries]
        z_boundaries = [eval(string) for string in string_z_boundaries]

         
        mask_tpc1 = ((hits['x'] > x_boundaries[2]) & (hits['x'] < (x_boundaries[2] + x_boundaries[3]) * 0.5)) & (hits['z'] > z_boundaries[2])
        mask_tpc2 = (hits['x'] > (x_boundaries[2] + x_boundaries[3]) * 0.5) & (hits['z'] > z_boundaries[2])

        mask_tpc3 = (hits['x'] < (x_boundaries[0] + x_boundaries[1]) * 0.5) & (hits['z'] > z_boundaries[2])
        mask_tpc4 = ((hits['x'] > (x_boundaries[0] + x_boundaries[1]) * 0.5) & (hits['x']< x_boundaries[1])) & (hits['z'] > z_boundaries[2])

        mask_tpc5 = ((hits['x'] > x_boundaries[2]) & (hits['x'] < (x_boundaries[2] + x_boundaries[3]) * 0.5)) & (hits['z'] < z_boundaries[1])
        mask_tpc6 = (hits['x'] > (x_boundaries[2] + x_boundaries[3]) * 0.5) & (hits['z'] < z_boundaries[1])

        mask_tpc7 = (hits['x'] < (x_boundaries[0] + x_boundaries[1]) * 0.5) & (hits['z'] < z_boundaries[1])
        mask_tpc8 = ((hits['x'] > (x_boundaries[0] + x_boundaries[1]) * 0.5) & (hits['x'] < x_boundaries[1])) & (hits['z'] < z_boundaries[1])

        hits_tpc1 = hits[mask_tpc1]
        hits_tpc2 = hits[mask_tpc2]

        hits_tpc3 = hits[mask_tpc3]
        hits_tpc4 = hits[mask_tpc4]

        hits_tpc5 = hits[mask_tpc5]
        hits_tpc6 = hits[mask_tpc6]

        hits_tpc7 = hits[mask_tpc7]
        hits_tpc8 = hits[mask_tpc8]

        return [hits_tpc1, hits_tpc2, hits_tpc3, hits_tpc4, hits_tpc5, hits_tpc6, hits_tpc7, hits_tpc8]
    
    #@staticmethod
    def get_segment_step_size(self, direction_vector, number_of_hits):
        #Steps in beam direction using pca fit(z)
        desired_factor = 200

        step_dz = number_of_hits/desired_factor

        scale = abs(step_dz/direction_vector[2])
        
        dx = np.linalg.norm(direction_vector) * scale
        
        if dx > 5:
            while dx > 5:
                desired_factor = desired_factor + 10
                step_dz = number_of_hits/desired_factor
                scale = step_dz/direction_vector[2]
                dx = np.linalg.norm(direction_vector) * scale

        elif dx < 1:
            while dx < 1:
                desired_factor = desired_factor - 10
                step_dz = number_of_hits/desired_factor
                scale = step_dz/direction_vector[2]
                dx = np.linalg.norm(direction_vector) * scale
                
        return step_dz, scale, dx
    
    #@staticmethod
    def grab_segment_info(self, direction_vector, segment, scale, mean_point, positive_values, negative_values, hit_ref, values,i):
        for hit in segment:
            hit_ref.append([self.segment_count,hit['id']])

        if (direction_vector[2] > 0 and values == negative_values) or (direction_vector[2] < 0 and values == positive_values):
            direction_vector = -direction_vector

        start_of_segment = mean_point + (abs(i)-1)*direction_vector*scale
        end_of_segment = mean_point + abs(i)*direction_vector*scale

        x_start, y_start, z_start = start_of_segment[0], start_of_segment[1], start_of_segment[2]
        x_end, y_end, z_end = end_of_segment[0], end_of_segment[1], end_of_segment[2]
        x_mid, y_mid, z_mid = (x_start+x_end)/2 , (y_start+y_end)/2 , (z_start+z_end)/2

        #dEdx

        Energy_of_segment = sum(segment['E'])

        # dQdx
        Charge_of_segment = sum(segment['Q'])

        #Drift time of segment
        drift_time = np.mean(segment['t_drift'])
    
        return x_start, y_start, z_start, Energy_of_segment, x_end, y_end, z_end, Charge_of_segment, x_mid, y_mid, z_mid, drift_time

    #@staticmethod
    def segments(self,muon_hits):
        hits_of_segment = []
        hit_segments = []
        
        segment_to_track_ref = []
        hit_ref = []
        
        track = muon_hits[0] #Makes sure hits go back to a (n,) shape instead of (1,n) shape
        
        ex_var, direction_vector, mean_point = self.PCAs(track)
         
        number_of_hits = len(track)
            
        step_size, scale, dx=self.get_segment_step_size(direction_vector, number_of_hits)
        
        hits_per_tpc = self.TPC_separation(track)
        
         
        for each_tpc in hits_per_tpc:
            
            if len(each_tpc) != 0:

                ex_var, direction_vector2, mean_point = self.PCAs(each_tpc)
                
                amount_of_steps_up = round(((np.max(each_tpc['z']) - mean_point[2])/step_size) +.5) 
                amount_of_steps_down = round(((mean_point[2]-np.min(each_tpc['z']))/step_size) +.5)
                 
                positive_values = range(1,amount_of_steps_up+1)
                negative_values = range(-1,-(amount_of_steps_down+1),-1)
                
                #Start at the mean point, jump up and make segments (positive values) then jump down and make segments (negative values)
                for values in [positive_values, negative_values]:
                    for i in values:
                        
                        if i > 0:
                            mask = ma.masked_where((each_tpc['z'] >= mean_point[2] + step_size*(i-1)) & (each_tpc['z'] < mean_point[2] + step_size*(i)),each_tpc)

                        
                        elif i < 0:
                            mask = ma.masked_where((each_tpc['z'] <= mean_point[2] - step_size*(abs(i)-1)) & (each_tpc['z'] > mean_point[2] - step_size*(abs(i))),each_tpc)
                            
                        indices_of_masked_values = np.ma.where(mask.mask)[0]
                    
                        min_number_of_hits = 3
                    
                        segment = each_tpc[indices_of_masked_values]
                        
                        #If segment has more than min_number_of_hits then consider this a good enough segment and append it to the list of segments
                        if len(segment) > min_number_of_hits:
                            self.segment_count += 1
                            hits_of_segment.append(segment)
                        #If there is no segments in hits of segment and # of hits of the considered segments in less than min number of hits then still append this segment. This stops code for messing at when first segment of track isn't long enough
                        elif (len(segment) >= 1) & (len(segment) <= min_number_of_hits) & (len(hits_of_segment) == 0):
                            self.segment_count += 1
                            hits_of_segment.append(segment)
                            
                        #If # of hits of the considered segments is below min hits and there is already segments then considered this hits as part of the previous segment, this is not making a new segment but adding to previous segment. Must recalculate the segment info for previous segment.
                        elif (len(segment) >= 1) & (len(segment) <= min_number_of_hits) & (len(hits_of_segment) != 0):
                            np.append(hits_of_segment[-1],segment)

                            x_start, y_start, z_start, Energy_of_segment, x_end, y_end, z_end, Charge_of_segment, x_mid, y_mid, z_mid, drift_time = self.grab_segment_info(direction_vector, hits_of_segment[-1], scale, mean_point, positive_values, negative_values, hit_ref, values,i)

                            hit_segments[-1] = [self.segment_count, x_start, y_start, z_start, Energy_of_segment, x_end, y_end, z_end, Charge_of_segment,dx, x_mid, y_mid, z_mid, drift_time]
                            
                            each_tpc = np.delete(each_tpc, indices_of_masked_values)

                            continue
                        else:
                            continue
                        each_tpc = np.delete(each_tpc, indices_of_masked_values)

                        x_start, y_start, z_start, Energy_of_segment, x_end, y_end, z_end, Charge_of_segment, x_mid, y_mid, z_mid, drift_time = self.grab_segment_info(direction_vector, hits_of_segment[-1], scale, mean_point, positive_values, negative_values, hit_ref, values, i)

                        hit_segments.append([self.segment_count, x_start, y_start, z_start, Energy_of_segment, x_end, y_end, z_end, Charge_of_segment,dx, x_mid, y_mid, z_mid,drift_time])

                        #Reference the segment to the track
                        segment_to_track_ref.append([self.track_count, self.segment_count])
               
        return hit_segments, hit_ref, segment_to_track_ref


    def run(self, source_name, source_slice, cache):
        
        super(RockMuonSelection, self).run(source_name, source_slice, cache)
                    
        event_id = np.r_[source_slice]
                
        PromptHits_ev = cache[self.FinalHits_dset_name][0]
        
        hit_indices = self.cluster(PromptHits_ev)
        
        for indices in hit_indices:
            if len(indices) > 10:
                hits = PromptHits_ev[indices]
                 
                muon_track,length_of_track, start_point, end_point, explained_var, direction_vector = self.select_muon_track(hits)
                 
                if len(muon_track) != 0:
                    #Loop through tracks and changes the DBSCAN cluster_id to a given track number
                    self.track_count += 1 
                    track_number = self.track_count
                     
                    #Get angle of track
                    theta_xz, theta_yz = self.angle(direction_vector)
                     
                    #Fill track info
                    track_info = [track_number,length_of_track, start_point[0],start_point[1],start_point[2], end_point[0],end_point[1],end_point[2], explained_var, theta_xz, theta_yz]
                    
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
                    
                    #print(track_ref)            
                    segment_track_ref = np.array([(x) for x in segment_track_ref])
                     
                    segment_hit_ref = np.array([(x) for x in segment_hit_ref])
                    
                    #Write References
                    self.data_manager.write_ref(self.rock_muon_hits_dset_name,self.PromptHits_dset_name, track_ref)
                
                    self.data_manager.write_ref(self.rock_muon_hits_dset_name,self.rock_muon_segments_dset_name, segment_track_ref)

                    self.data_manager.write_ref(self.rock_muon_segments_dset_name, self.PromptHits_dset_name, segment_hit_ref)
                # event -> hit
                #self.data_manager.write_ref(self.rock_muon_segments_dset_name, self.rock_muon_hits_dset_name, ref)
                
