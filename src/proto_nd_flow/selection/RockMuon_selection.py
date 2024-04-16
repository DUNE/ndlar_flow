#Imports
import numpy as np
import numpy.ma as ma

import h5py
from h5flow.core import H5FlowStage, resources
from h5flow import H5FLOW_MPI
import h5flow
from h5flow.data import dereference

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter

import statistics

import glob

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
    
    rock_muon_hit_dtype = np.dtype([('id', 'u4'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('t_drift', 'f8'), ('ts_pps', 'u8'), ('Q', 'f8'), ('E', 'f8'), ('rock_muon_id','u4'), ('rock_segment_id','u4')])
   
    rock_muon_segments_dtype = np.dtype([
        ('rock_segment_id', 'u8'),
        ('x_start', 'f8'),
        ('y_start','f8'),
        ('z_start','f8'),
        ('dEdX', 'f8'),
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

    
    rock_muon_hits_dset_name = 'rock_muon_hits'

    rock_muon_segments_dset_name = 'rock_muon_segments'
    
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
                                      dtype = self.rock_muon_hit_dtype)
        
        self.data_manager.create_dset(self.rock_muon_segments_dset_name,
                                      dtype = self.rock_muon_segments_dtype)

        self.data_manager.create_ref(self.events_dset_name, self.rock_muon_hits_dset_name) 
    
        self.data_manager.create_ref(self.rock_muon_segments_dset_name, self.rock_muon_hits_dset_name)
        
        self.data_manager.create_ref(source_name, self.rock_muon_hits_dset_name)
    
    #@staticmethod
    def cluster(self, PromptHits_ev):
        
        positions = np.column_stack((PromptHits_ev['x'], PromptHits_ev['y'], PromptHits_ev['z']))
        
        hit_cluster = DBSCAN(eps = 8, min_samples = 1).fit(positions)
         
        dummy_segment_id = np.full(len(positions), -1)
        
        hits_no_dtype = np.column_stack((PromptHits_ev['id'],PromptHits_ev['x'], PromptHits_ev['y'], PromptHits_ev['z'], PromptHits_ev['t_drift'],PromptHits_ev['ts_pps'], PromptHits_ev['Q'], PromptHits_ev['E'], hit_cluster.labels_, dummy_segment_id))
        
        hits = np.array([tuple(sub) for sub in hits_no_dtype], dtype = self.rock_muon_hit_dtype)
         
        return hits
    
    #@staticmethod
    
    def PCAs(self,hits_of_track):
        scaler = StandardScaler()

        positions = np.column_stack((hits_of_track['x'], hits_of_track['y'], hits_of_track['z']))
        X_train = positions
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)

        pca = PCA(1) # 1 component

        pca.fit(X_train)

        explained_var = pca.explained_variance_ratio_
        vector = pca.components_

        return  explained_var,vector
    
    #@staticmethod
    def length(self,hits):
        #Get Hit positions
        hit_positions = np.column_stack((hits['x'], hits['y'], hits['z']))

        #Check to see if a column has hit position of same value
        hit_result = np.all(hit_positions == hit_positions[0,:], axis = 0)
        same_result = np.where(hit_result == True)[0]

        if (len(same_result) == 0) & (len(hit_positions) > 4):
            hull = ConvexHull(hit_positions)
            vertexpoints = hit_positions[hull.vertices]
            hdist = cdist(vertexpoints, vertexpoints, metric='euclidean')


        elif (len(same_result) == 1) & (len(hit_positions) > 4):
            hits_not_same = np.delete(hit_positions, same_result, axis = 1)
            hull = ConvexHull(hits_not_same)
            vertexpoints = hits_not_same[hull.vertices]
            hdist = cdist(vertexpoints, vertexpoints, metric='euclidean')

        else:
            hdist = cdist(hit_positions, hit_positions)

        return np.max(hdist)

    #@staticmethod
    def select_muon_track(self,hits):

            tracks = np.unique(hits['rock_muon_id']) #This gets how many tracks there are
            
            muon_hits = []

            a = []

            #Boundaries of the detector
            x_boundaries = self.x_boundaries
            y_boundaries = self.y_boundaries
            z_boundaries = self.z_boundaries

            #Bottom face for each coordinate
            min_boundariess = np.array([x_boundaries[0], y_boundaries[0], z_boundaries[0]])
            min_boundaries = [eval(i) for i in min_boundariess] #turn the str values into floats

            #Top face for each coordinate
            max_boundariess = np.array([x_boundaries[-1], y_boundaries[-1], z_boundaries[-1]])
            max_boundaries = [eval(i) for i in max_boundariess] #turn the str values into floats
            
            for each_track in tracks:
                
                hits1 = hits['rock_muon_id'] == each_track #get the hits with track number equal to n

                hits_with_track = hits[hits1] #position of the hits with their associated track number

                hits_of_track_no_dtype = np.column_stack((hits_with_track['x'], hits_with_track['y'], hits_with_track['z'])) #hits without their track number
                
                hits_of_track = np.array([tuple(sub) for sub in hits_of_track_no_dtype], dtype = np.dtype([('x', 'f8'),('y', 'f8'),('z', 'f8')]))
                

                for i in range(len(max_boundaries)):
                    d = 2 #Distance away from a TPC face

                    v = self.MEVR #Minimum explained variance ratio

                    b = self.length_cut #minimum track length requirement

                    #Does the track penetrate two faces?
                    if min_boundaries[i]-d < np.min(hits_of_track_no_dtype[:,i]) < (min_boundaries[i] + d) and (max_boundaries[i] -d) < np.max(hits_of_track_no_dtype[:,i]) < max_boundaries[i]+d:
                        a, p  = self.PCAs(hits_of_track)
                        l = self.length(hits_of_track)
                        if np.logical_and(a > v, l > b):
                            muon_hits.append(hits_with_track)
                            break
                    elif min_boundaries[i]-d < np.min(hits_of_track_no_dtype[:,i]) < (min_boundaries[i] + d) and (max_boundaries[i-2] -d) < np.max(hits_of_track_no_dtype[:,i-2]) < (max_boundaries[i-2]+d):
                        a, p = self.PCAs(hits_of_track)
                        l = self.length(hits_of_track)
                        if np.logical_and(a > v, l > b):
                            muon_hits.append(hits_with_track)
                            break
                    elif min_boundaries[i]-d < np.min(hits_of_track_no_dtype[:,i]) < (min_boundaries[i] + d) and (max_boundaries[i-1] -d) < np.max(hits_of_track_no_dtype[:,i-1]) < max_boundaries[i-1]+d:
                        a, p = self.PCAs(hits_of_track)
                        l = self.length(hits_of_track)
                        if np.logical_and(a > v, l > b):
                            muon_hits.append(hits_with_track)
                            break
                    elif min_boundaries[i]-d < np.min(hits_of_track_no_dtype[:,i]) < (min_boundaries[i] + d) and (min_boundaries[i-1] -d) < np.min(hits_of_track_no_dtype[:,i-1]) < min_boundaries[i-1]+d:
                        a, p = self.PCAs(hits_of_track)
                        l = self.length(hits_of_track)
                        if np.logical_and(a > v, l > b):
                            muon_hits.append(hits_with_track)
                            break
                    elif min_boundaries[i]-d < np.min(hits_of_track_no_dtype[:,i]) < (min_boundaries[i] + d) and (min_boundaries[i-2] -d) < np.min(hits_of_track_no_dtype[:,i-2]) < min_boundaries[i-2]+d:
                        a, p = self.PCAs(hits_of_track)
                        l = self.length(hits_of_track)
                        if np.logical_and(a > v, l > b):
                            muon_hits.append(hits_with_track)
                            break
            return muon_hits
    

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
    def segments(self,muon_hits):
        hits_of_segment = []
        hit_segments = []

        ref = []
        
        for each_track in np.unique(muon_hits['rock_muon_id']):
            hits_wanted = muon_hits['rock_muon_id'] == each_track

            track = muon_hits[hits_wanted]

            ex_var, fit = self.PCAs(track)
            
            number_of_hits = len(track)
            
            #prompt_hits_constant = 100
            final_hits_constant = 80
            choice = len(track)/final_hits_constant
            #Steps in beam direction using pca fit(z)
            step_size = -choice*abs(fit[0][2])
            print(step_size)
            hits_per_tpc = self.TPC_separation(track)

            for each_tpc in hits_per_tpc:
                if len(each_tpc) != 0:
                    lowest_z = np.max(each_tpc['z'])

                    for i in range(1,1000):
                        mask = ma.masked_where((each_tpc['z'] <= lowest_z + step_size*(i-1)) & (each_tpc['z'] > lowest_z + step_size*(i)),each_tpc)

                        indices_of_masked_values = np.ma.where(mask.mask)[0]

                        min_number_of_hits = 1

                        segment = each_tpc[indices_of_masked_values]
                
                        #If segment has more than min_number_of_hits then consider this a good enough segment and append it to the list of segments
                        if len(segment) > min_number_of_hits:
                            self.segment_count += 1
                            hits_of_segment.append(segment)
                
                        #If there is no segments in hits of segment and # of hits of the considered segments in less than min number of hits then still append this segment. This stops code for messing at when first segment of track isn't long enough
                        elif (len(segment) == min_number_of_hits) & (len(hits_of_segment) == 0):
                            hits_of_segment.append(segment)

                        #If # of hits of the considered segments is below min hits and there is already segments then considered this hits as part of the previous segment
                        elif (len(segment) == min_number_of_hits) & (len(hits_of_segment) != 0):
                            np.append(hits_of_segment[-1],segment)
                        
                        else:
                            continue
                        
                        hits_wanted = np.where(np.in1d(muon_hits['id'],segment['id']))[0]
                
                        for hit in hits_wanted:
                            ref.append([hit,self.segment_count])

                        each_tpc = np.delete(each_tpc, indices_of_masked_values)         
                
                        #Starting, mid, and end points of segment
                        z_start = np.min(hits_of_segment[-1]['z'])

                        index1 = np.where(hits_of_segment[-1]['z'] == z_start)[0][0]

                        x_start, y_start, z_start = hits_of_segment[-1][index1]['x'], hits_of_segment[-1][index1]['y'], hits_of_segment[-1][index1]['z']

                        z_end = np.max(hits_of_segment[-1]['z'])

                        index2 = np.where(hits_of_segment[-1]['z'] == z_end)[0][0]

                        x_end, y_end, z_end = hits_of_segment[-1][index2]['x'], hits_of_segment[-1][index2]['y'], hits_of_segment[-1][index2]['z']

                        x_mid, y_mid, z_mid = (x_start+x_end)/2 , (y_start+y_end)/2 , (z_start+z_end)/2
                
                        #dEdx
                        Energy_of_segment = sum(hits_of_segment[-1]['E'])

                        Length_of_segment = self.length(hits_of_segment[-1])

                        # dQdx
                        Charge_of_segment = sum(hits_of_segment[-1]['Q'])
                        
                        #Drift time of segment
                        drift_time = np.mean(hits_of_segment[-1]['t_drift'])

                        hit_segments.append([self.segment_count, x_start, y_start, z_start, Energy_of_segment, x_end, y_end, z_end, Charge_of_segment,Length_of_segment, x_mid, y_mid, z_mid, drift_time])
                
                        if len(each_tpc) == 0:
                            break
        return hit_segments, ref


    def run(self, source_name, source_slice, cache):
        
        super(RockMuonSelection, self).run(source_name, source_slice, cache)
                
        event_id = np.r_[source_slice]
                
        PromptHits_ev = cache[self.FinalHits_dset_name]
        
        hits = self.cluster(PromptHits_ev[0])
        
        muon_tracks = self.select_muon_track(hits)
        
        if muon_tracks:
            #Loop through tracks and changes the DBSCAN cluster_id to a given track number
            for i in range(len(muon_tracks)):
                self.track_count += 1
                muon_tracks[i]['rock_muon_id'] = self.track_count 
            
            muon_hits_array = np.concatenate(muon_tracks) #Flatten multiple tracks to just an array of hits

            segments_list, hit_segment_ref = self.segments(muon_hits_array)
            
            #Reference hits to their segments
            for each_ref in hit_segment_ref:
                index = each_ref[0]

                muon_hits_array[index]['rock_segment_id'] = each_ref[1]

            nMuon_hits = len(muon_hits_array)

            #  1. reserve a new data region within the output dataset
            rock_muon_slice = self.data_manager.reserve_data(self.rock_muon_hits_dset_name, nMuon_hits)


            #  2. write the data to the new data region
            self.data_manager.write_data(self.rock_muon_hits_dset_name, rock_muon_slice, muon_hits_array)
            
            segments_array = np.array([tuple(sub) for sub in segments_list], dtype = self.rock_muon_segments_dtype) #Converts array of list to array of tuples
            
            nMuon_segments = len(segments_array)
            # 3. reserve a new data region within the rock muon segment dataset
            rock_muon_segments_slice = self.data_manager.reserve_data(self.rock_muon_segments_dset_name, nMuon_segments)

            # 4. Write the data into the rock muon segments data region
            self.data_manager.write_data(self.rock_muon_segments_dset_name, rock_muon_segments_slice, segments_array)
            
            #Write References
            event_ref = np.full(nMuon_hits, event_id)
            
            events_ref = np.stack((event_ref,np.r_[rock_muon_slice]), axis = -1)
             
            self.data_manager.write_ref(self.events_dset_name, self.rock_muon_hits_dset_name, events_ref)

            self.data_manager.write_ref(source_name, self.rock_muon_hits_dset_name, events_ref)
            # event -> hit
            #self.data_manager.write_ref(self.rock_muon_segments_dset_name, self.rock_muon_hits_dset_name, ref)
