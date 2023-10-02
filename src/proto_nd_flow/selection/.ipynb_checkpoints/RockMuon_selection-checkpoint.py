#Imports
import numpy as np

import h5py
from h5flow.core import H5FlowStage, resources
from h5flow import H5FLOW_MPI
import h5flow
from h5flow.data import dereference

from sklearn.cluster import DBSCAN

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

import statistics

import glob

class RockMuonSelection(H5FlowStage):
    '''
    This will perform a selection for rock muons. Rock muons are 
    determined by straight tracks that penetrate two faces of the 
    detector.
    '''

    #Detector Boundaries, Cuts
    defaults = dict([
    ('x_boundaries', np.array([-63.931, -3.069, 3.069, 63.931])), #cm
        
    ('y_boundaries', np.array([-268-42-19.8543, -268-42+103.8543])),#cm
        
    ('z_boundaries', np.array([1300-64.3163,  1300-2.6837, 1300+2.6837, 1300+64.3163])), #cm
    
    
    ('length_cut', 55), #cm
        
    ('MEVR', 0.974) #Miniumum explained variance ratio
    ])
    
    #Datasets
    PromptHits_dset_name = '"charge/events","charge/calib_prompt_hits"'
    
    Segs_PromptHits_dset_name = '"charge/events","charge/calib_prompt_hits","charge/packets","mc_truth/tracks"'
    
    Traj_PromptHits_dset_name = '"charge/events","charge/calib_prompt_hits","charge/packets","mc_truth/tracks","mc_truth/trajectories"'
    
    #Datatype wanted
    
    rock_muon_hit_dtype = np.dtype([
        ('x', 'f8'),
        ('y', 'f8'),
        ('z', 'f8'),
        ('cluster_id', 'u4')
        ('id', 'u4'),
        ('t_drift', 'f8'),
        ('ts_pps', 'u8'),
        ('Q', 'f8'),
        ('E', 'f8')
    ])

    
    rock_muon_hits_dset_name = 'rock_muon_hits'
    
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
        
        self.data_manager.set_attrs(self.path,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    **attrs)
        
        self.data_manager.create_dset(f'{self.path}/{self.rock_muon_hits_dset_name}',
                                      self.rock_muon_hit_dtype)
        
    @staticmethod
    def cluster(i_evt):

        hits = np.zeros((PromptHits_ev.shape[1],9))

        for i in range(PromptHits_ev.shape[1]):
            hits[i][0] = PromptHits_ev['x'].data[0][i]
            hits[i][1] = PromptHits_ev['y'].data[0][i]
            hits[i][2] = PromptHits_ev['z'].data[0][i]

        hit_cluster = DBSCAN(eps = 8, min_samples = 1).fit(hits)

        for i in range(PromptHits_ev.shape[1]):
            hits[i][3] = hit_cluster.labels_[i]
            hits[i][4] = PromptHits_ev['id'].data[0][i]
            hits[i][5] = PromptHits_ev['t_drift'].data[0][i]
            hits[i][6] = PromptHits_ev['ts_pps'].data[0][i]
            hits[i][7] = PromptHits_ev['Q'].data[0][i]
            hits[i][8] = PromptHits_ev['E'].data[0][i]
        return hits
    
    @staticmethod
    
    def PCAs(hits_of_track):
        scaler = StandardScaler()
        X_train = hits_of_track
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)

        pca = PCA(1) # 1 component

        pca.fit(X_train)

        explained_var = pca.explained_variance_ratio_
        variance = pca.explained_variance_

        return  explained_var,variance
    
    @staticmethod
    def select_muon_track(hits):

            tracks = np.unique(hits[:,3]) #This gets how many tracks there are

            muon_hits = []

            a = []

            #Boundaries of the detector
            x_boundaries = np.array([-63.931, -3.069, 3.069, 63.931])
            y_boundaries = np.array([-268-42-19.8543, -268-42+103.8543])
            z_boundaries = np.array([1300-64.3163,  1300-2.6837, 1300+2.6837, 1300+64.3163])

            #Bottom face for each coordinate
            min_boundaries = np.array([x_boundaries[0], y_boundaries[0], z_boundaries[0]])

            #Top face for each coordinate
            max_boundaries = np.array([x_boundaries[-1], y_boundaries[-1], z_boundaries[-1]])

            for n in range(len(np.unique(tracks))):

                hits1 = hits[:,3] == n #get the hits with track number equal to n

                hits_with_track = hits[hits1] #position of the hits with their associated track number

                hits_of_track = np.delete(hits_with_track, (3,4,5,6,7,8), axis = 1) #hits without their track number


                #print(np.shape(hitss_of_track))
                for i in range(len(max_boundaries)):
                    d = 2 #Distance away from a TPC face

                    v = 0.974 #Explained variance minimum

                    b = 55 #minimum track length requirement

                    #Does the track penetrate both z-boundaries?
                    if min_boundaries[i]-d < np.min(hits_of_track[:,i]) < (min_boundaries[i] + d) and (max_boundaries[i] -d) < np.max(hits_of_track[:,i]) < max_boundaries[i]+d:
                        a, p  = PCAs(hits_of_track)
                        l = length_track(hits_of_track)
                        if np.logical_and(a > v, l > b):
                            muon_hits.append(hits_with_track)
                            break
                    elif min_boundaries[i]-d < np.min(hits_of_track[:,i]) < (min_boundaries[i] + d) and (max_boundaries[i-2] -d) < np.max(hits_of_track[:,i-2]) < (max_boundaries[i-2]+d):
                        a, p = PCAs(hits_of_track)
                        l = length_track(hits_of_track)
                        if np.logical_and(a > v, l > b):
                            muon_hits.append(hits_with_track)
                            break
                    elif min_boundaries[i]-d < np.min(hits_of_track[:,i]) < (min_boundaries[i] + d) and (max_boundaries[i-1] -d) < np.max(hits_of_track[:,i-1]) < max_boundaries[i-1]+d:
                        a, p = PCAs(hits_of_track)
                        l = length_track(hits_of_track)
                        if np.logical_and(a > v, l > b):
                            muon_hits.append(hits_with_track)
                            break
                    elif min_boundaries[i]-d < np.min(hits_of_track[:,i]) < (min_boundaries[i] + d) and (min_boundaries[i-1] -d) < np.min(hits_of_track[:,i-1]) < min_boundaries[i-1]+d:
                        a, p = PCAs(hits_of_track)
                        l = length_track(hits_of_track)
                        if np.logical_and(a > v, l > b):
                            muon_hits.append(hits_with_track)
                            break
                    elif min_boundaries[i]-d < np.min(hits_of_track[:,i]) < (min_boundaries[i] + d) and (min_boundaries[i-2] -d) < np.min(hits_of_track[:,i-2]) < min_boundaries[i-2]+d:
                        a, p = PCAs(hits_of_track)
                        l = length_track(hits_of_track)
                        if np.logical_and(a > v, l > b):
                            muon_hits.append(hits_with_track) 
                            break
            return muon_hits

            


    def run(self, source_name, source_slice, cache):
        
        super(RockMuonSelection, self).run(source_name, source_slice, cache)
        
        event_id = np.r_[source_slice]
        
        PromptHits_ev = cache[self.PromptHits_dset_name]
        
        Segs_PromptHits = cache[self.Segs_PromptHits_dset_name]
        
        Traj_PromptHits = cache[self.Traj_PromptHits_dset_name]
        
        hits = cluster(event_id)
        
        muon_hits = select_muon_track(hits)
        
        #  1. reserve a new data region within the output dataset
        rock_muon_slice = self.data_manager.reserve_data(self.rock_muon_hits_dset_name, muon_hits)
        
        #  2. write the data to the new data region
        self.data_manager.write_data(self.rock_muon_hits_dset_name, rock_muon_slice)
        
        
        
