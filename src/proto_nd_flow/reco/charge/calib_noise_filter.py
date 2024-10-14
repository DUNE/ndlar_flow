import numpy as np
import numpy.ma as ma
import logging
from collections import defaultdict
import sys
from h5flow.core import H5FlowStage, resources
from sklearn.neighbors import KernelDensity
from pickle import dump, load

from proto_nd_flow.reco.charge.calib_prompt_hits import CalibHitBuilder

def unique_to_io_group(unique):
    return ((unique // (100*1000*1000)) % 1000)

def unique_channel_id(d):
    return ((d['io_group'].astype(int)*1000+d['io_channel'].astype(int))*1000 \
            + d['chip_id'].astype(int))*100 + d['channel_id'].astype(int)

def unique_to_channel_id(unique):
    return (unique % 100)


class low_current_filter:

    def __init__(self, cut=6.):
        self.cut = cut

    def filter(self, hits):
        return hits['Q']<self.cut

class correlated_post_trigger_filter:
    RANGE_Q = [-25, 25]
    SCALE_Q = RANGE_Q[1]-RANGE_Q[0]

    RANGE_T = [0, 3000]
    SCALE_T = RANGE_T[1]-RANGE_T[0]

    def __init__(self, param_filename='data/proto_nd_flow/filter_model_pars.pkl'):
        self.load_pars(param_filename)
        self.lrs_cut=1

    def load_pars(self, filename):
        
        with open(filename, 'rb') as f:
            model_pars = load(f)

        for key in model_pars.keys():
            setattr(self, key,  model_pars[key])

    def filter_chunked(self, hits):
        filter_mask=np.zeros(hits.shape, dtype=bool)

        for i in range(hits.shape[0]):
            filter_mask[i,:]=self.filter(hits[i,:])

        return filter_mask

    def filter(self, hits):
        if len(hits.shape)>1 and hits.shape[0]>1: 
            return self.filter_chunked(hits)
        lrs=np.zeros(hits.shape)
        un = unique_channel_id(hits)
        un_chip = (un//100)*100
        
        for chip in set(un_chip[~un_chip.mask]):
        
            mask = un_chip==chip
            
            n_chip_hits = np.sum(mask) 
            if np.sum(mask)<1: continue
            
            min_ts = np.min(hits[mask]['ts_pps'])
        
            
            for chan in set(hits[mask]['channel_id']):
                
                cc = unique_to_channel_id(chan)
                
                if not cc in [6,7,24]: continue
                
                m = np.logical_and(mask, hits['channel_id']==chan)
                
                chan_nhit = np.sum(m)
                
                m = np.logical_and(m, hits['Q']<self.RANGE_Q[1])
                m = np.logical_and(m, hits['Q']>self.RANGE_Q[0])
                
                ts = hits['ts_pps'][m].astype(int)-min_ts
                qs = hits['Q'][m]
                sumqs = np.array([np.sum(hits['Q'][m])]*ts.shape[0])
                if np.sum(m)<1: continue
                
                lrs[m] = self.get_lr(n_chip_hits, np.array([ts, qs, sumqs]).transpose(), cc, chan_nhit)
    
        return lrs > self.lrs_cut

    def get_lr(self, n_chip_hits, X, chan, chan_nhit):
        
        #scale data
        X[:, 0] = X[:, 0]/self.SCALE_T
        X[:, 1] = X[:, 1]/self.SCALE_Q
        if X.shape[1]==3: X[:, 2] = X[:, 2]/self.SCALE_Q
    
        nhit_lr=1
        chip_nhit=n_chip_hits-chan_nhit
    
        nhit_bins = self.nhit_bins

        if chip_nhit > 80: chip_nhit=80
        
        kdes = self.kdes
        kdes_cpth=self.kdes_cpth6
        if chan==6:
            nhit_lr = self.vals6[chan_nhit]/self.vals0[chan_nhit]
            nhit_lr *= self.chvals6[chan_nhit]/self.chvals0[chan_nhit]
            kdes_cpth=self.kdes_cpth6
        elif chan==7:
            nhit_lr = self.vals7[chan_nhit]/self.vals0[chan_nhit]
            nhit_lr *= self.chvals7[chan_nhit]/self.chvals0[chan_nhit]
            kdes_cpth=self.kdes_cpth7
        
        elif chan==24:
            nhit_lr = self.vals24[chan_nhit]/self.vals0[chan_nhit]
            nhit_lr *= self.chvals24[chan_nhit]/self.chvals0[chan_nhit]
            kdes_cpth=self.kdes_cpth24
        
        if n_chip_hits > nhit_bins[0][0] and n_chip_hits <= nhit_bins[0][1] :
            y = np.exp(kdes[0].score_samples(X))+1e-3
            y_cpt = np.exp(kdes_cpth[0].score_samples(X))+1e-5
            return nhit_lr*y_cpt/y
    
        if n_chip_hits > nhit_bins[1][0] and n_chip_hits <= nhit_bins[1][1] :
            y = np.exp(kdes[1].score_samples(X))+1e-3
            y_cpt = np.exp(kdes_cpth[1].score_samples(X))+1e-5
            return nhit_lr*y_cpt/y
    
        if n_chip_hits > nhit_bins[2][0] and n_chip_hits <= nhit_bins[2][1] :
            y = np.exp(kdes[2].score_samples(X))+1e-3
            y_cpt = np.exp(kdes_cpth[2].score_samples(X))+1e-5
            return nhit_lr*y_cpt/y
    
        else:
            y = np.exp(kdes[3].score_samples(X))+1e-3
            y_cpt = np.exp(kdes_cpth[3].score_samples(X))+1e-5
            return nhit_lr*y_cpt/y

class test_filter:

    def __init__(self):
        print('initializing test filter!')
        return

    def filter(self, hits):
        return hits['Q']>6

class CalibNoiseFilter(H5FlowStage):
    '''
        Noise Filter... documentation to come......
    '''
    class_version = '0.0.0'
    defaults = dict(
        events_dset_name = 'charge/events',
        hits_name = 'charge/calib_prompt_hits',
        hit_charge_name = 'charge/calib_prompt_hits',
        merged_name = 'charge/hits/calib_merged_hits',
        mc_hit_frac_dset_name = 'mc_truth/calib_final_hit_backtrack',
        filter_function_names = ['test_filter']
        )
    valid_filter_functions = ['test_filter', 'low_current_filter', 'correlated_post_trigger_filter']

    merged_dtype = CalibHitBuilder.calib_hits_dtype

    def __init__(self, **params):
        super(CalibNoiseFilter, self).__init__(**params)
        for key in self.defaults:
            setattr(self, key, params.get(key, self.defaults[key]))
        for f in self.filter_function_names:
            assert f in self.valid_filter_functions, f'invalid filter function name: {f}'

    def init(self, source_name):
        super(CalibNoiseFilter, self).init(source_name)
        self.data_manager.create_dset(self.merged_name, dtype=self.merged_dtype)
        if resources['RunData'].is_mc:
            self.data_manager.create_dset(self.mc_hit_frac_dset_name, dtype=self.hit_frac_dtype)
        self.data_manager.create_ref(self.hits_name, self.merged_name)
        self.data_manager.create_ref(source_name, self.merged_name)
        if resources['RunData'].is_mc:
            self.data_manager.create_ref(self.merged_name,self.mc_hit_frac_dset_name)
        self.data_manager.create_ref(self.events_dset_name, self.merged_name)
        
        self.init_filter_functions()

    def init_filter_functions(self):
        self.filter_functions=[]
        for filter_name in self.filter_function_names:
            self.filter_functions.append( getattr(sys.modules[__name__], filter_name  )() ) 
    
    def default_filter_function(self, hits):
        return hits['Q']>np.inf

    #@staticmethod
    def filter_hits(self, hits, seg_fracs):
        '''

        :param hits: original hits array, shape: (N,M)

        :param fracs: fractional contributions of true segments per packet

        :returns: new hit array, shape: (N,m), new hit charge array, shape: (N,m), and an index array with shape (L,2), [:,0] being the index into the original hit array and [:,1] being the flattened index into the compressed new array

        '''
        
        mask = hits.mask['id'].copy()
        new_hits = hits.data.copy()
        old_ids = hits.data['id'].copy()[...,np.newaxis]
        old_id_mask = hits.mask['id'].copy()[...,np.newaxis]
        filter_mask = self.default_filter_function(new_hits)
        for f in self.filter_functions:
            filter_mask = filter_mask | f.filter(hits)

        mask = filter_mask | mask

        new_ids = old_ids[~mask]
        back_track = None
        return (
            ma.array(new_hits, mask=mask),
            np.c_[new_ids, np.array(range(new_ids.shape[0]))],
            ma.array(back_track, mask=mask) if back_track is not None else None
            )

    def run(self, source_name, source_slice, cache):
        super(CalibNoiseFilter, self).run(source_name, source_slice, cache)

        event_id = np.r_[source_slice]
        packet_frac_bt = cache['packet_frac_backtrack']
        hits = cache[self.hits_name]

        merged, ref, back_track = self.filter_hits(hits, seg_fracs=packet_frac_bt)

        merged_mask = merged.mask['id']

        # first write the new merged hits to the file
        new_nhit = int((~merged_mask).sum())
        #print('hits after filter:', merged.shape, new_nhit)

        merge_slice = self.data_manager.reserve_data(self.merged_name, new_nhit)
        merge_idx = np.r_[merge_slice].astype(merged.dtype['id'])
        if new_nhit > 0:
            ref[:,1] += merge_idx[0] # offset references based on reserved region in output file
            np.place(merged['id'], ~merged_mask, merge_idx)

        self.data_manager.write_data(self.merged_name, merge_idx, merged[~merged_mask])

        # sort based on the ID of the prompt hit, to make analysis more convenient
        ref = ref[np.argsort(ref[:, 0])]
        self.data_manager.write_ref(self.hits_name, self.merged_name, ref)

        # if back tracking information was available, write the merged back tracking
        # dataset to file 
        if back_track is not None:
            merge_bt_slice = self.data_manager.reserve_data(self.mc_hit_frac_dset_name, new_nhit)
            self.data_manager.write_data(self.mc_hit_frac_dset_name, merge_idx, back_track[~merged_mask])
            self.data_manager.write_ref(self.merged_name,self.mc_hit_frac_dset_name,np.c_[merge_idx,merge_idx])

        ev_ref = np.c_[(np.indices(merged_mask.shape)[0] + source_slice.start)[~merged_mask], merge_idx]
        self.data_manager.write_ref(source_name, self.merged_name, ev_ref)
        self.data_manager.write_ref(self.events_dset_name, self.merged_name, ev_ref)
