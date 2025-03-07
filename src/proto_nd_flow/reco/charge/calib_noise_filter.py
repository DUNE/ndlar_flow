import numpy as np
import numpy.ma as ma
import logging
from collections import defaultdict
import sys
from h5flow.core import H5FlowStage, resources
from sklearn.neighbors import KernelDensity
from pickle import dump, load
import json
from proto_nd_flow.reco.charge.calib_prompt_hits import CalibHitBuilder

## Some useful functions for the filter classes
##
def unique_to_io_group(unique):
    return ((unique // (100*1000*1000)) % 1000)

def unique_channel_id(d):
    return ((d['io_group'].astype(int)*1000+d['io_channel'].astype(int))*1000 \
            + d['chip_id'].astype(int))*100 + d['channel_id'].astype(int)

def unique_to_channel_id(unique):
    return (unique % 100)

## Filter classes
##
class low_current_filter:
    '''
        Filters out hits with low indunced current. Specify threshold to remove hitsnwith Q<threshold
    '''

    def __init__(self, threshold=1., channel_threshold_file=''):
        self.threshold = float(threshold)
        print('using threshold:', self.threshold)
        self.channel_threshold_file=channel_threshold_file
        try:
            with open(self.channel_threshold_file, 'r') as fi:
                self.channel_thresholds=json.load(fi)
        except:
            self.channel_thresholds={}
            print('Unable to open channel threshold file! {}\nProceeding with default threshold for all channels.'.format(self.channel_threshold_file))
    
    def unique_channel_id(self, d):
        return ((d['io_group'].astype(int)*1000+d['io_channel'].astype(int))*1000 \
            + d['chip_id'].astype(int))*100 + d['channel_id'].astype(int)

    def filter(self, hits, default_threshold=5.0):
        
        #Get channel by channel thresholds
        tile_id = resources['Geometry'].tile_id[hits['io_group'],hits['io_channel']]
        hit_uniqueid = self.unique_channel_id(hits)

        charge_above_threshold = np.ones(hits.shape)*99.

        unique_ids, counts = np.unique(hit_uniqueid, return_counts=True)
        threshold=default_threshold
        for u in unique_ids:
            if not str(u) in self.channel_thresholds.keys():
                print('No threshold found for channel {}! Using default threshold of {} ke-!'.format(u, default_threshold))
            else:
                threshold = self.channel_thresholds[str(u)] 
            m = hit_uniqueid==u
            charge_above_threshold[m] = hits[m]['Q']-threshold

        return charge_above_threshold<self.threshold
        
class correlated_post_trigger_filter:
    '''
        Module 2 (v2b) specific filter for noise from charge injection from ADC reference instability
        Only affecting channel_id 6, 7, 24
        As these channels are not used in Moudle 0,1 and 3 (v2a), this filter will not affect the data from these modules
    '''

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
                
                m = np.logical_and(m, hits['Q_raw']<self.RANGE_Q[1])
                m = np.logical_and(m, hits['Q_raw']>self.RANGE_Q[0])
                
                ts = hits['ts_pps'][m].astype(int)-min_ts
                qs = hits['Q_raw'][m]
                sumqs = np.array([np.sum(hits['Q_raw'][m])]*ts.shape[0])
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

class hot_pixel_filter:

    def __init__(self, max_n_hits=35):
        self.max_n_hits = int(max_n_hits)

    def filter(self, hits):

        un = unique_channel_id(hits)

        chans, counts = np.unique( un, return_counts=True )
        
        hot_pixels = chans[counts>self.max_n_hits]

        mask = np.zeros( hits.shape ).astype(bool) 
        
        for pix in hot_pixels:
            chan_mask = un==pix
            event_mask = np.sum( chan_mask, axis=-1  ) > self.max_n_hits

            mask = mask | np.dot(event_mask,  chan_mask)

        return mask

#Main class defintion
##
class CalibNoiseFilter(H5FlowStage):
    '''
        Noise Filter for charge readout.
    '''
    class_version = '0.0.0'
    defaults = dict(
        events_dset_name = 'charge/events',
        hits_name = 'charge/calib_prompt_hits',
        hit_charge_name = 'charge/calib_prompt_hits',
        calib_hits_dset_name = 'charge/hits/calib_final_hits',
        mc_hit_frac_dset_name = 'mc_truth/calib_final_hit_backtrack',
        low_current_filter__threshold=6.0,
        hot_pixel_filter__max_n_hits=35,
        low_current_filter__channel_threshold_file='data/proto_nd_flow/thresholds_2x2.json',
        filter_function_names = ['hot_pixel_filter'],
        hit_ref = False
        )
    valid_filter_functions = ['low_current_filter', 'correlated_post_trigger_filter', 'hot_pixel_filter']

    hits_dtype = CalibHitBuilder.calib_hits_dtype

    def __init__(self, **params):
        super(CalibNoiseFilter, self).__init__(**params)
        for key in self.defaults:
            setattr(self, key, params.get(key, self.defaults[key]))
        for f in self.filter_function_names:
            assert f in self.valid_filter_functions, f'invalid filter function name: {f}'

    def init(self, source_name):
        super(CalibNoiseFilter, self).init(source_name)
        self.init_filter_functions()

    def init_filter_functions(self):
        self.filter_functions=[]
        for filter_name in self.filter_function_names:
            params={}
            for p in self.__dict__.keys(): 
                if filter_name in p: params[p.split('__')[-1]]=getattr(self, p)
            self.filter_functions.append( getattr(sys.modules[__name__], filter_name  )( **params ) ) 
    
    def default_filter_function(self, hits):
        return hits['Q']>np.inf

    #@staticmethod
    def filter_hits(self, hits, back_track):
        '''

        :param hits: original hits array, shape: (N,M)

        :param back_track: backtrack information per hit

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

        return (
            ma.array(new_hits, mask=mask),
            np.c_[new_ids, np.array(range(new_ids.shape[0]))],
            ma.array(back_track, mask=mask) if back_track is not None else None
            )

    def run(self, source_name, source_slice, cache):
        super(CalibNoiseFilter, self).run(source_name, source_slice, cache)

        # set up new datasets
        hits_frac_bt = None
        if resources['RunData'].is_mc:
            hits_frac_bt = cache['hits_frac_backtrack']
        has_mc_truth = hits_frac_bt is not None

        self.data_manager.create_dset(self.calib_hits_dset_name, dtype=self.hits_dtype)
        if has_mc_truth:
            self.data_manager.create_dset(self.mc_hit_frac_dset_name, dtype=hits_frac_bt.dtype)
        self.data_manager.create_ref(self.events_dset_name, self.calib_hits_dset_name)
        self.data_manager.create_ref(self.hits_name, self.calib_hits_dset_name)
        self.data_manager.create_ref(source_name, self.calib_hits_dset_name)
        if self.hit_ref and has_mc_truth:
            self.data_manager.create_ref(self.calib_hits_dset_name, self.mc_hit_frac_dset_name)

        event_id = np.r_[source_slice]
        if has_mc_truth:
            hits_frac_bt = np.squeeze(cache['hits_frac_backtrack'], axis=-1) # additional dimension from the reference
        hits = cache[self.hits_name]

        hits, hits_ref, back_track = self.filter_hits(hits, back_track=hits_frac_bt)

        hits_mask = hits.mask['id']

        # first write the new hits hits to the file
        new_nhit = int((~hits_mask).sum())

        hits_slice = self.data_manager.reserve_data(self.calib_hits_dset_name, new_nhit)
        if has_mc_truth:
            hit_bt_slice = self.data_manager.reserve_data(self.mc_hit_frac_dset_name, new_nhit)

        hits_idx = np.r_[hits_slice].astype(hits.dtype['id'])
        #FIXME Do we still need to renumber the hit id?
        #if new_nhit > 0:
        #    ref[:,1] += hits_idx[0] # offset references based on reserved region in output file
        #    np.place(hits['id'], ~hits_mask, hits_idx)

        new_hits = hits[~hits_mask]

        # write dataset and ref
        self.data_manager.write_data(self.calib_hits_dset_name, hits_slice, new_hits)

        if has_mc_truth:
            new_hits_frac_bt = back_track[~hits_mask]
            # make sure hitss and hits backtracking match in numbers
            if new_hits.shape[0] == new_hits_frac_bt.shape[0]:
                self.data_manager.write_data(self.mc_hit_frac_dset_name, hit_bt_slice, new_hits_frac_bt)
            else:
                raise Exception("The data hits and backtracking info do not match in size.")

        # prompt hit -> final hit
        # sort based on the ID of the prompt hit, to make analysis more convenient
        hits_ref = hits_ref[np.argsort(hits_ref[:, 0])]
        self.data_manager.write_ref(self.hits_name, self.calib_hits_dset_name, hits_ref)

        ev_ref = np.c_[(np.indices(hits_mask.shape)[0] + source_slice.start)[~hits_mask], hits_idx]

        # raw_event -> hit
        self.data_manager.write_ref(source_name, self.calib_hits_dset_name, ev_ref)

        # event -> hit
        self.data_manager.write_ref(self.events_dset_name, self.calib_hits_dset_name, ev_ref)

        # hit -> backtracking
        if self.hit_ref and has_mc_truth:
            self.data_manager.write_ref(self.calib_hits_dset_name,self.mc_hit_frac_dset_name,np.c_[new_hits['id'], new_hits['id']])

