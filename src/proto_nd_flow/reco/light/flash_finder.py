import numpy as np
import numpy.ma as ma
from collections import defaultdict

from h5flow.core import H5FlowStage, resources
from h5flow.data import dereference

from dbscan1d.core import DBSCAN1D

import proto_nd_flow.util.units as units

VERBOSE = False

class FlashFinder(H5FlowStage):
    '''
        ('id', 'u4'),                       Unique flash ID
        ('tpc', 'u1'),                      TPC ID (0-7)
        ('n_sum_hits', 'u4'),               Number of sum hits associated
        ('idx_range', 'u2', (2,)),          Min and Max sample index of sum hits
        ('hit_time_range', 'f4', (2,))      Min and Max timestamp of hit center relative to trigger time in ns (see busy_ns in hit definition)
        ('rising_spline_range', 'f4', (2,)) Min and Max timestamp of rising spline projections relative to trigger time in ns (see rising_spline in hit definition)
        ('tot_sum', 'f4'),                  Sum over hit sum values
        ('tot_max', 'f4'),                  Sum over hit max values
        ('tot_sum_spline', 'f4'),           Sum over hit spline sum values
        ('tot_max_spline', 'f4')            Sum over hit spline max values

    '''
    class_version = '1.0.0'

    flash_dset_name = 'light/flash'

    defaults=dict(
        cwvfm_dset_name = 'light/cwvfm',
        sipm_hits_dset_name = 'light/sipm_hits',
        sum_hits_dset_name = 'light/sum_hits',
        flash_dset_name = 'light/flash',
        eps = 5,
        min_samples = 1,
        nchantpc = 48
    )

    def flash_dtype(self,nchantpc):
        return np.dtype([
            ('id', 'u4'),
            ('tpc', 'u1'),
            ('n_sum_hits', 'u4'),
            ('idx_range', 'u2', (2,)),
            ('hit_time_range', 'f4', (2,)),
            ('rising_spline_range', 'f4', (2,)),
            ('tot_sum', 'f4'),
            ('tot_max', 'f4'),
            ('tot_sum_spline', 'f4'),
            ('tot_max_spline', 'f4'),
            ('deconv_sum', 'f4', (2,nchantpc//2)),
            ('deconv_max', 'f4', (2,nchantpc//2))
        ])


    def __init__(self, **params):
        super(FlashFinder, self).__init__(**params)

        # set up parameters
        for key,val in self.defaults.items():
            setattr(self, key, params.get(key, val))

        self.flash_dtype = self.flash_dtype(self.nchantpc)

    def init(self, source_name):
        super(FlashFinder, self).init(source_name)

        cwvfm_dset = self.data_manager.get_dset(self.cwvfm_dset_name)
        self.sum_hits_dset = self.data_manager.get_dset(self.sum_hits_dset_name)
        self.sipm_hits_dset = self.data_manager.get_dset(self.sipm_hits_dset_name)

        self.dbs = DBSCAN1D(eps=self.eps, min_samples=self.min_samples)

        # get waveform shape information
        self.nadc = cwvfm_dset.dtype['samples'].shape[0]
        self.nchan = cwvfm_dset.dtype['samples'].shape[1]
        self.ntpc = self.nadc                               #TOBEFIXED for ND implemenation
        self.nsamples = cwvfm_dset.dtype['samples'].shape[2]

        # Load channel map
        self.rel_pos_map = np.zeros((self.nadc,self.nchan,3))
        for adc in range(self.nadc):
            self.rel_pos_map[adc,:,:] = resources['Geometry'].sipm_rel_pos[(adc,range(64))]

        # create datasets and references
        self.data_manager.create_dset(self.flash_dset_name,
                                      dtype=self.flash_dtype)
        self.data_manager.create_ref(source_name, self.flash_dset_name)
        self.data_manager.create_ref(self.sum_hits_dset_name, self.flash_dset_name)
        #self.data_manager.create_ref(self.sipm_hits_dset_name, self.flash_dset_name)
        self.data_manager.set_attrs(self.flash_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    sum_hits_dset=self.sum_hits_dset_name,
                                    sipm_hits_dset=self.sipm_hits_dset_name
                                    )

    def get_tpc_channels(self,itpc):
        '''
        Returns array with (adc,channel) indices for given TPC
        The returnd array has size (2,nchantpc//2,2) where
        the first index is the side of the tpc (looking from cathode to anode, 0:left,1:right)
        and the second is the vertical position from bottom to top.
        The third axis is the index for adc (0) or channel (1).
        '''
        return_arr = np.zeros((2,self.nchantpc//2,2),dtype='i2')
        for iside in range(2):
            indices = np.where((self.rel_pos_map[..., 0] == itpc) & (self.rel_pos_map[..., 1] == iside))
            indexed_values = list(zip(indices[0], indices[1]))
            indexed_values.sort(key=lambda x: self.rel_pos_map[x[0], x[1], 2])
            return_arr[iside,:,:] = np.array(indexed_values)

        return(return_arr)

    def run(self, source_name, source_slice, cache):
        super(FlashFinder, self).run(source_name, source_slice, cache)
        events = cache[source_name]
        cwvfms = cache['light/cwvfm'].reshape(cache[source_name].shape)[
            'samples']
        
        #Get assosciate hits for events slice
        sum_hit_ref_dset, sum_hit_ref_dir = self.data_manager.get_ref(source_name,self.sum_hits_dset_name)
        sum_hit_ref_region = self.data_manager.get_ref_region(source_name,self.sum_hits_dset_name)
        sipm_hit_ref_dset, sipm_hit_ref_dir = self.data_manager.get_ref(source_name,self.sipm_hits_dset_name)
        sipm_hit_ref_region = self.data_manager.get_ref_region(source_name,self.sipm_hits_dset_name)

        

        sum_hits_idx = dereference(source_slice, sum_hit_ref_dset, region=sum_hit_ref_region,
                               ref_direction=sum_hit_ref_dir, indices_only=True)
        sum_hits = dereference(source_slice, sum_hit_ref_dset, data=self.sum_hits_dset, region=sum_hit_ref_region,
                               ref_direction=sum_hit_ref_dir)
        sipm_hits_idx = dereference(source_slice, sipm_hit_ref_dset, region=sipm_hit_ref_region,
                               ref_direction=sipm_hit_ref_dir, indices_only=True)
        sipm_hits = dereference(source_slice, sipm_hit_ref_dset, data=self.sipm_hits_dset, region=sipm_hit_ref_region,
                               ref_direction=sipm_hit_ref_dir)

        if VERBOSE: print("# events in slice: ",len(events))
        flash_list = []
        ev_ref_list = []
        sum_ref_list = []
        #sipm_ref_list = []

        for i, ev in enumerate(events):
            if VERBOSE: print("Event #",i)
            for itpc in range(self.ntpc):
                tpc_mask = (sum_hits[i,:]["tpc"] == itpc)
                tpc_hits = sum_hits[i,tpc_mask]
                tpc_hits_idx = sum_hits_idx[i,tpc_mask]
                if np.any(tpc_mask):
                    labels = self.dbs.fit_predict(tpc_hits["sample_idx"])

                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = np.count_nonzero(labels == -1)
                    tpc_flashes = np.empty((n_clusters+n_noise),dtype=self.flash_dtype)
                    ev_ref = np.empty((n_clusters+n_noise),dtype='u4') #ev ID for each flash
                    sum_ref = np.empty((len(tpc_hits_idx),2),dtype='u4')
                    tpc_flashes["tpc"] = itpc
                    
                    if VERBOSE:
                        print("    TPC #",itpc," #Clusters ",n_clusters)
                        print("       #Hits    ",sum_hits[i,tpc_mask].shape)
                        print("       Hits:    ",sum_hits[i,tpc_mask]["sample_idx"])
                        print("       Hits IDs:    ",sum_hits[i,tpc_mask]["id"])
                        print("       Hits IDs:    ",sum_hits_idx[i,tpc_mask])
                        print("       Labels:  ",labels)

                    #Handle clusters
                    # Note: Clusters pre-sorted in time by DBSCAN
                    for cl in range(n_clusters):
                        tpc_flashes[cl]["n_sum_hits"] = np.count_nonzero(labels==cl)

                        #Timing information
                        tpc_flashes[cl]["idx_range"] = get_extrema(tpc_hits[labels==cl]["sample_idx"])
                        tpc_flashes[cl]['hit_time_range'] = get_extrema(tpc_hits[labels==cl]["busy_ns"])
                        tpc_flashes[cl]['rising_spline_range'] = get_extrema(tpc_hits[labels==cl]["busy_ns"]
                                                                             +tpc_hits[labels==cl]["rising_spline"])

                        #Hit intensity information
                        tpc_flashes[cl]['tot_sum']=tpc_hits[labels==cl]["sum"].sum()
                        tpc_flashes[cl]['tot_max']=tpc_hits[labels==cl]["max"].sum()
                        tpc_flashes[cl]['tot_sum_spline']=tpc_hits[labels==cl]["sum_spline"].sum()
                        tpc_flashes[cl]['tot_max_spline']=tpc_hits[labels==cl]["max_spline"].sum()

                        #All channels information
                        ch_idx = self.get_tpc_channels(itpc)
                        flash_slice = slice(tpc_flashes[cl]["idx_range"][0], tpc_flashes[cl]["idx_range"][1]+1)
                        tpc_flashes[cl]['deconv_sum'] = np.sum(
                            cwvfms[i,ch_idx[...,0], ch_idx[...,1], flash_slice],
                            axis=-1)
                        tpc_flashes[cl]['deconv_max'] = np.max(
                            cwvfms[i,ch_idx[...,0], ch_idx[...,1], flash_slice],
                            axis=-1)

                    #Handle Noise events
                        # NOT NEEDED IF min_sample==1
                    
                    ev_ref[:] = np.r_[source_slice][i]
                    sum_ref[:,0] = tpc_hits_idx
                    sum_ref[:,1] = labels

                    flash_list.append(tpc_flashes)
                    ev_ref_list.append(ev_ref)
                    sum_ref_list.append(sum_ref)
        
        flash_data = np.concatenate(flash_list)

        # save data
        flash_slice = self.data_manager.reserve_data(
            self.flash_dset_name, len(flash_data))
        if len(flash_data):
            flash_data['id'] = np.r_[flash_slice]
        self.data_manager.write_data(self.flash_dset_name, flash_slice, flash_data)

        # save references
        ev_ref_data = np.concatenate(ev_ref_list)
        ref = np.array([(ev_idx,flash_data[flash_slice_idx]['id']) for flash_slice_idx, ev_idx in enumerate(ev_ref_data)])
        self.data_manager.write_ref(source_name, self.flash_dset_name, ref)

        flash_list_struc = [arr.shape[0] for arr in flash_list]
        flash_list = np.split(flash_data, np.cumsum(flash_list_struc)[:-1])
        for j, tpc_flash_slice in enumerate(flash_list):
            sum_ref_list[j] = np.c_[sum_ref_list[j][:,0], tpc_flash_slice["id"][sum_ref_list[j][:,1]]]
        ref_sum = np.concatenate(sum_ref_list)
        self.data_manager.write_ref(
            self.sum_hits_dset_name, self.flash_dset_name, ref_sum)
        #self.data_manager.write_ref(
        #    self.sum_hits_dset_name, self.flash_dset_name, ref_sipm)

@staticmethod
def get_extrema(input_array):
        return np.column_stack((
            input_array.min(),
            input_array.max()))