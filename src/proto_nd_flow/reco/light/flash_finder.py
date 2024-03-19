import numpy as np
import numpy.ma as ma
from collections import defaultdict

from h5flow.core import H5FlowStage, resources
from h5flow.data import dereference


import proto_nd_flow.util.units as units


class FlashFinder(H5FlowStage):
    '''
        TBD

    '''
    class_version = '1.0.0'

    default_flash_dset_name = 'light/flash'

    def flash_dtype(self, near_samples):
        return np.dtype([
            ('id', 'u4'),
            ('tpc', 'u1'),
        ])


    def __init__(self, **params):
        super(FlashFinder, self).__init__(**params)
        self.flash_dset_name = params.get('flash_dset_name',self.default_flash_dset_name)
        self.wvfm_dset_name = params.get('wvfm_dset_name')
        self.sum_hits_dset_name = params.get('sum_hits_dset_name')
        self.sipm_hits_dset_name = params.get('sum_hits_dset_name')

        self.flash_dtype = self.flash_dtype()

    def init(self, source_name):
        super(FlashFinder, self).init(source_name)

        wvfm_dset = self.data_manager.get_dset(self.wvfm_dset_name)
        sum_hits_dset = self.data_manager.get_dset(self.sum_hits_dset_name)
        sipm_hits_dset = self.data_manager.get_dset(self.sipm_hits_dset_name)

        # get waveform shape information
        self.ntpc = wvfm_dset.dtype['samples'].shape[0]
        self.nsamples = wvfm_dset.dtype['samples'].shape[2]

        # create datasets and references
        self.data_manager.create_dset(self.flash_dset_name,
                                      dtype=self.flash_dtype)
        self.data_manager.create_ref(source_name, self.flash_dset_name)
        self.data_manager.create_ref(self.sum_hits_dset_name, self.flash_dset_name)
        self.data_manager.create_ref(self.sipm_hits_dset_name, self.flash_dset_name)
        self.data_manager.set_attrs(self.flash_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    sum_hits_dset=self.sum_hits_dset_name,
                                    sipm_hits_dset=self.sipm_hits_dset_name
                                    )

    def run(self, source_name, source_slice, cache):
        super(FlashFinder, self).run(source_name, source_slice, cache)
        events = cache[source_name]
        wvfms = cache[self.wvfm_dset_name].reshape(cache[source_name].shape)[
            'samples']
        
        #Get assosciate hits for events slice
        sum_hit_ref_dset, sum_hit_ref_dir = self.data_manager.get_ref(source_name,self.sum_hits_dset_name)
        sum_hit_ref_region = self.data_manager.get_ref_region(source_name,self.sum_hits_dset_name)
        sipm_hit_ref_dset, sipm_hit_ref_dir = self.data_manager.get_ref(source_name,self.sipm_hits_dset_name)
        sipm_hit_ref_region = self.data_manager.get_ref_region(source_name,self.sipm_hits_dset_name)

        sum_hits = dereference(source_slice, sum_hit_ref_dset, region=sum_hit_ref_region,
                               ref_direction=sum_hit_ref_dir)
        sum_hits = dereference(source_slice, sipm_hit_ref_dset, region=sipm_hit_ref_region,
                               ref_direction=sipm_hit_ref_dir)
        
        for i, ev in enumerate(events):
            for itpc in self.ntpc:
                tpc_mask = (sum_hits[i,:]["tpc"] == itpc)
                print(sum_hits[i,tpc_mask].shape)


        flash_data = np.empty((1,), dtype=self.flash_dtype)
        
        # save data
        flash_slice = self.data_manager.reserve_data(
            self.flash_dset_name, len(flash_data))
        if len(flash_data):
            flash_data['id'] = np.r_[flash_slice]
        self.data_manager.write_data(self.flash_dset_name, flash_slice, flash_data)

        # save references
        self.data_manager.write_ref(source_name, self.flash_dset_name, ref)
        self.data_manager.write_ref(
            self.sum_hits_dset_name, self.flash_dset_name, ref_sum)
        self.data_manager.write_ref(
            self.sum_hits_dset_name, self.flash_dset_name, ref_sipm)