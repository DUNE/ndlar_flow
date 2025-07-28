import numpy as np
import numpy.ma as ma
import h5py
import logging
import warnings
from tqdm import tqdm
import os
import json
from collections import defaultdict

from h5flow.core import H5FlowGenerator, resources
from h5flow import H5FLOW_MPI
import proto_nd_flow.util.pixel_functions as pf

import proto_nd_flow.util.units as units

class GeneratePedestalJson(H5FlowGenerator):
    '''
    Calculate channel by channel pedestals.

    Produces a json file as well as an hdf5 file with the channel pedestals.

    Parameters:
     - ``vref_dac`` : ``int``, optional, vref_dac value for larpix configuration
     - ``vcm_dac`` : ``int``, optional, vcm_dac value for larpix configuration
     - ``adc_counts`` : ``int``, optional, total adc counts for vref/vcm calculation (normally 2^8)
     - ``vdda`` : ``int``, optional, vdda [mV]
     - ``mean_trunc`` : ``int``, adc counts around peak ADC to consider in pedestal mean
     - ``hist_dset_name`` : ``str``, required, input dataset path for pedestal histograms
     - ``pedestal_dset_name`` : ``str``, required, output dataset for pedestal values in hdf5 file
    '''
    class_version = '0.0.0'

    default_vref_dac = 223
    default_vcm_dac = 68
    default_adc_counts = 256
    default_vdda = 1800
    default_mean_trunc = 3
    default_hist_dset_name = 'charge/hist_data'
    default_pedestal_dset_name = 'charge/channel_pedestals'

    pedestal_dtype = np.dtype([
        ('id', 'u8'),
        ('unique_id', 'u8'),
        ('pedestal_mv', 'f4')
    ])

    def __init__(self, **params):
        super(GeneratePedestalJson, self).__init__(**params)
    
        self.vref_dac = params.get('vref_dac', self.default_vref_dac)
        self.vcm_dac = params.get('vcm_dac', self.default_vcm_dac)
        self.adc_counts = params.get('adc_counts', self.default_adc_counts)
        self.vdda = params.get('vdda', self.default_vdda)
        self.mean_trunc = params.get('mean_trunc', self.default_mean_trunc)
        self.hist_dset_name = params.get('hist_dset_name', self.default_hist_dset_name)
        self.pedestal_dset_name = params.get('pedestal_dset_name', self.default_pedestal_dset_name)

        # set up input file
        self.input_fh = h5py.File(self.input_filename, 'r')
        self.hist_data = self.input_fh[f'{self.hist_dset_name}/data']
        
        self.iteration = 0
        #self.unique_ids, inverse_indices = np.unique(self.hist_data['unique_id'], return_inverse=True)
        #self.unique_id_indices_dict = {}
        #for idx, val in enumerate(tqdm(self.unique_ids, desc='Finding unique id indices')):
        #    self.unique_id_indices_dict[val] = np.where(inverse_indices == idx)[0]

        self.unique_ids, inverse_indices = np.unique(self.hist_data['unique_id'], return_inverse=True)
        self.unique_id_indices_dict = defaultdict(list)
        for idx, inverse_idx in enumerate(inverse_indices):
            self.unique_id_indices_dict[self.unique_ids[inverse_idx]].append(idx)
        #self.unique_id_indices_dict = {key: np.array(val) for key, val in self.unique_id_indices_dict.items()}

        self.vref_mv = pf.dac2mv(self.vref_dac, self.vdda, self.adc_counts)
        self.vcm_mv = pf.dac2mv(self.vcm_dac, self.vdda, self.adc_counts)
        if self.start_position is None:
            self.start_position = 0
        if self.end_position is None or self.end_position > len(self.unique_ids):
            self.end_position = len(self.unique_ids)
        self.slices = [slice(st, st + 1) for st in range(self.start_position, self.end_position)]

    def __len__(self):
        return len(self.slices)

    def init(self):
        super(GeneratePedestalJson, self).init()

        # initialize data objects
        self.data_manager.create_dset(self.pedestal_dset_name, dtype=self.pedestal_dtype)
        self.data_manager.set_attrs(self.pedestal_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    buffer_size=1,
                                    start_position=self.start_position,
                                    end_position=self.end_position,
                                    input_filename=self.input_filename
                                   )
        
        self.pedestal_dict = {}
        self.channel_pedestal_mv = []
        self.channel_unique_id = []
    
    def finish(self):
        super(GeneratePedestalJson, self).finish()
        ### finish by saving pedestal data
        self.input_fh.close()
        json_path = self.data_manager.filepath.removesuffix(".FLOW.hdf5").removesuffix(".hdf5").removesuffix(".h5") + '.json'
        with open(json_path,'w') as fo:
            json.dump(self.pedestal_dict, fo, sort_keys=True, indent=4)
        
        pedestal_data = np.zeros((len(self.channel_unique_id),), dtype=self.pedestal_dtype)
        pedestal_data['id'] = np.arange(len(self.channel_pedestal_mv))
        pedestal_data['unique_id'] = np.array(self.channel_unique_id)
        pedestal_data['pedestal_mv'] = np.array(self.channel_pedestal_mv)
        
        sl = self.data_manager.reserve_data(self.pedestal_dset_name, len(self.channel_pedestal_mv))
        self.data_manager.write_data(self.pedestal_dset_name, sl, pedestal_data)
    
    def next(self):
        '''
            Calculate average pedestal for each pixel unique id.
        '''
        if self.iteration >= len(self.unique_ids):
            sl = H5FlowGenerator.EMPTY
            return sl
        else:
            unique_id = self.unique_ids[self.iteration]
            unique_id_indices = self.unique_id_indices_dict[unique_id]
            sl = self.slices[self.iteration]
        self.iteration += 1

        hist = np.zeros(self.hist_data[unique_id_indices[0]]['hist'].shape[0])
        bins = self.hist_data[unique_id_indices[0]]['bins']
        for index in unique_id_indices:
            hist += self.hist_data[index]['hist']
            
        peak_bin = np.argmax(hist)
        min_idx,max_idx = max(peak_bin-self.mean_trunc,0), min(peak_bin+self.mean_trunc,len(hist))
        ped_adc = np.average(bins[min_idx:max_idx]+0.5, weights=hist[min_idx:max_idx])
        pedestal_mv = pf.adc2mv(ped_adc, self.vref_mv, self.vcm_mv, self.adc_counts)
        self.channel_pedestal_mv.append(pedestal_mv)
        self.channel_unique_id.append(unique_id)
        self.pedestal_dict[str(unique_id)] = dict(
            pedestal_mv = pedestal_mv
        )
        return sl
        
     
        

        


        

    
