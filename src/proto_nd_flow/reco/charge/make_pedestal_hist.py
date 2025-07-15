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

class MakePedestalHist(H5FlowGenerator):
    '''
    Calculate channel by channel pedestals from data packets.

    Produces a json file as well as an hdf5 file with the channel pedestals.

    Parameters:
     - ``buffer_size`` : ``int``, optional, number of packets to load per iteration
     - ``vref_dac`` : ``int``, optional, vref_dac value for larpix configuration
     - ``vcm_dac`` : ``int``, optional, vcm_dac value for larpix configuration
     - ``adc_counts`` : ``int``, optional, total adc counts for vref/vcm calculation (normally 2^8)
     - ``vdda`` : ``int``, optional, vdda [mV]
     - ``mean_trunc`` : ``int``, adc counts around peak ADC to consider in pedestal mean
     - ``packets_dset_name`` : ``str``, required, input dataset path for packets
     - ``pedestal_dset_name`` : ``str``, required, output dataset for pedestal values in hdf5 file
    '''
    class_version = '0.0.0'

    default_buffer_size = 10000
    default_adc_counts = 256
    default_packets_dset_name = 'charge/packets'
    default_hist_dset_name = 'charge/hist_data'
    
    def __init__(self, **params):
        super(MakePedestalHist, self).__init__(**params)
    
        self.buffer_size = params.get('buffer_size', self.default_buffer_size)
        self.adc_counts = params.get('adc_counts', self.default_adc_counts)
        self.packets_dset_name = params.get('packets_dset_name', self.default_packets_dset_name)
        self.hist_dset_name = params.get('hist_dset_name', self.default_hist_dset_name)

        self.hist_dtype = np.dtype([
            ('id', 'i2'),
            ('unique_id', 'u8'),
            ('bins', ('i2', self.adc_counts)),
            ('hist', ('i2', self.adc_counts))
        ])
        # set up input file
        if H5FLOW_MPI:
            self.input_fh = h5py.File(self.input_filename, 'r', driver='mpio', comm=self.comm)
        else:
            self.input_fh = h5py.File(self.input_filename, 'r')
        self.packets = self.input_fh['packets']
        
        # set up loop variables
        if self.start_position is None:
            self.start_position = 0
        if self.end_position is None or self.end_position > len(self.packets):
            self.end_position = len(self.packets)
        self.slices = [slice(st, st + self.buffer_size) for st in range(self.start_position + self.rank * self.buffer_size, self.end_position, self.size * self.buffer_size)]
        self.iteration = 0
        
    def __len__(self):
        return len(self.slices)

    def init(self):
        super(MakePedestalHist, self).init()

        # initialize data objects
        if not self.data_manager.dset_exists(self.hist_dset_name):
            self.data_manager.create_dset(self.hist_dset_name, dtype=self.hist_dtype)
        self.data_manager.set_attrs(self.hist_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    buffer_size=self.buffer_size,
                                    start_position=self.start_position,
                                    end_position=self.end_position,
                                    input_filename=self.input_filename,
                                    packets_dset_name=self.packets_dset_name
                                    )
        self.dataword_dict = defaultdict(list)
        
        tile_ids = resources['Geometry'].tile_id[(self.packets['io_group'], self.packets['io_channel'])]
        self.pixel_unique_ids = pf.get_pixel_unique_ids(self.packets, tile_ids)

        self.mask = (self.packets['valid_parity'].astype(bool) & (self.packets['packet_type'] == 0))  # data packets
    
    def finish(self):
        super(MakePedestalHist, self).finish()
        ### finish by finding mean pedestal for all channels
        
        unique_id_set = set(list(self.dataword_dict.keys()))
        channel_pedestal_mv = []
        channel_unique_id = []
        pedestal_dict = {}

        vals_all = []
        bins_all = []
        unique_all = []
        for unique in sorted(unique_id_set):
            if unique == -1:
                continue
            vals, bins = np.histogram(self.dataword_dict[unique], bins = np.arange(self.adc_counts+1))
            vals_all.append(vals)
            bins_all.append(bins[:-1])
            unique_all.append(unique)
            
        self.input_fh.close()
        
        if len(unique_all):
            hist_data = np.zeros((len(unique_all)), dtype=self.hist_dtype)
            sl = self.data_manager.reserve_data(self.hist_dset_name, len(unique_all))
            hist_data['id'] = sl.start + np.arange(len(unique_all))
            hist_data['unique_id'] = np.array(unique_all)
            hist_data['bins'] = np.array(bins_all)
            hist_data['hist'] = np.array(vals_all)
            
            self.data_manager.write_data(self.hist_dset_name, sl, hist_data)
    
    def next(self):
        '''
            Read in a new block of LArPix packet data from the input file and collect datawords for individual channels.

            :returns: ``slice`` into the dataset given by ``packets_dset_name``
        '''
        if self.iteration >= len(self.slices):
            sl = H5FlowGenerator.EMPTY
        else:
            sl = self.slices[self.iteration]
        self.iteration += 1

        block = self.packets[sl]
        mask = self.mask[sl]
        packet_buffer = np.copy(block[mask])

        dataword = packet_buffer['dataword']
        pixel_unique_id = self.pixel_unique_ids[sl][mask]
        unique_pixel_unique_id = np.unique(pixel_unique_id)
        
        indices = np.argsort(pixel_unique_id)
        sorted_ids = pixel_unique_id[indices]
        sorted_data = dataword[indices]
        unique_ids, start_idx, counts = np.unique(sorted_ids, return_index=True, return_counts=True)
        
        for uid, start, count in zip(unique_ids, start_idx, counts):
            self.dataword_dict[uid].extend(sorted_data[start:start + count])
        
        return sl
        
     
        

        


        

    
