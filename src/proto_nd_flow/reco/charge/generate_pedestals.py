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

class GeneratePedestals(H5FlowGenerator):
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
    default_vref_dac = 223
    default_vcm_dac = 68
    default_adc_counts = 256
    default_vdda = 1800
    default_mean_trunc = 3
    default_packets_dset_name = 'charge/packets'
    default_pedestal_dset_name = 'charge/channel_pedestals'

    pedestal_dtype = np.dtype([
        ('id', 'u8'),
        ('unique_id', 'u8'),
        ('pedestal_mv', 'f4')
    ])

    def __init__(self, **params):
        super(GeneratePedestals, self).__init__(**params)
    
        self.buffer_size = params.get('buffer_size', self.default_buffer_size)
        self.vref_dac = params.get('vref_dac', self.default_vref_dac)
        self.vcm_dac = params.get('vcm_dac', self.default_vcm_dac)
        self.adc_counts = params.get('adc_counts', self.default_adc_counts)
        self.vdda = params.get('vdda', self.default_vdda)
        self.mean_trunc = params.get('mean_trunc', self.default_mean_trunc)
        self.packets_dset_name = params.get('packets_dset_name', self.default_packets_dset_name)
        self.pedestal_dset_name = params.get('pedestal_dset_name', self.default_pedestal_dset_name)

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
        super(GeneratePedestals, self).init()

        # initialize data objects
        self.data_manager.create_dset(self.pedestal_dset_name, dtype=self.pedestal_dtype)
        self.data_manager.set_attrs(self.pedestal_dset_name,
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
        super(GeneratePedestals, self).finish()
        ### finish by finding mean pedestal for all channels
        
        vref_mv = pf.dac2mv(self.vref_dac, self.vdda, self.adc_counts)
        vcm_mv = pf.dac2mv(self.vcm_dac, self.vdda, self.adc_counts)

        unique_id_set = set(list(self.dataword_dict.keys()))
        channel_pedestal_mv = []
        channel_unique_id = []
        pedestal_dict = {}
        for unique in sorted(unique_id_set):
            if unique == -1:
                continue
            vals, bins = np.histogram(self.dataword_dict[unique], bins = np.arange(257))
            peak_bin = np.argmax(vals)
            min_idx,max_idx = max(peak_bin-self.mean_trunc,0), min(peak_bin+self.mean_trunc,len(vals))
            ped_adc = np.average(bins[min_idx:max_idx]+0.5, weights=vals[min_idx:max_idx])
            pedestal_mv = pf.adc2mv(ped_adc, vref_mv, vcm_mv, self.adc_counts)
            channel_pedestal_mv.append(pedestal_mv)
            channel_unique_id.append(unique)
            pedestal_dict[str(unique)] = dict(
                pedestal_mv = pedestal_mv
            )
        self.input_fh.close()
        
        ### results output to two different files, json and hdf5
        json_path = self.data_manager.filepath.removesuffix(".FLOW.hdf5").removesuffix(".hdf5").removesuffix(".h5") + '.json'
        with open(json_path,'w') as fo:
            json.dump(pedestal_dict, fo, sort_keys=True, indent=4)
        
        pedestal_data = np.zeros((len(channel_unique_id),), dtype=self.pedestal_dtype)
        pedestal_data['id'] = np.arange(len(channel_pedestal_mv))
        pedestal_data['unique_id'] = np.array(channel_unique_id)
        pedestal_data['pedestal_mv'] = np.array(channel_pedestal_mv)
        
        sl = self.data_manager.reserve_data(self.pedestal_dset_name, len(channel_pedestal_mv))
        self.data_manager.write_data(self.pedestal_dset_name, sl, pedestal_data)
    
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
        
     
        

        


        

    
