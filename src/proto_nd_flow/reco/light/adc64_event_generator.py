import numpy as np
import h5py
import os
import logging
from math import ceil

from h5flow.core import H5FlowGenerator
from h5flow import H5FLOW_MPI

import adc64format


class LightADC64EventGenerator(H5FlowGenerator):
    '''
        Light system event builder - converts multiple ADC64-formatted light files to an event-packed
        h5flow-readable format. Uses the ``adc64format`` library.

        Parameters:
         - ``wvfm_dset_name`` : ``str``, required, path to dataset to store raw waveforms
         - ``sn_table`` : ``list`` of ``int``, required, serial number of each ADC (determines order of the ADCs in the output data type)
         - ``n_adcs`` : ``int``, number of ADC serial numbers (default = 2)
         - ``n_channels`` : ``int``, number of channels per ADC (default = 64)
         - ``sync_channel`` : ``int``, channel index to use for identifying sync events (default = 32)
         - ``sync_threshold`` : ``int``, threshold for identifying sync events (default = 5000) [ADC]
         - ``sync_buffer`` : ``int``, optional, number of events to scan to find the first sync for each event builder process, only relevant if using MPI
         - ``clock_timestamp_factor`` : ``float``, tick size for ``tai_ns`` in raw data [ns] (default = 0.625)
         - ``batch_size`` : ``int``, optional, number of events to buffer before initiating next loop iteration (default = 128)


        Generates a lightweight "event" dataset along with a dataset containing
        event-packed raw waveforms.

        Example config::

            flow:
                source: light_event_generator
                stages: []

            light_event_generator:
                classname: LightADC64EventGenerator
                path: module0_flow.reco.light.hdf5_event_generator
                dset_name: 'light/events'
                params:
                    wvfm_dset_name: 'light/wvfm'
                    n_adcs: 2
                    n_channels: 64
                    sync_channel: 32
                    sn_table:
                      - 175780172
                      - 175854781
                    batch_size: 128
                    utime_ms_window: 1000
                    tai_ns_window: 1000

        ``events`` datatype::

            id          u8,                     unique identifier per event
            event       i4,                     event number from source ROOT file
            sn          i4(n_adcs,),            serial number of adc
            ch          u1(n_adcs,n_channels),  channel id
            utime_ms    u8(n_adcs,n_channels),  unix time since epoch [ms]
            tai_ns      u8(n_adcs,n_channels),  time since PPS [ns]
            wvfm_valid  u1(n_adcs,n_channels),  boolean indicator if channel is present in event

        ``wvfm`` datatype::

            samples     i2(n_adc,n_channels,n_samples), sample 10-bit ADC value (lowest 6 bits are not used)
    '''
    defaults = dict(
        n_adcs = 8,
        n_channels = 64,
        batch_size = 64,
        sync_channel = 0,
        sync_threshold = 40000,
        sync_buffer = 200,        
        clock_timestamp_factor = 1.0,
        utime_ms_window = 1000,
        tai_ns_window = 1000,
        )

    def event_dtype(self): return np.dtype([
        ('id', 'u8'),  # unique identifier
        ('event', 'i4'),  # event number in data file
        ('sn', 'i4', (self.n_adcs,)),  # adc serial number
        ('ch', 'u1', (self.n_adcs, self.n_channels)),  # channel number
        ('utime_ms', 'u8', (self.n_adcs, self.n_channels)),  # unix time [ms since epoch]
        ('tai_ns', 'u8', (self.n_adcs, self.n_channels)),  # time since PPS [ns]
        ('wvfm_valid', 'u1', (self.n_adcs, self.n_channels))  # boolean, 1 if channel present in event
    ])

    def wvfm_dtype(self): return np.dtype([
        ('samples', 'i2', (self.n_adcs, self.n_channels, self.n_samples))  # sample value
    ])

    def __init__(self, **params):
        super(LightADC64EventGenerator, self).__init__(**params)

        # set up parameters
        for key,val in self.defaults.items():
            setattr(self, key, params.get(key,val))
        self.n_samples = 0

        self.wvfm_dset_name = params['wvfm_dset_name']
        self.event_dset_name = self.dset_name

        self.sn_table = params['sn_table']

        self.input_file = adc64format.ADC64Reader(self.input_filename,self.n_adcs)
        self.input_file.open()
        # Read run info
        _, self.nbytes_runinfo, self.runinfo =adc64format.parse_run_start(self.input_file.stream)
        # use the first file for the event reference
        _, self.chunk_size, test_event = adc64format.parse_chunk(self.input_file.stream)
        ndevices = len(test_event['data'])
        total_length_b = self.input_file.stream.seek(0, 2)-self.nbytes_runinfo
        self.input_file.reset()
        self.n_samples = test_event['data'][0].dtype['voltage'].shape[-1]

        self.end_position = total_length_b // self.chunk_size if self.end_position is None else min(self.end_position, total_length_b // self.chunk_size)
        self.start_position = self.nbytes_runinfo if self.start_position is None else self.start_position
        self.curr_position = self.start_position

        # skip to start position
        self.input_file.stream.seek(self.start_position, 0)

    def __len__(self):
        return (self.end_position - self.start_position) // (self.batch_size)

    def finish(self):
        self.input_file.close()
        super(LightADC64EventGenerator, self).finish()

    def init(self):
        super(LightADC64EventGenerator, self).init()

        # fix dataset dtypes
        self.event_dtype = self.event_dtype()
        self.wvfm_dtype = self.wvfm_dtype()

        # initialize data objects
        self.data_manager.create_dset(self.event_dset_name, dtype=self.event_dtype)
        self.data_manager.create_dset(self.wvfm_dset_name, dtype=self.wvfm_dtype)
        self.data_manager.create_ref(self.event_dset_name, self.wvfm_dset_name)
        params = dict([(key,getattr(self,key)) for key in self.defaults])
        params['sn_table'] = self.sn_table
        params['n_samples'] = self.n_samples
        self.data_manager.set_attrs(self.event_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    start_position=self.start_position,
                                    end_position=self.end_position,
                                    input_filename=self.input_filename,
                                    wvfm_dset_name=self.wvfm_dset_name,
                                    **params
                                    )

    def finish(self):
        super(LightADC64EventGenerator, self).finish()
        self.input_file.close()

    @staticmethod
    def valid_array(arr):
        return arr is not None

    def next(self):
        matched_events = None
        if self.rank == 0:
            # only read from single process
            matched_events = [self.input_file.next(self.batch_size)]  #FIXME: only works with batch_size=1 at the moment (to be fixed in adc64format)
        # format events into output shape / structure
        if matched_events[0] is not None:
            # create new event array / waveform array
            nevents = len(matched_events)
            event_arr = np.zeros(nevents, dtype=self.event_dtype)
            wvfm_arr = np.zeros(nevents, dtype=self.wvfm_dtype)
            for ievent,events in enumerate(matched_events):
                event = events['event']
                data = np.array(events['data'])
                device = np.array(events['device'])
                time = np.array(events['time'])
                event_arr[ievent]['event'] = event['event']
                for iadc, sn in enumerate(self.sn_table):
                    data_index = np.where(device["serial"] == sn)[0]
                    channels = data[data_index]['channel']
                    event_arr[ievent]['sn'][iadc] = device[data_index]['serial']
                    event_arr[ievent]['utime_ms'][iadc] = 0
                    event_arr[ievent]['tai_ns'][iadc] = time[data_index]['tai_s']*1e9 + time[data_index]['tai_ns']
                    event_arr[ievent]['wvfm_valid'][iadc, channels] = True
                    wvfm_arr[ievent]['samples'][iadc, channels] = data[data_index]['voltage']

            # apply different clock frequency
            event_arr['tai_ns'] = (event_arr['tai_ns'] * self.clock_timestamp_factor).astype(event_arr.dtype['tai_ns'].base)
            # mask off any totally empty events
            mask = np.any(event_arr['wvfm_valid'], axis=(-1,-2))
            event_arr = event_arr[mask]
            wvfm_arr = wvfm_arr[mask]
            self.curr_position += len(event_arr)
        else:
            event_arr = np.empty((0,), dtype=self.event_dtype)
            wvfm_arr = np.empty((0,), dtype=self.wvfm_dtype)
        # write event to file
        event_slice = self.data_manager.reserve_data(self.event_dset_name, len(event_arr))
        event_arr['id'] = np.arange(event_slice.start, event_slice.stop)
        self.data_manager.write_data(self.event_dset_name, event_slice, event_arr)

        self.data_manager.reserve_data(self.wvfm_dset_name, event_slice)
        self.data_manager.write_data(self.wvfm_dset_name, event_slice, wvfm_arr)

        # set up references
        #   just event -> wvfm 1:1 refs
        ref = np.c_[event_arr['id'], event_arr['id']]
        self.data_manager.write_ref(self.event_dset_name, self.wvfm_dset_name, ref)

        # if using MPI, divy up data across processes
        if H5FLOW_MPI:
            event_slice = self.comm.bcast(event_slice)
            n = ceil((event_slice.stop - event_slice.start) / self.size)
            start = event_slice.start + self.rank * n
            stop = min(event_slice.stop, event_slice.start + (self.rank+1) * n)
            event_slice = slice(start, stop)

        if len(event_arr):
            return event_slice
        return H5FlowGenerator.EMPTY
