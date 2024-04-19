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
        h5flow-readable format. Uses the ``adc64format`` library to synchronize and align the
        events in multiple files.

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
         - ``utime_ms_window``: ``float``, optional, DAQ unix time window to consider for event matching [ms] (default = 1000)
         - ``tai_ns_window``: ``float``, optional, event timestamp window to consider for event matching [ns] (default = 1000)


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
        n_adcs = 2,
        n_channels = 64,
        batch_size = 128,
        sync_channel = 32,
        sync_threshold = 5000,
        sync_buffer = 200,        
        clock_timestamp_factor = 0.625,
        utime_ms_window = 1000,
        tai_ns_window = 1000,
        )

    def event_dtype(self): return np.dtype([
        ('id', 'u8'),  # unique identifier
        ('event', 'i4'),  # event number in data file
        ('sn', 'i4', (self.n_adcs,)),  # adc serial number
        ('ch', 'u1', (self.n_adcs, self.n_channels)),  # channel number
        ('utime_ms', 'u8', (self.n_adcs)),  # unix time [ms since epoch]
        ('tai_ns', 'u8', (self.n_adcs)),  # time since PPS [ns]
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

        # set up input file
        # find other adc files
        # match input filename with a known adc
        hex_sn_repr = [hex(sn) for sn in self.sn_table]
        input_sn = [sn for sn in hex_sn_repr if sn[2:] in self.input_filename]
        if len(input_sn) != 1:
            raise RuntimeError(f'Unable to uniquely match {self.input_filename} to one of the declared ADCs: {hex_sn_repr}')
        input_sn = input_sn[0]

        # add other ADCs, if possible
        self.input_filenames = list()
        for sn in hex_sn_repr:
            test_filename = self.input_filename.replace(input_sn[2:], sn[2:])
            if sn == input_sn:
                self.input_filenames.append(self.input_filename)
            elif os.path.isfile(test_filename) and sn != input_sn:
                self.input_filenames.append(test_filename)
            else:
                raise RuntimeError(f'Unable to find a file for ADC {sn}, tried at {test_filename}')
        if self.rank == 0:
            logging.info(f'Successfully found a file for each ADC: {", ".join([str(t) for t in zip(hex_sn_repr, self.input_filenames)])}')

        self.input_file = adc64format.ADC64Reader(*self.input_filenames)
        self.input_file.SYNC_CHANNEL = self.sync_channel
        self.input_file.SYNC_THRESHOLD = self.sync_threshold
        self.input_file.UNIX_WINDOW = self.utime_ms_window
        self.input_file.TAI_NS_WINDOW = self.tai_ns_window / self.clock_timestamp_factor
        self.input_file.open()

        # use the first file for the event reference
        self.chunk_size = adc64format.chunk_size(self.input_file.streams[0])
        total_length_b = self.input_file.streams[0].seek(0, 2)
        self.input_file.streams[0].seek(0, 0)
        self.end_position = total_length_b // self.chunk_size if self.end_position is None else min(self.end_position, total_length_b // self.chunk_size)
        self.start_position = 0 if self.start_position is None else self.start_position
        self.curr_position = self.start_position

        # skip to start position
        self.input_file.skip(max(0, self.start_position - 1))

    def __len__(self):
        return (self.end_position - self.start_position) // (self.batch_size)

    def finish(self):
        self.input_file.close()
        super(LightADC64EventGenerator, self).finish()

    def init(self):
        super(LightADC64EventGenerator, self).init()

        # extract number of samples from file 0
        _, _, event = adc64format.parse_chunk(self.input_file.streams[0])
        self.n_samples = event['data'][0].dtype['voltage'].shape[-1]
        adc64format.skip_chunks(self.input_file.streams[0], -1)

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
                                    input_filenames=self.input_filenames,
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
            matched_events = self.input_file.next(self.batch_size)

        # format events into output shape / structure
        if matched_events is not None:
            # create new event array / waveform array
            nevents = len(matched_events[0]['header'])
            is_matched = np.array([[self.valid_array(entry) for entry in ev['header']] for ev in matched_events], dtype=bool)
            event_arr = np.zeros(nevents, dtype=self.event_dtype)
            wvfm_arr = np.zeros(nevents, dtype=self.wvfm_dtype)
            for iadc,events in enumerate(matched_events):
                for ievent in range(len(events['header'])):
                    if not is_matched[iadc, ievent]:
                        continue
                    else:
                        data = events['data'][ievent]
                        event = events['event'][ievent]
                        device = events['device'][ievent]
                        time = events['time'][ievent]
                        header = events['header'][ievent]
                        channels = data['channel']
                        event_arr[ievent]['event'] = event['serial']
                        event_arr[ievent]['sn'][iadc] = device['serial']
                        event_arr[ievent]['ch'][iadc, channels] = channels
                        event_arr[ievent]['utime_ms'][iadc] = header['unix']
                        event_arr[ievent]['tai_ns'][iadc] = time['tai_s']*1e9 + time['tai_ns']
                        event_arr[ievent]['wvfm_valid'][iadc, channels] = True
                        wvfm_arr[ievent]['samples'][iadc, channels] = data['voltage']

            # apply different clock frequency
            event_arr['tai_ns'] = (event_arr['tai_ns'] * self.clock_timestamp_factor).astype(event_arr.dtype['tai_ns'].base)

            # mask off any totally empty events
            mask = np.any(event_arr['wvfm_valid'], axis=(-1,-2))
            event_arr = event_arr[mask]
            wvfm_arr = wvfm_arr[mask]
            
            # mask off any extraneous events
            if self.curr_position + len(event_arr) > self.end_position:
                mask = self.curr_position + np.arange(len(event_arr)) < self.end_position
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
