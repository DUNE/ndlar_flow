import numpy as np
import h5py
import numpy.ma as ma
import ROOT
from collections import defaultdict
import logging

from h5flow.core import H5FlowGenerator

class LightEventGenerator(H5FlowGenerator):
    '''

    '''
    default_busy_channel = 0
    default_n_adcs = 2
    default_n_channels = 64
    default_n_samples = 256
    default_chunk_size = 128
    default_wvfm_dset_name = 'light/wvfm'

    buffer_dtype = lambda self : np.dtype([
        ('event', 'i4'), # event number in source ROOT file
        ('sn', 'i4'), # adc serial number
        ('ch', 'u1'), # channel number
        ('utime_ms', 'u8'), # unix time [ms since epoch]
        ('tai_ns', 'u8'), # time since PPS [ns]
        ('wvfm', 'i2', self.n_samples) # sample value
        ])
    event_dtype = lambda self : np.dtype([
        ('event_id', 'u8'), # unique identifier
        ('event', 'i4'), # event number in source ROOT file
        ('sn', 'i4', self.n_adcs), # adc serial number
        ('ch', 'u1', (self.n_adcs, self.n_channels)), # channel number
        ('utime_ms', 'u8', self.n_adcs), # unix time [ms since epoch]
        ('tai_ns', 'u8', self.n_adcs), # time since PPS [ns]
        ('alignment', 'i4', self.n_adcs), # busy signal rising edge sample
        ('wvfm_valid', 'u1', (self.n_adcs, self.n_channels)) # boolean, 1 if channel present in event
        ])
    wvfm_dtype = lambda self : np.dtype([
        ('samples', 'i2', (self.n_adcs, self.n_channels, self.n_samples)) # sample value
        ])

    def __init__(self, **params):
        super(LightEventGenerator,self).__init__(**params)

        # set up parameters
        self.busy_channel = params.get('busy_channel', self.default_busy_channel)
        self.n_adcs = params.get('n_adcs', self.default_n_adcs)
        self.n_channels = params.get('n_channels', self.default_n_channels)
        self.n_samples = params.get('n_samples', self.default_n_samples)
        self.chunk_size = params.get('chunk_size', self.default_chunk_size)
        self.wvfm_dset_name = params.get('wvfm_dset_name', self.default_wvfm_dset_name)
        self.event_dset_name = self.dset_name

        # fix dataset dtypes
        self.buffer_dtype = self.buffer_dtype()
        self.event_dtype = self.event_dtype()
        self.wvfm_dtype = self.wvfm_dtype()

        # set up input data buffers
        self.data_buffer = defaultdict(list) # serial number : [<buffered wvfm data>]
        self.event = np.zeros((1,), dtype=self.event_dtype)
        self.wvfms = np.zeros((1,), dtype=self.wvfm_dtype)
        self.event_buffer = list()
        self.curr_event = 0

        # set up input file
        self.root_file = ROOT.TFile(self.input_filename, 'r')
        self.rwf = self.root_file.Get('rwf')
        self.end_position = self.rwf.GetEntries() if self.end_position is None else min(self.end_position, self.rwf.GetEntries())
        self.start_position = 0 if self.start_position is None else self.start_position
        self.entry = self.rank

    def init(self):
        super(LightEventGenerator,self).init()

        if self.data_manager.dset_exists(self.event_dset_name):
            raise RuntimeError(f'{self.event_dset_name} already exists, refusing to append!')
        if self.data_manager.dset_exists(self.wvfm_dset_name):
            raise RuntimeError(f'{self.wvfm_dset_name} already exists, refusing to append!')

        # initialize data objects
        self.data_manager.create_dset(self.event_dset_name, dtype=self.event_dtype)
        self.data_manager.create_dset(self.wvfm_dset_name, dtype=self.wvfm_dtype)
        self.data_manager.create_ref(self.event_dset_name, self.wvfm_dset_name)
        self.data_manager.set_attrs(self.event_dset_name,
            classname=self.classname,
            class_version=self.class_version,
            busy_channel=self.busy_channel,
            n_adcs=self.n_adcs,
            n_channels=self.n_channels,
            n_samples=self.n_samples,
            chunk_size=self.chunk_size,
            wvfm_dset_name=self.wvfm_dset_name,
            start_position=self.start_position,
            end_position=self.end_position,
            input_filename=self.input_filename
            )

    def finish(self):
        self.root_file.Close()

    def next(self):
        if self.entry >= self.end_position:
            return H5FlowGenerator.EMPTY

        if self.rank == 0: # only use a single process

            subloop_flag = True
            while (len(self.event_buffer) < self.chunk_size * self.size) and \
                ((self.entry < self.end_position) or (all([len(buf) for buf in self.data_buffer.values()]))):
                # read until we've collected a large enough sample of events, stop once all data buffers are empty, and we've reached the end position
                while (self.entry < self.end_position) and subloop_flag:
                    # get next entry
                    self.rwf.GetEntry(self.entry)
                    self.entry += 1

                    # add entry to data buffer
                    self.store_entry()

                    # get data until data present in all buffers
                    subloop_flag = not all([len(buf) for buf in self.data_buffer.values()])

                # combine data into event array
                new_event = self.store_event(self.curr_event)
                if new_event != self.curr_event:
                    self.event_buffer.append((self.event.copy(), self.wvfms.copy()))

                    self.event = np.zeros((1,), dtype=self.event_dtype)
                    self.wvfms = np.zeros((1,), dtype=self.wvfm_dtype)

                # update position
                self.curr_event = new_event
                subloop_flag = True

        self.entry = self.comm.bcast(self.entry, root=0)
        logging.debug(f'entry {self.entry}/{self.end_position} buffers {[(key,len(val)) for key,val in self.data_buffer.items()]}')

        # distribute events to processes
        nevents = len(self.event_buffer)
        scatter_events = [self.event_buffer[nevents//self.size * i:nevents//self.size * (i+1)] for i in range(self.size)]
        events = self.comm.scatter(scatter_events, root=0)
        self.event_buffer = self.event_buffer[(nevents//self.size)*self.size:]
        nevents = len(events)

        events, wvfms = zip(*events)

        event_arr = np.concatenate(events, axis=0) if len(events) else np.empty(0, dtype=self.event_dtype)
        wvfm_arr = np.concatenate(wvfms, axis=0) if len(wvfms) else np.empty(0, dtype=self.wvfm_dtype)

        # write event to file
        event_slice = self.data_manager.reserve_data(self.event_dset_name, nevents)
        event_arr['event_id'] = np.arange(event_slice.start, event_slice.stop)
        self.data_manager.write_data(self.event_dset_name, event_slice, event_arr)

        self.data_manager.reserve_data(self.wvfm_dset_name, event_slice)
        self.data_manager.write_data(self.wvfm_dset_name, event_slice, wvfm_arr)

        # set up references
        #   just event -> wvfm refs for now
        self.data_manager.reserve_ref(self.event_dset_name, self.wvfm_dset_name, event_slice)
        ref = event_arr['event_id']
        self.data_manager.write_ref(self.event_dset_name, self.wvfm_dset_name, event_slice, ref)

        return event_slice

    def _sn_hash(self, sn):
        return sn % self.n_adcs

    def _ch_hash(self, ch):
        return ch % self.n_channels

    def store_entry(self):
        '''
            Convert TTree entry into a numpy type - this is heckin slow....
        '''
        # create new array
        arr = np.zeros((1,), dtype=self.buffer_dtype)

        # copy data (this also inverts the waveforms)
        arr['event'] = self.rwf.event
        arr['sn'] = self.rwf.sn
        arr['ch'] = self.rwf.ch
        arr['utime_ms'] = self.rwf.utime_ms
        arr['tai_ns'] = self.rwf.tai_ns
        arr['wvfm'] = -np.frombuffer(self.rwf.th1s_ptr.fArray, dtype='i2', count=self.n_samples)

        self.data_buffer[self.rwf.sn].append(arr)

    def store_event(self, event_number):
        '''
            Pull from event buffers and assemble into event (fills base array and wvfm array)

            :returns: ``event_number`` : event_number will be incremented when full event has been assembled
        '''
        while all([len(buf) for buf in self.data_buffer.values()]):
            # check if data in buffers match (either the current event or each other)
            sn = [key for key in self.data_buffer.keys()]
            utime_ms = np.array([self.data_buffer[key][0][0]['utime_ms'] for key in sn]).astype(int)
            tai_ns = np.array([self.data_buffer[key][0][0]['tai_ns'] for key in sn]).astype(int)

            valid_mask = np.any(self.event['wvfm_valid'], axis=-1)
            if np.any(valid_mask):
                # existing data in event, check if new data matches
                event_ms = self.event['utime_ms'][valid_mask].astype(int)
                event_ns = self.event['tai_ns'][valid_mask].astype(int)
                match_idcs = np.argwhere(
                    (np.abs(utime_ms-event_ms) <= 1000) & (np.abs(tai_ns-event_ns) <= 1000)
                    ).flatten()

                if len(match_idcs):
                    # there's a match (or more), so just grab one of them
                    i = None
                    for j in match_idcs:
                        data = self.data_buffer[sn[j]][0][0]
                        sn_hash = self._sn_hash(sn[j])
                        ch_hash = self._ch_hash(data['ch'])

                        if self.event['wvfm_valid'][0, sn_hash, ch_hash]:
                            # already placed into event, continue
                            continue
                        i = j
                        break
                    if i is None:
                        # no place for any data in current event, so declare a new event
                        return event_number + 1
                else:
                    # there's no match, so declare a new event
                    return event_number + 1
            else:
                # no existing data in event, fill with earliest
                idcs = np.argsort(tai_ns)

                if len(idcs) > 1:
                    # check for potential PPS rollover
                    if np.any(np.diff(tai_ns[idcs].astype(int)) > 1e8) and np.any(np.abs(np.diff(utime_ms[idcs].astype(int))) < 500):
                        i = idcs[-1]
                    # check for significant time offset (use utime ordering)
                    if np.any(np.abs(np.diff(utime_ms[idcs].astype(int))) > 500):
                        i = np.argsort(utime_ms)[0]
                    # default to tai ns ordering
                    else:
                        i = idcs[0]
                else:
                    i = idcs[0]

            data = self.data_buffer[sn[i]][0][0]
            sn_hash = self._sn_hash(sn[i])
            ch_hash = self._ch_hash(data['ch'])

            # fill base array
            self.event['event'] = event_number
            self.event['sn'][0, sn_hash] = data['sn']
            self.event['ch'][0, sn_hash, ch_hash] = data['ch']
            self.event['utime_ms'][0, sn_hash] = data['utime_ms']
            self.event['tai_ns'][0, sn_hash] = data['tai_ns']
            self.event['wvfm_valid'][0, sn_hash, ch_hash] = True

            # fill waveform array
            self.wvfms['samples'][0, sn_hash, ch_hash] = data['wvfm']

            # remove from buffer
            self.data_buffer[sn[i]] = self.data_buffer[sn[i]][1:]

        return event_number


