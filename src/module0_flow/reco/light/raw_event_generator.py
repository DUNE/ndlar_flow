import numpy as np
import h5py
import numpy.ma as ma
from collections import defaultdict
import logging

from h5flow.core import H5FlowGenerator, resources
from h5flow import H5FLOW_MPI


class LightEventGenerator(H5FlowGenerator):
    '''
        Light system event builder - converts ``rwf_XX.root`` files to an event-packed
        h5flow-readable format

        Parameters:
         - ``wvfm_dset_name`` : ``str``, required, path to dataset to store raw waveforms
         - ``n_adcs`` : ``int``, number of ADC serial numbers
         - ``n_channels`` : ``int``, number of channels per ADC
         - ``n_samples`` : ``int``, number of samples in waveform, optional if RunData resource exists
         - ``chunk_size`` : ``int``, optional, number of events to buffer before initiating loop

        Generates a lightweight "event" dataset along with a dataset containing
        event-packed raw waveforms.

        Requires RunData resource in workflow.

        Example config::

            flow:
                source: light_event_generator
                stages: []

            light_event_generator:
                classname: LightEventGenerator
                dset_name: 'light/events'
                params:
                    wvfm_dset_name: 'light/wvfm'
                    n_adcs: 2
                    n_channels: 64
                    n_samples: 256
                    chunk_size: 128
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
    default_n_adcs = 2
    default_n_channels = 64
    default_chunk_size = 128
    default_utime_ms_window = 1000
    default_tai_ns_window = 1000
    default_tai_ns_mod = 1000000000

    def buffer_dtype(self): return np.dtype([
        ('event', 'i4'),  # event number in source ROOT file
        ('sn', 'i4'),  # adc serial number
        ('ch', 'u1'),  # channel number
        ('utime_ms', 'u8'),  # unix time [ms since epoch]
        ('tai_ns', 'u8'),  # time since PPS [ns]
        ('wvfm', 'i2', self.n_samples)  # sample value
    ])

    def event_dtype(self): return np.dtype([
        ('id', 'u8'),  # unique identifier
        ('event', 'i4'),  # event number in source ROOT file
        ('sn', 'i4', self.n_adcs),  # adc serial number
        ('ch', 'u1', (self.n_adcs, self.n_channels)),  # channel number
        ('utime_ms', 'u8', (self.n_adcs, self.n_channels)),  # unix time [ms since epoch]
        ('tai_ns', 'u8', (self.n_adcs, self.n_channels)),  # time since PPS [ns]
        ('wvfm_valid', 'u1', (self.n_adcs, self.n_channels))  # boolean, 1 if channel present in event
    ])

    def wvfm_dtype(self): return np.dtype([
        ('samples', 'i2', (self.n_adcs, self.n_channels, self.n_samples))  # sample value
    ])

    def __init__(self, **params):
        super(LightEventGenerator, self).__init__(**params)

        # set up parameters
        self.n_adcs = params.get('n_adcs', self.default_n_adcs)
        self.n_channels = params.get('n_channels', self.default_n_channels)
        self.n_samples = params.get('n_samples')
        self.chunk_size = params.get('chunk_size', self.default_chunk_size)
        self.utime_ms_window = params.get('utime_ms_window', self.default_utime_ms_window)
        self.tai_ns_window = params.get('tai_ns_window', self.default_tai_ns_window)
        self.tai_ns_mod = int(params.get('tai_ns_mod', self.default_tai_ns_mod))
        self.wvfm_dset_name = params.get('wvfm_dset_name')
        self.event_dset_name = self.dset_name

        # set up input file
        import ROOT
        self.root_file = ROOT.TFile(self.input_filename, 'r')
        self.rwf = self.root_file.Get('rwf')
        self.end_position = self.rwf.GetEntries() if self.end_position is None else min(self.end_position, self.rwf.GetEntries())
        self.start_position = 0 if self.start_position is None else self.start_position
        self.entry = self.start_position

    def init(self):
        super(LightEventGenerator, self).init()

        if self.data_manager.dset_exists(self.event_dset_name):
            raise RuntimeError(f'{self.event_dset_name} already exists, refusing to append!')
        if self.data_manager.dset_exists(self.wvfm_dset_name):
            raise RuntimeError(f'{self.wvfm_dset_name} already exists, refusing to append!')

        if not self.n_samples:
            self.n_samples = resources['RunData'].light_samples

        # fix dataset dtypes
        self.buffer_dtype = self.buffer_dtype()
        self.event_dtype = self.event_dtype()
        self.wvfm_dtype = self.wvfm_dtype()

        # set up input data buffers
        self.data_buffer = defaultdict(list)  # serial number : [<buffered wvfm data>]
        self.event = np.zeros((1,), dtype=self.event_dtype)
        self.wvfms = np.zeros((1,), dtype=self.wvfm_dtype)
        self.event_buffer = list()
        self.curr_event = 0

        # initialize data objects
        self.data_manager.create_dset(self.event_dset_name, dtype=self.event_dtype)
        self.data_manager.create_dset(self.wvfm_dset_name, dtype=self.wvfm_dtype)
        self.data_manager.create_ref(self.event_dset_name, self.wvfm_dset_name)
        self.data_manager.set_attrs(self.event_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    n_adcs=self.n_adcs,
                                    n_channels=self.n_channels,
                                    n_samples=self.n_samples,
                                    chunk_size=self.chunk_size,
                                    utime_ms_window=self.utime_ms_window,
                                    tai_ns_window=self.tai_ns_window,
                                    wvfm_dset_name=self.wvfm_dset_name,
                                    start_position=self.start_position,
                                    end_position=self.end_position,
                                    input_filename=self.input_filename
                                    )

    def finish(self):
        super(LightEventGenerator, self).finish()
        self.root_file.Close()

    def next(self):
        if self.rank == 0:  # only use a single process

            subloop_flag = True
            while (len(self.event_buffer) < self.chunk_size * self.size) and \
                    ((self.entry < self.end_position) or (all([len(buf) for buf in self.data_buffer.values()]))):
                # read until we've collected a large enough sample of events
                # stop when all data buffers are empty or we've reached the end position
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
                    # logging.debug(f'~~~ NEW EVENT ~~~ (ch {np.sum(self.event["wvfm_valid"])})')
                    self.event_buffer.append((self.event.copy(), self.wvfms.copy()))

                    self.event = np.zeros((1,), dtype=self.event_dtype)
                    self.wvfms = np.zeros((1,), dtype=self.wvfm_dtype)

                # update position
                self.curr_event = new_event
                subloop_flag = True

        if H5FLOW_MPI:
            self.entry = self.comm.bcast(self.entry, root=0)
            if self.rank == 0:
                logging.debug(f'entry {self.entry-self.start_position}/{self.end_position-self.start_position} ({round(self.entry-self.start_position/(self.end_position-self.start_position), 3)}) buffers {[(key,len(val)) for key,val in self.data_buffer.items()]}')

        # distribute events to processes
        nevents = len(self.event_buffer)
        scatter_events = [self.event_buffer[nevents // self.size * i:nevents // self.size * (i + 1)] for i in range(self.size)]
        if H5FLOW_MPI:
            events = self.comm.scatter(scatter_events, root=0)
        else:
            events = scatter_events[0]

        self.event_buffer = self.event_buffer[(nevents // self.size) * self.size:]
        nevents = len(events)

        if nevents > 0:
            events, wvfms = zip(*events)
        else:
            events, wvfms = [], []

        event_arr = np.concatenate(events, axis=0) if len(events) else np.empty(0, dtype=self.event_dtype)
        wvfm_arr = np.concatenate(wvfms, axis=0) if len(wvfms) else np.empty(0, dtype=self.wvfm_dtype)

        # write event to file
        event_slice = self.data_manager.reserve_data(self.event_dset_name, nevents)
        event_arr['id'] = np.arange(event_slice.start, event_slice.stop)
        self.data_manager.write_data(self.event_dset_name, event_slice, event_arr)

        self.data_manager.reserve_data(self.wvfm_dset_name, event_slice)
        self.data_manager.write_data(self.wvfm_dset_name, event_slice, wvfm_arr)

        # set up references
        #   just event -> wvfm 1:1 refs for now
        ref = np.c_[event_arr['id'], event_arr['id']]
        self.data_manager.write_ref(self.event_dset_name, self.wvfm_dset_name, ref)

        if len(events) == 0:
            return H5FlowGenerator.EMPTY
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
            Pull from event buffers and assemble into event (fills event and wvfm arrays)

            :returns: ``event_number`` : event_number will be incremented when full event has been assembled
        '''
        while all([len(buf) for buf in self.data_buffer.values()]):
            # check if data in buffers match (either the current event or each other)
            sn = [key for key in self.data_buffer.keys()]
            utime_ms = np.array([self.data_buffer[key][0][0]['utime_ms'] for key in sn]).astype(int)
            tai_ns = np.array([self.data_buffer[key][0][0]['tai_ns'] for key in sn]).astype(int) % self.tai_ns_mod

            valid_mask = self.event['wvfm_valid'].astype(bool)
            if np.any(valid_mask):
                # existing data in event, check if new data matches
                event_ms = ma.array(self.event['utime_ms'].ravel(), mask=~valid_mask.ravel()).mean()
                event_ns = ma.array(self.event['tai_ns'].ravel(), mask=~valid_mask.ravel()).mean() % self.tai_ns_mod
                match_idcs = np.argwhere(
                    (np.abs(utime_ms - event_ms) <= self.utime_ms_window) & (np.abs(tai_ns - event_ns) <= self.tai_ns_window)
                ).ravel()

                if len(match_idcs):
                    # there's a match (or more), so just grab one of them
                    i = None
                    for j in match_idcs:
                        data = self.data_buffer[sn[j]][0][0]
                        sn_hash = self._sn_hash(sn[j])
                        ch_hash = self._ch_hash(data['ch'])

                        if valid_mask[0, sn_hash, ch_hash]:
                            # already placed into event, skip
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
                    if np.any(np.diff(tai_ns[idcs].astype(int)) > 1e8) and np.any(np.abs(np.diff(utime_ms[idcs].astype(int))) < 500):
                        # check for potential PPS rollover, => use reversed ordering
                        i = idcs[-1]
                    elif np.any(np.abs(np.diff(utime_ms[idcs].astype(int))) > 500):
                        # check for significant time offset, => use utime ordering
                        i = np.argsort(utime_ms)[0]
                    else:
                        # default is earlier tai ns ordering
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
            self.event['utime_ms'][0, sn_hash, ch_hash] = data['utime_ms']
            self.event['tai_ns'][0, sn_hash, ch_hash] = data['tai_ns'] % self.tai_ns_mod
            self.event['wvfm_valid'][0, sn_hash, ch_hash] = True

            # fill waveform array
            self.wvfms['samples'][0, sn_hash, ch_hash] = data['wvfm']

            # remove from buffer
            self.data_buffer[sn[i]] = self.data_buffer[sn[i]][1:]

        return event_number
