import numpy as np
import logging

from h5flow import H5FLOW_MPI
if H5FLOW_MPI:
    from mpi4py import MPI


class RawEventBuilder(object):
    '''
        Base class for raw event builder algorithms. Defines the following API
        for implementing new event-building algorithms:

    '''
    version = '0.0.0'

    def __init__(self, **params):
        '''
            Initialize given parameters for the class, each parameter is
            optional with a default provided by the implemented class
        '''
        pass

    def get_config(self):
        '''
            :returns: a `dict` of the instance configuration parameters
        '''
        return dict()

    def build_events(self, packets, unix_ts, mc_assn=None):
        '''
            Run the event builder on a sub-set of packet-formatted array data
            The unix timestamp for each packet is provided as additional meta-data

            :param packets: packet-formatted array (shape: ``(N,)``)

            :param unix_ts: Unix timestamp for each packet in ``packets`` (shape: ``(N,)``)

            :param mc_assn: array of mc truth associations for each packet in ``packets`` (shape: ``(N,)``)

            :returns: a `tuple` of `lists` of the packet array grouped into events, along with their corresponding unix timestamps
        '''
        raise NotImplementedError('Event building for this class has not been implemented!')

    def cross_rank_get_attrs(self, *attrs):
        '''
            Get an attribute from another MPI process. In particular:

             - ``N-1`` sends its stored attribute to ``0``
             - then, ``i`` receives the attribute from ``i-1``

            :param attrs: ``list`` of ``str`` specifying attributes to pass between ranks

        '''
        if H5FLOW_MPI:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            if rank == 0:
                for attr in attrs:
                    logging.debug(f'get {attr}: {getattr(self,attr).shape}')

            if size < 2:
                return
            # rank 1 get stored from rank N-1
            if rank == size - 1:
                # logging.debug('{}: {} -> {}'.format(attrs,rank,0))
                d = dict([(attr, getattr(self, attr)) for attr in attrs])
                comm.send(d, dest=0)
            # rank i give value to i+1
            source = rank - 1 if rank > 0 else size - 1
            # logging.debug('{}: {} <- {}'.format(attrs,rank,source))
            for attr, val in comm.recv(source=source).items():
                setattr(self, attr, val)

    def cross_rank_set_attrs(self, *attrs):
        '''
            Update an attribute and send to another MPI process. In particular:

             - ``i`` sends the attribute to ``i+1``
             - ``N-1`` does nothing

            :param attrs: ``list`` of ``str`` specifying attributes to pass between ranks

        '''
        if H5FLOW_MPI:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            if size < 2:
                return
            # rank N-1 store value for next iteration
            if rank != size - 1:
                # logging.debug('{}: {} -> {}'.format(attrs,rank,rank+1))
                d = dict([(attr, getattr(self, attr)) for attr in attrs])
                comm.send(d, dest=rank + 1)


class TimeDeltaRawEventBuilder(RawEventBuilder):
    '''
        Original "gap-based" event building

        Searches for separations in data greater than the ``event_dt`` parameter.
        Events are formed at these boundaries. Any events that are greater than
        ``max_event_dt`` in length are broken up into separate events at the
        ``max_event_dt`` boundaries.

        Configurable parameters::

            event_dt        - gap size to separate into different events
            max_event_dt    - maximum event length

    '''
    version = '0.0.0'

    default_event_dt = 1820
    default_max_event_dt = 1820 * 3

    def __init__(self, **params):
        super(TimeDeltaRawEventBuilder, self).__init__(**params)
        self.event_dt = params.get('event_dt', self.default_event_dt)
        self.max_event_dt = params.get('max_event_dt', self.default_max_event_dt)

        self.event_buffer = np.empty((0,))  # keep track of partial events from previous calls
        self.event_buffer_unix_ts = np.empty((0,), dtype='u8')
        self.event_buffer_mc_assn = np.empty((0,))

    def get_config(self):
        return dict(
            event_dt=self.event_dt,
            max_event_dt=self.max_event_dt
        )

    def build_events(self, packets, unix_ts, mc_assn=None):
        self.cross_rank_get_attrs('event_buffer', 'event_buffer_unix_ts', 'event_buffer_mc_assn')

        if len(packets) == 0:
            return ([], []) if mc_assn is None \
                else ([], [], [])

        # sort packets to fix 512 bug
        packets = np.append(self.event_buffer, packets) if len(self.event_buffer) else packets
        sorted_idcs = np.argsort(packets, order='timestamp')
        packets = packets[sorted_idcs]
        unix_ts = np.append(self.event_buffer_unix_ts, unix_ts)[sorted_idcs] if len(self.event_buffer_unix_ts) else unix_ts[sorted_idcs]
        if mc_assn is not None:
            mc_assn = np.append(self.event_buffer_mc_assn, unix_ts)[sorted_idcs] if len(self.event_buffer_mc_assn) else mc_assn[sorted_idcs]

        # cluster into events by delta t
        packet_dt = packets['timestamp'][1:] - packets['timestamp'][:-1]
        event_idx = np.argwhere(np.abs(packet_dt) > self.event_dt).ravel() - 1
        events = np.split(packets, event_idx)
        event_unix_ts = np.split(unix_ts, event_idx)
        if mc_assn is not None:
            event_mc_assn = np.split(mc_assn, event_idx)

        # reserve last event of every chunk for next iteration
        if len(events):
            self.event_buffer = np.copy(events[-1])
            self.event_buffer_unix_ts = np.copy(event_unix_ts[-1])
            del events[-1]
            del event_unix_ts[-1]
            if mc_assn is not None:
                self.event_buffer_mc_assn = np.copy(event_mc_assn[-1])
                del event_mc_assn[-1]
        self.cross_rank_set_attrs('event_buffer', 'event_buffer_unix_ts', 'event_buffer_mc_assn')

        # break up events longer than max window
        i = 0
        while i < len(events) and len(events[i]) \
                and events[i]['timestamp'][-1] - events[i]['timestamp'][0] > self.max_event_dt:
            event0, event1, unix_ts0, unix_ts1, mc_assn0, mc_assn1 = self.split_at_timestamp(
                events[i]['timestamp'][0] + self.max_event_dt,
                events[i],
                event_unix_ts[i]
            ), None, None if mc_assn is None else self.split_at_timestamp(
                events[i]['timestamp'][0] + self.max_event_dt,
                events[i],
                event_unix_ts[i],
                mc_assn[i]
            )

            events[i] = event0
            events.insert(i + 1, event1)
            event_unix_ts[i] = unix_ts0
            event_unix_ts.insert(i + 1, unix_ts1)
            if mc_assn is not None:
                event_mc_assn[i] = mc_assn0
                event_mc_assn.insert(i + 1, mc_assn1)
            i += 1
        
        
        
        '''
        # only return packets from events
        return zip(*[v for i, v in enumerate(zip(events, event_unix_ts)) if is_event[i]]) if mc_assn is None \
            else zip(*[v for i, v in enumerate(zip(events, event_unix_ts, event_mc_assn)) if is_event[i]])
        '''
        return events, event_unix_ts if mc_assn is None \
            else events, event_unix_ts, event_mc_assn

    @staticmethod
    def split_at_timestamp(timestamp, event, *args):
        '''
        Breaks event into two arrays at index where event['timestamp'] > timestamp
        Additional arrays can be specified with kwargs and will be split at the same
        index

        :returns: tuple of two event halves followed by any additional arrays (in pairs)
        '''
        args = list(args)
        timestamps = event['timestamp'].astype(int)
        indices = np.argwhere(timestamps > timestamp)
        if len(indices):
            idx = np.min(indices)
            args.insert(0, event)
            rv = [(arg[:idx], arg[idx:]) for arg in args]
            return tuple(v for vs in rv for v in vs)
        args.insert(0, event)
        rv = [(arg, np.array([], dtype=arg.dtype)) for arg in args]
        return tuple(v for vs in rv for v in vs)

    

class SymmetricWindowRawEventBuilder(RawEventBuilder):
    '''
        A sliding-window based event builder.

        Histograms the packets into bins of ``window`` width. Events are formed
        if a bin content is greater than ``threshold``. The event extent covers
        the bin of interest and +/- 1 bin. If multiple adjacent bins exceed
        the threshold, they are merged into a single event.

        Configurable parameters::

            window      - bin width
            threshold   - number of correlated hits to initiate event

    '''
    version = '0.0.2'

    default_window = 1820 // 2
    default_threshold = 10
    default_rollover_ticks = 10000000

    def __init__(self, **params):
        super(SymmetricWindowRawEventBuilder, self).__init__(**params)
        self.window = params.get('window', self.default_window)
        self.threshold = params.get('threshold', self.default_threshold)
        self.rollover_ticks = params.get('rollover_ticks', self.default_rollover_ticks)

        self.event_buffer = np.empty((0,))  # keep track of partial events from previous calls
        self.event_buffer_unix_ts = np.empty((0,), dtype='u8')
        self.event_buffer_mc_assn = np.empty((0,))

    def get_config(self):
        return dict(
            window=self.window,
            threshold=self.threshold,
            rollover_ticks=self.rollover_ticks,
        )

    def build_events(self, packets, unix_ts, mc_assn=None):
        # fetch attribute from appropriate process
        self.cross_rank_get_attrs('event_buffer', 'event_buffer_unix_ts', 'event_buffer_mc_assn')

        if len(packets) == 0:
            self.event_buffer = np.empty((0,), dtype=packets.dtype)
            self.event_buffer_unix_ts = np.empty((0,), dtype=unix_ts.dtype)
            if mc_assn is not None:
                self.event_buffer_mc_assn = np.empty((0,), dtype=mc_assn.dtype)
            self.cross_rank_set_attrs('event_buffer', 'event_buffer_unix_ts', 'event_buffer_mc_assn')
            return ([], []) if mc_assn is None \
                else ([], [], [])

        # sort packets to fix 512 bug
        packets = np.append(self.event_buffer, packets) if len(self.event_buffer) else packets

        # correct for rollovers
        rollover = np.zeros((len(packets),), dtype='i8')
        for io_group in np.unique(packets['io_group']):
            # find rollovers
            mask = (packets['io_group'] == io_group) & (packets['packet_type'] == 6) & (packets['trigger_type'] == 83)
            rollover[mask] = self.rollover_ticks
            # calculate sum of rollovers
            mask = (packets['io_group'] == io_group)
            rollover[mask] = np.cumsum(rollover[mask]) - rollover[mask]
            # correct for readout delay (only in real data)
            if mc_assn is None:
                mask = (packets['io_group'] == io_group) & (packets['packet_type'] == 0) \
                    & (packets['receipt_timestamp'].astype(int) - packets['timestamp'].astype(int) < 0)
                rollover[mask] -= self.rollover_ticks

        ts = packets['timestamp'].astype('i8') + rollover

        sorted_idcs = np.argsort(ts)
        ts = ts[sorted_idcs]
        packets = packets[sorted_idcs]
        unix_ts = np.append(self.event_buffer_unix_ts, unix_ts)[sorted_idcs] if len(self.event_buffer_unix_ts) else unix_ts[sorted_idcs]
        if mc_assn is not None:
            mc_assn = np.append(self.event_buffer_mc_assn, mc_assn)[sorted_idcs] if len(self.event_buffer_mc_assn) else mc_assn[sorted_idcs]

        # calculate time distance between hits
        min_ts, max_ts = np.min(ts), np.max(ts)
        bin_edges = np.linspace(min_ts - 1, max_ts + 1, int((max_ts - min_ts + 2) // self.window))
        hist, bin_edges = np.histogram(ts, bins=bin_edges)

        # find high correlation regions
        event_mask = (hist > self.threshold)
        # include ±1 bin
        event_mask[:-1] = event_mask[:-1] | event_mask[1:]
        event_mask[1:] = event_mask[:-1] | event_mask[1:]

        # find rising/falling edges
        event_edges = np.diff(event_mask.astype(int))
        event_start_timestamp = bin_edges[1:-1][event_edges > 0]
        event_end_timestamp = bin_edges[1:-1][event_edges < 0]

        if not np.any(event_mask):
            # no events
            self.event_buffer = np.empty((0,), dtype=packets.dtype)
            self.event_buffer_unix_ts = np.empty((0,), dtype=unix_ts.dtype)
            if mc_assn is not None:
                self.event_buffer_mc_assn = np.empty((0,), dtype=mc_assn.dtype)
            self.cross_rank_set_attrs('event_buffer', 'event_buffer_unix_ts', 'event_buffer_mc_assn')
            return ([], []) if mc_assn is None \
                else ([], [], [])
        if not len(event_start_timestamp):
            # first packet starts event
            event_start_timestamp = np.r_[min_ts, event_start_timestamp]
        if not len(event_end_timestamp):
            # last packet ends event, keep for next but return no events
            mask = ts >= event_start_timestamp[-1]
            self.event_buffer = packets[mask]
            self.event_buffer_unix_ts = unix_ts[mask]
            if mc_assn is not None:
                self.event_buffer_mc_assn = mc_assn[mask]
            self.cross_rank_set_attrs('event_buffer', 'event_buffer_unix_ts', 'event_buffer_mc_assn')
            return ([], []) if mc_assn is None \
                else ([], [], [])

        if event_end_timestamp[0] < event_start_timestamp[0]:
            # first packet is in first event, make sure you align the start/end idcs correctly
            event_start_timestamp = np.r_[min_ts, event_start_timestamp]
        if event_end_timestamp[-1] < event_start_timestamp[-1]:
            # last event is incomplete, reserve for next iteration
            mask = ts >= event_start_timestamp[-1]
            self.event_buffer = packets[mask]
            self.event_buffer_unix_ts = unix_ts[mask]
            if mc_assn is not None:
                self.event_buffer_mc_assn = mc_assn[mask]
            self.cross_rank_set_attrs('event_buffer', 'event_buffer_unix_ts', 'event_buffer_mc_assn')
            packets = packets[~mask]
            unix_ts = unix_ts[~mask]
            if mc_assn is not None:
                mc_assn = mc_assn[~mask]
            event_start_timestamp = event_start_timestamp[:-1]
        else:
            self.event_buffer = np.empty((0,), dtype=packets.dtype)
            self.event_buffer_unix_ts = np.empty((0,), dtype=unix_ts.dtype)
            if mc_assn is not None:
                self.event_buffer_mc_assn = np.empty((0,), dtype=mc_assn.dtype)
            self.cross_rank_set_attrs('event_buffer', 'event_buffer_unix_ts', 'event_buffer_mc_assn')

        # find starting event division for each packet
        event_idx_start = np.searchsorted(event_start_timestamp, ts, side='right')-1
        # find ending event division for each packet
        event_idx_end = np.searchsorted(event_end_timestamp, ts, side='left')
        # find packets within event boundaries
        event_mask = (event_idx_start == event_idx_end)
        # break packets at each event division
        event_idcs = np.argwhere((event_idx_start[1:] != event_idx_start[:-1]) | (event_idx_end[1:] != event_idx_end[:-1])).ravel() + 1
        # flag breaks that are events (and not gaps between events)
        is_event = np.r_[False, event_mask[event_idcs]]

        events = np.split(packets, event_idcs)
        event_unix_ts = np.split(unix_ts, event_idcs)
        if mc_assn is not None:
            event_mc_assn = np.split(mc_assn, event_idcs)
        
        # only return packets from events
        return zip(*[v for i, v in enumerate(zip(events, event_unix_ts)) if is_event[i]]) if mc_assn is None \
            else zip(*[v for i, v in enumerate(zip(events, event_unix_ts, event_mc_assn)) if is_event[i]])
      

class BeamTrigEventBuilder(RawEventBuilder):
    '''
    A beam trigger based event builder. Events are sliced such that they always follow a beam trigger and end at the following beam trigger (which starts the next event). Events also end after a time window of 182 units or at the end of the datastream.
    '''
    default_window = 1820 // 2
    default_threshold = 0
    default_rollover_ticks = 10000000
    
    def __init__(self, **params):
        super(BeamTrigEventBuilder, self).__init__(**params)
        self.window = params.get('window', self.default_window)
        self.threshold = params.get('threshold_', self.default_threshold)
        self.rollover_ticks = params.get('rollover_ticks', self.default_rollover_ticks)
        self.event_buffer = np.empty((0,))  
        self.event_buffer_unix_ts = np.empty((0,), dtype='u8')
        self.event_buffer_mc_assn = np.empty((0,))
        self.prepend_count = 0  
        self.last_beam_trigger_idx = None

    def get_config(self):
        return dict(
            window=self.window,
            threshold=self.threshold,
            rollover_ticks=self.rollover_ticks,
        )    
    
    def build_events(self, packets, unix_ts, mc_assn=None):
        rollover = np.zeros((len(packets),), dtype='i8')
        for io_group in np.unique(packets['io_group']):
            mask = (packets['io_group'] == io_group) & (packets['packet_type'] == 6) & (packets['trigger_type'] == 83)
            rollover[mask] = self.rollover_ticks
            mask = (packets['io_group'] == io_group)
            rollover[mask] = np.cumsum(rollover[mask]) - rollover[mask]
            if mc_assn is None:
                mask = (packets['io_group'] == io_group) & (packets['packet_type'] == 0) \
                    & (packets['receipt_timestamp'].astype(int) - packets['timestamp'].astype(int) < 0)
                rollover[mask] -= self.rollover_ticks

        ts = packets['timestamp'].astype('i8') + rollover
        #print("HERE IS TS non-sorted", ts)
        sorted_idcs = np.argsort(ts)
        ts = ts[sorted_idcs]
        #print("HERE IS TS sorted", ts)
        for i in ts:
            print(i)
        
        packets = packets[sorted_idcs]
        unix_ts = unix_ts[sorted_idcs]
        #print("HERE IS UNIX_TS", unix_ts)
        if mc_assn is not None:
            mc_assn = mc_assn[sorted_idcs]
        
        beam_trigger_idxs = np.where((packets['io_group'] == 1) & (packets['packet_type'] == 7))[0]
        if len(beam_trigger_idxs) == 0:
            return ([], []) if mc_assn is None else ([], [], [])
        
        events = []
        event_unix_ts = []
        event_mc_assn = [] if mc_assn is not None else None
        time_window = np.inf #1820*3  # Time window to check for event ending
        
        for i, start_idx in enumerate(beam_trigger_idxs):
            if i+1 < len(beam_trigger_idxs): # everything but last event
                next_start_idx = beam_trigger_idxs[i+1]
                start_ts_unix = unix_ts[start_idx]
                start_ts = ts[start_idx]
                print("HEREEEEEE", start_ts, len(beam_trigger_idxs))
                next_start_ts_unix = unix_ts[next_start_idx]
                next_start_ts = ts[next_start_idx]
                
                # if time stamp of next event is less than time window, then end this event at the next beam trigger (which also starts the next event)
                if next_start_ts - start_ts <= time_window:  
                    end_idx = next_start_idx
                # if time stamp of next event is greater than time window then (below)
                else:
                    time_diffs = ts[start_idx:next_start_idx] - start_ts #return all timestamps within range of start_idx to next_start_idx - start_ts
                    outside_window_idx = np.argmax(time_diffs > time_window) + start_idx # returns the index of the first occurrence where the condition time_diffs > time_window is True in the array time_diffs + start_idx
                    #end_idx = outside_window_idx if outside_window_idx > start_idx else next_start_idx # incase timestamps are not strictly increasing
                    end_idx = outside_window_idx
            else: #last event
                start_ts = ts[start_idx]
                if len(ts) - len(ts[start_idx:]) > 0: # are there timestamps, and therefore packet data, left to use to build a final event?
                    time_diffs = ts[start_idx:] - start_ts
                    if any(time_diffs > time_window): #if the time difference is greater than the window, anywhere
                        outside_window_idx = np.argmax(time_diffs > time_window) + start_idx #same as above ; grab the index of the window edge if the time condition is met (above)
                    else: 
                        outside_window_idx = None
                    #end_idx = outside_window_idx if outside_window_idx > start_idx else len(ts) # wrong logic 
                    last_index = len(ts) - 1
                    end_idx = outside_window_idx if outside_window_idx is not None else last_index  # set to window end or end of datastream
                else: #if no timestamps left
                    #end_idx = len(ts) #this would create an empty event 
                    continue # this should skip empty events

            events.append(packets[start_idx:end_idx])
            event_unix_ts.append(unix_ts[start_idx:end_idx])
            if mc_assn is not None:
                event_mc_assn.append(mc_assn[start_idx:end_idx])
                
        events_filtered, event_unix_ts_filtered, event_mc_assn_filtered = [], [], []        
        for i, event in enumerate(events):
            if len(event) > 0:  
                events_filtered.append(event)
                event_unix_ts_filtered.append(event_unix_ts[i])
                if mc_assn is not None:
                    event_mc_assn_filtered.append(event_mc_assn[i])

        return zip(*[v for v in zip(events_filtered, event_unix_ts_filtered)]) if mc_assn is None \
            else zip(*[v for v in zip(events_filtered, event_unix_ts_filtered, event_mc_assn_filtered)])

        #return self.filter_events(events, event_unix_ts, event_mc_assn)
    
'''
    def filter_events(self, events, event_unix_ts, event_mc_assn):
        events_filtered, event_unix_ts_filtered, event_mc_assn_filtered = [], [], []
        for i, event in enumerate(events):
            if len(event) > 0:  # Replace 0 with self.threshold if you use the threshold logic
                events_filtered.append(event)
                event_unix_ts_filtered.append(event_unix_ts[i])
                if mc_assn is not None:
                    event_mc_assn_filtered.append(event_mc_assn[i])

        return zip(*[v for v in zip(events_filtered, event_unix_ts_filtered)]) if mc_assn is None \
            else zip(*[v for v in zip(events_filtered, event_unix_ts_filtered, event_mc_assn_filtered)])
'''        
        

'''
class BeamTrigEventBuilder(RawEventBuilder):
    
'''
    #A beam trigger based event builder. Events are sliced such that they always follow a beam trigger. The end of a data stream marks the end of the last event in the data stream. There is a threshold check copied from SymmetricWindowRawEventBuilder - checks that events have a certain number of hits. It discards them if not. This threshold check may not actually be doing anything.
'''
    # copying from SymmetricWindowRawEventBuilder starts here
    default_window = 1820 // 2
    default_threshold = 0
    default_rollover_ticks = 10000000
    
    def __init__(self, **params):
        super(BeamTrigEventBuilder, self).__init__(**params)
        self.window = params.get('window', self.default_window)
        self.threshold = params.get('threshold_', self.default_threshold)
        self.rollover_ticks = params.get('rollover_ticks', self.default_rollover_ticks)

        self.event_buffer = np.empty((0,))  
        self.event_buffer_unix_ts = np.empty((0,), dtype='u8')
        self.event_buffer_mc_assn = np.empty((0,))
        
        self.prepend_count = 0  
        self.last_beam_trigger_idx = None

    def get_config(self):
        return dict(
            window=self.window,
            threshold=self.threshold,
            rollover_ticks=self.rollover_ticks,
        )    
    
    def build_events(self, packets, unix_ts, mc_assn=None):
        #if len(packets) == 0:
            #return ([], []) if mc_assn is None else ([], [], [])

        # correct for rollovers?
        # sort packets to fix 512 bug, appending incomplete events!?
        #packets = np.append(self.event_buffer, packets) if len(self.event_buffer) else packets  #this shouldnt be needed

        # correct for rollovers (copied from symmetric) 
        rollover = np.zeros((len(packets),), dtype='i8')
        for io_group in np.unique(packets['io_group']):
            # find rollovers
            mask = (packets['io_group'] == io_group) & (packets['packet_type'] == 6) & (packets['trigger_type'] == 83)
            rollover[mask] = self.rollover_ticks
            # calculate sum of rollovers
            mask = (packets['io_group'] == io_group)
            rollover[mask] = np.cumsum(rollover[mask]) - rollover[mask]
            # correct for readout delay (only in real data)
            if mc_assn is None:
                mask = (packets['io_group'] == io_group) & (packets['packet_type'] == 0) \
                    & (packets['receipt_timestamp'].astype(int) - packets['timestamp'].astype(int) < 0)
                rollover[mask] -= self.rollover_ticks

        ts = packets['timestamp'].astype('i8') + rollover

        # sort packets by timestamp
        sorted_idcs = np.argsort(ts)
        packets = packets[sorted_idcs]
        unix_ts = unix_ts[sorted_idcs]
        if mc_assn is not None:
            mc_assn = mc_assn[sorted_idcs]
            
        # copying from SymmetricWindowRawEventBuilder ends here

        # identify beam trigger indices
        beam_trigger_idxs = np.where((packets['io_group'] == 1) & (packets['packet_type'] == 7))[0]

        if len(beam_trigger_idxs) == 0:
            # no beam triggers found, no events are formed
            return ([], []) if mc_assn is None else ([], [], [])

        
        events = []
        event_unix_ts = []
        event_mc_assn = [] if mc_assn is not None else None

        # split events based on beam triggers
        for start, end in zip(beam_trigger_idxs[:-1], beam_trigger_idxs[1:]):
            events.append(packets[start:end])
            event_unix_ts.append(unix_ts[start:end])
            if mc_assn is not None:
                event_mc_assn.append(mc_assn[start:end])

        # handle the last event so it counts as an actual event
        events.append(packets[beam_trigger_idxs[-1]:])
        event_unix_ts.append(unix_ts[beam_trigger_idxs[-1]:])
        if mc_assn is not None:
            event_mc_assn.append(mc_assn[beam_trigger_idxs[-1]:])

        # filter events based on hit counts exceeding the threshold
        events_filtered, event_unix_ts_filtered, event_mc_assn_filtered = [], [], []
        for i, event in enumerate(events):
            #if len(event) > self.threshold:
            if len(event) > 0:
                events_filtered.append(event)
                event_unix_ts_filtered.append(event_unix_ts[i])
                if mc_assn is not None:
                    event_mc_assn_filtered.append(event_mc_assn[i])
                    
        return zip(*[v for v in zip(events_filtered, event_unix_ts_filtered)]) if mc_assn is None \
            else zip(*[v for v in zip(events_filtered, event_unix_ts_filtered, event_mc_assn_filtered)])

        #return (events_filtered, event_unix_ts_filtered) if mc_assn is None \
        #    else (events_filtered, event_unix_ts_filtered, event_mc_assn_filtered)    
    
'''    


    
