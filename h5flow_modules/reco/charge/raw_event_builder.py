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

        # break up by event
        event_mask = (ts.reshape(1, -1) > event_start_timestamp.reshape(-1, 1)) \
            & (ts.reshape(1, -1) < event_end_timestamp.reshape(-1, 1))
        event_idx = np.argmax(event_mask, axis=0)
        event_mask = np.any(event_mask, axis=0)
        event_diff = np.diff(event_idx, axis=-1)
        event_idcs = np.argwhere(event_diff | np.diff(event_mask, axis=-1)).ravel() + 1

        events = np.split(packets, event_idcs)
        event_unix_ts = np.split(unix_ts, event_idcs)
        if mc_assn is not None:
            event_mc_assn = np.split(mc_assn, event_idcs)
        is_event = np.r_[False, event_mask[event_idcs]]

        # only return packets from events
        return zip(*[v for i, v in enumerate(zip(events, event_unix_ts)) if is_event[i]]) if mc_assn is None \
            else zip(*[v for i, v in enumerate(zip(events, event_unix_ts, event_mc_assn)) if is_event[i]])
