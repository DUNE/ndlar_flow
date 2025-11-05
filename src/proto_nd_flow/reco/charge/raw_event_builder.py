from collections import defaultdict
import numpy as np
import logging

from h5flow import H5FLOW_MPI
if H5FLOW_MPI:
    from mpi4py import MPI

from h5flow.core import resources

from proto_nd_flow.util.array import fill_with_last, fill_with_next


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
        return dict(
            rollover_ticks=resources['RunData'].rollover_ticks,
        )

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

    def unroll_timestamps(self, packets: np.ndarray) -> np.ndarray:
        '''
            Calculates "unrolled" timestamps for an array of packets. The
            unrolled timestamps increase monotonically, rather than rolling over
            every ~second. Each SYNC packet introduces an additional cumulative
            offset (of rollover_ticks, e.g. 1E7) that gets added to each
            subsequent raw timestamp, giving the unrolled timestamps. We round
            the LArPix timestamp of the SYNC to the nearest rollover_ticks,
            which takes care of the case when a SYNC is missed by the PACMAN.
            Each IO group is treated independently here.
        '''
        rollover_ticks = resources['RunData'].rollover_ticks
        offsets = np.zeros((len(packets),), dtype='i8')
        for io_group in np.unique(packets['io_group']):
            mask = packets['io_group'] == io_group
            sync_mask = (mask &
                         (packets['packet_type'] == 6) &
                         (packets['trigger_type'] == 83))
            sync_ts = np.zeros_like(offsets)
            # Replace 0 with ~1E7 at each SYNC; ~2E7 if PACMAN missed prev SYNC
            # (assuming rollover_ticks is 1E7)
            sync_ts[sync_mask] = packets[sync_mask]['timestamp']
            # And round to the nearest 1E7 to prevent clock drift
            sync_ts[sync_mask] = (np.round(sync_ts[sync_mask] / rollover_ticks)
                                  * rollover_ticks)
            # Now get the cumulative sum of all _preceding_ increments
            # (subtracting sync_ts[mask] => "preceding")
            offsets[mask] = np.cumsum(sync_ts[mask]) - sync_ts[mask]
            # Finally: If the receipt_timestamp is less than the timestamp, this
            # means that a SYNC arrived while the packet was traveling across
            # the tile. In that case, subtract the timestamp of the preceding SYNC.
            oops_mask = (mask &
                         (packets['packet_type'] == 0) &
                         (packets['receipt_timestamp'] < packets['timestamp']))
            last_sync_ts = fill_with_last(sync_ts)
            offsets[oops_mask] -= last_sync_ts[oops_mask]

        # The offsets are already corrected for the cases when the SYNC was
        # missed by the PACMAN. Now the "% rollover_ticks" takes care of
        # LArPix ASICs (as opposed to PACMEN) that missed one or more SYNCs.
        ts = (packets['timestamp'].astype('i8') % rollover_ticks) + offsets

        # Timestamp packets require special treatment, since their timestamp
        # field is actually a unix timestamp. For these, we just assign the same
        # unrolled timestamp as the one in the next non-timestamp packet
        unix_mask = packets['packet_type'] == 4
        ts[unix_mask] = -1
        ts = fill_with_next(ts, marker=-1)

        return ts


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
        sorted_idcs = np.argsort(packets, order='timestamp', kind='stable')
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

    def __init__(self, **params):
        super(SymmetricWindowRawEventBuilder, self).__init__(**params)
        self.window = params.get('window', self.default_window)
        self.threshold = params.get('threshold', self.default_threshold)

        self.event_buffer = np.empty((0,))  # keep track of partial events from previous calls
        self.event_buffer_unix_ts = np.empty((0,), dtype='u8')
        self.event_buffer_mc_assn = np.empty((0,))

    def get_config(self):
        return dict(
            window=self.window,
            threshold=self.threshold,
            **super().get_config(),
        )

    def build_events(self, packets, unix_ts, mc_assn=None, ts=None, return_ts=False):
        # fetch attribute from appropriate process
        self.cross_rank_get_attrs('event_buffer', 'event_buffer_unix_ts', 'event_buffer_mc_assn')

        if len(packets) == 0:
            self.event_buffer = np.empty((0,), dtype=packets.dtype)
            self.event_buffer_unix_ts = np.empty((0,), dtype=unix_ts.dtype)
            if mc_assn is not None:
                self.event_buffer_mc_assn = np.empty((0,), dtype=mc_assn.dtype)
            self.cross_rank_set_attrs('event_buffer', 'event_buffer_unix_ts', 'event_buffer_mc_assn')
            if return_ts:
                return (([], []), []) if mc_assn is None \
                else (([], [], []), [])
            return ([], []) if mc_assn is None \
                else ([], [], [])

        # sort packets to fix 512 bug
        packets = np.append(self.event_buffer, packets) if len(self.event_buffer) else packets

        if ts is None:
            ts = self.unroll_timestamps(packets)

        sorted_idcs = np.argsort(ts, kind='stable')
        ts = ts[sorted_idcs]
        packets = packets[sorted_idcs]
        unix_ts = np.append(self.event_buffer_unix_ts, unix_ts)[sorted_idcs] if len(self.event_buffer_unix_ts) else unix_ts[sorted_idcs]
        if mc_assn is not None:
            mc_assn = np.append(self.event_buffer_mc_assn, mc_assn)[sorted_idcs] if len(self.event_buffer_mc_assn) else mc_assn[sorted_idcs]

        # calculate time distance between hits
        min_ts, max_ts = np.min(ts), np.max(ts)
        bin_edges = np.linspace(min_ts - 1, max_ts + 1, int((max_ts - min_ts + 2) // self.window))
        ts_data = ts[packets['packet_type'] == 0]
        hist, bin_edges = np.histogram(ts_data, bins=bin_edges)

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
            
            if return_ts: 
                return (([], []), []) if mc_assn is None \
                else (([], [], []), [])
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
            if return_ts: 
                return (([], []), []) if mc_assn is None \
                else (([], [], []), [])

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
        event_ts = np.split(ts, event_idcs)
        event_ts = list( [ np.min(times) for i, times in enumerate(event_ts) if is_event[i] ]  )
        event_unix_ts = np.split(unix_ts, event_idcs)
        if mc_assn is not None:
            event_mc_assn = np.split(mc_assn, event_idcs)
        
        # only return packets from events
        if return_ts:
            return (zip(*[v for i, v in enumerate(zip(events, event_unix_ts)) if is_event[i]]), event_ts) if mc_assn is None \
                else (zip(*[v for i, v in enumerate(zip(events, event_unix_ts, event_mc_assn)) if is_event[i]]), event_ts)

        return zip(*[v for i, v in enumerate(zip(events, event_unix_ts)) if is_event[i]]) if mc_assn is None \
            else zip(*[v for i, v in enumerate(zip(events, event_unix_ts, event_mc_assn)) if is_event[i]])
      

class ExtTrigRawEventBuilder(RawEventBuilder):
    '''
    An external trigger based event builder. Events are sliced such that they always follow an external trigger and the readout window is configurable. The default is set to 182 x 1.1 units (10% grace period). Note the event builder may contain more than one trigger if they are within a readout window time.
    '''
    default_window = 1820 * 1.1
    default_shifted_event_dt = 0 # This is to account for any offset between timing of trigger marker and corresponding event
    default_trig_io_grp = 1     # -1 -> all io groups
    default_extendable = False
    
    default_build_off_beam_events=False
    default_off_beam_window = 1820 // 2
    default_off_beam_threshold = 10

    def __init__(self, **params):
        super(ExtTrigRawEventBuilder, self).__init__(**params)
        self.window = params.get('window', self.default_window)
        self.trig_io_grp = params.get('trig_io_grp', self.default_trig_io_grp)
        self.extendable = params.get('extendable', self.default_extendable)
        self.build_off_beam_events = params.get('build_off_beam_events', self.default_build_off_beam_events)
        self.off_beam_window = params.get('off_beam_window', self.default_off_beam_window)
        self.off_beam_threshold = params.get('off_beam_threshold', self.default_off_beam_threshold)
        self.shifted_event_dt = params.get('shifted_event_dt', self.default_shifted_event_dt)

        self.event_buffer = np.empty((0,))  
        self.event_buffer_unix_ts = np.empty((0,), dtype='u8')
        self.event_buffer_mc_assn = np.empty((0,))
        self.prepend_count = 0  
        self.last_beam_trigger_idx = None

        if not isinstance(self.trig_io_grp, list):
            self.trig_io_grp = [self.trig_io_grp]
        self.window = self.to_iog_dict(self.window)
        self.shifted_event_dt = self.to_iog_dict(self.shifted_event_dt)
        self.extendable = self.to_iog_dict(self.extendable)

    def to_iog_dict(self, var):
        '''
            Convert a scalar or list VAR into a dict, keyed by io_group.
            If a scalar, use a defaultdict that always yields VAR.
            If a list, assume the i'th element corresponds to the i'th io_group
            in self.trig_io_grp.
        '''
        if isinstance(var, list):
            assert len(var) == len(self.trig_io_grp)
            return {self.trig_io_grp[i]: v for i, v in enumerate(var)}
        assert len(self.trig_io_grp) == 1
        if self.trig_io_grp == [-1]:
            return defaultdict(lambda: var)
        return {self.trig_io_grp[0]: var}

    def get_config(self):
        return dict(
            trig_io_grp=self.trig_io_grp,
            window=list(self.window.items()),
            shifted_event_dt=list(self.shifted_event_dt.items()),
            extendable=list(self.extendable.items()),
            **super().get_config(),
        )

    def build_events(self, packets, unix_ts, mc_assn=None):
        if len(packets) == 0:
            return ([], []) if mc_assn is None else ([], [], [])

        ts = self.unroll_timestamps(packets)

        trig_mask = packets['packet_type'] == 7
        if self.trig_io_grp != [-1]:
            iog_masks = [packets['io_group'] == iog for iog in self.trig_io_grp]
            trig_mask &= np.logical_or.reduce(iog_masks)
        
        trigger_idcs = np.where(trig_mask)[0]
            
        events = []
        event_unix_ts = []
        event_mc_assn = [] if mc_assn is not None else None
       
        start_times = []

        # Mask to keep track of packets associated to triggers
        used_mask = np.zeros( len(unix_ts) ) < -1
        used_trig_idcs = set()
        for i, start_idx in enumerate(trigger_idcs):
            if start_idx in used_trig_idcs:
                continue
            used_trig_idcs.add(start_idx)

            this_io_group = packets[start_idx]['io_group']
            this_trig_time = ts[start_idx] + self.shifted_event_dt[this_io_group]
            last_io_group, last_trig_time = this_io_group, this_trig_time
            start_times.append(this_trig_time)
            # FIXME & (ts % 1E7 != 0) is a hot fix for PPS signal
            hotfix_mask = (ts % 1E7 != 0) | ((ts % 1E7 == 0) & trig_mask)

            if self.extendable[this_io_group]:
                while True:
                    # Scan for further triggers in the window
                    pileup_trig_mask = ((ts - last_trig_time) > 0) \
                        & ((ts - last_trig_time) <= self.window[last_io_group]) \
                        & hotfix_mask \
                        & trig_mask
                    if not pileup_trig_mask.any():
                        break
                    for pileup_trig_idx in np.where(pileup_trig_mask)[0]:
                        iog = packets[pileup_trig_idx]['io_group']
                        if (self.trig_io_grp != -1) and (iog not in self.trig_io_grp):
                            continue
                        used_trig_idcs.add(pileup_trig_idx)
                        last_io_group = iog
                        last_trig_time = ts[pileup_trig_idx]
                        # If we find a non-extendable ("beam") trigger then we
                        # stop at the end of that trigger's window
                        if not self.extendable[last_io_group]:
                            break
                    else: # no break
                        continue # Scan over new window starting from last trig
                    break # Or, if we broke out of "for", break out of "while"

            mask = ((ts - this_trig_time) >= 0) \
                & ((ts - last_trig_time) <= self.window[last_io_group]) \
                & ~used_mask \
                & hotfix_mask

            events.append(packets[mask])
            event_unix_ts.append(unix_ts[mask])
            if mc_assn is not None:
                event_mc_assn.append(mc_assn[mask])

            used_mask = np.logical_or( used_mask, mask )

        
        if not self.build_off_beam_events:
            return zip(*[v for v in zip(events, event_unix_ts)]) if mc_assn is None \
                else zip(*[v for v in zip(events, event_unix_ts, event_mc_assn)])

        # build off beam events using SymmetricRawEventBuilder
        off_beam_config = {'window' : self.off_beam_window,
                           'threshold' : self.off_beam_threshold
                          }
        off_beam_builder = SymmetricWindowRawEventBuilder( **off_beam_config )
        
        off_beam_events, off_beam_event_unix_ts, off_beam_event_mc_assn = [], [], []
        this_mc_assn = mc_assn[~used_mask] if (mc_assn is not None) else None
        (off_beam_events_list, off_beam_ts) = off_beam_builder.build_events(packets[~used_mask], unix_ts[~used_mask], this_mc_assn, ts[~used_mask], return_ts=True)
        off_beam_events_list=list(off_beam_events_list)
        if off_beam_events_list:
            off_beam_events = list(off_beam_events_list[0])
            off_beam_event_unix_ts = list(off_beam_events_list[1])
            if mc_assn is not None:
                off_beam_event_mc_assn = list(off_beam_events_list[2])

        full_events = events + off_beam_events

        full_event_unix_ts = event_unix_ts + off_beam_event_unix_ts
        if not mc_assn is None: full_event_mc_assn = event_mc_assn + off_beam_event_mc_assn

        sorted_event_indices = np.argsort( start_times + off_beam_ts   )
        full_events = [full_events[index] for index in sorted_event_indices]
        full_event_unix_ts = [full_event_unix_ts[index] for index in sorted_event_indices]

        if not mc_assn is None: full_event_mc_assn = [full_event_mc_assn[index] for index in sorted_event_indices]

        return zip(*[v for v in zip(full_events, full_event_unix_ts)]) if mc_assn is None \
                else zip(*[v for v in zip(full_events, full_event_unix_ts, full_event_mc_assn)])
