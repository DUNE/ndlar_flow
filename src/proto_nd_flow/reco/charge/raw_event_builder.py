from collections import defaultdict
import numpy as np
import numpy.typing as npt
import logging
from typing import Optional

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

    def do_build_events(self, packets: npt.NDArray[np.void], ts: npt.NDArray[np.int64],
                        used_mask: npt.NDArray[np.bool]) \
            -> list[npt.NDArray[np.bool]]:
        '''
            This function does the actual event building and must be implemented
            by a subclass.

            :param packets: packet-formatted array (shape: ``(N,)``)

            :param ts: Absolute timestamp of each packet in ``packets`` (shape: ``(N,)``);
                units are ticks-since-epoch (i.e. 1E7*unix_time + larpix_timestamp)

            :param used_mask: Specifies packets that have already been used
                by a prior event builder

            :returns: a list of masks, specifying the set of packets for each event.
                (It is not necessary to include timestamp packets; they will be added
                automatically if missing.)
        '''
        packets, ts, used_mask             # suppress "unused variable" warning
        raise NotImplementedError('Event building for this class has not been implemented!')

    def build_events(self, packets: npt.NDArray[np.void], ts: npt.NDArray[np.int64],
                     used_mask: Optional[npt.NDArray[np.bool]]) \
            -> list[npt.NDArray[np.bool]]:
        '''
            Run the event builder on a sub-set of packet-formatted array data

            :param packets: packet-formatted array (shape: ``(N,)``)

            :param ts: Absolute timestamp of each packet in ``packets`` (shape: ``(N,)``);
                units are ticks-since-epoch (i.e. 1E7*unix_time + larpix_timestamp)

            :param used_mask: Optional; specifies packets that have already been used
                by a prior event builder

            :returns: a list of masks, specifying the set of packets for each event.
                (It is not necessary to include timestamp packets; they will be added
                automatically if missing.)
        '''
        if len(packets) == 0:
            return []

        if used_mask is None:
            used_mask = np.zeros(len(packets), dtype=np.bool)

        return self.do_build_events(packets, ts, used_mask)


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

    def get_config(self):
        return dict(
            window=self.window,
            threshold=self.threshold,
            **super().get_config(),
        )

    def do_build_events(self, packets, ts, used_mask):
        sorted_idcs = np.argsort(ts, kind='stable')
        ts_orig = ts
        ts = ts[sorted_idcs]
        packets = packets[sorted_idcs]

        # calculate time distance between hits
        min_ts, max_ts = np.min(ts), np.max(ts)
        bin_edges = np.linspace(min_ts - 1, max_ts + 1, int((max_ts - min_ts + 2) // self.window))
        ts_data = ts[packets['packet_type'] == resources['RunData'].data_packet_type
                     & ~used_mask]
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
            return []
        if not len(event_start_timestamp):
            # first packet starts event
            event_start_timestamp = np.r_[min_ts, event_start_timestamp]
        if not len(event_end_timestamp):
            # last packet ends event
            event_end_timestamp = np.r_[max_ts, event_end_timestamp]

        event_masks = [(ts_orig >= ts_start) & (ts_orig <= ts_end) & ~used_mask
                       for ts_start, ts_end
                       in zip(event_start_timestamp, event_end_timestamp)]
        
        return event_masks
      

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

    def do_build_events(self, packets, ts, used_mask):
        trig_mask = packets['packet_type'] == 7
        if self.trig_io_grp != [-1]:
            iog_masks = [packets['io_group'] == iog for iog in self.trig_io_grp]
            trig_mask &= np.logical_or.reduce(iog_masks) # ty: ignore
        trigger_idcs = np.where(trig_mask)[0]

        event_masks = []
        start_times = []
        used_trig_idcs = set()

        for start_idx in trigger_idcs:
            if start_idx in used_trig_idcs:
                continue
            used_trig_idcs.add(start_idx)

            this_io_group = packets[start_idx]['io_group']
            this_trig_time = ts[start_idx] + self.shifted_event_dt[this_io_group]
            last_io_group, last_trig_time = this_io_group, this_trig_time
            start_times.append(this_trig_time)

            if self.extendable[this_io_group]:
                last_io_group, last_trig_time = self.extend_window(
                    packets, ts, trig_mask,
                    used_trig_idcs, last_io_group, last_trig_time)

            mask = ((ts - this_trig_time) >= 0) \
                & ((ts - last_trig_time) <= self.window[last_io_group]) \
                & ~used_mask

            event_masks.append(mask)

            used_mask = np.logical_or( used_mask, mask )

        
        if not self.build_off_beam_events:
            return event_masks

        # build off beam events using SymmetricRawEventBuilder
        off_beam_config = {'window' : self.off_beam_window,
                           'threshold' : self.off_beam_threshold}
        off_beam_builder = SymmetricWindowRawEventBuilder( **off_beam_config )
        off_beam_event_masks = off_beam_builder.build_events(packets, ts, used_mask)

        return [*event_masks, *off_beam_event_masks]

    def extend_window(self, packets: np.ndarray, ts: npt.NDArray[np.int64],
                      trig_mask: npt.NDArray[np.bool],
                      used_trig_idcs: set[int],
                      last_io_group: int, last_trig_time: int) -> tuple[int, int]:
        hotfix_mask = (ts % 1E7 != 0) | ((ts % 1E7 == 0) & trig_mask)

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

        return last_io_group, last_trig_time

