#!/usr/bin/env python3

# initialize unix_ts and unix_ts_usec to zero; len = len(packets)
# for each iog:
#   set unix_ts to timestamp of preceding timestamp packet (for iog)
#   find all unix_ts jumps
#   find all pps rollovers
#   define next_pps_jump_idx, next_unix_jump_idx (handle edge case)
#   define on_right accordingly
#   decrement unix_ts for on_right hits* w/ ts + delay < 1e7        *non-timestamp
#   decrement unix_ts for hits* w/ timestamp > receipt_teimstamp
#   set* unix_ts_usec to (timestamp + delay) % 1e7

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from h5flow.core import resources
from proto_nd_flow.util.array import fill_with_last, fill_with_next


@dataclass
class Timestamps:
    unix_ts: npt.NDArray[np.uint64]
    unix_ts_usec: npt.NDArray[np.float64]


def _next_tagged(B: np.ndarray, N: int, fill=None):
    if fill is None:
        fill = N
    Bs = np.append(B, fill)                                 # sentinel
    return Bs[np.searchsorted(B, np.arange(N), side='right')]


def _prev_tagged(B: np.ndarray, N: int):
    assert B[0] == 0
    return B[np.searchsorted(B, np.arange(N), side='right') - 1]


def _find_unix_jumps(unix_ts: npt.NDArray[np.uint64]) \
        -> npt.NDArray[np.uint64]:
    sel = unix_ts[1:] - unix_ts[:-1] == 1
    return 1 + np.where(sel)[0]


def _find_pps_jumps(pps_ts: npt.NDArray[np.uint64]) \
        -> npt.NDArray[np.uint64]:
    pps_ts = fill_with_next(pps_ts)
    sel = pps_ts[:-1] > pps_ts[1:]
    sel &= pps_ts[:-1] - pps_ts[1:] > 9E6
    return 1 + np.where(sel)[0]


def _get_delay(pps_delays: npt.NDArray[np.void], iog: int) -> float:
    sel = pps_delays['io_group'] == iog
    return np.median(pps_delays[sel]['delay_ticks'])


def get_true_timestamps(packets: npt.NDArray[np.void],
                        pps_delays: Optional[npt.NDArray[np.void]]) -> Timestamps:
    all_unix_ts = np.zeros(packets.shape, dtype=np.uint64)
    all_unix_ts_usec = np.zeros(packets.shape, dtype=np.float64)

    for iog in sorted(np.unique(packets['io_group'])):
        delay = _get_delay(pps_delays, iog) if (pps_delays is not None) else 0
        sel = packets['io_group'] == iog
        map2all = np.where(sel)[0]
        p = packets[sel]

        unix_ts = np.zeros(p.shape, dtype=np.uint64)

        is_unix = p['packet_type'] == 4
        unix_ts[is_unix] = p[is_unix]['timestamp']
        unix_ts = fill_with_last(unix_ts)

        unix_jumps = _find_unix_jumps(unix_ts)
        pps_jumps = _find_pps_jumps(p['receipt_timestamp'])

        next_unix_jumps = _next_tagged(unix_jumps, p.shape[0])
        next_pps_jumps = _next_tagged(pps_jumps, p.shape[0])
        on_right = next_pps_jumps < next_unix_jumps # right of unix jump

        to_corr1 = on_right & (p['timestamp'] + delay < 1E7) & ~is_unix
        unix_ts[to_corr1] -= 1
        is_sync = p['packet_type'] == 6
        to_corr2 = (p['timestamp'] > p['receipt_timestamp']) & ~is_unix & ~is_sync
        unix_ts[to_corr2] -= 1
        # decrement twice when both conditions apply

        unix_ts_usec = ((p['timestamp'] + delay) % 1E7 / 10)
        all_unix_ts[map2all] = unix_ts
        all_unix_ts_usec[map2all] = unix_ts_usec

    return Timestamps(all_unix_ts, all_unix_ts_usec)


def unroll_timestamps(packets: np.ndarray) -> np.ndarray:
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
    data_packet_type = resources['RunData'].data_packet_type
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
        offsets[mask] = np.cumsum(sync_ts[mask]) # - sync_ts[mask]

        # Apply correction for clogged UARTs
        clog_mask = (mask &
                     (packets['packet_type'] != 7) &
                     (packets['packet_type'] != 6) &
                     (packets['timestamp'].astype(np.int32)
                      - packets['receipt_timestamp'].astype(np.int32) > 1E6))
        offsets[clog_mask] -= rollover_ticks

        # Finally: If the receipt_timestamp is a bit less than the timestamp, this
        # means that a SYNC arrived while the packet was traveling across
        # the tile. In that case, subtract the timestamp of the preceding SYNC.
        oops_mask = (mask &
                     (packets['packet_type'] == data_packet_type) &
                     (packets['receipt_timestamp'] < packets['timestamp']) &
                     (packets['timestamp'].astype(np.int32)
                      - packets['receipt_timestamp'].astype(np.int32) <= 1E6))
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


def add_timestamp_packets(event_masks: list[npt.NDArray[np.bool]],
                          packets: npt.NDArray[np.void]):
    is_unix = packets['packet_type'] == 4
    unix_idcs = np.where(is_unix)[0]
    prev_unix_idcs = _prev_tagged(unix_idcs, packets.shape[0])
    for mask in event_masks:
        our_unix_idcs = prev_unix_idcs[mask]
        mask[our_unix_idcs] = True


def get_unix_timestamps(packets: npt.NDArray[np.void]):
    is_unix = packets['packet_type'] == 4
    unix_idcs = np.where(is_unix)[0]
    prev_unix_idcs = _prev_tagged(unix_idcs, packets.shape[0])
    return packets['timestamp'][prev_unix_idcs]


def get_event_unix_ts(packets, packet_unix_ts_usec, event_masks):
    event_unix_ts = np.zeros(len(event_masks), dtype=np.uint64)
    event_unix_ts_usec = np.zeros(len(event_masks), dtype=np.float64)
    dpkt_type = resources['RunData'].data_packet_type
    for i, mask in enumerate(event_masks):
        p = packets[mask]
        data_mask = p['packet_type'] == dpkt_type
        assert np.any(data_mask)
        unix_ts = get_unix_timestamps(p)
        rcpt_ts, ts = \
            p['receipt_timestamp'].astype(np.int32), p['timestamp']
        clean_mask = data_mask & (rcpt_ts - ts > 0) & (rcpt_ts - ts < 1E5)
        if not np.any(clean_mask):
            clean_mask = data_mask
        event_unix_ts[i] = np.min(unix_ts[clean_mask])
        event_unix_ts_usec[i] = packet_unix_ts_usec[mask][clean_mask][0]
    return event_unix_ts, event_unix_ts_usec
