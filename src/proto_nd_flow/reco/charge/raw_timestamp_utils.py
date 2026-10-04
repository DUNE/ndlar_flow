#!/usr/bin/env python3

import itertools
from typing import Optional

import numpy as np
import numpy.typing as npt

from h5flow.core import resources
from proto_nd_flow.util.array import fill_with_last, fill_with_next


def _prev_tagged(B: np.ndarray, N: int):
    assert B[0] == 0
    return B[np.searchsorted(B, np.arange(N), side='right') - 1]


def _get_delay(pps_delays: npt.NDArray[np.void], iog: int) -> float:
    sel = pps_delays['io_group'] == iog
    return np.median(pps_delays[sel]['delay_ticks'])


def get_unix_ts_usec(packets: npt.NDArray[np.void],
                     pps_delays: Optional[npt.NDArray[np.void]]) \
        -> npt.NDArray[np.float64]:
    all_unix_ts_usec = np.zeros(packets.shape, dtype=np.float64)
    iogs = sorted(np.unique(packets['io_group']))

    delay = 0.
    if pps_delays is not None:
        delay = np.median([_get_delay(pps_delays, iog)
                           for iog in iogs])
    for iog in iogs:
        sel = packets['io_group'] == iog
        map2all = np.where(sel)[0]
        p = packets[sel]

        unix_ts_usec = ((p['timestamp'] + delay) % 1E7 / 10)
        all_unix_ts_usec[map2all] = unix_ts_usec

    return all_unix_ts_usec


def unroll_timestamps(packets: np.ndarray, last_offsets) \
        -> tuple[np.ndarray, dict]:
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
    new_last_offsets = {}

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

        if io_group == 7:
            unix_ts = get_unix_timestamps(packets)
            sel = sync_mask & (unix_ts == 1720502219)
            if np.any(sel):
                ixs = np.where(sel)[0]
                assert len(ixs) == 1
                sync_ts[ixs[0]] += rollover_ticks

        # Now get the cumulative sum of all _preceding_ increments
        # (subtracting sync_ts[mask] => "preceding")
        offsets[mask] = np.cumsum(sync_ts[mask]) # - sync_ts[mask]

        new_last_offsets[io_group] = last_offsets[io_group] + offsets[mask][-1]
        offsets[mask] += last_offsets[io_group]

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

    return ts, new_last_offsets


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


def get_event_unix_ts(packets, packet_unix_ts, packet_unix_ts_usec, event_masks,
                      abs_ticks):
    event_unix_ts = np.zeros(len(event_masks), dtype=np.uint64)
    event_unix_ts_usec = np.zeros(len(event_masks), dtype=np.float64)
    dpkt_type = resources['RunData'].data_packet_type
    for i, mask in enumerate(event_masks):
        p = packets[mask]
        submask = p['packet_type'] == 7
        is_trig = True
        if not np.any(submask):
            is_trig = False
            submask = p['packet_type'] == dpkt_type
            if not np.any(submask):
                submask = p['packet_type'] != 4
                assert np.any(submask)
        event_unix_ts[i] = np.min(packet_unix_ts[mask][submask])
        event_unix_ts_usec[i] = packet_unix_ts_usec[mask][submask][0]
        ## the "168" file is with the following uncommented:
        # if not is_trig:
        #     ts2 = abs_ticks[mask][submask]
        #     p2 = p[submask]
        #     event_larpix_ts = np.floor(np.min(p2['timestamp']) / 100) * 100
        #     # imin = np.argmin(p2['timestamp'])
        #     imin = np.argmin(ts2)
        #     delta1 = p2['timestamp'][imin] - event_larpix_ts
        #     delta2 = ts2[0] - ts2[imin]
        #     delta_usec = (delta1 + delta2) / 10
        #     event_unix_ts_usec[i] -= delta_usec

    return event_unix_ts, event_unix_ts_usec


def get_anchor_unix_ts(packets, packet_msg_unix_ts, pps_delays, threshold=30):
    result = {}
    iogs = np.unique(packets['io_group'])

    for iog in iogs:
        sel = packets['io_group'] == iog
        p = packets[sel]
        u = packet_msg_unix_ts[sel]
        anchor0 = u[0]
        for anchor in itertools.count(anchor0):
            anchor_pkts = p[u == anchor]
            anchor_unix_pkts = anchor_pkts[anchor_pkts['packet_type'] == 4]
            if len(anchor_unix_pkts) > threshold:
                break
        delay = np.median(pps_delays['delay_ticks'][pps_delays['io_group'] == iog])
        first_receipt_ts = p[p['packet_type'] != 4]['receipt_timestamp'][0]
        if first_receipt_ts + delay > 1E7: 
            result[iog] = anchor + 1
        else:
            result[iog] = anchor

    assert all(result[iog] == result[iogs[0]] for iog in iogs)
    return result[iogs[0]]


def maybe_insert_unix_ts(packets):
    if packets[0]['packet_type'] == 4:
        return packets
    iog = packets[0]['io_group']
    for p in packets:
        if p['io_group'] == iog and p['packet_type'] == 4:
            return np.insert(packets, [0], p)
    raise RuntimeError(f'Could not find timestamp packet for io_group {iog}')


def clear_timestamp_high_bit(packets):
    ts_mask = packets['packet_type'] == 4
    packets[~ts_mask]['timestamp'] = \
        packets[~ts_mask]['timestamp'].astype(int) % (2**31)
