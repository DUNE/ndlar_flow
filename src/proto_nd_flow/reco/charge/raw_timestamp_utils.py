#!/usr/bin/env python3

import itertools
import warnings
from typing import Optional

import numpy as np
import numpy.typing as npt

from h5flow.core import resources
from proto_nd_flow.util.array import fill_with_last, fill_with_next

from .pps_delay_extractor import PPSDelayExtractor

def _prev_tagged(B: np.ndarray, N: int):
    assert B[0] == 0
    return B[np.searchsorted(B, np.arange(N), side='right') - 1]


def get_unix_ts_usec(packets: npt.NDArray[np.void],
                     pps_delays: Optional[npt.NDArray[np.void]]) \
        -> npt.NDArray[np.float64]:
    all_unix_ts_usec = np.zeros(packets.shape, dtype=np.float64)
    iogs = sorted(np.unique(packets['io_group']))

    delay = 0.
    if pps_delays is not None:
        # delay = np.median([_get_delay(pps_delays, iog)
        #                    for iog in iogs])
        delay = np.median(pps_delays['delay_ticks'])
    for iog in iogs:
        sel = packets['io_group'] == iog
        map2all = np.where(sel)[0]
        p = packets[sel]

        unix_ts_usec = ((p['timestamp'] + delay) % 1E7 / 10)
        all_unix_ts_usec[map2all] = unix_ts_usec

    return all_unix_ts_usec


def unroll_timestamps(packets: np.ndarray, last_offsets, sync_sums) \
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
        ``sync_sums`` (io_group => float) carries each io_group's running total
        of SYNC timestamps from one buffer to the next.
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
        # Round the running total, not each SYNC: a burst of short SYNCs
        # (e.g. during configuration) that add up to one second must still
        # count as one second. Rounding the total also prevents clock drift.
        if sync_mask.any():
            total = sync_sums[io_group] + np.cumsum(
                packets[sync_mask]['timestamp'].astype('f8'))
            before = np.r_[sync_sums[io_group], total[:-1]]
            sync_ts[sync_mask] = ((np.round(total / rollover_ticks)
                                   - np.round(before / rollover_ticks))
                                  * rollover_ticks).astype('i8')
            sync_sums[io_group] = float(total[-1])

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


def add_timestamp_packets(event_idcs: list[npt.NDArray[np.int64]],
                          packets: npt.NDArray[np.void]):
    is_unix = packets['packet_type'] == 4
    unix_idcs = np.where(is_unix)[0]
    prev_unix_idcs = _prev_tagged(unix_idcs, packets.shape[0])
    for i, idcs in enumerate(event_idcs):
        mask = np.zeros_like(packets, dtype=bool)
        mask[idcs] = True
        our_unix_idcs = prev_unix_idcs[idcs]
        mask[our_unix_idcs] = True
        event_idcs[i] = np.where(mask)[0]


def get_unix_timestamps(packets: npt.NDArray[np.void]):
    is_unix = packets['packet_type'] == 4
    unix_idcs = np.where(is_unix)[0]
    prev_unix_idcs = _prev_tagged(unix_idcs, packets.shape[0])
    return packets['timestamp'][prev_unix_idcs]


def get_event_unix_ts(packets, packet_unix_ts, packet_unix_ts_usec, event_idcs,
                      abs_ticks):
    event_unix_ts = np.zeros(len(event_idcs), dtype=np.uint64)
    event_unix_ts_usec = np.zeros(len(event_idcs), dtype=np.float64)
    dpkt_type = resources['RunData'].data_packet_type
    for i, idcs in enumerate(event_idcs):
        p = packets[idcs]
        submask = p['packet_type'] == 7
        is_trig = True
        if not np.any(submask):
            is_trig = False
            submask = p['packet_type'] == dpkt_type
            if not np.any(submask):
                submask = p['packet_type'] != 4
                assert np.any(submask)
        event_unix_ts[i] = np.min(packet_unix_ts[idcs][submask])
        event_unix_ts_usec[i] = packet_unix_ts_usec[idcs][submask][0]

    return event_unix_ts, event_unix_ts_usec


def get_anchor_unix_ts(packets,
                       packet_msg_unix_ts,
                       delay_extractor: PPSDelayExtractor | None,
                       threshold=30):
    result = {}
    iogs = np.unique(packets['io_group'])
    dpkt_type = resources['RunData'].data_packet_type

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
        delay = 0.
        if delay_extractor is not None:
            pps_delays = delay_extractor.data
            delay = np.median(pps_delays['delay_ticks'][pps_delays['io_group'] == iog])
        # 0 is the data packet. important to run-3 as $74 comes first
        first_receipt_ts = p[p['packet_type'] == dpkt_type]['receipt_timestamp'][0]
        if first_receipt_ts + delay > 1E7: 
            result[iog] = anchor - 1
        else:
            result[iog] = anchor

    # Failsafe for run-3:
    # The unix-second labels come per io_group from the software timestamp packets, so near
    # the second flip an io_group can land 1 s off (or more if it is nearly silent).
    # All io_groups share one PPS, so take the value most io_groups agree on.
    values, counts = np.unique(list(result.values()), return_counts=True)
    anchor = values[np.argmax(counts)]
    if len(values) > 1:
        warnings.warn(f'get_anchor_unix_ts: io_groups disagree {result}; using the majority {anchor}')
    return anchor


def get_start_unix_ts(packets, abs_ticks, pps_delay):
    '''
        Unix second of the file's start (the second where the SYNC count N is 0).
        Use all packets of a iogroup to vote on the first second to avoid using the unreliable packets at the start
        Then take majority vote across all iogroups 
    '''
    rollover = resources['RunData'].rollover_ticks
    is_ts = packets['packet_type'] == 4
    result = {}
    for iog in np.unique(packets['io_group']):
        idx = np.where(packets['io_group'] == iog)[0]
        ts = is_ts[idx]
        # position of the last timestamp packet up to each packet of this io_group
        pos = np.maximum.accumulate(np.where(ts, np.arange(len(idx)), -1))
        use = (pos >= 0) & ~ts
        if not use.any():
            continue
        label = packets['timestamp'][idx][pos[use]].astype('i8')
        sec = np.floor((abs_ticks[idx][use] + pps_delay) / rollover).astype('i8')
        vals, counts = np.unique(label - sec, return_counts=True)
        result[int(iog)] = int(vals[np.argmax(counts)])

    values, counts = np.unique(list(result.values()), return_counts=True)
    start = int(values[np.argmax(counts)])
    if len(values) > 1:
        warnings.warn(f'get_start_unix_ts: io_groups disagree {result}; using the majority {start}')
    return start


def get_lineup_shifts(packets, abs_ticks):
    '''
        Ticks to add to each io_group so that the same pulse gets the same SYNC count on every
        io_group. Each io_group's count is compared with a ruler io_group
        (the one with most SYNCs) at the nearest SYNC in the buffer; the shift is
        relative to the most common difference over all io_groups, so the majority decides.
    '''
    rollover = resources['RunData'].rollover_ticks
    is_sync = (packets['packet_type'] == 6) & (packets['trigger_type'] == 83)

    # Step 1: each io_group's SYNC count at each of its SYNCs.
    # io_group => (positions of its SYNCs in the buffer, SYNC count at each)
    syncs = {}
    for iog in np.unique(packets['io_group']):
        idx = np.where(packets['io_group'] == iog)[0]  # positions of this io_group's packets
        sel = idx[is_sync[idx]]                        # positions of its SYNCs
        if len(sel):
            # For a SYNC, abs_ticks = (raw timestamp % rollover) + offset. Take away the
            # raw timestamp and divide by the rollover: the number of seconds counted so far
            count = (abs_ticks[sel] - packets['timestamp'][sel].astype('i8') % rollover) // rollover
            syncs[int(iog)] = (sel, count)
    # An io_group without a SYNC in this buffer is not in syncs, so it is never shifted
    if not syncs:
        return {}

    def mode(a):
        # Most common value (on a tie, the lowest)
        v, c = np.unique(np.asarray(a), return_counts=True)
        return int(v[np.argmax(c)])

    # Step 2: choose a reference io_group to compare with: the one with most SYNCs (on a tie,
    # the lowest io_group). It is only a ruler, not assumed to be right: step 4 decides that
    ref = max(syncs, key=lambda g: len(syncs[g][0]))
    r_where, r_count = syncs[ref]

    # Step 3: how many more SYNCs each io_group has counted than the reference
    extra = {}
    for g, (where, count) in syncs.items():
        # For each SYNC of g, find the reference SYNC nearest in position in the buffer
        # (position, not time: assumes SYNCs of the same pulse are written close together)
        if len(r_where) > 1:
            # searchsorted: first reference SYNC at or after it; then take the one before
            # instead if that is closer
            j = np.clip(np.searchsorted(r_where, where), 1, len(r_where) - 1)
            j = j - (np.abs(r_where[j - 1] - where) < np.abs(r_where[j] - where))
        else:
            j = np.zeros(len(where), int)
        # Most common difference, so that one bad match does not decide (0 for the reference)
        extra[g] = mode(count - r_count[j])

    # Step 4: the difference most io_groups have is taken as right, even if the reference is
    # the odd one out. v - base = +1: g counted one SYNC too many, its hits are 1 s late, so
    # shift it by -1 s. io_groups that agree with the majority are not returned (no shift)
    base = mode(list(extra.values()))
    return {g: -(v - base) * rollover for g, v in extra.items() if v != base}


def maybe_insert_unix_ts(packets):
    if packets[0]['packet_type'] == 4:
        return packets
    iog = packets[0]['io_group']
    for p in packets:
        if p['io_group'] == iog and p['packet_type'] == 4:
            return np.insert(packets, [0], p)
    # Use another io_group if this iogroup has no timestamp packet
    other = np.where(packets['packet_type'] == 4)[0]
    if len(other):
        warnings.warn(f'No timestamp packet for io_group {iog} in this buffer; '
                      f'using one from io_group {packets[other[0]]["io_group"]}')
        return np.insert(packets, [0], packets[other[0]])
    raise RuntimeError(f'Could not find timestamp packet for io_group {iog}')


def clear_timestamp_high_bit(packets):
    ts_mask = packets['packet_type'] == 4
    packets[~ts_mask]['timestamp'] = \
        packets[~ts_mask]['timestamp'].astype(int) % (2**31)
