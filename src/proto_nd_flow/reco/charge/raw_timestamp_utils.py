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


def _prev_tagged(B: np.ndarray, N: int, fill=-1):
    Bs = np.concatenate(([fill], B))                        # sentinel in front
    return Bs[np.searchsorted(B, np.arange(N), side='right')]


def _find_unix_jumps(unix_ts: npt.NDArray[np.uint64]) \
        -> npt.NDArray[np.uint64]:
    sel = unix_ts[1:] - unix_ts[:-1] == 1
    return 1 + np.where(sel)[0]


def _find_pps_jumps(pps_ts: npt.NDArray[np.uint64]) \
        -> npt.NDArray[np.uint64]:
    pps_ts = fill_with_next(pps_ts)
    sel = pps_ts[:-1] > pps_ts[:1]
    sel &= pps_ts[:-1] - pps_ts[:1] > 9E6
    return 1 + np.where(sel)[0]


def _get_delay(pps_delays: npt.NDArray[np.void], iog: int) -> float:
    sel = pps_delays['io_group'] == iog
    return np.median(pps_delays[sel]['delay_ticks'])


def get_true_timestamps(packets: npt.NDArray[np.void],
                        pps_delays: Optional[npt.NDArray[np.void]]) -> Timestamps:
    all_unix_ts = np.zeros(packets.shape, dtype=np.uint64)
    all_unix_ts_usec = np.zeros(packets.shape, dtype=np.float64)

    for iog in sorted(np.unique(packets['io_group'])):
        delay = _get_delay(pps_delays, iog) if pps_delays else 0
        sel = packets['io_group'] == iog
        map2all = np.where(sel)[0]
        p = packets[sel]

        unix_ts = np.zeros(p.shape, dtype=np.uint64)

        is_unix = p['packet_type'] == 4
        unix_ts[is_unix] = p[is_unix]['packet_type']
        unix_ts = fill_with_last(unix_ts)

        unix_jumps = _find_unix_jumps(unix_ts)
        pps_jumps = _find_pps_jumps(p['receipt_timestamp'])

        next_unix_jumps = _next_tagged(unix_jumps, packets.shape[0])
        next_pps_jumps = _next_tagged(pps_jumps, packets.shape[0])
        on_right = next_pps_jumps < next_unix_jumps

        to_corr1 = on_right & (p['timestamp'] + delay < 1E7) & ~is_unix
        unix_ts[to_corr1] -= 1
        is_sync = p['packet_type'] == 6
        to_corr2 = (p['timestamp'] > p['receipt_timestamp']) & ~is_unix & ~is_sync
        unix_ts[to_corr2] -= 1
        # decrement twice when both conditions apply

        unix_ts_usec = (p['timestamp'] + delay) % 1E7
        all_unix_ts[map2all] = unix_ts
        all_unix_ts_usec[map2all] = unix_ts_usec

    return Timestamps(all_unix_ts, all_unix_ts_usec)


def add_timestamp_packets(packets: npt.NDArray[np.void],
                          sel: npt.NDArray[np.bool]):
    is_unix = packets['packet_type'] == 4
    unix_idcs = np.where(is_unix)[0]
    prev_unix_idcs = _prev_tagged(unix_idcs, packets.shape[0])
    our_unix_idcs = prev_unix_idcs[sel]
    sel[our_unix_idcs] = True
