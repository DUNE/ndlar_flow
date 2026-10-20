from typing import Any

import numpy as np
import numpy.typing as npt
from dbscan1d import DBSCAN1D

from h5flow.core import resources
from h5flow.data import H5FlowDataManager


class PPSDelayExtractor:
    delay_dtype = np.dtype([
        ('unix_ts', '<u8'), ('delay_ticks', '<u8'), ('io_group', '<u2')
    ])

    default_window: int = 100
    default_dbscan_eps: int = 100
    default_min_packets: int = 50

    # The params are forwarded from the RawEventGenerator
    def __init__(self, **params: Any):
        self.packets_dset_name: str = params['packets_dset_name']

        k = 'pps_delay_extractor_config'
        self.window: int = params[k].get('window', self.default_window)
        self.dbscan_eps: int = params[k].get('dbscan_eps',
                                             self.default_dbscan_eps)
        self.min_packets: int = params[k].get('min_packets',
                                              self.default_min_packets)
        self.debug_mode: bool = params[k].get('debug_mode', False)

        self.unix_ts: list[np.uint32] = []
        self.delay_ticks: list[np.float64] = []
        self.io_group: list[np.uint16] = []

        self.data_manager: H5FlowDataManager | None = None

    def setup(self, data_manager: H5FlowDataManager):
        self.data_manager = data_manager
        if self.debug_mode:
            self.data_manager.create_dset('charge/pps_delay',
                                          dtype=self.delay_dtype)

    def update(self, packets: npt.NDArray[Any]):
        for iog in np.unique(packets['io_group']):
            all_pkts = packets[packets['io_group'] == iog]
            ts2pkt = np.where(all_pkts['packet_type'] == 4)[0]
            all_ts = all_pkts[ts2pkt]['timestamp']
            jumps = np.where(all_ts[1:] - all_ts[:-1] == 1)[0]
            clusters = DBSCAN1D(eps=self.dbscan_eps, min_samples=1) \
                .fit(jumps.reshape(-1, 1))

            assert clusters.labels_ is not None
            for i in list(range(max(clusters.labels_))):
                ts_idcs = jumps[clusters.labels_ == i]
                pkt_idcs = ts2pkt[ts_idcs]  # indices of ts pkts in this cluster

                ts_pkts = all_pkts[pkt_idcs]
                unix_ts = ts_pkts['timestamp']
                assert len(np.unique(unix_ts) == 2)
                t = np.max(unix_ts)

                pkts = all_pkts[np.min(pkt_idcs) - self.window
                                :np.max(pkt_idcs) + self.window]
                rollover = resources['RunData'].rollover_ticks

                data_packet_type = resources['RunData'].data_packet_type
                data_pkts = pkts[pkts['packet_type'] == data_packet_type]
                if len(data_pkts) < self.min_packets:
                    continue
                delay = np.float64(rollover) - np.median(data_pkts['timestamp'])

                self.unix_ts.append(t)
                self.delay_ticks.append(delay)
                self.io_group.append(iog)

    def finish(self):
        data = np.empty(len(self.unix_ts), dtype=self.delay_dtype)
        order = np.argsort(self.unix_ts)
        data['unix_ts'] = np.array(self.unix_ts)[order]
        data['delay_ticks'] = np.array(self.delay_ticks)[order]
        data['io_group'] = np.array(self.io_group)[order]

        delays: list[tuple[int, int]] = [] # iog -> median delay

        for iog in np.sort(np.unique(data['io_group'])):
            sel = data['io_group'] == iog
            delay = np.median(data[sel]['delay_ticks'])
            delays.append((int(iog), int(delay)))

        assert self.data_manager is not None, "call setup plz"
        self.data_manager.set_attrs(self.packets_dset_name,
                                    pps_delays=delays)

        if self.debug_mode:
            name = 'charge/pps_delay'
            self.data_manager.create_dset(name, dtype=self.delay_dtype)
            sl = self.data_manager.reserve_data(name, len(self.unix_ts))
            self.data_manager.write_data(name, sl, data)
