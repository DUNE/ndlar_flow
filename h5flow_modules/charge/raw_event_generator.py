import numpy as np
from numpy.lib import recfunctions as rfn
import h5py
import logging

from h5flow.core import H5FlowGenerator

try:
    from raw_event_builder import *
except:
    from .raw_event_builder import *

class RawEventGenerator(H5FlowGenerator):
    '''
        Low-level event builder - generates packet groups according to the
        specified algorithm from a larpix packet datalog file

        Parameters:
         - ``packets_dset_name`` : ``str``, required, output dataset path for packet groups
         - ``buffer_size`` : ``int``, optional, number of packets to load at a time
         - ``nhit_cut`` : ``int``, optional, minimum number of packets in an event
         - ``sync_noise_cut_enabled`` : ``bool``, optional, remove hits occuring soon after a SYNC event
         - ``sync_noise_cut`` : ``int``, optional, if ``sync_noise_cut_enabled`` removes all events that have a timestamp less than this value
         - ``event_builder_class`` : ``str``, optional, event builder algorithm to use (see ``raw_event_builder.py``)
         - ``event_builder_config`` : ``dict``, optional, modify parameters of the event builder algorithm (see ``raw_event_builder.py``)
         - ``mc_tracks_dset_name`` : ``str``, optional, output dataset path for mc truth tracks (if present)

        The ``dset_name`` points to a lightweight array used to organize
        low-level event references.

        Example config::

            raw_event_generator:
                classname: RawEventGenerator
                dset_name: 'charge/raw_events'
                params:
                    packets_dset_name: 'charge/packets'
                    buffer_size: 38400
                    nhit_cut: 100
                    sync_noise_cut: 100000
                    sync_noise_cut_enabled: True
                    event_builder_class: 'SymmetricWindowRawEventBuilder'
                    event_builder_config:
                        window: 910
                        threshold: 10
                        rollover_ticks: 10000000

        ``raw_event`` datatype::

            id          u8, unique event identifier
            unix_ts     u8, unix timestamp of event [s since epoch]

    '''
    class_version = '0.1.0'

    default_buffer_size = 38400
    default_nhit_cut = 100
    default_sync_noise_cut = [100000, 10000000]
    default_sync_noise_cut_enabled = True
    default_event_builder_class = 'SymmetricWindowRawEventBuilder'
    default_event_builder_config = dict()
    default_packets_dset_name = 'charge/packets'
    default_mc_tracks_dset_name = 'mc_truth/tracks'

    raw_event_dtype = np.dtype([
        ('id', 'u8'),
        ('unix_ts', 'u8')
        ])

    def __init__(self, **params):
        super(RawEventGenerator,self).__init__(**params)

        # set up parameters
        self.buffer_size = params.get('buffer_size', self.default_buffer_size)
        self.nhit_cut = params.get('nhit_cut', self.default_nhit_cut)
        self.sync_noise_cut = params.get('sync_noise_cut', self.default_sync_noise_cut)
        self.sync_noise_cut_enabled = params.get('sync_noise_cut_enabled', self.default_sync_noise_cut_enabled)
        self.event_builder_class = params.get('event_builder_class', self.default_event_builder_class)
        self.event_builder_config = params.get('event_builder_config', self.default_event_builder_config)

        # create event builder
        self.event_builder = globals()[self.event_builder_class](**self.event_builder_config)

        # set up input file
        self.input_fh = h5py.File(self.input_filename, 'r', driver='mpio', comm=self.comm)
        self.packets = self.input_fh['packets']
        self.is_mc = 'mc_packets_assn' in self.input_fh
        self.mc_assn = self.input_fh['mc_packets_assn'] if self.is_mc else None
        self.mc_tracks = self.input_fh['tracks'] if self.is_mc else None

        # set up new data objects
        self.packets_dtype = self.packets.dtype
        self.packets_dset_name = params.get('packets_dset_name', self.default_packets_dset_name)
        self.raw_event_dset_name = self.dset_name
        self.mc_tracks_dtype = self.mc_tracks.dtype if self.is_mc else None
        self.mc_tracks_dset_name = params.get('mc_tracks_dset_name', self.default_mc_tracks_dset_name)

        # set up loop variables
        if self.start_position is None:
            self.start_position = 0
        if self.end_position is None or self.end_position > len(self.packets):
            self.end_position = len(self.packets)
        self.slices = [slice(st, st + self.buffer_size) for st in range(self.start_position + self.rank * self.buffer_size, self.end_position, self.size * self.buffer_size)]
        self.iteration = 0

        self.last_unix_ts = self.packets[np.argmax(self.packets['packet_type']==4)] # first timestamp packet

    def __len__(self):
        return len(self.slices)

    def init(self):
        super(RawEventGenerator,self).init()

        if self.data_manager.dset_exists(self.raw_event_dset_name):
            raise RuntimeError(f'{self.raw_event_dset_name} already exists, refusing to append!')
        if self.data_manager.dset_exists(self.packets_dset_name):
            raise RuntimeError(f'{self.packets_dset_name} already exists, refusing to append!')

        # initialize data objects
        self.data_manager.create_dset(self.raw_event_dset_name, dtype=self.raw_event_dtype)
        self.data_manager.create_dset(self.packets_dset_name, dtype=self.packets_dtype)
        self.data_manager.create_ref(self.raw_event_dset_name, self.packets_dset_name)
        self.data_manager.set_attrs(self.raw_event_dset_name,
            classname=self.classname,
            class_version=self.class_version,
            nhit_cut=self.nhit_cut,
            buffer_size=self.buffer_size,
            sync_noise_cut=self.sync_noise_cut,
            sync_noise_cut_enabled=self.sync_noise_cut_enabled,
            event_builder_class=self.event_builder_class,
            event_builder_class_version=self.event_builder.version,
            start_position=self.start_position,
            end_position=self.end_position,
            input_filename=self.input_filename,
            is_mc=self.is_mc,
            packets_dset_name=self.packets_dset_name,
            **self.event_builder.get_config()
            )
        if self.is_mc:
            self.data_manager.set_attrs(self.raw_event_dset_name,
                mc_tracks_dset_name=self.mc_tracks_dset_name)

    def finish(self):
        self.input_fh.close()

    def next(self):
        '''
            Read in a new block of LArPix packet data from the input file and
            apply the raw event building algorithm. Save packets to a dataset
            (``packets_dset_name``) and create references to a raw event
            (``raw_event_dset_name``).

            :returns: ``slice`` into the dataset given by ``raw_event_dset_name``
        '''
        if self.iteration >= len(self.slices):
            sl = H5FlowGenerator.EMPTY
        else:
            sl = self.slices[self.iteration]
        self.iteration += 1

        block = self.packets[sl]
        if self.is_mc:
            mc_assn = self.mc_assn[sl]
        else:
            mc_assn = None

        mask = (block['valid_parity'].astype(bool) & (block['packet_type'] == 0)) # data packets
        mask = mask | (block['packet_type'] == 4) # timestamp packets
        mask = mask | (block['packet_type'] == 7) # external trigger packets
        mask = mask | (block['packet_type'] == 6) # sync packets

        packet_buffer = np.copy(block[mask])
        self.pass_last_unix_ts(packet_buffer)
        packet_buffer = np.insert(packet_buffer, [0], self.last_unix_ts)
        if self.is_mc:
            mc_assn = mc_assn[mask]

        # find unix timestamp groups
        ts_mask = packet_buffer['packet_type'] == 4
        ts_grps = np.split(packet_buffer, np.argwhere(ts_mask).ravel())
        unix_ts_grps = [np.full(len(ts_grp[1:]), ts_grp[0], dtype=packet_buffer.dtype) for ts_grp in ts_grps if len(ts_grp) > 1]
        unix_ts = np.concatenate(unix_ts_grps, axis=0) \
            if len(unix_ts_grps) else np.empty((0,), dtype=packet_buffer.dtype)
        packet_buffer = packet_buffer[~ts_mask]
        if self.is_mc:
            mc_assn = mc_assn[~ts_mask]
        packet_buffer['timestamp'] = packet_buffer['timestamp'].astype(int) % (2**31) # ignore 32nd bit from pacman triggers
        self.last_unix_ts = unix_ts[-1] if len(unix_ts) else self.last_unix_ts

        if self.sync_noise_cut_enabled:
            # remove all packets that occur before the cut
            sync_noise_mask = (packet_buffer['timestamp'] > self.sync_noise_cut[0]) & (packet_buffer['timestamp'] < self.sync_noise_cut[1])
            packet_buffer = packet_buffer[sync_noise_mask]
            unix_ts = unix_ts[sync_noise_mask]
            if self.is_mc:
                mc_assn = mc_assn[sync_noise_mask]

        # run event builder
        eb_rv = self.event_builder.build_events(packet_buffer, unix_ts, mc_assn)
        events, event_unix_ts = eb_rv if not self.is_mc else eb_rv[:-1]
        if self.is_mc:
            event_mc_assn = eb_rv[-1]
        else:
            event_mc_assn = None

        # apply nhit cut
        nhit_filtered = list(filter(lambda x: len(x[0]) >= self.nhit_cut, zip(events, event_unix_ts)))
        if self.is_mc
            mc_assn_filtered = list(filter(lambda x: len(x) >= self.nhit_cut, event_mc_assn))

        if len(nhit_filtered):
            events, event_unix_ts = zip(*nhit_filtered)
            if self.is_mc:
                event_mc_assn = mc_assn_filtered
        else:
            events, event_unix_ts = list(),list()
            if self.is_mc:
                event_mc_assn = list()
        nevents = len(events)

        # write event to file
        raw_event_array = np.zeros((nevents,), dtype=self.raw_event_dtype)
        raw_event_slice = self.data_manager.reserve_data(self.raw_event_dset_name, nevents)
        raw_event_idcs = np.arange(raw_event_slice.start, raw_event_slice.stop, dtype=int)
        if nevents:
            raw_event_array['unix_ts'] = [p[0]['timestamp'] for p in event_unix_ts]
            raw_event_array['id'] = raw_event_idcs
        self.data_manager.write_data(self.raw_event_dset_name, raw_event_slice, raw_event_array)

        # write packets to file
        packets_array = np.concatenate(events, axis=0) if nevents else np.empty((0,), dtype=self.packets_dtype)
        packets_slice = self.data_manager.reserve_data(self.packets_dset_name, len(packets_array))
        packets_idcs = np.arange(packets_slice.start, packets_slice.stop)
        self.data_manager.write_data(self.packets_dset_name, packets_slice, packets_array)

        # set up references
        #   event -> packet refs
        ev_idcs = np.concatenate([np.full(len(ev),i_ev) for i_ev,ev in zip(raw_event_idcs,events)], axis=0) \
            if len(events) else np.empty(0, dtype=raw_event_idcs.dtype)
        ref = np.c_[ev_idcs, packets_idcs]
        self.data_manager.write_ref(self.raw_event_dset_name, self.packets_dset_name, ref)

        if self.is_mc:
            # write mc data to file
            track_src_idx, track_src_idx_inv = np.unique(np.concatenate(event_mc_assn, axis=0), return_inverse=True)
            tracks = self.mc_tracks[track_src_idx] if len(track_src_idx) else np.empty((0,), dtype=self.mc_tracks_dtype)
            tracks_slice = self.data_manager.reserve_data(self.mc_tracks_dset_name, len(tracks_array))
            tracks_dest_idx = np.r_[tracks_slice]

            # set up packet references
            ref = np.c_[packets_index, tracks_dest_idx[track_src_idx_inv]]
            self.data_manager.write_ref(self.packets_dset_name, self.mc_tracks_dset_name, ref)

            # set up event references
            ref = np.unique(np.c_[ev_idcs, tracks_dest_idx[track_src_idx_inv]], axis=0)
            self.data_manager.write_ref(self.raw_event_dset_name, self.mc_tracks_dset_name, ref)

        return raw_event_slice if sl is not H5FlowGenerator.EMPTY else H5FlowGenerator.EMPTY

    def pass_last_unix_ts(self, packets):
        if self.size < 2:
            return

        # rank 1 get stored from rank N-1
        if self.rank == self.size-1:
            self.comm.send(self.last_unix_ts, dest=0)
        # rank i give max unix timestamp to i+1
        mask = packets['packet_type'] == 4
        max_unix_ts = packets[mask][np.argmax(packets[mask]['timestamp'])] if np.any(mask) else self.last_unix_ts
        self.last_unix_ts = self.comm.recv(source=self.rank-1 if self.rank > 0 else self.size-1)
        # rank N-1 store max unix timestamp for next iteration
        if self.rank != self.size-1:
            self.comm.send(max_unix_ts, dest=self.rank+1)

