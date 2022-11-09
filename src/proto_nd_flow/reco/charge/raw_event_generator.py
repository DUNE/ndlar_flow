import numpy as np
import numpy.ma as ma
from numpy.lib import recfunctions as rfn
import h5py
import logging
from math import ceil
from tqdm import tqdm

from h5flow.core import H5FlowGenerator, resources
from h5flow.data import dereference
from h5flow import H5FLOW_MPI

from module0_flow.reco.charge.raw_event_builder import *
import module0_flow.util.units as units


class RawEventGenerator(H5FlowGenerator):
    '''
        Low-level event builder - generates packet groups according to the
        specified algorithm from a larpix packet datalog file.

        For simulated data, also creates the following datasets:
         - ``mc_truth/events``: Dataset containing event-level truth
         - ``mc_truth/trajectories``: Dataset containing true particle trajectories
         - ``mc_truth/tracks``: Dataset containing true edep-sim segments
        and references:
         - ``mc_truth/events -> mc_truth/trajectories``
         - ``mc_truth/events -> mc_truth/tracks``
         - ``mc_truth/trajectories -> mc_truth/tracks``
         - ``charge/raw_events -> mc_truth/events``
         - ``charge/packets -> mc_truth/tracks``

        Parameters:
         - ``packets_dset_name`` : ``str``, required, output dataset path for packet groups
         - ``buffer_size`` : ``int``, optional, number of packets to load at a time
         - ``nhit_cut`` : ``int``, optional, minimum number of packets in an event
         - ``sync_noise_cut_enabled`` : ``bool``, optional, remove hits occuring soon after a SYNC event
         - ``sync_noise_cut`` : ``int``, optional, if ``sync_noise_cut_enabled`` removes all events that have a timestamp less than this value
         - ``event_builder_class`` : ``str``, optional, event builder algorithm to use (see ``raw_event_builder.py``)
         - ``event_builder_config`` : ``dict``, optional, modify parameters of the event builder algorithm (see ``raw_event_builder.py``)
         - ``mc_events_dset_name`` : ``str``, optional, output dataset for mc events (if present)
         - ``mc_tracks_dset_name`` : ``str``, optional, output dataset path for mc truth tracks (if present)
         - ``mc_trajectories_dset_name`` : ``str``, optional, output dataset path for mc truth trajectories (if present)
         - ``mc_packet_fraction_dset_name`` : ``str``, optional, output dataset path for packet charge fraction truth (if present)

        ``dset_name`` points to a lightweight array used to organize low-level
        event references.

        Requires Units, RunData, and LArData resources in workflow.

        Example config::

            raw_event_generator:
                classname: RawEventGenerator
                dset_name: 'charge/raw_events'
                params:
                    packets_dset_name: 'charge/packets'
                    buffer_size: 38400
                    nhit_cut: 100
                    sync_noise_cut: [100000, 10000000]
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
    class_version = '0.3.0'

    default_buffer_size = 38400
    default_nhit_cut = 100
    default_sync_noise_cut = [100000, 10000000]
    default_sync_noise_cut_enabled = True
    default_event_builder_class = 'SymmetricWindowRawEventBuilder'
    default_event_builder_config = dict()
    default_packets_dset_name = 'charge/packets'
    default_mc_events_dset_name = 'mc_truth/events'
    default_mc_tracks_dset_name = 'mc_truth/tracks'
    default_mc_trajectories_dset_name = 'mc_truth/trajectories'
    default_mc_packet_fraction_dset_name = 'mc_truth/packet_fraction'

    raw_event_dtype = np.dtype([
        ('id', 'u8'),
        ('unix_ts', 'u8')
    ])

    mc_event_dtype = np.dtype([
        ('id', 'u8'),
    ])

    def __init__(self, **params):
        super(RawEventGenerator, self).__init__(**params)

        # set up parameters
        self.buffer_size = params.get('buffer_size', self.default_buffer_size)
        self.nhit_cut = params.get('nhit_cut', self.default_nhit_cut)
        self.sync_noise_cut = params.get('sync_noise_cut', self.default_sync_noise_cut)
        self.sync_noise_cut_enabled = params.get('sync_noise_cut_enabled', self.default_sync_noise_cut_enabled)
        self.event_builder_class = params.get('event_builder_class', self.default_event_builder_class)
        self.event_builder_config = params.get('event_builder_config', self.default_event_builder_config)

        # set up new dataset paths
        self.packets_dset_name = params.get('packets_dset_name', self.default_packets_dset_name)
        self.raw_event_dset_name = self.dset_name
        self.mc_events_dset_name = params.get('mc_events_dset_name', self.default_mc_events_dset_name)
        self.mc_tracks_dset_name = params.get('mc_tracks_dset_name', self.default_mc_tracks_dset_name)
        self.mc_trajectories_dset_name = params.get('mc_trajectories_dset_name', self.default_mc_trajectories_dset_name)
        self.mc_packet_fraction_dset_name = params.get('mc_packet_fraction_dset_name', self.default_mc_packet_fraction_dset_name)

        # create event builder
        self.event_builder = globals()[self.event_builder_class](**self.event_builder_config)

        # set up input file
        if H5FLOW_MPI:
            self.input_fh = h5py.File(self.input_filename, 'r', driver='mpio', comm=self.comm)
        else:
            self.input_fh = h5py.File(self.input_filename, 'r')
        self.packets = self.input_fh['packets']

        # set up loop variables
        if self.start_position is None:
            self.start_position = 0
        if self.end_position is None or self.end_position > len(self.packets):
            self.end_position = len(self.packets)
        self.slices = [slice(st, st + self.buffer_size) for st in range(self.start_position + self.rank * self.buffer_size, self.end_position, self.size * self.buffer_size)]
        self.iteration = 0

    def __len__(self):
        return len(self.slices)

    def _convert_mc_truth_tracks(self, tracks):
        ''' Apply geometry transformation from edep-sim coordinates to larnd-sim / module0_flow coordinates '''
        tracks_copy = tracks.copy()
        for field in ('x_start', 'y_start', 'z_start', 'x', 'y', 'z', 'x_end',
                      'y_end', 'z_end'):
            if 'x' in field or 'z' in field:
                tracks_copy[field] = tracks[field] * units.cm
            elif 'y' in field:
                tracks_copy[field] = (tracks[field] + 21.8236) * units.cm
        for field in ('tran_diff', 'dx', 'long_diff'):
            tracks_copy[field] = tracks[field] * units.cm
        tracks_copy['dE'] = tracks['dE'] * units.MeV
        tracks_copy['dEdx'] = tracks['dEdx'] * (units.MeV / units.cm)
        for field in ('t', 't_start', 't_end'):
            tracks_copy[field] = tracks[field] * units.us
        return tracks_copy

    def _convert_mc_truth_trajectories(self, traj):
        traj_copy = traj.copy()
        for field in ('xyz_start', 'xyz_end'):
            traj_copy[field][:, 0] = traj[field][:, 2] * units.mm
            traj_copy[field][:, 1] = traj[field][:, 1] * units.mm + 218.236
            traj_copy[field][:, 2] = traj[field][:, 0] * units.mm
        for field in ('pxyz_start', 'pxyz_end'):
            traj_copy[field][:, 0] = traj[field][:, 2] * units.MeV
            traj_copy[field][:, 1] = traj[field][:, 1] * units.MeV
            traj_copy[field][:, 2] = traj[field][:, 0] * units.MeV
        for field in ('t_start', 't_end'):
            traj_copy[field] = traj[field] * units.ns
        return traj_copy

    def init(self):
        super(RawEventGenerator, self).init()

        if self.data_manager.dset_exists(self.raw_event_dset_name):
            raise RuntimeError(f'{self.raw_event_dset_name} already exists, refusing to append!')
        if self.data_manager.dset_exists(self.packets_dset_name):
            raise RuntimeError(f'{self.packets_dset_name} already exists, refusing to append!')

        self.is_mc = resources['RunData'].is_mc

        self.packets_dtype = self.packets.dtype
        if self.is_mc:
            self.mc_assn = self.input_fh['mc_packets_assn']
            self.mc_tracks = self.input_fh['tracks']
            self.mc_trajectories = self.input_fh['trajectories']
            self.mc_tracks_dtype = self.mc_tracks.dtype

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
                                    packets_dset_name=self.packets_dset_name,
                                    **self.event_builder.get_config()
                                    )
        if self.is_mc:
            self.data_manager.set_attrs(self.raw_event_dset_name,
                                        mc_tracks_dset_name=self.mc_tracks_dset_name,
                                        mc_trajectories_dset_name=self.mc_trajectories_dset_name,
                                        mc_packet_fraction_dset_name=self.mc_packet_fraction_dset_name)

            self.data_manager.create_dset(self.mc_packet_fraction_dset_name, dtype=self.mc_assn['fraction'].dtype)
            self.data_manager.create_ref(self.packets_dset_name, self.mc_packet_fraction_dset_name)

            # copy datasets from source file
            self.data_manager.create_dset(self.mc_events_dset_name, dtype=self.mc_event_dtype)

            self.data_manager.create_dset(self.mc_tracks_dset_name, dtype=self.mc_tracks_dtype)
            ntracks = len(self.mc_tracks)

            track_sl = slice(
                min(ntracks, ceil(ntracks / self.size) * self.rank),
                min(ntracks, ceil(ntracks / self.size) * (self.rank + 1)))
            self.data_manager.reserve_data(self.mc_tracks_dset_name, track_sl)
            self.data_manager.write_data(
                self.mc_tracks_dset_name, track_sl,
                self._convert_mc_truth_tracks(self.mc_tracks[track_sl]))

            self.data_manager.create_dset(self.mc_trajectories_dset_name, dtype=self.mc_trajectories.dtype)
            ntraj = len(self.mc_trajectories)
            traj_sl = slice(
                min(ntracks, ceil(ntraj / self.size * self.rank)),
                min(ntraj, ceil(ntraj / self.size * (self.rank + 1))))
            self.data_manager.reserve_data(self.mc_trajectories_dset_name, traj_sl)
            self.data_manager.write_data(
                self.mc_trajectories_dset_name, traj_sl,
                self._convert_mc_truth_trajectories(self.mc_trajectories[traj_sl]))

            # set up references
            self.data_manager.create_ref(self.raw_event_dset_name, self.mc_events_dset_name)
            self.data_manager.create_ref(self.packets_dset_name, self.mc_tracks_dset_name)
            self.data_manager.create_ref(self.mc_events_dset_name, self.mc_trajectories_dset_name)
            self.data_manager.create_ref(self.mc_events_dset_name, self.mc_tracks_dset_name)
            self.data_manager.create_ref(self.mc_trajectories_dset_name, self.mc_tracks_dset_name)

            # create references between trajectories and tracks
            traj_evid = self.mc_trajectories['eventID'][:]
            tracks_evid = self.mc_tracks['eventID'][:]
            evs, ev_traj_start, ev_track_start = np.intersect1d(
                traj_evid, tracks_evid, return_indices=True)
            evs, ev_traj_end, ev_track_end = np.intersect1d(
                traj_evid[::-1], tracks_evid[::-1], return_indices=True)
            ev_traj_end = len(self.mc_trajectories['eventID']) - ev_traj_end
            ev_track_end = len(self.mc_tracks['eventID']) - ev_track_end
            truth_slice = slice(
                ceil(len(evs) / self.size) * self.rank,
                ceil(len(evs) / self.size) * (self.rank + 1))

            # create placeholder events data
            mc_events_slice = self.data_manager.reserve_data(self.mc_events_dset_name, len(evs[truth_slice]))
            mc_events_data = np.empty(len(evs[truth_slice]), dtype=self.mc_event_dtype)
            mc_events_data['id'] = evs[truth_slice]
            self.data_manager.write_data(
                self.mc_events_dset_name, mc_events_slice, mc_events_data)

            traj_trackid = self.mc_trajectories['trackID'][:]
            tracks_trackid = self.mc_tracks['trackID'][:]
            iter_ = tqdm(range(truth_slice.start, truth_slice.stop), smoothing=1, desc='generating truth references') if self.rank == 0 else range(truth_slice.start, truth_slice.stop)
            for i in iter_:
                if i < len(evs):
                    ev = evs[i]
                    traj_start, traj_end = ev_traj_start[i], ev_traj_end[i]
                    track_start, track_end = ev_track_start[i], ev_track_end[i]
                    traj_trackid_block = np.expand_dims(traj_trackid[traj_start:traj_end], -1)
                    track_trackid_block = np.expand_dims(tracks_trackid[track_start:track_end], 0)
                    traj_evid_block = np.expand_dims(traj_evid[traj_start:traj_end], -1)
                    track_evid_block = np.expand_dims(tracks_evid[track_start:track_end], 0)
                    ref = np.argwhere((traj_trackid_block == track_trackid_block) &
                                      (traj_evid_block == track_evid_block))
                    ref[:, 0] += traj_start
                    ref[:, 1] += track_start
                    self.data_manager.write_ref(self.mc_trajectories_dset_name, self.mc_tracks_dset_name, ref)
                    ref = np.argwhere(ev == traj_evid_block)
                    ref[:, 0] += traj_start
                    ref[:, 1] = i + mc_events_slice.start
                    self.data_manager.write_ref(self.mc_trajectories_dset_name, self.mc_events_dset_name, ref)
                    ref = np.argwhere(ev == track_evid_block)
                    ref[:, 0] = i + mc_events_slice.start
                    ref[:, 1] += track_start
                    self.data_manager.write_ref(self.mc_events_dset_name, self.mc_tracks_dset_name, ref)
                else:
                    self.data_manager.write_ref(self.mc_trajectories_dset_name, self.mc_tracks_dset_name, np.empty((0,2)))
                    self.data_manager.write_ref(self.mc_trajectories_dset_name, self.mc_events_dset_name, np.empty((0,2)))
                    self.data_manager.write_ref(self.mc_events_dset_name, self.mc_tracks_dset_name, np.empty((0,2)))                                        

        # if self.is_mc:
        #     # copy meta-data from input file
        #     resources['LArData'].data['v_drift'] = self.input_fh['configs'].attrs['vdrift'] * \
        #         (units.cm / units.us)

        # get first timestamp packet from file, without loading the full dataset
        self.last_unix_ts = np.empty((0,), dtype=self.packets_dtype)
        for p in self.packets:
            if p['packet_type'] == 4:
                self.last_unix_ts = p
                break

    def finish(self):
        super(RawEventGenerator, self).finish()
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

        mask = (block['valid_parity'].astype(bool) & (block['packet_type'] == 0))  # data packets
        mask = mask | (block['packet_type'] == 4)  # timestamp packets
        mask = mask | (block['packet_type'] == 7)  # external trigger packets
        mask = mask | (block['packet_type'] == 6)  # sync packets

        packet_buffer = np.copy(block[mask])
        self.pass_last_unix_ts(packet_buffer)
        packet_buffer = np.insert(packet_buffer, [0], self.last_unix_ts)
        if self.is_mc:
            mc_assn = mc_assn[mask]

        # find unix timestamp groups
        ts_mask = packet_buffer['packet_type'] == 4
        ts_grps = np.split(packet_buffer, np.argwhere(ts_mask).ravel())
        unix_ts_grps = [np.full(len(ts_grp[1:]), ts_grp[0], dtype=packet_buffer.dtype) for ts_grp in ts_grps if len(ts_grp)]
        unix_ts = np.concatenate(unix_ts_grps, axis=0) \
            if len(unix_ts_grps) else np.empty((0,), dtype=packet_buffer.dtype)
        packet_buffer = packet_buffer[~ts_mask]
        if self.is_mc:
            mc_assn = mc_assn[~ts_mask[1:]]
        packet_buffer['timestamp'] = packet_buffer['timestamp'].astype(int) % (2**31)  # ignore 32nd bit from pacman triggers
        self.last_unix_ts = unix_ts[-1] if len(unix_ts) else self.last_unix_ts

        if self.sync_noise_cut_enabled and not self.is_mc:
            # remove all packets that occur before the cut
            sync_noise_mask = (packet_buffer['timestamp'] > self.sync_noise_cut[0]) & (packet_buffer['timestamp'] < self.sync_noise_cut[1])
            packet_buffer = packet_buffer[sync_noise_mask]
            unix_ts = unix_ts[sync_noise_mask]
            if self.is_mc:
                mc_assn = mc_assn[sync_noise_mask]

        # run event builder
        eb_rv = list(self.event_builder.build_events(packet_buffer, unix_ts, mc_assn))
        events, event_unix_ts = eb_rv if not self.is_mc else eb_rv[:-1]
        if self.is_mc:
            event_mc_assn = eb_rv[-1]
        else:
            event_mc_assn = None

        # apply nhit cut
        nhit_filtered = list(filter(lambda x: len(x[0]) >= self.nhit_cut, zip(events, event_unix_ts)))
        if self.is_mc:
            mc_assn_filtered = list(filter(lambda x: len(x) >= self.nhit_cut, event_mc_assn))

        if len(nhit_filtered):
            events, event_unix_ts = zip(*nhit_filtered)
            if self.is_mc:
                event_mc_assn = mc_assn_filtered
        else:
            events, event_unix_ts = list(), list()
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
        ev_idcs = np.concatenate([np.full(len(ev), i_ev) for i_ev, ev in zip(raw_event_idcs, events)], axis=0) \
            if len(events) else np.empty(0, dtype=raw_event_idcs.dtype)
        ref = np.c_[ev_idcs, packets_idcs]
        self.data_manager.write_ref(self.raw_event_dset_name, self.packets_dset_name, ref)

        if self.is_mc:
            # write mc data to file
            mc_assn = (np.concatenate(event_mc_assn, axis=0)
                       if len(event_mc_assn) else np.full((0,), -1, dtype=self.mc_assn.dtype))
            mc_assn_mask = (mc_assn['track_ids'] == -1) | (mc_assn['fraction'] == 0.)
            event_tracks = ma.array(mc_assn['track_ids'], mask=mc_assn_mask)
            event_packet_fraction = ma.array(mc_assn['fraction'], mask=mc_assn_mask)

            # set up packet references
            packets_idcs = np.broadcast_to(packets_idcs[:, np.newaxis], event_tracks.shape)
            ref = np.c_[packets_idcs.ravel(), event_tracks.ravel()]
            ref = np.unique(ref[~event_tracks.mask.ravel()], axis=0) \
                if len(ref) else ref
            self.data_manager.write_ref(self.packets_dset_name, self.mc_tracks_dset_name, ref)
            sl = self.data_manager.reserve_data(self.mc_packet_fraction_dset_name, len(ref))
            self.data_manager.write_data(self.mc_packet_fraction_dset_name, sl, event_packet_fraction.compressed())
            self.data_manager.write_ref(self.packets_dset_name, self.mc_packet_fraction_dset_name, np.c_[ref[:,0], sl])

            # find events associated with tracks
            if H5FLOW_MPI:
                self.comm.barrier()
            ref_dset, ref_dir = self.data_manager.get_ref(self.mc_tracks_dset_name, self.mc_events_dset_name)
            ref_region = self.data_manager.get_ref_region(self.mc_tracks_dset_name, self.mc_events_dset_name)
            mc_evs = dereference(ref[:, 1], ref_dset, region=ref_region,
                                 ref_direction=ref_dir, indices_only=True)

            ev_idcs = np.broadcast_to(np.expand_dims(ev_idcs, axis=-1), event_tracks.shape)
            ref = np.c_[ev_idcs[~event_tracks.mask].ravel(), mc_evs.ravel()]
            ref = np.unique(ref, axis=0) if len(ref) else ref
            self.data_manager.write_ref(self.raw_event_dset_name, self.mc_events_dset_name, ref)

        return raw_event_slice if nevents else H5FlowGenerator.EMPTY

    def pass_last_unix_ts(self, packets):
        if self.size < 2:
            return

        # rank 0 get stored from rank N-1
        if self.rank == self.size - 1:
            self.comm.send(self.last_unix_ts, dest=0)
        # rank i give max unix timestamp to i+1
        mask = packets['packet_type'] == 4
        max_unix_ts = packets[mask][np.argmax(packets[mask]['timestamp'])] if np.any(mask) else self.last_unix_ts
        self.last_unix_ts = self.comm.recv(source=self.rank - 1 if self.rank > 0 else self.size - 1)
        # rank N-1 store max unix timestamp for next iteration
        if self.rank != self.size - 1:
            self.comm.send(max_unix_ts, dest=self.rank + 1)
