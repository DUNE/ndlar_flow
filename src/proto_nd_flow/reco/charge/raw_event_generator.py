import numpy as np
import numpy.ma as ma
from numpy.lib import recfunctions as rfn
import h5py
import logging
import warnings
from math import ceil
from tqdm import tqdm

from h5flow.core import H5FlowGenerator, resources
from h5flow.data import dereference
from h5flow import H5FLOW_MPI

from proto_nd_flow.reco.charge.raw_event_builder import *
import proto_nd_flow.util.units as units


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
         - ``nhit_limit`` : ``int``, optional, maximum number of packets in an event
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
                    nhit_limit: 1e9
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
    default_nhit_limit= 1e9
    default_sync_noise_cut = [100000, 10000000]
    default_sync_noise_cut_enabled = True
    
    default_event_builder_class = 'SymmetricWindowRawEventBuilder'
    default_event_builder_config = dict()
    default_packets_dset_name = 'charge/packets'
    default_mc_events_dset_name = 'mc_truth/interactions'
    default_mc_stack_dset_name = 'mc_truth/stack'
    default_mc_tracks_dset_name = 'mc_truth/segments'
    default_mc_trajectories_dset_name = 'mc_truth/trajectories'
    default_mc_packet_fraction_dset_name = 'mc_truth/packet_fraction'

    raw_event_dtype = np.dtype([
        ('id', 'u8'),
        ('unix_ts', 'u8')
    ])

    # mc_event_dtype = np.dtype([
        # ('id', 'u8'),
    # ])

    def __init__(self, **params):
        super(RawEventGenerator, self).__init__(**params)

        # set up parameters
        self.buffer_size = params.get('buffer_size', self.default_buffer_size)
        self.nhit_cut = params.get('nhit_cut', self.default_nhit_cut)
        self.nhit_limit = params.get('nhit_limit', self.default_nhit_limit)
        self.sync_noise_cut = params.get('sync_noise_cut', self.default_sync_noise_cut)
        self.sync_noise_cut_enabled = params.get('sync_noise_cut_enabled', self.default_sync_noise_cut_enabled)
        self.event_builder_class = params.get('event_builder_class', self.default_event_builder_class)
        self.event_builder_config = params.get('event_builder_config', self.default_event_builder_config)
        
        # set up new dataset paths
        self.packets_dset_name = params.get('packets_dset_name', self.default_packets_dset_name)
        self.raw_event_dset_name = self.dset_name
        self.mc_events_dset_name = params.get('mc_events_dset_name', self.default_mc_events_dset_name)
        self.mc_stack_dset_name = params.get('mc_stack_dset_name', self.default_mc_stack_dset_name)
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

    def init(self):
        super(RawEventGenerator, self).init()

        if self.data_manager.dset_exists(self.raw_event_dset_name):
            raise RuntimeError(f'{self.raw_event_dset_name} already exists, refusing to append!')
        if self.data_manager.dset_exists(self.packets_dset_name):
            raise RuntimeError(f'{self.packets_dset_name} already exists, refusing to append!')

        self.is_mc = resources['RunData'].is_mc

        self.packets_dtype = self.packets.dtype
        if self.is_mc:
            self.is_mc_neutrino = True
            self.mc_assn = self.input_fh['mc_packets_assn']
            self.mc_tracks = self.input_fh['segments']
            self.mc_trajectories = self.input_fh['trajectories']
            try:
                self.mc_events = self.input_fh['mc_hdr']
            except:
                self.mc_events = self.input_fh['vertices']
            try:
                self.mc_stack = self.input_fh['mc_stack']
            except:
                self.is_mc_neutrino = False
                print("Hope you are not processing neutrino simulation! There is no information for neutrino interactions.")
                pass

            # set up attribute name for vertex_id and traj_id
            if 'file_vertex_id' in self.input_fh['vertices'].dtype.names:
                self.vertex_id_name = 'file_vertex_id'
            else:
                self.vertex_id_name = 'vertex_id'
                warnings.warn("Using 'vertex_id'(unique for beam simulation, but not for mpvmpr) instead of 'file_vertex_id'.")

            if 'file_traj_id' in self.input_fh['trajectories'].dtype.names:
                self.traj_id_name = 'file_traj_id'
                if self.is_mc_neutrino and 'file_traj_id' not in self.input_fh['mc_stack'].dtype.names:
                    self.traj_id_name = 'traj_id'
            else:
                self.traj_id_name = 'traj_id'
                warnings.warn("Using 'traj_id' instead of 'file_traj_id'. 'traj_id' is not unique across the file and will cause reference issues.")

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
                                        mc_packet_fraction_dset_name=self.mc_packet_fraction_dset_name,
                                        mc_events_dset_name=self.mc_events_dset_name)
            if self.is_mc_neutrino:
                self.data_manager.set_attrs(self.raw_event_dset_name,
                                            mc_stack_dset_name=self.mc_stack_dset_name)

            self.data_manager.create_dset(self.mc_packet_fraction_dset_name, dtype=self.mc_assn.dtype)
            self.data_manager.create_ref(self.packets_dset_name, self.mc_packet_fraction_dset_name)

            # copy datasets from source file
            if self.is_mc_neutrino:
                # MC generator particle stack
                self.data_manager.create_dset(self.mc_stack_dset_name, dtype=self.mc_stack.dtype)
                nstack = len(self.mc_stack)
                stack_sl = slice(
                    ceil(nstack / self.size * self.rank),
                    ceil(nstack / self.size * (self.rank + 1)))
                self.data_manager.reserve_data(self.mc_stack_dset_name, stack_sl)
                self.data_manager.write_data(self.mc_stack_dset_name, stack_sl, self.mc_stack[stack_sl])

            # MC interaction summary info
            self.data_manager.create_dset(self.mc_events_dset_name, dtype=self.mc_events.dtype)
            ninter = len(self.mc_events)
            inter_sl = slice(
                ceil(ninter / self.size * self.rank),
                ceil(ninter / self.size * (self.rank + 1)))
            self.data_manager.reserve_data(self.mc_events_dset_name, inter_sl)
            self.data_manager.write_data(self.mc_events_dset_name, inter_sl,
                                         self.mc_events[inter_sl])

            # edep-sim energy segments/deposits
            self.data_manager.create_dset(self.mc_tracks_dset_name, dtype=self.mc_tracks.dtype)
            ntracks = len(self.mc_tracks)
            # track_sl = slice(
                # min(ntracks, ceil(ntracks / self.size) * self.rank),
                # min(ntracks, ceil(ntracks / self.size) * (self.rank + 1)))
            track_sl = slice(
                ceil(ntracks / self.size * self.rank),
                ceil(ntracks / self.size * (self.rank + 1)))
            self.data_manager.reserve_data(self.mc_tracks_dset_name, track_sl)
            self.data_manager.write_data(
                self.mc_tracks_dset_name, track_sl,
                self.mc_tracks[track_sl])

            # edep-sim trajectories
            self.data_manager.create_dset(self.mc_trajectories_dset_name, dtype=self.mc_trajectories.dtype)
            ntraj = len(self.mc_trajectories)
            # traj_sl = slice(
                # min(ntracks, ceil(ntraj / self.size * self.rank)),
                # min(ntraj, ceil(ntraj / self.size * (self.rank + 1))))
            traj_sl = slice(
                ceil(ntraj / self.size * self.rank),
                ceil(ntraj / self.size * (self.rank + 1)))
            self.data_manager.reserve_data(self.mc_trajectories_dset_name, traj_sl)
            self.data_manager.write_data(
                self.mc_trajectories_dset_name, traj_sl,
                self.mc_trajectories[traj_sl])

            # set up references
            self.data_manager.create_ref(self.packets_dset_name, self.mc_tracks_dset_name)
            self.data_manager.create_ref(self.mc_trajectories_dset_name, self.mc_tracks_dset_name)
            self.data_manager.create_ref(self.raw_event_dset_name, self.mc_events_dset_name)
            self.data_manager.create_ref(self.mc_events_dset_name, self.mc_trajectories_dset_name)
            self.data_manager.create_ref(self.mc_events_dset_name, self.mc_tracks_dset_name)
            if self.is_mc_neutrino:
                self.data_manager.create_ref(self.mc_events_dset_name, self.mc_stack_dset_name)
                self.data_manager.create_ref(self.mc_stack_dset_name, self.mc_trajectories_dset_name)

            # create references between trajectories and tracks
            # eventID --> vertexID for latest production files
            if self.is_mc_neutrino:
                stack_evid = self.mc_stack[self.vertex_id_name][:]
            intr_evid = self.mc_events[self.vertex_id_name][:]
            traj_evid = self.mc_trajectories[self.vertex_id_name][:]
            tracks_evid = self.mc_tracks[self.vertex_id_name][:]
            evs, ev_traj_start, ev_track_start = np.intersect1d(
                traj_evid, tracks_evid, return_indices=True)
            evs, ev_traj_end, ev_track_end = np.intersect1d(
                traj_evid[::-1], tracks_evid[::-1], return_indices=True)
            ev_traj_end = len(self.mc_trajectories[self.vertex_id_name]) - ev_traj_end
            ev_track_end = len(self.mc_tracks[self.vertex_id_name]) - ev_track_end
            truth_slice = slice(
                ceil(len(evs) / self.size * self.rank),
                ceil(len(evs) / self.size * (self.rank + 1)))

            if self.is_mc_neutrino:
                stack_trackid = self.mc_stack[self.traj_id_name][:]
            traj_trackid = self.mc_trajectories[self.traj_id_name][:]
            tracks_trackid = self.mc_tracks[self.traj_id_name][:]
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

                    # Create refs for traj --> tracks
                    ref = np.argwhere((traj_trackid_block == track_trackid_block) &
                                      (traj_evid_block == track_evid_block))
                    ref[:, 0] += traj_start
                    ref[:, 1] += track_start
                    self.data_manager.write_ref(self.mc_trajectories_dset_name, self.mc_tracks_dset_name, ref)

                    # Create refs for interactions --> traj
                    intr_evid_block = np.expand_dims(intr_evid[:], 0) # Might need to modify for MPI running
                    ref = np.argwhere((ev == intr_evid_block) & (ev == traj_evid_block))
                    ref[:, 0] += traj_start
                    ref[:, 1] += 0 #i + inter_sl.start # Might need to modify for MPI running
                    self.data_manager.write_ref(self.mc_trajectories_dset_name, self.mc_events_dset_name, ref)

                    # Create refs for interactions --> tracks
                    intr_evid_block = np.expand_dims(intr_evid[:], -1) # Might need to modify for MPI running
                    ref = np.argwhere((ev == track_evid_block) & (ev == intr_evid_block))
                    ref[:, 0] += 0 #i + inter_sl.start # Might need to modify for MPI running
                    ref[:, 1] += track_start
                    self.data_manager.write_ref(self.mc_events_dset_name, self.mc_tracks_dset_name, ref)

                    if self.is_mc_neutrino:
                        # Create refs for interactions --> generator particle stack
                        stack_evid_block = np.expand_dims(stack_evid[:], 0) # Might need to modify for MPI running
                        ref = np.argwhere((ev == intr_evid_block) & (ev == stack_evid_block))
                        # ref[:, 0] += 0 # Placeholders for now.
                        # ref[:, 1] += 0 # This extra offset might be needed for future MPI running
                        self.data_manager.write_ref(self.mc_events_dset_name, self.mc_stack_dset_name, ref)

                        # Create refs for generator particle stack --> traj
                        stack_trackid_block = np.expand_dims(stack_trackid[:], -1) # Might need to modify for MPI running
                        traj_trackid_block = np.transpose(traj_trackid_block)
                        stack_evid_block = np.transpose(stack_evid_block) # Might need to modify for MPI running
                        traj_evid_block = np.transpose(traj_evid_block)
                        ref = np.argwhere((stack_trackid_block == traj_trackid_block) & (stack_evid_block == traj_evid_block))
                        ref[:, 0] += 0 # Might need to modify for MPI running
                        ref[:, 1] += traj_start
                        self.data_manager.write_ref(self.mc_stack_dset_name, self.mc_trajectories_dset_name, ref)
                else:
                    self.data_manager.write_ref(self.mc_trajectories_dset_name, self.mc_tracks_dset_name, np.empty((0,2)))
                    self.data_manager.write_ref(self.mc_trajectories_dset_name, self.mc_events_dset_name, np.empty((0,2)))
                    self.data_manager.write_ref(self.mc_events_dset_name, self.mc_tracks_dset_name, np.empty((0,2)))
                    if self.is_mc_neutrino:
                        self.data_manager.write_ref(self.mc_events_dset_name, self.mc_stack_dset_name, np.empty((0,2)))
                        self.data_manager.write_ref(self.mc_stack_dset_name, self.mc_trajectories_dset_name, np.empty((0,2)))

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
        events, event_unix_ts, event_mc_assn = [], [], None
        eb_rv = list(self.event_builder.build_events(packet_buffer, unix_ts, mc_assn))
        if eb_rv:
            events, event_unix_ts = eb_rv[:2]
            if self.is_mc:
                event_mc_assn = eb_rv[2]

        # apply nhit cut
        nhit_filtered = list(filter(lambda x: (len(x[0]) >= self.nhit_cut) and (len(x[0]) <= self.nhit_limit), zip(events, event_unix_ts)))
        if self.is_mc:
            mc_assn_filtered = list(filter(lambda x: (len(x) >= self.nhit_cut) and (len(x) <= self.nhit_limit), event_mc_assn))

        if len(nhit_filtered):
            events, event_unix_ts = zip(*nhit_filtered)
            if self.is_mc:
                event_mc_assn = mc_assn_filtered
        else:
            events, event_unix_ts = list(), list()
            if self.is_mc:
                event_mc_assn = list()
        nevents = len(events)
        if not nevents:
            return H5FlowGenerator.EMPTY

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

            # packet -> mc_packet_assn
            ref = np.c_[packets_idcs.ravel(), packets_idcs.ravel()]
            sl = self.data_manager.reserve_data(self.mc_packet_fraction_dset_name, len(ref))
            self.data_manager.write_data(self.mc_packet_fraction_dset_name, sl, np.concatenate(event_mc_assn))
            self.data_manager.write_ref(self.packets_dset_name, self.mc_packet_fraction_dset_name, ref)

            # packet -> segment
            mc_assn = (np.concatenate(event_mc_assn, axis=0)
                       if len(event_mc_assn) else np.full((0,), -1, dtype=self.mc_assn.dtype))
            id_field = 'segment_ids' if 'segment_ids' in mc_assn.dtype.fields else 'track_ids'
            mc_assn_mask = (mc_assn[id_field] == -1) | (mc_assn['fraction'] == 0.)
            event_tracks = ma.array(mc_assn[id_field], mask=mc_assn_mask)

            packets_idcs = np.broadcast_to(packets_idcs[:, np.newaxis], event_tracks.shape)
            packets_idcs = packets_idcs.ravel()[~event_tracks.mask.ravel()]

            segment_id_idcs = {segment_id:idcs for idcs,segment_id in enumerate(self.mc_tracks['segment_id'])}
            segment_idcs = [segment_id_idcs[seg_id] for seg_id in event_tracks.ravel()[~event_tracks.mask.ravel()]]

            if len(packets_idcs) != len(segment_idcs):
                raise Exception("packets_idcs and segment_idcs do not match in size!")
            ref = np.c_[packets_idcs, segment_idcs]
            ref = np.unique(ref, axis=0) if len(ref) else ref
            self.data_manager.write_ref(self.packets_dset_name, self.mc_tracks_dset_name, ref)

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

