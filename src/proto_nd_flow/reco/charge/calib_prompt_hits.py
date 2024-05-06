import numpy as np
import numpy.lib.recfunctions as rfn
from collections import defaultdict
import json

from h5flow.core import H5FlowStage, resources
import proto_nd_flow.util.units as units


class CalibHitBuilder(H5FlowStage):
    '''
        Converts larpix data packets into hits - assigns geometric properties,
        filters by packet type, and performs the conversion from ADC -> mV above
        pedestal.

        The external data files used for ``pedestal_file`` and
        ``configuration_file`` are searched for in the current working
        directory, if the paths are not specified as global paths.

        Parameters:
         - ``hits_dset_name`` : ``str``, required, output dataset path
         - ``events_dset_name`` : ``str``, required, input dataset path for high-level events
         - ``packets_dset_name`` : ``str``, required, input dataset path for packets
         - ``packets_index_name`` : ``str``, required, input dataset path for packet index (defaults to ``{packets_dset_name}_index'``)
         - ``ts_dset_name`` : ``str``, required, input dataset path for clock-corrected packet timestamps
         - ``pedestal_file`` : ``str``, optional, path to a pedestal json file
         - ``configuration_file`` : ``str``, optional, path to a vref/vcm config json file

        ``packets_dset_name``, ``ts_dset_name``, and ``packets_index_name`` are required in
        the data cache. ``packets_index_name`` must point to the index for ``packets_dset_name``.

        Requires RunData resource in workflow.

        Example config::

            calib_hit_builder:
                classname: CalibHitBuilder
                requires:
                    - 'charge/packets'
                    - 'charge/raw_hits'
                    - 'combined/t0'
                    - name: 'charge/packets_index'
                      path: 'charge/packets'
                      index_only: True
                params:
                    hits_dset_name: 'charge/raw_hits'
                    events_dset_name: 'charge/events'
                    packets_dset_name: 'charge/packets'
                    packets_index_name: 'charge/packets_index'
                    t0_dset_name: 'combined/t0'
                    pedestal_file: 'datalog_2021_04_02_19_00_46_CESTevd_ped.json'
                    configuration_file: 'evd_config_21-03-31_12-36-13.json'

        ``calib_prompt_hits`` datatype::

            x              f8, pixel x location [cm]
            y              f8, pixel y location [cm]
            z              f8, pixel z location [cm]
            t_drift        u8, drift time [ticks???]
            ts_pps         f8, PPS packet timestamp [ns]
            io_group       u8, io group ID (PACMAN number)
            io_channel     u8, io channel ID (related to PACMAN number & PACMAN UART Number)
            Q              f8, hit charge [ke-]
            E              f8, hit energy [MeV]

    '''
    class_version = '1.0.0'

    #: ASIC ADC configuration lookup table
    configuration = defaultdict(lambda: dict(
        vref_mv=1300,
        vcm_mv=288
    ))

    #: pixel pedestal value
    pedestal = defaultdict(lambda: dict(
        pedestal_mv=580
    ))

    calib_hits_dtype = np.dtype([
        ('id', 'u4'),
        ('x', 'f8'),
        ('y', 'f8'),
        ('z', 'f8'),
        ('t_drift', 'f8'),
        ('ts_pps', 'u8'),
        ('io_group', 'u8'),
        ('io_channel', 'u8'),
        ('Q', 'f8'),
        ('E', 'f8')
    ])

    def __init__(self, **params):
        super(CalibHitBuilder, self).__init__(**params)

        self.events_dset_name = params.get('events_dset_name')
        self.raw_hits_dset_name = params.get('raw_hits_dset_name')
        self.calib_hits_dset_name = params.get('calib_hits_dset_name')
        self.mc_hit_frac_dset_name = params.get('mc_hit_frac_dset_name')
        self.packets_dset_name = params.get('packets_dset_name')
        self.packets_index_name = params.get('packets_index_name', self.packets_dset_name + '_index')
        self.t0_dset_name = params.get('t0_dset_name')
        self.pedestal_file = params.get('pedestal_file', '')
        self.configuration_file = params.get('configuration_file', '')
        self.max_contrib_segments = params.get('max_contrib_segments','')

    def init(self, source_name):
        super(CalibHitBuilder, self).init(source_name)
        self.load_pedestals()
        self.load_configurations()

        self.hit_frac_dtype = np.dtype([
            ('fraction', f'({self.max_contrib_segments},)f8'),
            ('segment_id', f'({self.max_contrib_segments},)u8')
        ])

        # save all config info
        self.data_manager.set_attrs(self.calib_hits_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    source_dset=source_name,
                                    packets_dset=self.packets_dset_name,
                                    t0_dset=self.t0_dset_name,
                                    pedestal_file=self.pedestal_file,
                                    configuration_file=self.configuration_file
                                    )

        # then set up new datasets
        self.data_manager.create_dset(self.calib_hits_dset_name, dtype=self.calib_hits_dtype)
        self.data_manager.create_dset(self.mc_hit_frac_dset_name, dtype=self.hit_frac_dtype)
        self.data_manager.create_ref(source_name, self.calib_hits_dset_name)
        self.data_manager.create_ref(self.calib_hits_dset_name, self.packets_dset_name)
        self.data_manager.create_ref(self.events_dset_name, self.calib_hits_dset_name)
        self.data_manager.create_ref(self.calib_hits_dset_name, self.mc_hit_frac_dset_name)

    def run(self, source_name, source_slice, cache):
        super(CalibHitBuilder, self).run(source_name, source_slice, cache)
        events_data = cache[self.events_dset_name]
        packets_data = cache[self.packets_dset_name]
        packets_index = cache[self.packets_index_name]
        packet_frac_bt = cache['packet_frac_backtrack']
        packet_seg_bt = cache['packet_seg_backtrack']
        t0_data = cache[self.t0_dset_name]
        raw_hits = cache[self.raw_hits_dset_name]

        mask = ~rfn.structured_to_unstructured(packets_data.mask).any(axis=-1)
        rh_mask = ~rfn.structured_to_unstructured(raw_hits.mask).any(axis=-1)

        # get event boundaries
        if np.count_nonzero(mask):
            raw_hits_arr = raw_hits.data[rh_mask]
            mask = (packets_data['packet_type'] == 0) & mask
            n = np.count_nonzero(mask)
            packets_arr = packets_data.data[mask]
            packet_frac_bt_arr = packet_frac_bt.data[mask]
            packet_seg_bt_arr = packet_seg_bt.data[mask]
            index_arr = packets_index.data[mask]
        else:
            n = 0
            index_arr = np.zeros((0,), dtype=packets_index.dtype)

        has_mc_truth = packet_seg_bt is not None

        # reserve new data
        calib_hits_slice = self.data_manager.reserve_data(self.calib_hits_dset_name, n)
        if has_mc_truth:
            hit_bt_slice = self.data_manager.reserve_data(self.mc_hit_frac_dset_name,n)

        # convert to hits array
        calib_hits_arr = np.zeros((n,), dtype=self.calib_hits_dtype)
        if n:

            # For now, use the event time as the t0 for each hit
            # this should eventually be improved to match each hit
            # to the correct light trigger and use that timing.
            # Given optical pileup, we can have multiple triggers
            # per event. There is probably a cleaner way to use h5flow
            # associations, but for now this will do...
            hit_t0 = np.full(len(raw_hits_arr['ts_pps']),0)

            if not len(raw_hits) == len(t0_data['ts']):
                print("event dividers for raw hits and t0 inconsistent")
                exit
            else:
                first_index = 0
                for t0_it, t0 in enumerate(t0_data['ts']):
                    n_masked = np.ma.count_masked(raw_hits[t0_it]['id'],axis=0)
                    n_not_masked = len(raw_hits[t0_it]['id']) - n_masked
                    last_index = first_index + n_not_masked
                    print(t0_it,n_not_masked,first_index,first_index+n_not_masked,t0)
                    hit_t0[first_index:last_index] = np.full(n_not_masked,t0)
                    first_index += n_not_masked

            drift_t = raw_hits_arr['ts_pps'] - hit_t0

            drift_d = drift_t * (resources['LArData'].v_drift * resources['RunData'].crs_ticks) / units.cm # convert mm -> cm
            x = resources['Geometry'].get_drift_coordinate(packets_arr['io_group'],packets_arr['io_channel'],drift_d)

            zy = resources['Geometry'].pixel_coordinates_2D[packets_arr['io_group'],
                                                packets_arr['io_channel'], packets_arr['chip_id'], packets_arr['channel_id']]
            tile_id = resources['Geometry'].tile_id[packets_arr['io_group'],packets_arr['io_channel']]
            hit_uniqueid = (((packets_arr['io_group'].astype(int)) * 100000
                             + packets_arr['io_channel'].astype(int)) * 1000
                            + packets_arr['chip_id'].astype(int)) * 64 \
                + packets_arr['channel_id'].astype(int)
            hit_uniqueid_str = hit_uniqueid.astype(str)
            vref = np.array(
                [self.configuration[unique_id]['vref_mv'] for unique_id in hit_uniqueid_str])
            vcm = np.array([self.configuration[unique_id]['vcm_mv']
                            for unique_id in hit_uniqueid_str])
            ped = np.array([self.pedestal[unique_id]['pedestal_mv']
                            for unique_id in hit_uniqueid_str])
            calib_hits_arr['id'] = calib_hits_slice.start + np.arange(n, dtype=int)
            calib_hits_arr['x'] = x
            calib_hits_arr['y'] = zy[:,1]
            calib_hits_arr['z'] = zy[:,0]
            calib_hits_arr['ts_pps'] = raw_hits_arr['ts_pps']
            calib_hits_arr['t_drift'] = drift_t
            calib_hits_arr['io_group'] = packets_arr['io_group']
            calib_hits_arr['io_channel'] = packets_arr['io_channel']
            calib_hits_arr['Q'] = self.charge_from_dataword(packets_arr['dataword'],vref,vcm,ped)
            #!!! hardcoding W_ion, R=0.7, and not accounting for finite electron lifetime
            calib_hits_arr['E'] = self.charge_from_dataword(packets_arr['dataword'],vref,vcm,ped) * 23.6e-3 * 0.7

            # create truth-level backtracking dataset
            if has_mc_truth:
                back_track = np.full(shape=packets_arr.shape,fill_value=0.,dtype=self.hit_frac_dtype)
                for hit_it, pack in np.ndenumerate(packets_arr):
                    back_track[hit_it]['fraction'] = packet_frac_bt_arr[hit_it] 
                    back_track[hit_it]['segment_id'] = packet_seg_bt_arr[hit_it]['segment_id']

        # if back tracking information was available, write the merged back tracking
        # dataset to file 
        if has_mc_truth:
            self.data_manager.write_data(self.mc_hit_frac_dset_name, hit_bt_slice, back_track)

        # write
        self.data_manager.write_data(self.calib_hits_dset_name, calib_hits_slice, calib_hits_arr)

        # save references
        raw_ev_id = np.broadcast_to(np.expand_dims(np.r_[source_slice], axis=-1), packets_data.shape)
        ref = np.c_[raw_ev_id[mask], calib_hits_arr['id']]
        # raw_event -> hit
        self.data_manager.write_ref(source_name, self.calib_hits_dset_name, ref)

        # event -> hit
        self.data_manager.write_ref(self.events_dset_name, self.calib_hits_dset_name, ref)

        # hit -> packet
        ref = np.c_[calib_hits_arr['id'], index_arr]
        self.data_manager.write_ref(self.calib_hits_dset_name, self.packets_dset_name, ref)

        # hit -> backtracking
        if has_mc_truth:
            self.data_manager.write_ref(self.calib_hits_dset_name,self.mc_hit_frac_dset_name,np.c_[calib_hits_arr['id'],calib_hits_arr['id']])


    @staticmethod
    def charge_from_dataword(dw, vref, vcm, ped):
        return (dw / 256. * (vref - vcm) + vcm - ped) / 4. # hardcoding 4 mV/ke- conv.

    def load_pedestals(self):
        if self.pedestal_file != '' and not resources['RunData'].is_mc:
            with open(self.pedestal_file, 'r') as infile:
                for key, value in json.load(infile).items():
                    self.pedestal[key] = value

    def load_configurations(self):
        if self.configuration_file != '' and not resources['RunData'].is_mc:
            with open(self.configuration_file, 'r') as infile:
                for key, value in json.load(infile).items():
                    self.configuration[key] = value
