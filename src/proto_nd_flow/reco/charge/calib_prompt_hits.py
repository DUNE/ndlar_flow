import numpy as np
import numpy.lib.recfunctions as rfn
from collections import defaultdict
import json

from h5flow.core import H5FlowStage, resources
import proto_nd_flow.util.units as units


class CalibHitBuilder(H5FlowStage):
    '''
        Converts larpix data packets into calibrated hits - assigns geometric
        properties, filters by packet type, and performs the conversion from ADC
        -> mV above pedestal.

        The external data files used for ``pedestal_file`` and
        ``configuration_file`` are searched for in the current working
        directory, if the paths are not specified as global paths.

        Parameters:
         - ``hits_dset_name`` : ``str``, required, output dataset path
         - ``events_dset_name`` : ``str``, required, input dataset path for high-level events
         - ``packets_dset_name`` : ``str``, required, input dataset path for packets
         - ``packets_index_name`` : ``str``, required, input dataset path for packet index (defaults to ``{packets_dset_name}_index'``)
         - ``ts_dset_name`` : ``str``, required, input dataset path for clock-corrected packet timestamps

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
                    
        ``calib_prompt_hits`` datatype::

            x              f8, pixel x location [cm]
            y              f8, pixel y location [cm]
            z              f8, pixel z location [cm]
            t_drift        u8, drift time [tick = 100ns]
            ts_pps         f8, PPS packet timestamp [tick = 100ns]
            io_group       u8, io group ID (PACMAN number)
            io_channel     u8, io channel ID (related to PACMAN number & PACMAN UART Number)
            chip_id        u8, chip_id on tile 
            channel_id     u8, channel_id on single chip (0-63)
            Q_raw          f8, hit charge [ke-] (uncalibrated for ADC droop)
            Q              f8, calibrated hit charge with ADC droop corrections
            E              f8, hit energy [MeV]

    '''
    class_version = '1.0.0'

    calib_hits_dtype = np.dtype([
        ('id', 'u4'),
        ('x', 'f8'),
        ('y', 'f8'),
        ('z', 'f8'),
        ('t_drift', 'f8'),
        ('ts_pps', 'u8'),
        ('io_group', 'u8'),
        ('io_channel', 'u8'),
        ('chip_id', 'u8'),
        ('channel_id', 'u8'),
        ('Q_raw', 'f8'),
        ('Q', 'f8'),
        ('E', 'f8'),
        ('is_disabled', '?')
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
        self.adc_droop_calibration = params.get('adc_droop_calibration', False)
        self.hit_ref = params.get('hit_ref', True)

    def init(self, source_name):
        super(CalibHitBuilder, self).init(source_name)

    def run(self, source_name, source_slice, cache):
        super(CalibHitBuilder, self).run(source_name, source_slice, cache)
        events_data = cache[self.events_dset_name]
        packets_data = cache[self.packets_dset_name]
        packets_index = cache[self.packets_index_name]
        
        if resources['RunData'].is_mc:
            packet_frac_bt = cache['packet_frac_backtrack']
            #packet_seg_bt = cache['packet_seg_backtrack']

        t0_data = cache[self.t0_dset_name]
        raw_hits = cache[self.raw_hits_dset_name]

        has_mc_truth = resources['RunData'].is_mc and (packet_frac_bt is not None)

        mask = ~rfn.structured_to_unstructured(packets_data.mask).any(axis=-1)
        rh_mask = ~rfn.structured_to_unstructured(raw_hits.mask).any(axis=-1)

        # get event boundaries
        if np.count_nonzero(mask):
            raw_hits_arr = raw_hits.data[rh_mask]
            mask = (packets_data['packet_type'] == 0) & mask
            n = np.count_nonzero(mask)
            packets_arr = packets_data.data[mask]
            if resources['RunData'].is_mc:
                packet_frac_bt_arr = packet_frac_bt.data[mask]
                packet_frac_bt_arr = np.concatenate(packet_frac_bt_arr)
                #packet_seg_bt_arr = packet_seg_bt.data[mask]
            index_arr = packets_index.data[mask]
        else:
            n = 0
            index_arr = np.zeros((0,), dtype=packets_index.dtype)

        #if has_mc_truth and ('x_true_seg_t' not in self.calib_hits_dtype.fields):
        #    self.calib_hits_dtype = np.dtype(self.calib_hits_dtype.descr + [('x_true_seg_t', f'({packet_seg_bt.shape[-1]},)f8'), ('E_true_recomb_elife', f'({packet_seg_bt.shape[-1]},)f8')])

        # save all config info
        self.data_manager.set_attrs(self.calib_hits_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    source_dset=source_name,
                                    packets_dset=self.packets_dset_name,
                                    t0_dset=self.t0_dset_name,
                                    adc_droop_calibration=self.adc_droop_calibration
                                    )
        
        
        # then set up new datasets
        self.data_manager.create_dset(self.calib_hits_dset_name, dtype=self.calib_hits_dtype)
        if has_mc_truth:
            self.data_manager.create_dset(self.mc_hit_frac_dset_name, dtype=packet_frac_bt_arr.dtype)
        self.data_manager.create_ref(source_name, self.calib_hits_dset_name)
        self.data_manager.create_ref(self.events_dset_name, self.calib_hits_dset_name)
        if self.hit_ref:
            self.data_manager.create_ref(self.calib_hits_dset_name, self.packets_dset_name)
            if has_mc_truth:
                self.data_manager.create_ref(self.calib_hits_dset_name, self.mc_hit_frac_dset_name)

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
                    hit_t0[first_index:last_index] = np.full(n_not_masked,t0)
                    first_index += n_not_masked

            drift_t = raw_hits_arr['ts_pps'].astype('f8') - hit_t0 #ticks

            # If this event crosses a PPS reset, and if the t0 is post-reset,
            # then correct the drift time for pre-reset hits. Identify those
            # hits as those having an absurdly large (positive) drift_t.
            before_sync_mask = drift_t > 1e5
            # For those hits, subtract out the rollover period.
            drift_t[before_sync_mask] -= resources['RunData'].rollover_ticks
            # TODO: Instead of the nominal rollover_ticks, use the actual
            # timestamps of the SYNC. Need to wire in those SYNC timestamps and
            # the io group of each hit (or use the average SYNC timestamps; see
            # ave_pps_ts in timestamp_corrector.py)

            # Now handle the case where the t0 is pre-reset. The post-reset hits
            # will have absurdly negative drift_t.
            after_sync_mask = drift_t < -1e5
            # This time we add the rollover period instead of subtracting.
            drift_t[after_sync_mask] += resources['RunData'].rollover_ticks

            drift_d = drift_t * (resources['LArData'].v_drift * resources['RunData'].crs_ticks) / units.cm # convert mm -> cm
            x = resources['Geometry'].get_drift_coordinate(packets_arr['io_group'],packets_arr['io_channel'],drift_d)

            ## true drift position pair
            #if has_mc_truth:
            #    drift_t_true = packet_seg_bt_arr['t'] #us
            #    drift_d_true = drift_t_true * (resources['LArData'].v_drift) / units.cm # convert mm -> cm
            #    x_true_seg_t = resources['Geometry'].get_drift_coordinate(packets_arr['io_group'],packets_arr['io_channel'],drift_d_true)

            zy = resources['Geometry'].pixel_coordinates_2D[packets_arr['io_group'],
                                                packets_arr['io_channel'], packets_arr['chip_id'], packets_arr['channel_id']]
            if resources['RunData'].is_mc and np.isnan(zy).any():
                raise Exception("For simulation, all the channel keys should be valid. Please check your configuration.")

            calib_hits_arr['id'] = calib_hits_slice.start + np.arange(n, dtype=int)
            calib_hits_arr['x'] = x
            #if has_mc_truth:
            #    calib_hits_arr['x_true_seg_t'] = x_true_seg_t
            calib_hits_arr['y'] = zy[:,1]
            calib_hits_arr['z'] = zy[:,0]
            calib_hits_arr['ts_pps'] = raw_hits_arr['ts_pps']
            calib_hits_arr['t_drift'] = drift_t
            calib_hits_arr['io_group'] = packets_arr['io_group']
            calib_hits_arr['io_channel'] = packets_arr['io_channel']
            calib_hits_arr['chip_id'] = packets_arr['chip_id']
            calib_hits_arr['channel_id'] = packets_arr['channel_id']
            hits_charge = resources['Calibrate'].charge_from_dataword(packets_arr) # ke-
            calib_hits_arr['Q_raw'] = hits_charge # ke-
            if self.adc_droop_calibration: 
                hits_charge_calibrated = resources['Calibrate'].charge_from_dataword_corrected(packets_arr) # ke- 
                calib_hits_arr['Q'] = hits_charge_calibrated # ke-
            else:
                calib_hits_arr['Q'] = hits_charge

            #FIXME supply more realistic dEdx in the recombination; also apply measured electron lifetime
            calib_hits_arr['E'] = calib_hits_arr['Q'] * (1000 * units.e) / resources['LArData'].ionization_recombination(mode=2,dEdx=2) * (resources['LArData'].ionization_w / units.MeV) # MeV
            #if has_mc_truth:
            #    true_recomb = resources['LArData'].ionization_recombination(mode=2,dEdx=packet_seg_bt_arr['dEdx'])
            #    calib_hits_arr['E_true_recomb_elife'] = np.divide(hits_charge.reshape((hits_charge.shape[0],1)) * (1000 * units.e), true_recomb, out=np.zeros_like(true_recomb), where=true_recomb!=0) / resources['LArData'].charge_reduction_lifetime(t_drift=drift_t_true) * (resources['LArData'].ionization_w / units.MeV) # MeV

            mask_disabled_channels = np.isin(packets_arr[['io_group', 'io_channel', 'chip_id', 'channel_id']], resources['Geometry'].disabled_channels)

            mask_disabled_chips = np.isin(packets_arr[['io_group', 'io_channel', 'chip_id']], resources['Geometry'].disabled_chips)

            calib_hits_arr['is_disabled'] = mask_disabled_channels | mask_disabled_chips

        # if back tracking information was available, write the merged back tracking
        # dataset to file 
        if has_mc_truth:
            # make sure packets and packet backtracking match in numbers
            if packets_arr.shape[0] == packet_frac_bt_arr.shape[0]:
                self.data_manager.write_data(self.mc_hit_frac_dset_name, hit_bt_slice, packet_frac_bt_arr)
            else:
                raise Exception("The data packet and backtracking info do not match in size.")

        # write
        self.data_manager.write_data(self.calib_hits_dset_name, calib_hits_slice, calib_hits_arr)

        # save references
        raw_ev_id = np.broadcast_to(np.expand_dims(np.r_[source_slice], axis=-1), packets_data.shape)
        ref = np.c_[raw_ev_id[mask], calib_hits_arr['id']]
        # raw_event -> hit
        self.data_manager.write_ref(source_name, self.calib_hits_dset_name, ref)

        # event -> hit
        self.data_manager.write_ref(self.events_dset_name, self.calib_hits_dset_name, ref)

        if self.hit_ref:
            # hit -> packet
            ref = np.c_[calib_hits_arr['id'], index_arr]
            self.data_manager.write_ref(self.calib_hits_dset_name, self.packets_dset_name, ref)

            # hit -> backtracking
            if has_mc_truth:
                self.data_manager.write_ref(self.calib_hits_dset_name,self.mc_hit_frac_dset_name,np.c_[calib_hits_arr['id'],calib_hits_arr['id']])
