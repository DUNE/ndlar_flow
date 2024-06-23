import numpy as np
import numpy.lib.recfunctions as rfn
from collections import defaultdict
import json

from h5flow.core import H5FlowStage, resources


class RawHitBuilder(H5FlowStage):
    '''
        Converts larpix data packets into hits - assigns geometric properties,
        filters by packet type, and performs the conversion from ADC -> mV above
        pedestal.

        The external data files used for ``pedestal_file`` and
        ``configuration_file`` are searched for in the current working
        directory, if the paths are not specified as global paths.

        Parameters:
         - ``hits_dset_name`` : ``str``, required, output dataset path
         - ``packets_dset_name`` : ``str``, required, input dataset path for packets
         - ``packets_index_name`` : ``str``, required, input dataset path for packet index (defaults to ``{packets_dset_name}_index'``)
         - ``ts_dset_name`` : ``str``, required, input dataset path for clock-corrected packet timestamps
         - ``pedestal_file`` : ``str``, optional, path to a pedestal json file
         - ``configuration_file`` : ``str``, optional, path to a vref/vcm config json file

        ``packets_dset_name``, ``ts_dset_name``, and ``packets_index_name`` are required in
        the data cache. ``packets_index_name`` must point to the index for ``packets_dset_name``.

        Requires RunData resource in workflow.

        Example config::

            raw_hit_builder:
                classname: RawHitBuilder
                requires:
                    - 'charge/packets'
                    - 'charge/packets_corr_ts'
                    - name: 'charge/packets_index'
                      path: 'charge/packets'
                      index_only: True
                params:
                    hits_dset_name: 'charge/raw_hits'
                    packets_dset_name: 'charge/packets'
                    packets_index_name: 'charge/packets_index'
                    ts_dset_name: 'charge/packets_corr_ts'
                    pedestal_file: 'datalog_2021_04_02_19_00_46_CESTevd_ped.json'
                    configuration_file: 'evd_config_21-03-31_12-36-13.json'

        ``raw_hits`` datatype::

            x_pix          f8, pixel x location [cm]
            y_pix          f8, pixel y location [cm]
            z_pix          f8, pixel z location [cm]
            ts_pps         u8, PPS packet timestamp [ticks]
            ADC            u1, hit charge [ADC]
            iogroup        u1, io group id
            iochannel      u1, io channel id

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

    #hits_dtype = np.dtype([
    #    ('id', 'u4'),
    #    ('px', 'f8'),
    #    ('py', 'f8'),
    #    ('ts', 'f8'),
    #    ('ts_raw', 'u8'),
    #    ('q', 'f8'),
    #    ('iogroup', 'u1'), ('iochannel', 'u1'), ('chipid', 'u1'), ('channelid', 'u1'),
    #    ('geom', 'i8')
    #])

    hits_dtype = np.dtype([
        ('id', 'u4'),
        ('x_pix', 'f8'),
        ('y_pix', 'f8'),
        ('z_pix', 'f8'),
        ('ts_pps', 'u8'),
        ('ADC', 'u1'),
        ('iogroup', 'u1'),
        ('iochannel', 'u1')
    ])

    def __init__(self, **params):
        super(RawHitBuilder, self).__init__(**params)

        self.hits_dset_name = params.get('hits_dset_name')
        self.packets_dset_name = params.get('packets_dset_name')
        self.packets_index_name = params.get('packets_index_name', self.packets_dset_name + '_index')
        self.ts_dset_name = params.get('ts_dset_name')
        self.pedestal_file = params.get('pedestal_file', '')
        self.configuration_file = params.get('configuration_file', '')

    def init(self, source_name):
        super(RawHitBuilder, self).init(source_name)
        self.load_pedestals()
        self.load_configurations()

        # save all config info
        self.data_manager.set_attrs(self.hits_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    source_dset=source_name,
                                    packets_dset=self.packets_dset_name,
                                    ts_dset=self.ts_dset_name,
                                    pedestal_file=self.pedestal_file,
                                    configuration_file=self.configuration_file
                                    )

        # then set up new datasets
        self.data_manager.create_dset(self.hits_dset_name, dtype=self.hits_dtype)
        self.data_manager.create_ref(source_name, self.hits_dset_name)
        self.data_manager.create_ref(self.hits_dset_name, self.packets_dset_name)

    def run(self, source_name, source_slice, cache):
        super(RawHitBuilder, self).run(source_name, source_slice, cache)
        packets_data = cache[self.packets_dset_name]
        packets_index = cache[self.packets_index_name]
        ts_data = cache[self.ts_dset_name].reshape(packets_data.shape)

        mask = ~rfn.structured_to_unstructured(packets_data.mask).any(axis=-1)

        # get event boundaries
        if np.count_nonzero(mask):
            mask = (packets_data['packet_type'] == 0) & mask
            n = np.count_nonzero(mask)
            packets_arr = packets_data.data[mask]
            ts_arr = ts_data.data[mask]
            index_arr = packets_index.data[mask]
        else:
            n = 0
            index_arr = np.zeros((0,), dtype=packets_index.dtype)

        # reserve new data
        raw_hits_slice = self.data_manager.reserve_data(self.hits_dset_name, n)

        # convert to hits array
        raw_hits_arr = np.zeros((n,), dtype=self.hits_dtype)
        if n:
            zy = resources['Geometry'].pixel_coordinates_2D[packets_arr['io_group'],
                                                packets_arr['io_channel'], packets_arr['chip_id'], packets_arr['channel_id']]
            tile_id = resources['Geometry'].tile_id[packets_arr['io_group'],packets_arr['io_channel']]
            print(min(tile_id), max(tile_id))
            x = resources['Geometry'].anode_drift_coordinate[(tile_id,)]

            raw_hits_arr['id'] = raw_hits_slice.start + np.arange(n, dtype=int)
            raw_hits_arr['x_pix'] = x
            raw_hits_arr['y_pix'] = zy[:,1]
            raw_hits_arr['z_pix'] = zy[:,0]
            raw_hits_arr['ts_pps'] = ts_arr['ts']
            raw_hits_arr['ADC'] = packets_arr['dataword']
            raw_hits_arr['iogroup'] = packets_arr['io_group']
            raw_hits_arr['iochannel'] = packets_arr['io_channel']

        # write
        self.data_manager.write_data(self.hits_dset_name, raw_hits_slice, raw_hits_arr)

        # save references
        ev_id = np.broadcast_to(np.expand_dims(np.r_[source_slice], axis=-1), packets_data.shape)
        # event -> hit
        ref = np.c_[ev_id[mask], raw_hits_arr['id']]
        self.data_manager.write_ref(source_name, self.hits_dset_name, ref)

        # hit -> packet
        ref = np.c_[raw_hits_arr['id'], index_arr]
        self.data_manager.write_ref(self.hits_dset_name, self.packets_dset_name, ref)

    @staticmethod
    def charge_from_dataword(dw, vref, vcm, ped):
        return dw / 256. * (vref - vcm) + vcm - ped

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
