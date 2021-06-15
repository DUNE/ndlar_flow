import numpy as np
from collections import defaultdict
import logging
import yaml
import json

from h5flow.core import H5FlowStage

class HitBuilder(H5FlowStage):
    '''
        Converts larpix data packets into hits - assigns geometric properties,
        filters by packet type, and performs the conversion from ADC -> mV above
        pedestal.

        Parameters:
         - ``hits_dset_name`` : ``str``, required, output dataset path
         - ``packets_dset_name`` : ``str``, required, input dataset path for packets
         - ``ts_dset_name`` : ``str``, required, input dataset path for clock-corrected packet timestamps
         - ``geometry_file`` : ``str``, optional, path to a pixel geometry yaml file
         - ``pedestal_file`` : ``str``, optional, path to a pedestal json file
         - ``configuration_file`` : ``str``, optional, path to a vref/vcm config json file

        Both the ``packets_dset_name`` and ``ts_dset_name`` are required in
        the data cache.

        Example config::

            hit_builder:
                classname: HitBuilder
                requires:
                    - 'charge/packets'
                    - 'charge/packets_corr_ts'
                params:
                    hits_dset_name: 'charge/hits'
                    packets_dset_name: 'charge/packets'
                    ts_dset_name: 'charge/packets_corr_ts'
                    geometry_file: 'multi_tile_layout-2.1.16.yaml'
                    pedestal_file: 'datalog_2021_04_02_19_00_46_CESTevd_ped.json'
                    configuration_file: 'evd_config_21-03-31_12-36-13.json'

        ``hits`` datatype::

            id          u4, unique identifier per hit
            px          f8, pixel x location [mm]
            py          f8, pixel y location [mm]
            ts          f8, PPS timestamp (corrected for clock frequency) [ticks]
            ts_raw      u8, PPS timestamp [ticks]
            q           f8, hit charge [mV]
            iogroup     u1, io group id (PACMAN number)
            iochannel   u1, io channel id (PACMAN UART number)
            chipid      u1, chip id (ASIC number on PACMAN UART)
            channelid   u1, channel id (channel number on ASIC)
            geom        u1, unused

    '''
    class_version = '0.0.0'

    hits_dtype = np.dtype([
        ('id', 'u4'),
        ('px', 'f8'),
        ('py', 'f8'),
        ('ts', 'f8'),
        ('ts_raw', 'u8'),
        ('q', 'f8'),
        ('iogroup', 'u1'), ('iochannel', 'u1'), ('chipid', 'u1'), ('channelid', 'u1'),
        ('geom', 'i8')
        ])

    def __init__(self, **params):
        super(HitBuilder,self).__init__(**params)

        self.hits_dset_name = params.get('hits_dset_name')
        self.packets_dset_name = params.get('packets_dset_name')
        self.ts_dset_name = params.get('ts_dset_name')
        self.geometry_file = params.get('geometry_file','')
        self.pedestal_file = params.get('pedestal_file','')
        self.configuration_file = params.get('configuration_file','')

        self.load_geometry()
        self.load_pedestals()
        self.load_configurations()

    def init(self, source_name):
        # save all config info
        self.data_manager.set_attrs(self.hits_dset_name,
            classname=self.classname,
            class_version=self.class_version,
            source_dset=source_name,
            packets_dset=self.packets_dset_name,
            ts_dset=self.ts_dset_name,
            geometry_file=self.geometry_file,
            pedestal_file=self.pedestal_file,
            configuration_file=self.configuration_file
            )

        # then set up new datasets
        self.data_manager.create_dset(self.hits_dset_name, dtype=self.hits_dtype)
        self.data_manager.create_ref(source_name, self.hits_dset_name)

    def run(self, source_name, source_slice, cache):
        packets_data = cache[self.packets_dset_name]
        ts_data = cache[self.ts_dset_name]

        # get event boundaries
        if len(packets_data):
            masks = [packets['packet_type'] == 0 for packets in packets_data]
            lengths = [np.count_nonzero(mask) for mask in masks]
            n = int(np.sum(lengths))
            packets_arr = np.concatenate(packets_data, axis=0)
            ts_arr = np.concatenate(ts_data, axis=0)
            mask = np.concatenate(masks, axis=0)
            packets_arr = packets_arr[mask]
            ts_arr = ts_arr[mask]
        else:
            n = 0

        # reserve new data
        hits_slice = self.data_manager.reserve_data(self.hits_dset_name, n)

        # convert to hits array
        hits_arr = np.zeros((n,), dtype=self.hits_dtype)
        if n:
            hits_arr['id'] = hits_slice.start + np.arange(n)
            hits_arr['ts'] = ts_arr['ts']
            hits_arr['ts_raw'] = packets_arr['timestamp']
            hits_arr['iogroup'] = packets_arr['io_group']
            hits_arr['iochannel'] = packets_arr['io_channel']
            hits_arr['chipid'] = packets_arr['chip_id']
            hits_arr['channelid'] = packets_arr['channel_id']
            hit_uniqueid = (((packets_arr['io_group'].astype(int))*256
                             + packets_arr['io_channel'].astype(int))*256
                            + packets_arr['chip_id'].astype(int))*64 \
                + packets_arr['channel_id'].astype(int)
            hit_uniqueid_str = hit_uniqueid.astype(str)
            if self.is_multi_tile:
                xy = self.geometry[self.geometry_hash(packets_arr['io_group'], packets_arr['io_channel'], packets_arr['chip_id'], packets_arr['channel_id'])]
                # xy = np.array([self.geometry[(io_group, io_channel, chip_id, channel_id)]
                #                for io_group, io_channel, chip_id, channel_id in zip(packets_arr['io_group'], packets_arr['io_channel'], packets_arr['chip_id'], packets_arr['channel_id'])])
            else:
                xy = np.array([self.geometry[(
                    1, 1, (unique_id//64) % 256, unique_id % 64)] for unique_id in hit_uniqueid])

            vref = np.array(
                [self.configuration[unique_id]['vref_mv'] for unique_id in hit_uniqueid_str])
            vcm = np.array([self.configuration[unique_id]['vcm_mv']
                           for unique_id in hit_uniqueid_str])
            ped = np.array([self.pedestal[unique_id]['pedestal_mv']
                           for unique_id in hit_uniqueid_str])
            hits_arr['px'] = xy[:, 0]
            hits_arr['py'] = xy[:, 1]
            hits_arr['q'] = self.charge_from_dataword(packets_arr['dataword'], vref, vcm, ped)

        # write
        self.data_manager.write_data(self.hits_dset_name, hits_slice, hits_arr)

        # save references
        self.data_manager.reserve_ref(source_name, self.hits_dset_name, source_slice)
        ref = [slice(sum(lengths[:i])+hits_slice.start, sum(lengths[:i+1])+hits_slice.start) for i in range(len(packets_data))]
        self.data_manager.write_ref(source_name, self.hits_dset_name, source_slice, ref)

    @staticmethod
    def charge_from_dataword(dw, vref, vcm, ped):
        return dw/256. * (vref-vcm) + vcm - ped

    @staticmethod
    def _default_pxy():
        return (0., 0.)

    @staticmethod
    def _rotate_pixel(pixel_pos, tile_orientation):
        return pixel_pos[0]*tile_orientation[2], pixel_pos[1]*tile_orientation[1]

    def load_geometry(self):
        # self.geometry = defaultdict(self._default_pxy)
        self.io_group_io_channel_to_tile = defaultdict(int)
        self.is_multi_tile = False

        if self.geometry_file != '':
            with open(self.geometry_file) as gf:
                geometry_yaml = yaml.load(gf, Loader=yaml.FullLoader)

            if 'multitile_layout_version' in geometry_yaml.keys():
                pixel_pitch = geometry_yaml['pixel_pitch']
                self.is_multi_tile = True
                chip_channel_to_position = geometry_yaml['chip_channel_to_position']
                tile_orientations = geometry_yaml['tile_orientations']
                tile_positions = geometry_yaml['tile_positions']
                tpc_centers = geometry_yaml['tpc_centers']
                tile_indeces = geometry_yaml['tile_indeces']
                xs = np.array(list(chip_channel_to_position.values()))[
                    :, 0] * pixel_pitch
                ys = np.array(list(chip_channel_to_position.values()))[
                    :, 1] * pixel_pitch
                x_size = max(xs)-min(xs)+pixel_pitch
                y_size = max(ys)-min(ys)+pixel_pitch

                io_groups = [
                    geometry_yaml['tile_chip_to_io'][tile][chip]//1000
                    for tile in geometry_yaml['tile_chip_to_io']
                    for chip in geometry_yaml['tile_chip_to_io'][tile]
                    ]
                io_channels = [
                    geometry_yaml['tile_chip_to_io'][tile][chip]%1000
                    for tile in geometry_yaml['tile_chip_to_io']
                    for chip in geometry_yaml['tile_chip_to_io'][tile]
                    ]
                chip_ids = [
                    chip_channel//1000
                    for chip_channel in geometry_yaml['chip_channel_to_position']
                    ]
                channel_ids = [
                    chip_channel%1000
                    for chip_channel in geometry_yaml['chip_channel_to_position']
                    ]
                self.geometry_hash, max_hash = self.geometry_hash_factory(
                    (np.min(io_groups), np.max(io_groups)),
                    (np.min(io_channels), np.max(io_channels)),
                    (np.min(chip_ids), np.max(chip_ids)),
                    (np.min(channel_ids), np.max(channel_ids))
                    )
                self.geometry = np.zeros((max_hash+1, 2)) # pixel xy
                logging.debug(f'max geometry hash value: {max_hash}')

                for tile in geometry_yaml['tile_chip_to_io']:
                    tile_orientation = tile_orientations[tile]
                    for chip in geometry_yaml['tile_chip_to_io'][tile]:
                        io_group_io_channel = geometry_yaml['tile_chip_to_io'][tile][chip]
                        io_group = io_group_io_channel//1000
                        io_channel = io_group_io_channel % 1000
                        self.io_group_io_channel_to_tile[(
                            io_group, io_channel)] = tile

                    for chip_channel in geometry_yaml['chip_channel_to_position']:
                        chip = chip_channel // 1000
                        channel = chip_channel % 1000
                        try:
                            io_group_io_channel = geometry_yaml['tile_chip_to_io'][tile][chip]
                        except KeyError:
                            continue

                        io_group = io_group_io_channel // 1000
                        io_channel = io_group_io_channel % 1000
                        x = chip_channel_to_position[chip_channel][0] * \
                            pixel_pitch + pixel_pitch / 2 - x_size / 2
                        y = chip_channel_to_position[chip_channel][1] * \
                            pixel_pitch + pixel_pitch / 2 - y_size / 2

                        x, y = self._rotate_pixel((x, y), tile_orientation)
                        x += tile_positions[tile][2] + \
                            tpc_centers[tile_indeces[tile][1]][0]
                        y += tile_positions[tile][1] + \
                            tpc_centers[tile_indeces[tile][1]][1]

                        self.geometry[self.geometry_hash(np.array(io_group), np.array(io_channel), np.array(chip), np.array(channel))] = np.array([[x, y]])

            else:
                import larpixgeometry.layouts
                geo = larpixgeometry.layouts.load(
                    self.geometry_file)  # open geometry yaml file
                self.is_multi_tile = False
                for chip, pixels in geo['chips']:
                    for channel, pixel_id in enumerate(pixels):
                        if pixel_id is not None:
                            self.geometry[(1, 1, chip, channel)
                                          ] = geo['pixels'][pixel_id][1:3]

    def load_pedestals(self):
        self.pedestal = defaultdict(lambda: dict(
            pedestal_mv=580
        ))
        if self.pedestal_file != '':
            with open(self.pedestal_file, 'r') as infile:
                for key, value in json.load(infile).items():
                    self.pedestal[key] = value

    def load_configurations(self):
        self.configuration = defaultdict(lambda: dict(
            vref_mv=1300,
            vcm_mv=288
        ))
        if self.configuration_file != '':
            with open(self.configuration_file, 'r') as infile:
                for key, value in json.load(infile).items():
                    self.configuration[key] = value

    @staticmethod
    def geometry_hash_factory(min_max_io_group, min_max_io_channel, min_max_chip_id, min_max_channel_id):
        '''
            Generates a hashing function to translate (io_group, io_channel,
            chip_id, channel_id) into a unique index in an array

            :param min_max_...: ``tuple`` of (min value, max value)

            :returns: ``tuple`` of hashing function, max hash value

        '''
        len_io_group = np.diff(min_max_io_group)+1
        len_io_channel = np.diff(min_max_io_channel)+1
        len_chip_id = np.diff(min_max_chip_id)+1
        len_channel_id = np.diff(min_max_channel_id)+1

        max_hash = (((min_max_io_group[1]-min_max_io_group[0])*len_io_channel + min_max_io_channel[1]-min_max_io_channel[0])*len_chip_id + min_max_chip_id[1]-min_max_chip_id[0])*len_channel_id + min_max_channel_id[1]-min_max_channel_id[0] + 1
        def geometry_hash(io_group, io_channel, chip_id, channel_id):
            val = (((io_group-min_max_io_group[0])*len_io_channel + io_channel-min_max_io_channel[0])*len_chip_id + chip_id-min_max_chip_id[0])*len_channel_id + channel_id-min_max_channel_id[0] + 1
            val[val < 0] = 0
            val[val > max_hash] = 0
            return val
        return geometry_hash, int(max_hash)


