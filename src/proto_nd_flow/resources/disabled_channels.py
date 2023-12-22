import logging
import json
import numpy as np

from h5flow.core import H5FlowResource, resources

import proto_nd_flow.util.units as units
from proto_nd_flow.util.lut import LUT, write_lut, read_lut
from proto_nd_flow.util.compat import assert_compat_version


class DisabledChannels(H5FlowResource):
    '''
        Provides helper functions for identifying the positions of disabled
        channels.

        Requires ``RunData`` and ``Geometry`` resources within workflow.

        Parameters:
         - ``path``: ``str``, path to stored geometry data within file
         - ``disabled_channels_timestamp_dict``: ``str``, path to file mapping disabled channel file timestamps to data file timestamps
         - ``disabled_channels_file_dir``: ``str``, path to directory with time dependent disabled channel files
         - ``disabled_channels_common_filename``: ``str``, common beginning part of disabled channel file filenames
         - ``disabled_channels_file_format``: ``str``, file format for disabled channel files
         - ``missing_asic_list``: ``str``, path to file specifying disabled coordinates not in geometry file

        Provides:
         - ``disabled_pixel_coords``: 2D coordinates of all disabled channels
         - ``disabled_channel_lut``: lookup table to find if a pixel 2D coordinate is disabled
         - ``is_active()``: helper function for determining if a 3D point in in an active region

        Example usage::

            from h5flow.core import resources

            resources['DisabledChannels'].disabled_channel_lut[(io_group,z,y)]

        Example config::

            resources:
                - classname: DisabledChannels
                  params:
                    path: 'disabled_channels'
                    disabled_channels_timestamp_dict: 'data/module0_flow/module1_config_to_data_map.json'
                    disabled_channels_file_dir: '/global/cfs/cdirs/dune/www/data/Module1/TPC12/disabled/' 
                    disabled_channels_common_filename: 'disabled_channels_'
                    disabled_channels_file_format: '.json'
                    missing_asic_list: 'data/module1_flow/module1-network-absent-ASICs.json'

    '''
    class_version = '0.0.0'

    default_path = 'disabled_channels'

    def __init__(self, **params):
        super(DisabledChannels, self).__init__(**params)

        self.path = params.get('path', self.default_path)
        self.disabled_channels_timestamp_dict = params.get('disabled_channels_timestamp_dict', None)
        self.disabled_channels_file_dir = params.get('disabled_channels_file_dir', None)
        self.disabled_channels_common_filename = params.get('disabled_channels_common_filename', None)
        self.disabled_channels_file_format = params.get('disabled_channels_file_format', None)
        self.missing_asic_list = params.get('missing_asic_list', None)

    def init(self, source_name):
        super(DisabledChannels, self).init(source_name)

        # create group (if not present)
        self.data_manager.set_attrs(self.path)
        # load data (if present)
        self.data = dict(self.data_manager.get_attrs(self.path))

        if not self.data:
            # no data stored in file, generate it
            self._disabled_channel_lut, self._disabled_pixel_coords = self.load_disabled_channels_lut(
                self.disabled_channels_list, self.missing_asic_list)
            self.data['classname'] = self.classname
            self.data['class_version'] = self.class_version
            self.data['disabled_channels_list'] = (self.disabled_channels_list
                                                   if self.disabled_channels_list is not None
                                                   else '')
            self.data['missing_asic_list'] = (self.missing_asic_list
                                              if self.missing_asic_list is not None
                                              else '')
            self.data_manager.set_attrs(self.path, **self.data)
            zy_dtype = np.dtype([('z', self._disabled_pixel_coords.dtype), ('y', self._disabled_pixel_coords.dtype)])
            self.data_manager.create_dset(self.path + '/zy', dtype=zy_dtype)
            sl = self.data_manager.reserve_data(self.path + '/zy', slice(0, len(self._disabled_pixel_coords)))
            self.data_manager.write_data(self.path + '/zy', sl, self._disabled_pixel_coords.view(zy_dtype).ravel())

            write_lut(self.data_manager, self.path, self.disabled_channel_lut,
                      'lut')
        else:
            assert_compat_version(self.class_version, self.data['class_version'])

            self._disabled_channel_lut = read_lut(self.data_manager, self.path,
                                                  'lut')
            self._disabled_pixel_coords = np.c_[self.data_manager[self.path+'/zy/data']['z'], self.data_manager[self.path+'/zy/data']['y']]

        if self.rank == 0:
            logging.info(f'N disabled channels: {len(self.disabled_pixel_coords)}')
            logging.info(f'Disabled channel LUT size: '
                         f'{self.disabled_channel_lut.nbytes/1024/1024:0.02f}MB')

        self._pixel_pitch = resources['Geometry'].pixel_pitch
        self._pixel_z_hi_edge = np.sort(np.unique(resources['Geometry'].pixel_coordinates_2D.compress((0,)))) + self._pixel_pitch/2
        self._pixel_y_hi_edge = np.sort(np.unique(resources['Geometry'].pixel_coordinates_2D.compress((1,)))) + self._pixel_pitch/2
        io_group,io_channel,_,_ = resources['Geometry'].pixel_coordinates_2D.keys()
        tile_id = resources['Geometry'].tile_id[(io_group,io_channel)]
        self._anode_drift_coordinate, idx = np.unique(resources['Geometry'].anode_drift_coordinate[(tile_id,)], return_index=True)
        self._tpc_lookup = io_group[idx]

    @property
    def disabled_pixel_coords(self):
        return self._disabled_pixel_coords

    @property
    def disabled_channel_lut(self):
        return self._disabled_channel_lut

    def is_active(self, xyz):
        '''
        Lookup a specific position to determine if it would fall onto an active pixel

        :param xyz: 3D position ``shape: (..., 3)``

        :returns: boolean array with ``True == active``, ``shape: (...,)``

        '''
        pixel_z = self._pixel_z_hi_edge[np.clip(np.digitize(xyz[...,2], bins=self._pixel_z_hi_edge), 0, len(self._pixel_z_hi_edge)-1)] - self._pixel_pitch/2
        pixel_y = self._pixel_y_hi_edge[np.clip(np.digitize(xyz[...,1], bins=self._pixel_y_hi_edge), 0, len(self._pixel_y_hi_edge)-1)] - self._pixel_pitch/2
        tpc = self._tpc_lookup[np.argmin(np.abs(xyz[...,2:3] - self._anode_drift_coordinate.reshape([1,]*(xyz.ndim-1)+[-1])), axis=-1)]
        disabled = self.disabled_channel_lut[(tpc.astype(int), pixel_z.astype(int), pixel_y.astype(int))]
        return ~disabled

    @staticmethod
    def load_disabled_channels_lut(disabled_channels_list=None,
                                   missing_asic_list=None):
        '''
        Loads a disabled channels lookup-table from the json formatted filenames::

            disabled_channels_*.json 
            missing_asic_list

        ``disabled_channels_*.json`` files contain ``chip-key: [channel_id]`` pairs of
        disabled channels that are defined within the geometry, but should be
        considered as disabled. The ``Geometry`` resource is used to find the 2D
        locations of these pixels.

        ``missing_asic_list`` contains ``io_group: [[z,y], ...]`` pixel positions
        that should be considered as disabled regions.

        Creates a boolean lookup table with keys of
        ``(io_group, int(pixel_z), int(pixel_y))`` to determine if a given
        pixel position falls onto a disabled channel.

        :returns: ``tuple`` of boolean ``proto_nd_flow.util.lut.LUT`` and ``list`` of pixel 2D coordinates for each disabled channel

        '''
        io_group = list()
        zy = np.empty((0, 2))

        if disabled_channels_list is not None:
            # first load disabled channels list
            with open(disabled_channels_list, 'r') as fi:
                data = json.load(fi)

            # get disabled channels from file
            io_channel = list()
            chip_id = list()
            channel_id = list()
            for key in data:
                if key == 'All':
                    continue
                io_group_, io_channel_, chip_id_ = key.split('-')
                for ch in data[key]:
                    io_group.append(int(io_group_))
                    io_channel.append(int(io_channel_))
                    chip_id.append(int(chip_id_))
                    channel_id.append(int(ch))

                    if resources['Geometry'].network_agnostic == True:
                        # add additional entries for each io channel
                        n_io_channels_per_tile = resources['Geometry'].n_io_channels_per_tile
                        start_io_channel = ((io_channel_-1)//n_io_channels_per_tile)*n_io_channels_per_tile + 1
                        for io_channel in range(start_io_channel, start_io_channel+n_io_channels_per_tile):
                            io_group.append(int(io_group_))
                            io_channel.append(int(io_channel))
                            chip_id.append(int(chip_id_))
                            channel_id.append(int(ch))

            pixel_coordinates_2D = resources['Geometry'].pixel_coordinates_2D
            chip_key = (np.array(io_group), np.array(io_channel),
                        np.array(chip_id), np.array(channel_id))
            zy = pixel_coordinates_2D[chip_key]

        if missing_asic_list is not None:
            # then load missing asic pixels
            with open(missing_asic_list, 'r') as fi:
                data = json.load(fi)

            # add to lists
            for io_group_ in data:
                for asic in data[io_group_]:
                    io_group.append(int(io_group_))
                    zy = np.append(zy, np.array([asic]), axis=0)

        disable_channels_lut = LUT(bool,
                                   (min(io_group), max(io_group)),
                                   (min(zy[:, 0].astype(int)) - 1,
                                    max(zy[:, 0].astype(int)) + 1),
                                   (min(zy[:, 1].astype(int)) - 1,
                                    max(zy[:, 1].astype(int)) + 1),
                                   default=False)
        # apply a fudge factor to account for any rounding errors
        for dz in (+1, 0, -1):
            for dy in (+1, 0, -1):
                disable_channels_lut[(io_group, zy[:, 0].astype(int) + dz,
                                      zy[:, 1].astype(int) + dy)] = True

        return disable_channels_lut, zy


""" TODO: Add version of the following code/methods for time dependent lookup functionality for dis. ch. list

    disabled_channels_config = self.disabled_channels_timestamp_dict
    charge_filename = resources['RunData'].charge_filename

    def convert_ts_str_to_float(filename):

        filename = filename.strip('CET')
        file_ts_arr = np.array([float(x)/100 for x in filename.split('_') if x and float(x)/100 < 1.])
        file_ts_float = 0.
        len_file_ts_arr = len(file_ts_arr)
        for i in range(len_file_ts_arr): 
            file_ts_float += file_ts_arr[i]*10**(len_file_ts_arr*2 - i*2)

        return file_ts_float

    print("File timestamp float:", convert_ts_to_float(charge_filename))
    file_ts = convert_ts_to_float(charge_filename)

    def lookup_disabled_channel_file_ts(self, filename): # use self

        dc_file_ts = ''
        dc_config_file = open(self.disabled_channels_timestamp_dict)
        dc_config = json.load(dc_config_file)
        for ts in dc_config.keys():
        
            dc_ts = convert_ts_to_float(ts)

            if file_ts > dc_ts:
                dc_file_ts = ts
                continue
            else:
                break

        if dc_file_ts == '': 
            raise ValueError("Disabled channel file timestamp not found.")

        return dc_file_ts

    print("Disabled Channel File Timestamp:", lookup_disabled_channel_file_ts(disabled_channels_config, charge_filename))

    



"""