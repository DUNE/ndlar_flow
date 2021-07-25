import numpy as np
import logging
import yaml

from h5flow.core import H5FlowResource, resources

class Geometry(H5FlowResource):
    class_version = '0.0.0'

    default_path = 'geometry_info'
    default_crs_geometry_file = 'multi_tile_layout-2.2.16.yaml'
    # default_lrs_geometry_file = 'lrs_layout-0.0.0.yaml'

    # for light detectors (a proposal):
    # module -> panel_ids
    # panel_id -> channels [adc, channel]

    # panel_id -> (i_x, i_y, i_z) -> panel_x, panel_y, panel_z
    # panel_id -> panel_shape
    # panel_x -> x
    # panel_y -> y
    # panel_z -> z
    # panel_shape -> (dx,dy,dz) [half-width]

    def __init__(self, **params):
        super(Geometry,self).__init__(**params)

        self.path = params.get('path', self.default_path)
        self.crs_geometry_file = params.get('crs_geometry_file', self.default_crs_geometry_file)
        # self.lrs_geometry_file = params.get('lrs_geometry_file', self.default_lrs_geometry_file)

    def init(self, source_name):
        # create group (if not present)
        self.data_manager.set_attrs(self.path)
        # load data (if present)
        self.data = dict(self.data_manager.get_attrs(self.path))

        if not self.data:
            self.load_geometry()

            self.data_manager.set_attrs(self.path,
                pixel_pitch=self.pixel_pitch,
                crs_geometry_file=self.crs_geometry_file
                )
            self.write_lut('pixel_xy', self.pixel_xy)
            self.write_lut('tile_id', self.tile_id)
            self.write_lut('anode_z', self.anode_z)
            self.write_lut('drift_dir', self.drift_dir)
        else:
            self._pixel_pitch = self.data['pixel_pitch']
            self._pixel_xy = self.read_lut('pixel_xy')
            self._tile_id = self.read_lut('tile_id')
            self._anode_z = self.read_lut('anode_z')
            self._drift_dir = self.read_lut('drift_dir')

    def write_lut(self, name, lut):
        lut_meta, lut_arr = lut.to_array()
        self.data_manager.create_dset(self.path+'/'+name, dtype=lut_arr.dtype)
        self.data_manager.reserve_data(self.path+'/'+name, slice(0,len(lut_arr)))
        self.data_manager.write_data(self.path+'/'+name, slice(0,len(lut_arr)), lut_arr)
        self.data_manager.set_attrs(self.path+'/'+name, meta=lut_meta)
        self.data_manager.set_attrs(self.path, **{name+'_table': self.path+'/'+name})

    def read_lut(self, name):
        lut_arr = self.data_manager.get_dset(self.path+'/'+name)
        lut_meta = self.data_manager.get_attrs(self.path+'/'+name)['meta'][0]
        return LUT.from_array(lut_meta, lut_arr)

    @property
    def pixel_pitch(self):
        ''' Pixel pitch in mm '''
        return self._pixel_pitch

    @property
    def pixel_xy(self):
        '''
            Lookup table for pixel xy coordinate, usage::

                resource['Geometry'].pixel_xy[(io_group,io_channel,chip_id,channel_id)]

        '''
        return self._pixel_xy

    @property
    def tile_id(self):
        '''
            Lookup table for tile id, usage::

                resource['Geometry'].tile_id[(io_group,io_channel)]

        '''
        return self._tile_id

    @property
    def anode_z(self):
        '''
            Lookup table for anode z coordinate, usage::

                resource['Geometry'].anode_z[(tile_id,)]

        '''
        return self._anode_z

    @property
    def drift_dir(self):
        '''
            Lookup table for drift direction, usage::

                resource['Geometry'].drift_dir[(tile_id,)]

        '''
        return self._drift_dir


    def get_z_coordinate(self, io_group, io_channel, drift):
        '''
            Convert a drift distance on a set of ``(io group, io channel)`` to
            a z-coordinate.

            :param io_group: io group to calculate z coordinate, ``shape: (N,)``

            :param io_channel: io channel to calculate z coordinate, ``shape: (N,)``

            :param drift: drift distance [mm], ``shape: (N,)``

            :returns: z coordinate [mm], ``shape: (N,)``

        '''
        tile_id = self.tile_id[io_group, io_channel]
        z_anode = self.anode_z[np.array(tile_id)]
        drift_direction = self.drift_dir[np.array(tile_id)]

        return z_anode + drift_direction * drift

    @staticmethod
    def _rotate_pixel(pixel_pos, tile_orientation):
        return pixel_pos[0]*tile_orientation[2], pixel_pos[1]*tile_orientation[1]

    def load_geometry(self):
        logging.warning(f'Loading geometry from {self.crs_geometry_file}...')

        with open(self.crs_geometry_file) as gf:
            geometry_yaml = yaml.load(gf, Loader=yaml.FullLoader)

        if 'multitile_layout_version' not in geometry_yaml.keys():
            raise RuntimeError('Only multi-tile geometry configurations are accepted')

        self._pixel_pitch = geometry_yaml['pixel_pitch']
        chip_channel_to_position = geometry_yaml['chip_channel_to_position']
        tile_orientations = geometry_yaml['tile_orientations']
        tile_positions = geometry_yaml['tile_positions']
        tpc_centers = geometry_yaml['tpc_centers']
        tile_indeces = geometry_yaml['tile_indeces']
        xs = np.array(list(chip_channel_to_position.values()))[
            :, 0] * self.pixel_pitch
        ys = np.array(list(chip_channel_to_position.values()))[
            :, 1] * self.pixel_pitch
        x_size = max(xs)-min(xs)+self.pixel_pitch
        y_size = max(ys)-min(ys)+self.pixel_pitch

        tiles = [tile for tile in geometry_yaml['tile_chip_to_io']]
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

        pixel_xy_min_max = [(min(v),max(v)) for v in (io_groups, io_channels, chip_ids, channel_ids)]
        self._pixel_xy = LUT('f4', *pixel_xy_min_max, shape=(2,))
        self._pixel_xy.default = 0.

        tile_min_max = [(min(v),max(v)) for v in (io_groups, io_channels)]
        self._tile_id = LUT('i4', *tile_min_max)
        self._tile_id.default = -1

        anode_min_max = [(min(tiles),max(tiles))]
        self._anode_z = LUT('f4', *anode_min_max)
        self._anode_z.default = 0.
        self._drift_dir = LUT('i1', *anode_min_max)
        self._drift_dir.default = 0.

        self._anode_z[(tiles,)] = [tile_positions[tile][0] for tile in tiles]
        self._drift_dir[(tiles,)] = [tile_orientations[tile][0] for tile in tiles]

        for tile in geometry_yaml['tile_chip_to_io']:
            tile_orientation = tile_orientations[tile]
            for chip in geometry_yaml['tile_chip_to_io'][tile]:
                io_group_io_channel = geometry_yaml['tile_chip_to_io'][tile][chip]
                io_group = io_group_io_channel//1000
                io_channel = io_group_io_channel % 1000
                self._tile_id[([io_group], [io_channel])] = tile

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
                    self.pixel_pitch + self.pixel_pitch / 2 - x_size / 2
                y = chip_channel_to_position[chip_channel][1] * \
                    self.pixel_pitch + self.pixel_pitch / 2 - y_size / 2

                x, y = self._rotate_pixel((x, y), tile_orientation)
                x += tile_positions[tile][2] + \
                    tpc_centers[tile_indeces[tile][1]][0]
                y += tile_positions[tile][1] + \
                    tpc_centers[tile_indeces[tile][1]][1]

                self._pixel_xy[([io_group], [io_channel], [chip], [channel])] = np.array([x, y])

class LUT(object):
    '''
        Creates a lookup table that can be used to quickly access data based
        on tuples of integers. Works best if keys are contiguous within
        each position of the tuple. E.g.::

            key0 = [0,1,2]
            key1 = [30,31,32]

        is 10x more efficient than::

            key0 = [10,20,30]
            key1 = [300,310,320]

        Initialize with tuples of min and max values for each of the used key
        values::

            key0 = [0,1,2,3]
            key1 = [5,6,7,8]
            shape = (2,)
            dtype = 'f8'
            lut = LUT(shape, dtype, (min(key0), max(key0)), (min(key1), max(key1)))

        Data can then be stored in the table using a tuple of key arrays::

            lut[(key0,key1)] = np.array([[0,0],[1,1],[2,2],[3,3]])

        and accessed::

            lut[(key0,key1)] # np.array([[0,0],[1,1],[2,2],[3,3]])

        A default value should be set for keys that are not found in the table::

            lut.default = np.array([-1,-1])
            lut[([0],[0])] # np.array([-1,-1])

    '''
    def __init__(self, dtype, *min_max_keys, default=None, shape=None):
        self.min_max_keys = min_max_keys
        self.lengths = [max_-min_+1 for min_,max_ in self.min_max_keys]
        self.max_hash = int(self._hash(*[max_ for min_,max_ in min_max_keys]))
        shape = (self.max_hash+1,)+shape if shape else (self.max_hash+1,)
        self._data = np.empty(shape, dtype=dtype)
        self._filled = np.zeros_like(self._data, dtype=bool)
        if default is not None:
            self.default = default

    def __repr__(self):
        str_ = 'LUT('
        str_ += repr(self._data.dtype)
        for min_max in self.min_max_keys:
            str_ += ', ' + repr(min_max)
        str_ += f', default={repr(self.default)}'
        str_ += f', shape={repr(self._data.shape[1:])}'
        str_ += ')'
        return str_

    @staticmethod
    def from_array(meta_arr, data_arr):
        min_max_keys = meta_arr['min_max_keys']
        default = meta_arr['default']
        data = data_arr['data']
        filled = data_arr['filled']

        lut = LUT(data.dtype, *min_max_keys, shape=data.shape[1:])
        # initialization order is important here!
        lut._data = data
        lut._filled = filled
        lut.default = default

        return lut

    def to_array(self):
        dtype_meta = np.dtype([
            ('min_max_keys', 'i8', (len(self.min_max_keys),2)),
            ('default', self._data.dtype, self._data.shape[1:])
            ])
        dtype_data = np.dtype([
            ('data', self._data.dtype, self._data.shape[1:]),
            ('filled', self._filled.dtype, self._filled.shape[1:])
            ])
        meta_arr =np.empty((1,), dtype=dtype_meta)
        meta_arr['min_max_keys'] = self.min_max_keys
        meta_arr['default'] = self.default

        data_arr = np.empty(self._data.shape[0], dtype=dtype_data)
        data_arr['data'] = self._data
        data_arr['filled'] = self._filled

        return meta_arr, data_arr

    def _hash(self, *keys):
        val = 1
        for i,key in enumerate(keys):
            val += (np.array(key)-self.min_max_keys[i][0]) * sum(self.lengths[:i])
        return val.astype(int).ravel()

    def hash(self, *keys):
        '''
            Generate a hash index from arrays of key values

            :param *keys: arrays of each key value, ``shape: (N,)``

            :returns: array of hash index, ``shape: (N,)``
        '''
        val = self._hash(*keys)
        val[val < 0] = 0
        val[val > self.max_hash] = 0
        return val

    @property
    def default(self):
        '''
            Default value to return if key not found in table. Datatype is
            same as lookup table
        '''
        return self._data[0] # position 0 is reserved for the default value

    @default.setter
    def default(self,val):
        new_default = np.broadcast_to(np.expand_dims(np.array(val), axis=0),
            self._data.shape)
        self._data[~self._filled] = new_default[~self._filled]

    def clear(self, *keys):
        '''
            Remove stored value for specified keys

            :param *keys: arrays of key values, ``shape: (N,)``
        '''
        idx = self.hash(*keys)
        self._data[idx] = self.default
        self._filled[idx] = False

    def __getitem__(self, keys):
        return self._data[self.hash(*keys)]

    def __setitem__(self, keys, val):
        idx = self.hash(*keys)
        self._data[idx] = val
        self._filled[idx] = True
