import numpy as np
import numpy.ma as ma
import logging
import yaml

from h5flow.core import H5FlowResource, resources

from module0_flow.util.lut import LUT, write_lut, read_lut
from module0_flow.util.compat import assert_compat_version


class Geometry(H5FlowResource):
    class_version = '0.0.1'

    default_path = 'geometry_info'
    default_crs_geometry_file = '-'
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
        super(Geometry, self).__init__(**params)

        self.path = params.get('path', self.default_path)
        self.crs_geometry_file = params.get('crs_geometry_file', self.default_crs_geometry_file)
        # self.lrs_geometry_file = params.get('lrs_geometry_file', self.default_lrs_geometry_file)
        self._regions = None  # active TPC regions

    def init(self, source_name):
        super(Geometry, self).init(source_name)

        # create group (if not present)
        self.data_manager.set_attrs(self.path)
        # load data (if present)
        self.data = dict(self.data_manager.get_attrs(self.path))

        if not self.data:
            # first time loading geometry, save to file
            self.load_geometry()

            self.data_manager.set_attrs(self.path,
                                        classname=self.classname,
                                        class_version=self.class_version,
                                        pixel_pitch=self.pixel_pitch,
                                        crs_geometry_file=self.crs_geometry_file
                                        )
            write_lut(self.data_manager, self.path, self.pixel_xy, 'pixel_xy')
            write_lut(self.data_manager, self.path, self.tile_id, 'tile_id')
            write_lut(self.data_manager, self.path, self.anode_z, 'anode_z')
            write_lut(self.data_manager, self.path, self.drift_dir, 'drift_dir')
        else:
            assert_compat_version(self.class_version, self.data['class_version'])

            # load geometry from file
            self._pixel_pitch = self.data['pixel_pitch']
            self._pixel_xy = read_lut(self.data_manager, self.path, 'pixel_xy')
            self._tile_id = read_lut(self.data_manager, self.path, 'tile_id')
            self._anode_z = read_lut(self.data_manager, self.path, 'anode_z')
            self._drift_dir = read_lut(self.data_manager, self.path, 'drift_dir')

        lut_size = (self.pixel_xy.nbytes + self.tile_id.nbytes
                    + self.anode_z.nbytes + self.drift_dir.nbytes)
        if self.rank == 0:
            logging.info(f'Geometry LUT(s) size: {lut_size/1024/1024:0.02f}MB')

    def _create_regions(self):
        self._regions = []

        io_group, io_channel, chip_id, channel_id = self.pixel_xy.keys()
        xy = self.pixel_xy[(io_group, io_channel, chip_id, channel_id)]
        tile_id = self.tile_id[(io_group, io_channel)]
        anode_z = self.anode_z[(tile_id,)]
        drift_dir = self.drift_dir[(tile_id,)]

        anode_zs, inv = np.unique(anode_z, return_inverse=True)
        for i, z in enumerate(anode_zs):
            mask = (inv == i)

            min_x, max_x = xy[mask, 0].min(), xy[mask, 0].max()
            min_y, max_y = xy[mask, 1].min(), xy[mask, 1].max()
            min_z, max_z = (z * (drift_dir[mask][0] > 0), z * (drift_dir[mask][0] < 0))

            self._regions.append(np.array([[min_x, min_y, min_z],
                                           [max_x, max_y, max_z]]))

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

    @property
    def regions(self):
        if self._regions is None:
            self._create_regions()
        return self._regions

    def in_fid(self, xyz, cathode_fid=0.0, field_cage_fid=0.0):
        '''
            Check if xyz point is contained in the specified fiducial volume

            :param xyz: point to check, array ``shape: (N,3)``

            :param cathode_fid: fiducial boundary for cathode and anode, ``float``, optional

            :param field_cage_fid: fiducial boundary for field cage walls, ``float``, optional

            :returns: boolean array, ``shape: (N,)``, True indicates point is within fiducial volume

        '''
        fid = np.array([field_cage_fid, field_cage_fid, cathode_fid])
        coord_in_fid = ma.concatenate([np.expand_dims((xyz < np.expand_dims(boundary[1] - fid, 0))
                                                      & (xyz > np.expand_dims(boundary[0] + fid, 0)), axis=-1)
                                       for boundary in self.regions], axis=-1)
        in_fid = ma.all(coord_in_fid, axis=1)
        in_any_fid = ma.any(in_fid, axis=-1)
        return in_any_fid

    def get_z_coordinate(self, io_group, io_channel, drift):
        '''
            Convert a drift distance on a set of ``(io group, io channel)`` to
            a z-coordinate.

            :param io_group: io group to calculate z coordinate, ``shape: (N,)``

            :param io_channel: io channel to calculate z coordinate, ``shape: (N,)``

            :param drift: drift distance [mm], ``shape: (N,)``

            :returns: z coordinate [mm], ``shape: (N,)``

        '''
        tile_id = self.tile_id[(io_group, io_channel)]
        z_anode = self.anode_z[(np.array(tile_id),)]
        drift_direction = self.drift_dir[(np.array(tile_id),)]

        return z_anode.reshape(drift.shape) + \
            drift_direction.reshape(drift.shape) * drift

    @staticmethod
    def _rotate_pixel(pixel_pos, tile_orientation):
        return pixel_pos[0] * tile_orientation[2], pixel_pos[1] * tile_orientation[1]

    def load_geometry(self):
        if self.rank == 0:
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
        x_size = max(xs) - min(xs) + self.pixel_pitch
        y_size = max(ys) - min(ys) + self.pixel_pitch

        tiles = [tile for tile in geometry_yaml['tile_chip_to_io']]
        io_groups = [
            geometry_yaml['tile_chip_to_io'][tile][chip] // 1000
            for tile in geometry_yaml['tile_chip_to_io']
            for chip in geometry_yaml['tile_chip_to_io'][tile]
        ]
        io_channels = [
            geometry_yaml['tile_chip_to_io'][tile][chip] % 1000
            for tile in geometry_yaml['tile_chip_to_io']
            for chip in geometry_yaml['tile_chip_to_io'][tile]
        ]
        chip_ids = [
            chip_channel // 1000
            for chip_channel in geometry_yaml['chip_channel_to_position']
        ]
        channel_ids = [
            chip_channel % 1000
            for chip_channel in geometry_yaml['chip_channel_to_position']
        ]

        pixel_xy_min_max = [(min(v), max(v)) for v in (io_groups, io_channels, chip_ids, channel_ids)]
        self._pixel_xy = LUT('f4', *pixel_xy_min_max, shape=(2,))
        self._pixel_xy.default = 0.

        tile_min_max = [(min(v), max(v)) for v in (io_groups, io_channels)]
        self._tile_id = LUT('i4', *tile_min_max)
        self._tile_id.default = -1

        anode_min_max = [(min(tiles), max(tiles))]
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
                io_group = io_group_io_channel // 1000
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
