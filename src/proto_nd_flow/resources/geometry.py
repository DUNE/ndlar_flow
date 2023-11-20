import numpy as np
import numpy.ma as ma
import logging
import yaml

from h5flow.core import H5FlowResource
from h5flow.core import resources

from proto_nd_flow.util.lut import LUT, write_lut, read_lut
from proto_nd_flow.util.compat import assert_compat_version
import proto_nd_flow.util.units as units


class Geometry(H5FlowResource):
    '''
        Provides helper functions for looking up geometric properties

        Parameters:
         - ``path``: ``str``, path to stored geometry data within file
         - ``crs_geometry_file``: ``str``, path to yaml file describing charge readout system geometry
         - ``lrs_geometry_file``: ``str``, path to yaml file describing light readout system

        Provides (for charge geometry):
         - ``pixel_pitch``: pixel pitch in mm
         - ``pixel_coordinates_2D``: lookup table for pixel coordinates in pixel plane (2D)
         - ``tile_id``: lookup table for io channel tile ids
         - ``anode_drift_coordinate``: lookup table for tile drift coordinate (x as of Spring 2023)
         - ``drift_dir``: lookup table for tile drift direction (±x direction as of Spring 2023;
                          hard to use bc cathode is not at x=0 in 2x2)
         - ``drift_regions``: drift regions minimum and maximum corners of TPC drift regions
                              cathode drift coordinate is treated as minimum x coordinate and anode
                              drift coordinate is treated as maximum x coordinate
         - ``in_fid()``: helper function for defining fiducial volumes
         - ``get_drift_coordinate()``: helper function for converting drift time to drift coordinate 
                                       (x as of Spring 2023)

        Provides (for light geometry):
         - ``tpc_id``: lookup table for TPC number for light detectors
         - ``det_id``: lookup table for detector number from adc, channel id
         - ``det_bounds``: lookup table for detector minimum and maximum corners light detectors
         - ``solid_angle()``: helper function for determining the solid angle of a given detector

        Example usage::

            from h5flow.core import resources

            resources['Geometry'].pixel_pitch

        Example config::

            resources:
                - classname: Geometry
                  params:
                    path: 'geometry_info'
                    det_geometry_file: 'data/prot_nd_flow/2x2.yaml'
                    crs_geometry_file: 'data/proto_nd_flow/multi_tile_layout-3.0.40.yaml'
                    lrs_geometry_file: 'data/proto_nd_flow/light_module_desc-0.0.0.yaml'

    '''
    class_version = '0.1.0'

    default_path = 'geometry_info'
    default_det_geometry_file = '-'
    default_crs_geometry_file = '-'
    default_lrs_geometry_file = '-'


    def __init__(self, **params):
        super(Geometry, self).__init__(**params)

        self.path = params.get('path', self.default_path)
        self.det_geometry_file = params.get('det_geometry_file', self.default_crs_geometry_file)
        self.crs_geometry_file = params.get('crs_geometry_file', self.default_crs_geometry_file)
        self.lrs_geometry_file = params.get('lrs_geometry_file', self.default_lrs_geometry_file)
        self._drift_regions = None  # active TPC drift regions


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
                                        drift_regions=self.drift_regions,
                                        crs_geometry_file=self.crs_geometry_file
                                        )
            write_lut(self.data_manager, self.path, self.pixel_coordinates_2D, 'pixel_coordinates_2D')
            write_lut(self.data_manager, self.path, self.tile_id, 'tile_id')
            write_lut(self.data_manager, self.path, self.anode_drift_coordinate, 'anode_drift_coordinate')
            write_lut(self.data_manager, self.path, self.drift_dir, 'drift_dir')

            write_lut(self.data_manager, self.path, self.tpc_id, 'tpc_id')
            write_lut(self.data_manager, self.path, self.det_id, 'det_id')
            write_lut(self.data_manager, self.path, self.det_bounds, 'det_bounds')
        else:
            assert_compat_version(self.class_version, self.data['class_version'])

            # load geometry from file
            self._pixel_pitch = self.data['pixel_pitch']
            self._drift_regions = self.data['drift_regions']
            self._pixel_coordinates_2D = read_lut(self.data_manager, self.path, 'pixel_coordinates_2D')
            self._tile_id = read_lut(self.data_manager, self.path, 'tile_id')
            self._anode_drift_coordinate = read_lut(self.data_manager, self.path, 'anode_drift_coordinate')
            self._drift_dir = read_lut(self.data_manager, self.path, 'drift_dir')

            self._tpc_id = read_lut(self.data_manager, self.path, 'tpc_id')
            self._det_id = read_lut(self.data_manager, self.path, 'det_id')
            self._det_bounds = read_lut(self.data_manager, self.path, 'det_bounds')

        lut_size = (self.pixel_coordinates_2D.nbytes + self.tile_id.nbytes
                    + self.anode_drift_coordinate.nbytes + self.drift_dir.nbytes
                    + self.tpc_id.nbytes + self.det_id.nbytes
                    + self.det_bounds.nbytes) * 4

        if self.rank == 0:
            logging.info(f'Geometry LUT(s) size: {lut_size/1024/1024:0.02f}MB')

    @property
    def pixel_pitch(self):
        ''' Pixel pitch in mm '''
        return self._pixel_pitch


    @property
    def pixel_coordinates_2D(self):
        '''
            Lookup table for pixel coordinates (2D), usage::

                resource['Geometry'].pixel_coordinates_2D[(io_group,io_channel,chip_id,channel_id)]

        '''
        return self._pixel_coordinates_2D


    @property
    def tile_id(self):
        '''
            Lookup table for tile id, usage::

                resource['Geometry'].tile_id[(io_group,io_channel)]

        '''
        return self._tile_id


    @property
    def anode_drift_coordinate(self):
        '''
            Lookup table for anode drift coordinate, usage::

                resource['Geometry'].anode_drift_coordinate[(tile_id,)]

        '''
        return self._anode_drift_coordinate


    @property
    def drift_dir(self):
        '''
            Lookup table for drift direction, usage::

                resource['Geometry'].drift_dir[(tile_id,)]

        '''
        return self._drift_dir


    @property
    def drift_regions(self):
        '''
            List of active volume extent for each TPC, each shape: ``(2,3)``
            representing the minimum xyz coordinate and the maximum xyz
            coordinate
        '''
        return self._drift_regions


    def in_fid(self, xyz, cathode_fid=0.0, field_cage_fid=0.0, anode_fid=0.0):
        '''
            Check if xyz point is contained in the specified fiducial volume

            :param xyz: point to check, array ``shape: (N,3)``

            :param cathode_fid: fiducial boundary for cathode and anode, ``float``, optional

            :param field_cage_fid: fiducial boundary for field cage walls, ``float``, optional

            :returns: boolean array, ``shape: (N,)``, True indicates point is within fiducial volume

        '''
        # Define xyz coordinates of fiducial boundaries separately for each limiting boundary
        fid_cathode = np.array([cathode_fid, 0., 0.])
        fid_anode = np.array([anode_fid, 0., 0.])
        fid_field_cage = np.array([0., field_cage_fid, field_cage_fid])

        drift_regions_fid = self.drift_regions
        num_tpcs = len(drift_regions_fid)
        drift_dir = np.ones(num_tpcs, dtype=int)
        coord_in_fid = np.zeros(num_tpcs, dtype=bool)

        # Loop through drift regions
        # Notes: Cases are separated by drift direction because drift region boundaries are 
        #       defined using the convention [[cathode_x, min_y, min_z], [anode_x, max_y, max_z]]
        #       Positive drift direction: anode_x > cathode_x
        #       Negative drift direction: anode_x < cathode_x
        for i in range(num_tpcs):

            # Redefine drift regions to account for fiducial boundaries
            if drift_regions_fid[i][0][0] < drift_regions_fid[i][1][0]:

                drift_regions_fid[i][0] = drift_regions_fid[i][0] + fid_cathode + fid_field_cage
                drift_regions_fid[i][1] = drift_regions_fid[i][1] - fid_anode - fid_field_cage
            
            elif drift_regions_fid[i][0][0] > drift_regions_fid[i][1][0]:

                drift_dir[i] = -1
                drift_regions_fid[i][0] = drift_regions_fid[i][0] - fid_cathode + fid_field_cage
                drift_regions_fid[i][1] = drift_regions_fid[i][1] + fid_anode - fid_field_cage

            else:
                raise ValueError('Drift distance is 0.')

            # Check if xyz point is in each fiducial drift region
            if drift_dir[i] == 1:
                coord_in_fid[i] = ma.all(ma.concatenate(\
                    [np.expand_dims((xyz > np.expand_dims(drift_regions_fid[i][0], 0))
                                  & (xyz < np.expand_dims(drift_regions_fid[i][1], 0)), axis=-1)]), axis=1)
            elif drift_dir[i] == -1:
                coord_in_fid[i] = ma.all(ma.concatenate(\
                    [np.expand_dims((xyz[0] < np.expand_dims(drift_regions_fid[i][0][0], 0))
                                  & (xyz[0] > np.expand_dims(drift_regions_fid[i][1][0], 0))
                                  & (xyz[1:] > np.expand_dims(drift_regions_fid[i][0][1:], 0))
                                  & (xyz[1:] < np.expand_dims(drift_regions_fid[i][1][1:], 0)), axis=-1)]), axis=1)

            else: 
                raise ValueError('Drift direction is invalid.')
        
        in_any_fid = ma.any(coord_in_fid, axis=-1)
        return in_any_fid


    def get_drift_coordinate(self, io_group, io_channel, drift):
        '''
            Convert a drift distance on a set of ``(io group, io channel)`` to
            the drift coordinate.

            :param io_group: io group to calculate z coordinate, ``shape: (N,)``

            :param io_channel: io channel to calculate z coordinate, ``shape: (N,)``

            :param drift: drift distance [mm], ``shape: (N,)``

            :returns: drift coordinate [mm], ``shape: (N,)``

        '''
        tile_id = self.tile_id[(io_group, io_channel)]
        anode_drift_coord = self.anode_drift_coordinate[(np.array(tile_id),)]
        drift_direction = self.drift_dir[(np.array(tile_id),)]

        return anode_drift_coord.reshape(drift.shape) + \
            drift_direction.reshape(drift.shape) * drift

    @staticmethod
    def _rotate_pixel(pixel_pos, tile_orientation):
        return pixel_pos[0] * tile_orientation[2], pixel_pos[1] * tile_orientation[1]


    @property
    def tpc_id(self):
        '''
            Lookup table for TPC id, usage::

                resource['Geometry'].tpc_id[(adc_index, channel_index)]

        '''
        return self._tpc_id


    @property
    def det_id(self):
        '''
            Lookup table for detector id within a TPC, usage::

                resource['Geometry'].det_id[(adc_index, channel_index)]

        '''
        return self._det_id


    @property
    def det_bounds(self):
        '''
            Lookup table for detector min and max xyz coordinate, usage::

                resource['Geometry'].det_bounds[(tpc_id, det_id)]

        '''
        return self._det_bounds


    @staticmethod
    def _rect_solid_angle_sign(coord, rect_min, rect_max):
        overlapping = (coord >= rect_min) & (coord <= rect_max)
        inverted = np.abs(rect_min - coord) < np.abs(rect_max - coord)

        sign_min = overlapping + ~overlapping * (1 - 2*inverted)
        sign_max = overlapping + ~overlapping * (2*inverted - 1)

        return sign_min, sign_max


    def solid_angle(self, xyz, tpc_id, det_id):
        '''
        Calculate the solid angle of a rectangular detector ``det_id`` in TPC
        ``tpc_id`` as seen from the point ``xyz``, under the assumption
        that the detector is oriented along the drift direction

        Note: this method does not consider cathode / field cage visibilty.

        :param xyz: array shape: ``(N,3)``

        :param tpc_id: array shape: ``(M,)``

        :param det_id: array shape: ``(M,)``

        :returns: array shape: ``(N, M)``

        '''
        x,y,z = xyz[...,0:1,np.newaxis], xyz[...,1:2,np.newaxis], xyz[...,2:3,np.newaxis]
        det_bounds = self.det_bounds[(tpc_id, det_id)]
        det_bounds = det_bounds.reshape((1,)+det_bounds.shape)
        det_min = det_bounds[...,0,:]
        det_max = det_bounds[...,1,:]

        det_x = (det_min[...,0] + det_max[...,0])/2
        det_y_sign_min, det_y_sign_max = self._rect_solid_angle_sign(
            y, det_min[...,1], det_max[...,1])
        det_z_sign_min, det_z_sign_max = self._rect_solid_angle_sign(
            z, det_min[...,2], det_max[...,2])

        omega = np.zeros(det_y_sign_min.shape, dtype=float)
        for det_y,det_y_sign in ((det_max[...,1], det_y_sign_max), (det_min[...,1], det_y_sign_min)):
            for det_z,det_z_sign in ((det_max[...,2], det_z_sign_max), (det_min[...,2], det_z_sign_min)):
                d = np.sqrt((x-det_x)**2 + (y-det_y)**2 + (z-det_z)**2)
                omega += det_y_sign * det_z_sign * np.arctan2(np.abs(det_y-y) * np.abs(det_z-z), np.abs(det_x-x)* d)

        return omega


    def load_geometry(self):
        self._load_charge_geometry()
        self._load_light_geometry()


    def _load_light_geometry(self):
        if self.rank == 0:
            logging.warning(f'Loading geometry from {self.lrs_geometry_file}...')

        with open(self.lrs_geometry_file) as gf:
            geometry = yaml.load(gf, Loader=yaml.FullLoader)

        # enforce that light geometry formatting is as expected
        assert_compat_version(geometry['format_version'], '0.0.0')

        tpc_ids = np.array([v for v in geometry['tpc_center'].keys()])
        det_ids = np.array([v for v in geometry['det_center'].keys()])
        max_chan = max([len(chan) for tpc in geometry['det_chan'].values() for chan in tpc.values()])

        shape = tpc_ids.shape + det_ids.shape
        det_adc = np.full(shape, -1, dtype=int)
        det_chan = np.full(shape + (max_chan,), -1, dtype=int)
        det_chan_mask = np.zeros(shape + (max_chan,), dtype=bool)
        det_bounds = np.zeros(shape + (2,3), dtype=float)
        for i, tpc in enumerate(tpc_ids):
            for j, det in enumerate(det_ids):
                det_adc[i,j] = geometry['det_adc'][tpc][det]
                det_chan[i,j,:len(geometry['det_chan'][tpc][det])] = geometry['det_chan'][tpc][det]

                tpc_center = np.array(geometry['tpc_center'][tpc])
                det_geom = geometry['geom'][geometry['det_geom'][det]]
                det_center = np.array(geometry['det_center'][det])
                det_bounds[i,j,0] = tpc_center + det_center + np.array(det_geom['min'])
                det_bounds[i,j,1] = tpc_center + det_center + np.array(det_geom['max'])

        det_chan_mask = det_chan != -1

        det_adc, det_chan, tpc_ids, det_ids = np.broadcast_arrays(
            det_adc[...,np.newaxis], det_chan,
            tpc_ids[...,np.newaxis,np.newaxis], det_ids[...,np.newaxis])

        adc_chan_min_max = [(min(det_adc[det_chan_mask]), max(det_adc[det_chan_mask])),
                            (min(det_chan[det_chan_mask]), max(det_chan[det_chan_mask]))]
        self._tpc_id = LUT('i4', *adc_chan_min_max)
        self._tpc_id.default = -1

        self._det_id = LUT('i4', *adc_chan_min_max)
        self._det_id.default = -1

        det_min_max = [(min(tpc_ids[det_chan_mask]), max(tpc_ids[det_chan_mask])),
                       (min(det_ids[det_chan_mask]), max(det_ids[det_chan_mask]))]
        self._det_bounds = LUT('f4', *det_min_max, shape=(2,3))
        self._det_bounds.default = 0.

        self._tpc_id[(det_adc[det_chan_mask], det_chan[det_chan_mask])] = tpc_ids[det_chan_mask]
        self._det_id[(det_adc[det_chan_mask], det_chan[det_chan_mask])] = det_ids[det_chan_mask]

        tpc_ids, det_ids, det_chan_mask = tpc_ids[...,0], det_ids[...,0], det_chan_mask[...,0]
        self._det_bounds[(tpc_ids[det_chan_mask], det_ids[det_chan_mask])] = det_bounds[det_chan_mask]


    def _load_charge_geometry(self):
        if self.rank == 0:
            logging.warning(f'Loading geometry from {self.crs_geometry_file}...')

        with open(self.crs_geometry_file) as gf:
            geometry_yaml = yaml.load(gf, Loader=yaml.FullLoader)

        with open(self.det_geometry_file) as dgf:
            det_geometry_yaml = yaml.load(dgf, Loader=yaml.FullLoader)

        if 'multitile_layout_version' not in geometry_yaml.keys():
            raise RuntimeError('Only multi-tile geometry configurations are accepted')

        self._pixel_pitch = geometry_yaml['pixel_pitch']
        chip_channel_to_position = geometry_yaml['chip_channel_to_position']
        tile_orientations = geometry_yaml['tile_orientations']
        tile_positions = geometry_yaml['tile_positions']
        tpc_centers = geometry_yaml['tpc_centers']
        tile_indeces = geometry_yaml['tile_indeces']
        mod_centers = det_geometry_yaml['tpc_offsets']
        tile_chip_to_io = geometry_yaml['tile_chip_to_io']

        zs = np.array(list(chip_channel_to_position.values()))[:, 0] * self.pixel_pitch
        ys = np.array(list(chip_channel_to_position.values()))[:, 1] * self.pixel_pitch
        z_size = max(zs) - min(zs) + self.pixel_pitch
        y_size = max(ys) - min(ys) + self.pixel_pitch

        tile_geometry = {}

        tiles = np.arange(1,len(geometry_yaml['tile_chip_to_io'])*len(det_geometry_yaml['module_to_io_groups'])+1)
        io_groups = [
            geometry_yaml['tile_chip_to_io'][tile][chip] // 1000 * (mod-1)*2
            for tile in geometry_yaml['tile_chip_to_io']
            for chip in geometry_yaml['tile_chip_to_io'][tile]
            for mod in det_geometry_yaml['module_to_io_groups']
        ]
        io_channels = [
            geometry_yaml['tile_chip_to_io'][tile][chip] % 1000
            for tile in geometry_yaml['tile_chip_to_io']
            for chip in geometry_yaml['tile_chip_to_io'][tile]
            for mod in det_geometry_yaml['module_to_io_groups']
        ]
        chip_ids = [
            chip_channel // 1000
            for chip_channel in geometry_yaml['chip_channel_to_position']
            for mod in det_geometry_yaml['module_to_io_groups']
        ]
        channel_ids = [
            chip_channel % 1000
            for chip_channel in geometry_yaml['chip_channel_to_position']
            for mod in det_geometry_yaml['module_to_io_groups']
        ]
 
        pixel_coordinates_2D_min_max = [(min(v), max(v)) for v in (io_groups, io_channels, chip_ids, channel_ids)]
        self._pixel_coordinates_2D = LUT('f4', *pixel_coordinates_2D_min_max, shape=(2,))
        self._pixel_coordinates_2D.default = 0.
    
        tile_min_max = [(min(v), len(det_geometry_yaml['module_to_io_groups'])*max(v)) for v in (io_groups, io_channels)]
        self._tile_id = LUT('i4', *tile_min_max)
        self._tile_id.default = -1
    
        anode_min_max = [(min(tiles), len(det_geometry_yaml['module_to_io_groups'])*max(tiles))]
        self._anode_drift_coordinate = LUT('f4', *anode_min_max)
        self._anode_drift_coordinate.default = 0.
        self._drift_dir = LUT('i1', *anode_min_max)
        self._drift_dir.default = 0.

        self._anode_drift_coordinate[(tiles,)] = [tile_positions[(tile-1)%16+1][0]+10.*mod_centers[((tile-1)//16)%4][0] for tile in tiles]

        self._drift_dir[(tiles,)] = [tile_orientations[(tile-1)%16+1][0] for tile in tiles]
        for module_id in det_geometry_yaml['module_to_io_groups']:
            for tile in tile_chip_to_io:
                tile_orientation = tile_orientations[tile]
                tile_geometry[tile] = tile_positions[tile], tile_orientations[tile]

                for chip in tile_chip_to_io[tile]:
                    io_group_io_channel = tile_chip_to_io[tile][chip]
                    io_group = io_group_io_channel//1000 + (module_id-1)*len(det_geometry_yaml['module_to_io_groups'][module_id])
                    io_channel = io_group_io_channel % 1000
                    self._tile_id[([io_group], [io_channel])] = tile+(module_id-1)*len(tile_chip_to_io)

                for chip_channel in chip_channel_to_position:
                    chip = chip_channel // 1000
                    channel = chip_channel % 1000

                    try:
                        io_group_io_channel = tile_chip_to_io[tile][chip]
                    except KeyError:
                        continue

                    io_group = io_group_io_channel // 1000 + (module_id-1)*len(det_geometry_yaml['module_to_io_groups'][module_id])
                    io_channel = io_group_io_channel % 1000

                    z = chip_channel_to_position[chip_channel][0] * \
                        self.pixel_pitch - z_size / 2 + self.pixel_pitch / 2
                    y = chip_channel_to_position[chip_channel][1] * \
                        self.pixel_pitch - y_size / 2 + self.pixel_pitch / 2

                    z, y = self._rotate_pixel((z, y), tile_orientation)

                    z += tile_positions[tile][2]
                    y += tile_positions[tile][1]
                    z += mod_centers[module_id-1][2]*units.cm # det geo yaml is in cm; here we convert to mm 
                    y += mod_centers[module_id-1][1]*units.cm # det geo yaml is in cm; here we convert to mm
                    self._pixel_coordinates_2D[(io_group, io_channel, chip, channel)] = z, y

        self._drift_regions = []

        io_group, io_channel, chip_id, channel_id = self.pixel_coordinates_2D.keys()
        cathode_drift_coordinates = np.unique(np.array(mod_centers)[:,0])*units.cm # det geo yaml is in cm; here we convert to mm
        unique_io_group, inv = np.unique(io_group, return_inverse=True)
        
        # Loop through io_groups
        for i, group in enumerate(unique_io_group):
            mask = (inv == i)

            # Get zy coordinates for io_group
            zy = self.pixel_coordinates_2D[(io_group[mask], io_channel[mask], chip_id[mask], channel_id[mask])]
            
            # Assign min and max y,z coordinates for drift region
            min_y, max_y = zy[:,1].min(), zy[:,1].max()
            min_z, max_z = zy[:,0].min(), zy[:,0].max()

            # Get x coordinates for anode and cathode corresponding to io_group
            tile_id = self.tile_id[(io_group[mask], io_channel[mask])]
            anode_drift_coordinate = np.unique(self.anode_drift_coordinate[(tile_id,)])[0]
            cathode_drift_coordinate = cathode_drift_coordinates[0]
            if abs(anode_drift_coordinate - cathode_drift_coordinates[1]) \
                < abs(anode_drift_coordinate - cathode_drift_coordinate):
                cathode_drift_coordinate = cathode_drift_coordinates[1]

            # Append drift region to _drift_regions list
            self._drift_regions.append(np.array([[cathode_drift_coordinate, min_y, min_z],
                                                 [anode_drift_coordinate, max_y, max_z]]))
            
        print("Drift Regions:", self._drift_regions)