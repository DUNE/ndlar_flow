import numpy as np
import numpy.ma as ma
import logging
import warnings
import yaml

from h5flow.core import H5FlowResource
from h5flow.core import resources

from proto_nd_flow.util.lut import LUT, write_lut, read_lut
from proto_nd_flow.util.compat import assert_compat_version
import proto_nd_flow.util.units as units


class Geometry(H5FlowResource):
    '''
        Provides helper functions for looking up geometric properties. 

        Input charge geometry file is assumed to use distance units of [mm] and input 
        detector geometry file is assumed to use units of [cm].

        **!! All output CHARGE attributes and datasets are saved in units of cm !!**

        Parameters:
         - ``path``: ``str``, path to stored geometry data within file
         - ``network_agnostic``: ``bool``, optional, ignore the (io_channel % 4), useful if the io_channel mapping changes run-to-run (``True == ignore``)
         - ``n_io_channels_per_tile``: ``int``, optional, only used with ``network_agnostic == True``, sets the number of io channels to have same geometry info
         - ``crs_geometry_file``: ``str``, path to yaml file describing charge readout system geometry
         - ``det_geometry_file``: ``str``, path to yaml file describing overall detector geometry
         - ``lrs_geometry_file``: ``str``, path to yaml file describing light readout system
         - ``beam_direction``   : ``str``, Cartesian coordinate of beam direction, e.g. 'x', 'y', 'z'
         - ``drift_direction``  : ``str``, Cartesian coordinate of drift direction, e.g. 'x', 'y', 'z'

        Provides (for charge geometry):
         - ``beam_direction``    [param->attr]: Cartesian coordinate of beam direction
         - ``crs_geometry_file`` [param->attr]: path to yaml file describing charge 
                                                readout system geometry
         - ``drift_direction``   [param->attr]: Cartesian coordinate of drift direction

         - ``cathode_thickness``        [attr]: thickness of cathode [cm]
         - ``lar_detector_bounds``      [attr]: min and max xyz coordinates for full LAr detector [cm]
         - ``max_drift_distance``       [attr]: max drift distance in each LArTPC (2 TPCs/module) [cm]
         - ``module_RO_bounds``         [attr]: min and max xyz coordinates for each module [cm]   
         - ``pixel_pitch``              [attr]: distance between adjacent pixel centers [cm]

         - ``anode_drift_coordinate``   [dset]: lookup table for tile drift coordinate [cm]
         - ``drift_dir``                [dset]: lookup table for tile drift direction (either ±1)
         - ``pixel_coordinates_2D``     [dset]: lookup table for pixel coordinates in 2D pixel plane [cm]
         - ``tile_id``                  [dset]: lookup table for io channel tile ids 

         - ``get_drift_coordinate()``   [mthd]: class method converting drift time to drift coordinate [cm]
         - ``in_fid()``                 [mthd]: class method for determing whether an xyz coordinate [cm] 
                                                is in the LAr fiducial volume

        Provides (for light geometry):
         - ``det_rel_pos``: lookup table for relative position (TPC,side,vertical position from bottom) of light detectors (Full ArCLight or LCM)
         - ``sipm_rel_pos``: lookup table for lookup table for relative position (TPC,side,vertical position from bottom) of SiPMs (Single SiPM)
         - ``det_id``: lookup table for detector number from adc, channel id
         - ``det_bounds``: lookup table for detector minimum and maximum corners light detectors
         - ``sipm_abs_pos``: lookup table for sipm absolute position (x,y,z) in cm
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
    class_version = '0.2.0'

    default_path = 'geometry_info'
    default_network_agnostic = False
    default_n_io_channels_per_tile = 4
    default_det_geometry_file = '-'
    default_crs_geometry_file = ['-']
    default_lrs_geometry_file = '-'
    default_beam_direction    = 'z'
    default_drift_direction   = 'x'
    default_crs_geometry_to_module = [0]


    def __init__(self, **params):
        super(Geometry, self).__init__(**params)

        self.path = params.get('path', self.default_path)
        self.network_agnostic = params.get('network_agnostic', self.default_network_agnostic)
        self.n_io_channels_per_tile = params.get('n_io_channels_per_tile', self.default_n_io_channels_per_tile)
        self.crs_geometry_files = params.get('crs_geometry_files', self.default_crs_geometry_file)
        self.crs_geometry_to_module = params.get('crs_geometry_to_module', self.default_crs_geometry_to_module)
        self.det_geometry_file = params.get('det_geometry_file', self.default_det_geometry_file)
        self.lrs_geometry_file = params.get('lrs_geometry_file', self.default_lrs_geometry_file)
        self.beam_direction = params.get('beam_direction', self.default_beam_direction)
        self.drift_direction = params.get('drift_direction', self.default_drift_direction)
        self._cathode_thickness = 0.0 # thickness of cathode [cm]
        self._lar_detector_bounds = None # min and max xyz coordinates for full LAr detector
        self._max_drift_distance = None # max drift distance in each LArTPC (2 TPCs per module)
        self._module_RO_bounds = None # min and max xyz coordinates for each pixel LArTPC module

    def init(self, source_name):
        super(Geometry, self).init(source_name)

        # create group (if not present)
        self.data_manager.set_attrs(self.path)
        # load data (if present)
        self.data = dict(self.data_manager.get_attrs(self.path))

        if not self.data:
            # first time loading geometry, save to file

            with open(self.det_geometry_file) as dgf:
                self.det_geometry_yaml = yaml.load(dgf, Loader=yaml.FullLoader)

            with open(self.lrs_geometry_file) as gf:
                self.lrs_geometry_yaml = yaml.load(gf, Loader=yaml.FullLoader)

            self.load_geometry()

            self.data_manager.set_attrs(self.path,
                                        classname=self.classname,
                                        class_version=self.class_version,
                                        beam_direction=self.beam_direction,
                                        crs_geometry_files=self.crs_geometry_files, 
                                        crs_geometry_to_module=self.crs_geometry_to_module, 
                                        drift_direction=self.drift_direction,
                                        cathode_thickness=self.cathode_thickness,
                                        lar_detector_bounds=self.lar_detector_bounds,
                                        max_drift_distance=self.max_drift_distance,
                                        module_RO_bounds=self.module_RO_bounds,
                                        pixel_pitch=self.pixel_pitch,
                                        network_agnostic=self.network_agnostic,
                                        n_io_channels_per_tile=self.n_io_channels_per_tile
                                        )
            write_lut(self.data_manager, self.path, self.anode_drift_coordinate, 'anode_drift_coordinate')
            write_lut(self.data_manager, self.path, self.drift_dir, 'drift_dir')
            write_lut(self.data_manager, self.path, self.pixel_coordinates_2D, 'pixel_coordinates_2D')
            write_lut(self.data_manager, self.path, self.tile_id, 'tile_id')

            write_lut(self.data_manager, self.path, self.det_rel_pos, 'det_rel_pos')
            write_lut(self.data_manager, self.path, self.sipm_rel_pos, 'sipm_rel_pos')
            write_lut(self.data_manager, self.path, self.det_id, 'det_id')
            write_lut(self.data_manager, self.path, self.det_bounds, 'det_bounds')
            write_lut(self.data_manager, self.path, self.sipm_abs_pos, 'sipm_abs_pos')
        else:
            assert_compat_version(self.class_version, self.data['class_version'])

            # load geometry from file
            self._cathode_thickness = self.data['cathode_thickness']
            self._lar_detector_bounds = self.data['lar_detector_bounds']
            self._max_drift_distance = self.data['max_drift_distance']
            self._module_RO_bounds = self.data['module_RO_bounds']
            self._pixel_pitch = self.data['pixel_pitch']

            self._anode_drift_coordinate = read_lut(self.data_manager, self.path, 'anode_drift_coordinate')
            self._drift_dir = read_lut(self.data_manager, self.path, 'drift_dir')
            self._pixel_coordinates_2D = read_lut(self.data_manager, self.path, 'pixel_coordinates_2D')
            self._tile_id = read_lut(self.data_manager, self.path, 'tile_id')
            self._det_rel_pos = read_lut(self.data_manager, self.path, 'det_rel_pos')
            self._sipm_rel_pos = read_lut(self.data_manager, self.path, 'sipm_rel_pos')

            self._det_id = read_lut(self.data_manager, self.path, 'det_id')
            self._det_bounds = read_lut(self.data_manager, self.path, 'det_bounds')
            self._sipm_abs_pos = read_lut(self.data_manager, self.path, 'sipm_abs_pos')

        lut_size = (self.anode_drift_coordinate.nbytes + self.drift_dir.nbytes
                    + self.pixel_coordinates_2D.nbytes + self.tile_id.nbytes
                    + self.det_rel_pos.nbytes + self.det_rel_pos.nbytes 
                    + self.det_id.nbytes + self.det_bounds.nbytes
                    + self.sipm_abs_pos.nbytes)

        if self.rank == 0:
            logging.info(f'Geometry LUT(s) size: {lut_size/1024/1024:0.02f}MB')


    ## Charge geometry attributes, datasets, and methods ##
    @property
    def cathode_thickness(self):
        ''' Thickness of cathode [cm] '''
        return self._cathode_thickness
    

    @property
    def lar_detector_bounds(self):
        '''
            Array of shape ``(2,3)`` representing the minimum xyz coordinate 
            and the maximum xyz coordinate for the full LAr detector being studied
            (e.g. single module, 2x2, ND-LAr, etc.) [cm]
        '''
        return self._lar_detector_bounds
    

    @property
    def max_drift_distance(self):
        '''
            Maximum possible drift distance for ionization electrons in each TPC (2 TPCs
            per module). This is the distance between the surface of the cathode and the
            surface of one of the two anodes in a module [cm]
        '''
        return self._max_drift_distance


    @property
    def module_RO_bounds(self):
        '''
            Array of active volume extent for each module shape: ``(# modules,2,3)`` 
            representing the minimum xyz coordinate and the maximum xyz coordinate  
            for each module in the LAr detector [cm]
        '''
        return self._module_RO_bounds
    

    @property
    def pixel_pitch(self):
        ''' Distance between pixel centers [cm] '''
        return self._pixel_pitch


    @property
    def anode_drift_coordinate(self):
        '''
            Lookup table for anode drift coordinate [cm], usage::

                resource['Geometry'].anode_drift_coordinate[(tile_id,)]

        '''
        return self._anode_drift_coordinate


    @property
    def drift_dir(self):
        '''
            Lookup table for drift direction (+/-1), usage::

                resource['Geometry'].drift_dir[(tile_id,)]

        '''
        return self._drift_dir 


    @property
    def pixel_coordinates_2D(self):
        '''
            Lookup table for pixel coordinates (2D) [cm], usage::

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


    def get_drift_coordinate(self, io_group, io_channel, drift):
        '''
            Convert a drift distance on a set of ``(io group, io channel)`` to
            the drift coordinate.

            :param io_group: io group to calculate z coordinate, ``shape: (N,)``

            :param io_channel: io channel to calculate z coordinate, ``shape: (N,)``

            :param drift: drift distance [cm], ``shape: (N,)``

            :returns: drift coordinate [cm], ``shape: (N,)``

        '''
        tile_id = self.tile_id[(io_group, io_channel)]
        anode_drift_coord = self.anode_drift_coordinate[(np.array(tile_id),)]
        drift_direction = self.drift_dir[(np.array(tile_id),)]

        return anode_drift_coord.reshape(drift.shape) + \
            drift_direction.reshape(drift.shape) * drift


    def in_fid(self, xyz, cathode_fid=0.0, field_cage_fid=0.0, anode_fid=0.0):
        '''
            Check if xyz point is contained in the specified fiducial volume

            :param xyz: point to check, array ``shape: (N,3)`` [cm]

            :param cathode_fid: fiducial boundary for cathode and anode [cm], ``float``, optional

            :param field_cage_fid: fiducial boundary for field cage walls [cm], ``float``, optional

            :returns: boolean array, ``shape: (N,)``, True indicates point is within fiducial volume

        '''
        # Define xyz coordinates of fiducial boundaries
        fid_cathode = np.array([cathode_fid, field_cage_fid, field_cage_fid])
        fid_anode = np.array([anode_fid, field_cage_fid, field_cage_fid])

        # Define drift regions
        positive_drift_regions = np.array([bound for bound in self.module_RO_bounds])
        negative_drift_regions = np.array([bound for bound in self.module_RO_bounds])
        
        for i in range(len(self.module_RO_bounds)):
            positive_drift_regions[i][0][0] = positive_drift_regions[i][1][0] - self.max_drift_distance
            negative_drift_regions[i][1][0] = negative_drift_regions[i][0][0] + self.max_drift_distance

        # Define fiducial boundaries for each drift region
        fid_positive_drift = np.array([[fid_cathode, fid_anode] for module in self.module_RO_bounds])
        fid_negative_drift = np.array([[fid_anode, fid_cathode] for module in self.module_RO_bounds])

        # Check if coordinate is in fiducial volume for any drift region
        # In very rare situations, a coordinate very close (<1e-7 cm) to a fiducial volume boundary
        # may be classified as outside the fiducial volume due to floating point precision
        # errors. To avoid this, we should explore rounding position coordinates based on detector
        # resolution. For now, we can tolerate treating these hits as outside the fiducial volume.
        coord_in_positive_drift_fid = ma.concatenate([np.expand_dims(\
                                    (xyz < np.expand_dims(boundary[1] - fid_positive_drift[i][1], 0)) &\
                                    (xyz > np.expand_dims(boundary[0] + fid_positive_drift[i][0], 0)), axis=-1)\
                                    for i,boundary in enumerate(positive_drift_regions)], axis=-1)
        coord_in_negative_drift_fid = ma.concatenate([np.expand_dims(\
                                    (xyz < np.expand_dims(boundary[1] - fid_negative_drift[i][1], 0)) &\
                                    (xyz > np.expand_dims(boundary[0] + fid_negative_drift[i][0], 0)), axis=-1)\
                                    for i,boundary in enumerate(negative_drift_regions)], axis=-1)
        in_positive_fid = ma.all(coord_in_positive_drift_fid, axis=1)
        in_negative_fid = ma.all(coord_in_negative_drift_fid, axis=1)
        in_any_positive_fid = ma.any(in_positive_fid, axis=-1)
        in_any_negative_fid = ma.any(in_negative_fid, axis=-1)
        in_any_fid = in_any_positive_fid | in_any_negative_fid
        return in_any_fid
    

    def _get_module_RO_bounds(self):
        '''
            Get module_RO_bounds from pre-saved 2D pixel coordinates and anode drift coordinates 
        '''
        with open(self.det_geometry_file) as dgf:
            det_geometry_yaml = yaml.load(dgf, Loader=yaml.FullLoader)

        module_to_io_groups = det_geometry_yaml['module_to_io_groups']

        self._module_RO_bounds = []

        # Loop through modules
        for module_id in module_to_io_groups:  
            io_group, io_channel, chip_id, channel_id = self.pixel_coordinates_2D.keys()
            min_coord = np.finfo(self.pixel_coordinates_2D.dtype).min
            max_coord = np.finfo(self.pixel_coordinates_2D.dtype).max
            min_x, max_x = min_coord, max_coord
            min_y, max_y = min_coord, max_coord
            min_z, max_z = min_coord, max_coord
            
            # Loop through io_groups
            for iog in module_to_io_groups[module_id]:
                
                mask = (io_group == iog)

                # Get zy coordinates for io_group
                zy = self.pixel_coordinates_2D[(io_group[mask], io_channel[mask], chip_id[mask], channel_id[mask])]
            
                if (min_y == min_coord) and (max_y == max_coord) \
                    and (min_z == min_coord) and (max_z == max_coord):

                    # Assign min and max y,z coordinates for initial io_group
                    min_y, max_y = zy[:,1].min(), zy[:,1].max()
                    min_z, max_z = zy[:,0].min(), zy[:,0].max()

                else:
                    # Update min and max y,z coordinates based on subsequent io_group
                    min_y, max_y = min(min_y, zy[:,1].min()), max(max_y, zy[:,1].max())
                    min_z, max_z = min(min_z, zy[:,0].min()), max(max_z, zy[:,0].max())

                # Get x coordinates for anode corresponding to io_group
                tile_id = self.tile_id[(io_group[mask], io_channel[mask])]
                anode_drift_coordinate = np.unique(self.anode_drift_coordinate[(tile_id,)])[0]

                # For first io_group in loop, set min_x and max_x to io_group anode drift coordinate
                if (min_x == min_coord) and (max_x == max_coord):

                    min_x, max_x = anode_drift_coordinate, anode_drift_coordinate

                # For subsequent io_groups, update min_x and max_x based on new io_group anode drift coordinates
                else: 
                    min_x, max_x = min(min_x, anode_drift_coordinate), max(max_x, anode_drift_coordinate)


            # Append module boundaries to module readout bounds list
            # Subtract/add half of pixel pitch to pixel 2D coordinates (yz here) to get true module boundaries
            self._module_RO_bounds.append(np.array([[min_x, min_y-self.pixel_pitch[module_id-1]/2., min_z-self.pixel_pitch[module_id-1]/2.],
                                                    [max_x, max_y+self.pixel_pitch[module_id-1]/2., max_z+self.pixel_pitch[module_id-1]/2.]]))
            
        self._module_RO_bounds = np.array(self._module_RO_bounds)


    @staticmethod
    def _rotate_pixel(pixel_pos, tile_orientation):
        return pixel_pos[0] * tile_orientation[2], pixel_pos[1] * tile_orientation[1]


    ## Light geometry methods ##
    @property
    def det_rel_pos(self):
        '''
            Lookup table for detector relative position, usage::

                resource['Geometry'].det_rel_pos[(tpc_index, detector_index)]

        '''
        return self._det_rel_pos

    @property
    def sipm_rel_pos(self):
        '''
            Lookup table for detector relative position, usage::

                resource['Geometry'].sipm_rel_pos[(adc_index, channel_index)]

        '''
        return self._sipm_rel_pos


    @property
    def det_id(self):
        '''
            Lookup table for TPC and detector id, usage::

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

    @property
    def sipm_abs_pos(self):
        '''
            Lookup table for SiPM center xyz coordinate, usage::

                resource['Geometry'].sipm_abs_pos[(adc_index, channel_index)]

        '''
        return self._sipm_abs_pos



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

    def get_sipm_rel_pos(self, adc, channel):
        ''' Returns 
        - TPC number starting at 0 (Attention not to be confused wiht CRS IO group)
        - TPC side with 0: -z direction 1: +z direction
        - Vertical position starting from bottom
        if channel not used, returns NaN,NaN,NaN
        '''

        # Get TPC/det number
        tpc = -1
        det = -1
        for tpc_temp, det_map in self.lrs_geometry_yaml["det_adc"].items():
            for det_temp, adc_map in det_map.items():
                if adc_map == adc:
                    if channel in self.lrs_geometry_yaml["det_chan"][tpc_temp][det_temp]:
                        tpc = tpc_temp
                        det = det_temp

        # Return NaN if adc-channel does not exist
        if tpc == -1 or det == -1:
            return [-1,-1,-1]

        det_type = self.lrs_geometry_yaml["adc_to_det_type"][adc]

        # Get TPC side
        side = self.lrs_geometry_yaml["det_side"][det]

        # Get vertical position
        # Get Y pos
        if det_type == 0:
            vert_pos = self.lrs_geometry_yaml["ch_to_vert_bin"][0][channel]
        else:
            vert_pos = self.lrs_geometry_yaml["ch_to_vert_bin"][1][channel]

        return tpc, side, vert_pos


    def get_sipm_abs_pos(self,adc,channel):
        '''Returns x,y,z position of each SiPM in cm (z=beam direction)
        if channel not used, returns NaN,NaN,NaN'''

        tpc, side, vert_pos = self.get_sipm_rel_pos(adc,channel)
        tpc_channel = vert_pos + side*(len(self.lrs_geometry_yaml["sipm_center"])//2)

        if np.isnan(tpc):
            return [-1,-1,-1]

        # Get X pos
        x_pos = self.det_geometry_yaml["tpc_offsets"][tpc//2][0] + self.lrs_geometry_yaml["tpc_center_offset"][tpc][0] 
        if tpc % 2 == 0:
            x_pos += self.lrs_geometry_yaml["sipm_center"][tpc_channel][0]
        else:
            x_pos -= self.lrs_geometry_yaml["sipm_center"][tpc_channel][0]

        # Get Y pos
        y_pos = self.det_geometry_yaml["tpc_offsets"][tpc//2][1] + self.lrs_geometry_yaml["tpc_center_offset"][tpc][1] 
        y_pos += self.lrs_geometry_yaml["sipm_center"][tpc_channel][1]

        # Get Z pos
        z_pos = self.det_geometry_yaml["tpc_offsets"][tpc//2][2] + self.lrs_geometry_yaml["tpc_center_offset"][tpc][2]
        if tpc % 2 == 0:
            z_pos += self.lrs_geometry_yaml["sipm_center"][tpc_channel][2]
        else:
            z_pos -= self.lrs_geometry_yaml["sipm_center"][tpc_channel][2]

        return x_pos, y_pos, z_pos


    ## Load light and charge geometry ##
    def load_geometry(self):
        self._load_charge_geometry()
        self._load_light_geometry()


    def _load_light_geometry(self):
        if self.rank == 0:
            logging.warning(f'Loading geometry from {self.lrs_geometry_file}...')

        # enforce that light geometry formatting is as expected
        assert_compat_version(self.lrs_geometry_yaml['format_version'], '0.2.0')

        mod_ids = np.array([v for v in self.det_geometry_yaml['module_to_tpcs'].keys()])
        tpc_ids = np.array([v for v in self.lrs_geometry_yaml['tpc_center_offset'].keys()])
        det_ids = np.array([v for v in self.lrs_geometry_yaml['det_center'].keys()])
        adc_ids = np.array([v for v in self.lrs_geometry_yaml['adc_to_det_type'].keys()])
        max_chan_per_det = max([len(chan) for tpc in self.lrs_geometry_yaml['det_chan'].values() for chan in tpc.values()])
        chan_ids = np.unique(sum([chan for tpc in self.lrs_geometry_yaml['det_chan'].values() for chan in tpc.values()],[]))

        tpc_mod = np.full(tpc_ids.shape, -1, dtype=int)
        for i, mod in enumerate(self.det_geometry_yaml["module_to_tpcs"]):
            for j, tpc in enumerate(self.det_geometry_yaml["module_to_tpcs"][mod]):
                tpc_mod[tpc] = i

        det_min_max = [(min(tpc_ids), max(tpc_ids)),
                       (min(det_ids), max(det_ids))]
        self._det_rel_pos = LUT('i4', *det_min_max, shape=(3,))
        self._det_rel_pos.default = -1

        shape = tpc_ids.shape + det_ids.shape
        det_adc = np.full(shape, -1, dtype=int)
        det_side = np.full(shape, -1, dtype=int)
        det_vert_pos = np.full(shape, -1, dtype=int)
        det_chan = np.full(shape + (max_chan_per_det,), -1, dtype=int)
        det_chan_mask = np.zeros(shape + (max_chan_per_det,), dtype=bool)
        det_bounds = np.zeros(shape + (2,3), dtype=float)
        for i, tpc in enumerate(tpc_ids):
            for j, det in enumerate(det_ids):
                det_adc[i,j] = self.lrs_geometry_yaml['det_adc'][tpc][det]
                det_side[i,j] = self.lrs_geometry_yaml['det_side'][det]
                det_vert_pos[i,j] = [key for key, value in self.lrs_geometry_yaml['det_side'].items() if value == det_side[i,j]].index(det)
                det_chan[i,j,:len(self.lrs_geometry_yaml['det_chan'][tpc][det])] = self.lrs_geometry_yaml['det_chan'][tpc][det]
                tpc_center = (np.array(self.lrs_geometry_yaml['tpc_center_offset'][tpc])
                    + np.array(self.det_geometry_yaml["tpc_offsets"][tpc_mod[i]]))
                det_geom = self.lrs_geometry_yaml['geom'][self.lrs_geometry_yaml['det_geom'][det]]
                det_center = np.array(self.lrs_geometry_yaml['det_center'][det])
                det_bounds[i,j,0] = tpc_center + det_center + np.array(det_geom['min'])
                det_bounds[i,j,1] = tpc_center + det_center + np.array(det_geom['max'])
                self._det_rel_pos[i,j] = np.array((tpc,det_side[i,j],det_vert_pos[i,j]))

        det_chan_mask = det_chan != -1

        det_adc, det_chan, tpc_ids, det_ids = np.broadcast_arrays(
            det_adc[...,np.newaxis],
            det_chan, tpc_ids[...,np.newaxis,np.newaxis], det_ids[...,np.newaxis])

        adc_chan_min_max = [(min(adc_ids), max(adc_ids)), 
                            (min(chan_ids), max(chan_ids))]
        self._sipm_abs_pos = LUT('f4', *adc_chan_min_max, shape=(3,))
        self._sipm_abs_pos.default = -1

        self._sipm_rel_pos = LUT('i4', *adc_chan_min_max, shape=(3,))
        self._sipm_rel_pos.default = -1

        self._det_id = LUT('i4', *adc_chan_min_max)
        self._det_id.default = -1

        self._det_bounds = LUT('f4', *det_min_max, shape=(2,3))
        self._det_bounds.default = 0.

        self._det_id[(det_adc[det_chan_mask], det_chan[det_chan_mask])] = det_ids[det_chan_mask]

        for adc in adc_ids:
            for chan in chan_ids:
                self._sipm_rel_pos[(adc,chan)] = np.array(self.get_sipm_rel_pos(adc,chan))
                self._sipm_abs_pos[(adc,chan)] = np.array(self.get_sipm_abs_pos(adc,chan))

        tpc_ids, det_ids, det_chan_mask = tpc_ids[...,0], det_ids[...,0], det_chan_mask[...,0]
        self._det_bounds[(tpc_ids[det_chan_mask], det_ids[det_chan_mask])] = det_bounds[det_chan_mask]


    def _load_charge_geometry(self):
        if self.rank == 0:
            logging.warning(f'Loading geometry from {self.crs_geometry_files}...')

        geometry_yamls = []
        for crs_geometry_file in self.crs_geometry_files:
            with open(crs_geometry_file) as gf:
                geometry_yamls.append(yaml.load(gf, Loader=yaml.FullLoader))
                if 'multitile_layout_version' not in geometry_yamls[-1].keys():
                    raise RuntimeError('Only multi-tile geometry configurations are accepted')

        with open(self.det_geometry_file) as dgf:
            det_geometry_yaml = yaml.load(dgf, Loader=yaml.FullLoader)

        self._max_drift_distance = det_geometry_yaml['drift_length'] # det geo yaml is already in cm

        module_to_io_groups = det_geometry_yaml['module_to_io_groups']

        tile_geometry = {}

        ## warning, this is assuming same number of tiles in all modules for now
        tiles = np.arange(1,len(geometry_yamls[0]['tile_chip_to_io'])*len(det_geometry_yaml['module_to_io_groups'])+1)
        io_groups = [
            geometry_yamls[self.crs_geometry_to_module[mod-1]]['tile_chip_to_io'][tile][chip] // 1000 + (mod-1)*2
            for mod in module_to_io_groups
            for tile in geometry_yamls[self.crs_geometry_to_module[mod-1]]['tile_chip_to_io']
            for chip in geometry_yamls[self.crs_geometry_to_module[mod-1]]['tile_chip_to_io'][tile]
        ]
        io_channels = [
            geometry_yamls[self.crs_geometry_to_module[mod-1]]['tile_chip_to_io'][tile][chip] % 1000
            for mod in module_to_io_groups
            for tile in geometry_yamls[self.crs_geometry_to_module[mod-1]]['tile_chip_to_io']
            for chip in geometry_yamls[self.crs_geometry_to_module[mod-1]]['tile_chip_to_io'][tile]
        ]
        chip_ids = [
            chip_channel // 1000
            for mod in module_to_io_groups
            for chip_channel in geometry_yamls[self.crs_geometry_to_module[mod-1]]['chip_channel_to_position']
        ]
        channel_ids = [
            chip_channel % 1000
            for mod in module_to_io_groups
            for chip_channel in geometry_yamls[self.crs_geometry_to_module[mod-1]]['chip_channel_to_position']
        ]
 
        pixel_coordinates_2D_min_max = [(min(v), max(v)) for v in (io_groups, io_channels, chip_ids, channel_ids)]
        self._pixel_coordinates_2D = LUT('f4', *pixel_coordinates_2D_min_max, shape=(2,))
        self._pixel_coordinates_2D.default = np.nan
    
        tile_min_max = [(min(v), len(module_to_io_groups)*max(v)) for v in (io_groups, io_channels)]
        self._tile_id = LUT('i4', *tile_min_max)
        self._tile_id.default = -1
    
        anode_min_max = [(min(tiles), len(module_to_io_groups)*max(tiles))]
        self._anode_drift_coordinate = LUT('f4', *anode_min_max)
        self._anode_drift_coordinate.default = 0.
        self._drift_dir = LUT('i1', *anode_min_max)
        self._drift_dir.default = 0.

        mod_centers = det_geometry_yaml['tpc_offsets']
        n_modules = len(det_geometry_yaml['module_to_io_groups'])
        n_tiles = 0
        for entry in tile_map:
            n_tiles += len(tile_map)

        print("IN THE GEOMETRY")
        print(n_tiles)
        print(n_modules)
        # DOUBLE WARNING!: I'm doing a terrible thing and hardcoding things based on
        #                  the first geometry file option in the list...
        #                  Please, fix me! (move into loop below)
        tile_or = geometry_yamls[0]['tile_orientations']
        tile_pos = geometry_yamls[0]['tile_positions']
        self._anode_drift_coordinate[(tiles,)] = [tile_pos[(tile-1)%n_tiles+1][0]/units.cm+mod_centers[((tile-1)//n_tiles)%n_modules][0] for tile in tiles] # convert mm -> cm for crs yaml; det geo yaml in cm already

        self._drift_dir[(tiles,)] = [tile_or[(tile-1)%n_tiles+1][0] for tile in tiles]
        self._module_RO_bounds = []
        self._pixel_pitch = [0.]*n_modules

        # Loop through modules
        for module_id in module_to_io_groups:
            geometry_yaml = geometry_yamls[self.crs_geometry_to_module[module_id-1]]
            pixel_pitch = geometry_yaml['pixel_pitch'] / units.cm # convert mm -> cm
            self._pixel_pitch[module_id-1] = pixel_pitch
            chip_channel_to_position = geometry_yaml['chip_channel_to_position']
            tile_orientations = geometry_yaml['tile_orientations']
            tile_positions = geometry_yaml['tile_positions']
            tile_chip_to_io = geometry_yaml['tile_chip_to_io']
            zs = np.array(list(chip_channel_to_position.values()))[:, 0] * pixel_pitch
            ys = np.array(list(chip_channel_to_position.values()))[:, 1] * pixel_pitch
            z_size = max(zs) - min(zs) + pixel_pitch
            y_size = max(ys) - min(ys) + pixel_pitch
            for tile in tile_chip_to_io:
                tile_orientation = tile_orientations[tile]
                tile_geometry[tile] = [pos / units.cm for pos in tile_positions[tile]], tile_orientations[tile] # convert mm -> cm

                for chip in tile_chip_to_io[tile]:
                    io_group_io_channel = tile_chip_to_io[tile][chip]
                    io_group = io_group_io_channel//1000 + (module_id-1)*len(det_geometry_yaml['module_to_io_groups'][module_id])
                    io_channel = io_group_io_channel % 1000
                    self._tile_id[([io_group], [io_channel])] = tile+(module_id-1)*len(tile_chip_to_io)

                    if self.network_agnostic == True:
                        # if we don't care about the network configuration, then we
                        # can just loop over every N io channels and add them to the LUT
                        start_io_channel = ((io_channel-1)//self.n_io_channels_per_tile)*self.n_io_channels_per_tile + 1
                        for io_channel in range(start_io_channel, start_io_channel+self.n_io_channels_per_tile):
                            self._tile_id[([io_group], [io_channel])] = tile

                for chip_channel in chip_channel_to_position:
                    chip = chip_channel // 1000
                    channel = chip_channel % 1000

                    try:
                        io_group_io_channel = tile_chip_to_io[tile][chip]
                    except KeyError:
                        if self.network_agnostic == True:
                            warnings.warn('Encountered an out-of-network chip, but because you enabled ``network_agnostic``, we will carry on with assumptions about the io group and io channel')
                            # using the info about the first chip on the tile for all others
                            io_group_io_channel = list(geometry_yaml['tile_chip_to_io'][tile].values())[0]
                        else:
                            continue

                    io_group = io_group_io_channel // 1000 + (module_id-1)*len(det_geometry_yaml['module_to_io_groups'][module_id])
                    io_channel = io_group_io_channel % 1000

                    z = chip_channel_to_position[chip_channel][0] * \
                        pixel_pitch - z_size / 2 + pixel_pitch / 2
                    y = chip_channel_to_position[chip_channel][1] * \
                        pixel_pitch - y_size / 2 + pixel_pitch / 2

                    z, y = self._rotate_pixel((z, y), tile_orientation)

                    z += tile_positions[tile][2]/units.cm # convert mm -> cm 
                    y += tile_positions[tile][1]/units.cm # convert mm -> cm
                    z += mod_centers[module_id-1][2] # det geo yaml is already in cm
                    y += mod_centers[module_id-1][1] # det geo yaml is already in cm
                    self._pixel_coordinates_2D[(io_group, io_channel, chip, channel)] = z, y

        # Determine module readout bounds
        self._get_module_RO_bounds()

        # Determine LAr detector bounds
        self._lar_detector_bounds = np.array([np.min(np.array([bound[0] for bound in self._module_RO_bounds]), axis=0),
                                              np.max(np.array([bound[1] for bound in self._module_RO_bounds]), axis=0)])
        
        # Determine cathode thickness
        cathode_x_coords = np.unique(np.array(mod_centers)[:,0])
        anode_to_cathode = np.min(np.array([abs(self.lar_detector_bounds[0][0] - cathode_x)
                                            for cathode_x in cathode_x_coords]))
        
        if self.max_drift_distance < anode_to_cathode:
            # Difference b/w max drift dist and anode-cathode dist is 1/2 cathode thickness
            self._cathode_thickness = abs(anode_to_cathode - self.max_drift_distance) * 2.0
        else: 
            self._cathode_thickness = 0.0
