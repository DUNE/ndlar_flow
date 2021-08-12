import logging
import json
import numpy as np

from h5flow.core import H5FlowResource, resources

from module0_flow.util.lut import LUT, write_lut, read_lut
from module0_flow.util.compat import assert_compat_version


class DisabledChannels(H5FlowResource):
    class_version = '0.0.0'

    default_path = 'disabled_channels'

    def __init__(self, **params):
        super(DisabledChannels, self).__init__(**params)

        self.path = params.get('path', self.default_path)
        self.disabled_channels_list = params.get('disabled_channels_list', None)
        self.missing_asic_list = params.get('missing_asic_list', None)

    def init(self, source_name):
        # create group (if not present)
        self.data_manager.set_attrs(self.path)
        # load data (if present)
        self.data = dict(self.data_manager.get_attrs(self.path))

        if not self.data:
            # no data stored in file, generate it
            self._disabled_channel_lut, disabled_xy = self.load_disabled_channels_lut(self.disabled_channels_list, self.missing_asic_list)
            self.data['classname'] = self.classname
            self.data['class_version'] = self.class_version
            self.data['disabled_channels_list'] = (self.disabled_channels_list
                                                   if self.disabled_channels_list is not None
                                                   else '')
            self.data['missing_asic_list'] = (self.missing_asic_list
                                              if self.missing_asic_list is not None
                                              else '')
            self.data['disabled_xy'] = disabled_xy
            self.data_manager.set_attrs(self.path, **self.data)

            write_lut(self.data_manager, self.path, self.disabled_channel_lut,
                      'lut')
        else:
            assert_compat_version(self.class_version, self.data['class_version'])

            self._disabled_channel_lut = read_lut(self.data_manager, self.path,
                                                  'lut')

        logging.info(f'N disabled channels: {len(self.disabled_xy)}')
        logging.info(f'Disabled channel LUT size: '
                     f'{self.disabled_channel_lut.nbytes/1024/1024:0.02f}MB')

    @property
    def disabled_xy(self):
        return self.data['disabled_xy']

    @property
    def disabled_channel_lut(self):
        return self._disabled_channel_lut

    @staticmethod
    def load_disabled_channels_lut(disabled_channels_list=None,
                                   missing_asic_list=None):
        '''
        Loads a disabled channels lookup-table from the json formatted filenames::

            disabled_channels_list
            missing_asic_list

        ``disabled_channels_list`` contains ``chip-key: [channel_id]`` pairs of
        disabled channels that are defined within the geometry, but should be
        considered as disabled. The ``Geometry`` resource is used to find the xy
        locations of these pixels.

        ``missing_asic_list`` contains ``io_group: [[x,y], ...]`` pixel positions
        that should be considered as disabled regions.

        :returns: a boolean ``module0_flow.util.lut.LUT`` instance, with keys of ``(io_group, pixel_x.astype(int), pixel_y.astype(int))`` and a ``list`` of xy coordinates for each disabled channel

        '''
        io_group = list()
        xy = np.empty((0, 2))

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

            pixel_xy = resources['Geometry'].pixel_xy
            chip_key = (np.array(io_group), np.array(io_channel),
                        np.array(chip_id), np.array(channel_id))
            xy = pixel_xy[chip_key]

        if missing_asic_list is not None:
            # then load missing asic pixels
            with open(missing_asic_list, 'r') as fi:
                data = json.load(fi)

            # add to lists
            for io_group_ in data:
                for asic in data[io_group_]:
                    io_group.append(int(io_group_))
                    xy = np.append(xy, np.array([asic]), axis=0)

        disable_channels_lut = LUT(bool,
                                   (min(io_group), max(io_group)),
                                   (min(xy[:, 0].astype(int)) - 1,
                                    max(xy[:, 0].astype(int)) + 1),
                                   (min(xy[:, 1].astype(int)) - 1,
                                    max(xy[:, 1].astype(int)) + 1),
                                   default=False)
        # apply a fudge factor to account for any rounding errors
        for dx in (+1, 0, -1):
            for dy in (+1, 0, -1):
                disable_channels_lut[(io_group, xy[:, 0].astype(int) + dx,
                                      xy[:, 1].astype(int) + dy)] = True

        return disable_channels_lut, xy
