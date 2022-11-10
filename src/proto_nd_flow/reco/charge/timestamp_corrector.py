import numpy as np
import numpy.lib.recfunctions as rfn
from collections import defaultdict
import logging

from h5flow.core import H5FlowStage, resources


class TimestampCorrector(H5FlowStage):
    '''
        Corrects larpix clock timestamps due to slightly different PACMAN clock
        frequencies - creates a new dataset with 1:1 relationship to packets, but
        filled with a single value representing the true number of 10MHz clock
        cycles since the SYNC.

        The applied correction factor is given by::

            ts_corrected = (ts_original - correction_factor[0]) / (1. + correction_factor[1])

        Parameters:
         - ``ts_dset_name`` : ``str``, required, output dataset path
         - ``packets_dset_name`` : ``str``, required, input dataset path for packets
         - ``correction`` : ``dict``, optional, ``iogroup : [<constant offset>, <slope>]`` pairs

        The ``packets_dset_name`` is required in the data cache along with
        its indices under the name of ``'{packets_dset_name}_index'``.

        Requires RunData resource in workflow.

        Example config::

            timestamp_corrector:
                classname: TimestampCorrector
                requires:
                    - 'charge/packets'
                    - name: 'charge/packets_index'
                      path: 'charge/packets'
                      index_only: True
                params:
                    ts_dset_name: 'charge/packets_corr_ts'
                    packets_dset_name: 'charge/packets'
                    correction:
                        1: [-9.597, 4.0021e-6]
                        2: [-9.329, 1.1770e-6]

        ``ts`` datatype:

            id  u8, unique identifier
            ts  f8, PPS timestamp after correction for timestamp drift [ticks]

    '''
    class_version = '1.0.0'

    ts_dtype = np.dtype([
        ('id', 'u8'),  # unique identifier
        ('ts', 'f8')  # PPS timestamp after correcting for timestamp drift [ticks]
    ])
    correction_dtype = np.dtype([('iogroup', 'u1'), ('offset', 'f8'), ('slope', 'f8')])

    def __init__(self, **params):
        super(TimestampCorrector, self).__init__(**params)

        self.ts_dset_name = params.get('ts_dset_name')
        self.packets_dset_name = params.get('packets_dset_name')

        self.correction = defaultdict(self._default_correction)
        for key, val in params.get('correction', dict()).items():
            self.correction[key] = val

    @staticmethod
    def _default_correction():
        return (0., 0.)

    def init(self, source_name):
        super(TimestampCorrector, self).init(source_name)

        # check if MC
        if resources['RunData'].is_mc:
            # bypass correction for MC
            self.correction = defaultdict(self._default_correction)

        # write all configuration variables to the dataset
        self.data_manager.set_attrs(self.ts_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    source_dset=source_name,
                                    packets_dset=self.packets_dset_name
                                    )
        correction_arr = np.empty((len(self.correction.keys()),), dtype=self.correction_dtype)
        for i, (key, val) in enumerate(self.correction.items()):
            correction_arr[i]['iogroup'] = key
            correction_arr[i]['slope'] = val[1]
            correction_arr[i]['offset'] = val[0]
        self.data_manager.set_attrs(self.ts_dset_name,
                                    correction=correction_arr
                                    )

        # then set up new datasets
        self.data_manager.create_dset(self.ts_dset_name, dtype=self.ts_dtype)
        self.data_manager.create_ref(self.packets_dset_name, self.ts_dset_name)

    def run(self, source_name, source_slice, cache):
        super(TimestampCorrector, self).run(source_name, source_slice, cache)

        # get packet data from cache
        packets_data = cache[self.packets_dset_name]
        packets_index = cache[self.packets_dset_name + '_index']

        mask = ~rfn.structured_to_unstructured(packets_data.mask).any(axis=-1)

        packets_data = packets_data.data[mask]
        packets_index = packets_index.data[mask]

        # apply timestamp correction
        ts_corr_data = np.empty((len(packets_data),), dtype=self.ts_dtype)
        if len(packets_data):
            for io_group in np.unique(packets_data['io_group']):
                mask = packets_data['io_group'] == io_group
                ts_corr_data['ts'][mask] = (packets_data[mask]['timestamp'].astype('f8') - self.correction[io_group][0]) / (1. + self.correction[io_group][1])

        # save corrected timestamps
        ts_slice = self.data_manager.reserve_data(self.ts_dset_name, len(ts_corr_data))
        if len(ts_corr_data):
            ts_corr_data['id'] = packets_index
        self.data_manager.write_data(self.ts_dset_name, ts_slice, ts_corr_data)

        # save references
        #   packet -> packet_ts (1:1)
        ref = np.c_[packets_index, ts_corr_data['id']] if len(packets_data) else np.empty((0, 2))
        self.data_manager.write_ref(self.packets_dset_name, self.ts_dset_name, ref)
