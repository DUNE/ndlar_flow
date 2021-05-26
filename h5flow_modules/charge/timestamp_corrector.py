import numpy as np
from collections import defaultdict
import logging

from h5flow.core import H5FlowStage

class TimestampCorrector(H5FlowStage):
    '''
        Corrects larpix clock timestamps due to slightly different PACMAN clock
        frequencies - creates a new dataset with 1:1 relationship to packets.

        The applied correction factor is given by::

            ts_corrected = ts_original / (1. + correction_factor)

        Parameters:
         - ``ts_dset_name`` : ``str``, required, output dataset path
         - ``packets_dset_name`` : ``str``, required, input dataset path for packets
         - ``correction`` : ``dict``, optional, ``iogroup : [<constant offset>, <slope>]`` pairs

        The ``packets_dset_name`` is required in the data cache.

        Example config::

            timestamp_corrector:
                classname: TimestampCorrector
                requires:
                    - 'charge/packets'
                params:
                    ts_dset_name: 'charge/packets_corr_ts'
                    packets_dset_name: 'charge/packets'
                    correction:
                        1: [-9.5, 3.56e-6]
                        2: [-9.5, 0.93e-6]

    '''
    class_version = '0.0.0'

    default_correction = lambda : 0.

    ts_dtype = np.dtype([
        ('id','u4'), # unique identifier
        ('ts','f8') # PPS timestamp after correcting for timestamp drift [ticks]
        ])
    correction_dtype = np.dtype([('iogroup','u1'),('offset','f8'),('slope','f8')])

    def __init__(self, **params):
        super(TimestampCorrector, self).__init__(**params)

        self.ts_dset_name = params.get('ts_dset_name')
        self.packets_dset_name = params.get('packets_dset_name')

        self.correction = defaultdict(self.default_correction)
        for key,val in params.get('correction', dict()).items():
            self.correction[key] = val

    def init(self, source_name):
        # write all configuration variables to the dataset
        self.data_manager.set_attrs(self.ts_dset_name,
            classname=self.classname,
            class_version=self.class_version,
            source_dset=source_name,
            packets_dset=self.packets_dset_name
            )
        correction_arr = np.empty((len(self.correction.keys()),), dtype=self.correction_dtype)
        for i,(key,val) in enumerate(self.correction.items()):
            correction_arr[i]['iogroup'] = key
            correction_arr[i]['slope'] = val[1]
            correction_arr[i]['offset'] = val[0]
        self.data_manager.set_attrs(self.ts_dset_name,
            correction=correction_arr
            )

        # then set up new datasets
        self.data_manager.create_dset(self.ts_dset_name, dtype=self.ts_dtype)
        self.data_manager.create_ref(source_name, self.ts_dset_name)

    def run(self, source_name, source_slice, cache):
        # manipulate data from cache
        packets_data = cache[self.packets_dset_name]
        packets_arr = np.concatenate(packets_data, axis=0) if len(packets_data) else np.empty((0,))

        # apply timestamp correction
        ts_corr_data = np.empty((len(packets_arr),), dtype=self.ts_dtype)
        if len(packets_arr):
            unique_io_groups = np.unique(packets_arr['io_group'])
            for io_group in unique_io_groups:
                mask = packets_arr['io_group'] == io_group
                ts_corr_data[mask]['ts'] = (packets_arr[mask]['timestamp'] - self.correction[io_group][0]) / (1. + self.correction[io_group][1])

        # save corrected timestamps
        ts_slice = self.data_manager.reserve_data(self.ts_dset_name, len(ts_corr_data))
        if len(ts_corr_data):
            ts_corr_data['id'] = np.arange(ts_slice.start, ts_slice.stop)
        self.data_manager.write_data(self.ts_dset_name, ts_slice, ts_corr_data)

        # save references
        event_lengths = [len(p) for p in packets_data]
        self.data_manager.reserve_ref(source_name, self.ts_dset_name, source_slice)
        ref = [ts_corr_data['id'][sum(event_lengths[:i]), sum(event_lengths[:i+1])] for i in range(len(packets_data))]
        self.data_manager.write_ref(source_name, self.ts_dset_name, source_slice, ref)
