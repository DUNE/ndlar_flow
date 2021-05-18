import numpy as np
from collections import defaultdict
import logging

from h5flow.core import H5FlowStage

class TimestampCorrector(H5FlowStage):
    class_version = '0.0.0'

    default_correction = lambda : 0.

    ts_dtype = 'f8' #
    correction_dtype = np.dtype([('iogroup','u1'),('slope','f8')])

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
            correction_arr[i]['slope'] = val
        self.data_manager.set_attrs(self.ts_dset_name,
            correction=correction_arr
            )

        # then set up new datasets
        self.data_manager.create_dset(self.ts_dset_name, dtype=self.ts_dtype)
        self.data_manager.create_ref(source_name, self.ts_dset_name)

    def run(self, source_name, source_slice, cache):
        # manipulate data from cache
        packets_data = cache[self.packets_dset_name]
        packets_arr = np.concatenate(packets_data, axis=0)

        # apply timestamp correction
        ts_corr_data = np.empty((len(packets_arr),), dtype=self.ts_dtype)
        unique_io_groups = np.unique(packets_arr['io_group'])
        for io_group in unique_io_groups:
            mask = packets_arr['io_group'] == io_group
            ts_corr_data[mask] = packets_arr[mask]['timestamp'] / (1. + self.correction[io_group])

        # save corrected timestamps
        ts_slice = self.data_manager.reserve_data(self.ts_dset_name, len(ts_corr_data))
        self.data_manager.write_data(self.ts_dset_name, ts_slice, ts_corr_data)

        # save references
        event_lengths = [len(p) for p in packets_data]
        self.data_manager.reserve_ref(source_name, self.ts_dset_name, source_slice)
        ref = [slice(sum(event_lengths[:i])+ts_slice.start, sum(event_lengths[:i+1])+ts_slice.start) for i in range(len(packets_data))]
        self.data_manager.write_ref(source_name, self.ts_dset_name, source_slice, ref)
