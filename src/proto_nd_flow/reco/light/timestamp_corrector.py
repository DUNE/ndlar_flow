import numpy as np
import numpy.lib.recfunctions as rfn
from collections import defaultdict
import logging

from h5flow.core import H5FlowStage


class LightTimestampCorrector(H5FlowStage):
    '''
        Applies a linear correction term to the light system ``tai_ns`` timestamps
        to account for different clock frequencies.

        Creates a new timestamp for each event equal to::

            t_ns[adc] = tai_ns[adc] / (1. + slope[adc])

        Parameters:
         - ``t_ns_dset_name``: ``str``, path to output dataset
         - ``slope``: ``dict`` of ``adc_index : slope`` in units of seconds/second

        ``t_ns`` datatype::

            t_ns    f8(nadcs,), event PPS timestamp for each adc [ns]

    '''
    class_version = '0.0.0'

    default_t_ns_dset_name = 'light/t_ns'
    default_slope = defaultdict(float)

    def t_ns_dtype(self, nadcs, nchannels): return np.dtype([
        ('t_ns', 'f8', (nadcs, nchannels))
    ])

    def __init__(self, **params):
        super(LightTimestampCorrector, self).__init__(**params)

        self.t_ns_dset_name = params.get('t_ns_dset_name')

        self.slope = self.default_slope
        for key, val in params.get('slope', dict()).items():
            self.slope[key] = val

    def init(self, source_name):
        super(LightTimestampCorrector, self).init(source_name)

        events_dset = self.data_manager.get_dset(source_name)

        self.t_ns_dtype = self.t_ns_dtype(*events_dset.dtype['tai_ns'].shape[0:2])

        self.slope_array = np.zeros(self.t_ns_dtype['t_ns'].shape)
        for key, val in self.slope.items():
            self.slope_array[key] = val

        self.data_manager.set_attrs(self.t_ns_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    source_dset=source_name,
                                    slope=self.slope_array
                                    )
        # then set up new datasets
        self.data_manager.create_dset(self.t_ns_dset_name, dtype=self.t_ns_dtype)
        self.data_manager.create_ref(source_name, self.t_ns_dset_name)

    def run(self, source_name, source_slice, cache):
        super(LightTimestampCorrector, self).run(source_name, source_slice, cache)

        tai_ns = cache[source_name]['tai_ns']

        if len(tai_ns):
            t_ns = tai_ns / (1. + self.slope_array.reshape((1,) + tai_ns.shape[1:]))

            t_ns_data = np.empty(len(t_ns), dtype=self.t_ns_dtype)
            t_ns_data['t_ns'] = t_ns
        else:
            t_ns_data = np.empty((0,), dtype=self.t_ns_dtype)

        t_ns_slice = self.data_manager.reserve_data(self.t_ns_dset_name, len(t_ns_data))
        self.data_manager.write_data(self.t_ns_dset_name, t_ns_slice, t_ns_data)

        ref = np.c_[source_slice, t_ns_slice]
        self.data_manager.write_ref(source_name, self.t_ns_dset_name, ref)
