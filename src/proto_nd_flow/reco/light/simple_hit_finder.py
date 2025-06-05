import numpy as np
import numpy.ma as ma
from collections import defaultdict
import scipy.interpolate
from scipy.signal import butter, filtfilt

from h5flow.core import H5FlowStage, resources

import module0_flow.util.units as units
import time

class WaveformHitFinder(H5FlowStage):
    '''
        Find waveforms from light events that surpass a simple threshold. This script is meant to run be run on filtered summed raw waveforms.

        Parameters:
         - ``sum_wvfm_dset_name``: ``str``, path to input filtered summed waveforms
         - ``hits_dset_name``: ``str``, path to output hits dataset
         - ``threshold``: ``dict`` of ``dict`` containing sets of ``tpc_index: {channel_index: threshold, ...}`` used for hit finding. A fixed global value can also be specified with a single ``float`` value

         ``sum_wvfm_dset_name`` and ``t_ns_dset_name`` are required in the cache.

         Requires RunData and Geometry resources in workflow.

         ``hits`` datatype::

            id                  u4,             unique identifier
            tpc                 u1,             tpc (for sum_hit)
            sum_chan            u1,             detector id
            boundary            f4(3),          (x,y,z) boundaries of det
            samples             f4(nsamples,),  waveform adc values
            amplitude           f4,             peak adc value
    '''
    class_version = '2.0.0'

    default_hits_dset_name = 'light/simple_hits'
    default_threshold = 500
    default_mask = []

    def __init__(self, **params):
        super(WaveformHitFinder, self).__init__(**params)
        self.sum_wvfm_dset_name = params.get('sum_wvfm_dset_name')
        self.hits_dset_name = params.get('hits_dset_name', self.default_hits_dset_name)
        self.threshold = params.get('threshold', self.default_threshold)
            
    def init(self, source_name):
        super(WaveformHitFinder, self).init(source_name)

        self.hits_dtype = np.dtype([
                    ('id', 'u4'),
                    ('tpc', 'u1'),
                    ('sum_chan', 'u1'),
                    ('trap_type', 'u1'),
                    ('boundary', 'f4', (2,3)),
                    #('samples', 'f4', (nsamples,)),
                    ('amplitude', 'f4'),
                    ('ts_pps', 'f8'),
                    ('unix', 'i8')
                ])
        
        # create datasets and references
        self.data_manager.create_dset(self.hits_dset_name,
                                      dtype=self.hits_dtype)
        self.data_manager.create_ref(source_name, self.hits_dset_name)
        self.data_manager.set_attrs(self.hits_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    wvfm_dset=self.sum_wvfm_dset_name
                                    )

    def run(self, source_name, source_slice, cache):
        super(WaveformHitFinder, self).run(source_name, source_slice, cache)
        wvfms = np.array(cache[self.sum_wvfm_dset_name].reshape(cache[source_name].shape)[
            'samples'])  # 1:1 relationship
        events = np.array(cache[source_name])
        
        hits_data = np.zeros((0,), dtype=self.hits_dtype)
        hits_event_id = []
        for i in range(len(events)):
            event = events[i]
            
            tai_ns = np.unique(event['tai_ns'])
            tai_ns = tai_ns[tai_ns != 0][0]*1e-3
            utime_ms = np.unique(event['utime_ms'])
            utime_ms = int(utime_ms[utime_ms != 0][0]*1e-3)

            wvfms_arr = wvfms[i]
            max_of_wvfms = np.max(wvfms_arr, axis=2)
            indices = np.where(max_of_wvfms > self.threshold)

            for tpc, sum_chan in zip(indices[0], indices[1]):
                hit_data = np.zeros((1,), dtype=self.hits_dtype)
                hit_data['tpc'] = tpc
                hit_data['sum_chan'] = sum_chan
                hit_data['trap_type'] = resources['Geometry'].sum_chan_to_trap_type[(tpc, sum_chan)]
                hit_data['boundary'] = resources['Geometry'].sum_chan_bounds[(tpc, sum_chan)]
                #hit_data['samples'] = wvfms_arr[tpc, sum_chan, :]
                hit_data['amplitude'] = max_of_wvfms[tpc, sum_chan]
                hit_data['ts_pps'] = tai_ns
                hit_data['unix'] = utime_ms
                hits_data = np.concatenate((hits_data, hit_data))
                hits_event_id.append(event['id'])
                
        hit_slice = self.data_manager.reserve_data(self.hits_dset_name, len(hits_data))
        hits_data['id'] = hit_slice.start + np.arange(len(hits_data), dtype=int)
        self.data_manager.write_data(self.hits_dset_name, hit_slice, hits_data)

        hits_event_id = np.array(hits_event_id)
        if len(hits_data):
            ref = np.c_[hits_event_id, hits_data['id']]
        else:
            ref = np.empty((0, 2))
        self.data_manager.write_ref(source_name, self.hits_dset_name, ref)