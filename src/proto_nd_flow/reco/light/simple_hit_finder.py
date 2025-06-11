import numpy as np
import numpy.ma as ma

from h5flow.core import H5FlowStage, resources

import module0_flow.util.units as units

class WaveformHitFinder(H5FlowStage):
    '''
        Find waveforms from light events that surpass a simple threshold. This script is meant to run be run on filtered summed raw waveforms.

        Parameters:
         - ``sum_wvfm_dset_name``: ``str``, path to input filtered summed waveforms
         - ``hits_dset_name``: ``str``, path to output hits dataset
         - ``threshold``: ``dict`` of ``dict`` containing sets of ``tpc_index: {channel_index: threshold, ...}`` used for hit finding. A fixed global value can also be specified with a single ``float`` value

         ``sum_wvfm_dset_name`` is required in the cache.

         Requires RunData and Geometry resources in workflow.

         ``hits`` datatype::

            id                  u4,             unique identifier
            tpc                 u1,             tpc (for sum_hit)
            sum_chan            u1,             detector id
            trap_type           u2,             light detector type (LCM = 1, ACL = 0)
            boundary            f4(3),          (x,y,z) boundaries of det
            amplitude           f4,             peak adc value
            ts_pps              f4,             PPS timestamp of light event
            unix                i8,             UNIX timestamp of light event with second-level precision
    '''
    class_version = '0.0.0'

    default_hits_dset_name = 'light/simple_hits'
    default_threshold = 500
    
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
                    ('amplitude', 'f4'),
                    ('ts_pps', 'f8'),
                    ('unix', 'i8')
                ])
        
        # create datasets and references
        self.data_manager.create_dset(self.hits_dset_name,
                                      dtype=self.hits_dtype)
        self.data_manager.create_ref(source_name, self.hits_dset_name)
        #self.data_manager.set_attrs(self.hits_dset_name,
        #                            classname=self.classname,
        #                            class_version=self.class_version,
        #                            wvfm_dset=self.sum_wvfm_dset_name
        #                            )

    def run(self, source_name, source_slice, cache):
        super(WaveformHitFinder, self).run(source_name, source_slice, cache)
        wvfms = np.array(cache[self.sum_wvfm_dset_name].reshape(cache[source_name].shape)[
            'samples'])  # 1:1 relationship
        events = np.array(cache[source_name])
        
        hits_data = np.zeros((0,), dtype=self.hits_dtype)
        hits_event_id = []

        events_tai_ns = []
        events_utime_ms = []
        for i in range(len(events)):
            event = events[i]
            tai_ns = event['tai_ns'][event['tai_ns'] != 0][0]*1e-3
            utime_ms = int(event['utime_ms'][event['utime_ms'] != 0][0]*1e-3)
            events_tai_ns.append(tai_ns)
            events_utime_ms.append(utime_ms)
            
        max_of_wvfms = np.max(wvfms, axis=3)
        indices = np.where(max_of_wvfms > self.threshold)
        for i, (event_index, tpc, sum_chan) in enumerate(zip(indices[0], indices[1], indices[2])):
            hit_data = np.zeros((1,), dtype=self.hits_dtype)
            hit_data['tpc'] = tpc
            hit_data['sum_chan'] = sum_chan
            hit_data['trap_type'] = resources['Geometry'].sum_chan_to_trap_type[(tpc, sum_chan)]
            hit_data['boundary'] = resources['Geometry'].sum_chan_bounds[(tpc, sum_chan)]
            hit_data['amplitude'] = max_of_wvfms[event_index, tpc, sum_chan]
            hit_data['ts_pps'] = events_tai_ns[event_index]
            hit_data['unix'] = events_utime_ms[event_index]
            hits_data = np.concatenate((hits_data, hit_data))
            hits_event_id.append(events[event_index]['id'])

        hit_slice = self.data_manager.reserve_data(self.hits_dset_name, len(hits_data))
        hits_data['id'] = hit_slice.start + np.arange(len(hits_data), dtype=int)
        self.data_manager.write_data(self.hits_dset_name, hit_slice, hits_data)
        hits_event_id = np.array(hits_event_id)
        if len(hits_data):
            ref = np.c_[hits_event_id, hits_data['id']]
        else:
            ref = np.empty((0, 2))
        self.data_manager.write_ref(source_name, self.hits_dset_name, ref)
