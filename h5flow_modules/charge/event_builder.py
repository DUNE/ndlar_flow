import numpy as np
from collections import defaultdict
import logging

from h5flow.core import H5FlowStage

class EventBuilder(H5FlowStage):
    '''
        High-level event builder - converts raw_events with external trigger and
        hit associations into a high level event

        Parameters:
         - ``events_dset_name`` : ``str``, required, output dataset path
         - ``hits_dset_name`` : ``str``, required, input dataset path for hits
         - ``ext_trigs_dset_name`` : ``str``, required, input dataset path for external triggers

        Both the ``hits_dset_name`` and ``ext_trigs_dset_name`` are required in
        the data cache.

        Example config::

            event_builder:
                classname: EventBuilder
                requires:
                    - 'charge/hits'
                    - 'charge/ext_trigs'
                params:
                    events_dset_name: 'charge/events'
                    hits_dset_name: 'charge/hits'
                    ext_trigs_dset_name: 'charge/ext_trigs'

    '''
    class_version = '0.0.0'

    events_dtype = np.dtype([
        ('id', 'u4'), # unique identifier
        ('nhit', 'u4'), # number of hits in event
        ('q', 'f8'), # total charge in event [mV]
        ('ts_start', 'f8'), ('ts_end', 'f8'), # minimum and maximum corrected PPS timestamp [ticks]
        ('n_ext_trigs', 'u4'), # number of external triggers
        ('unix_ts', 'u8'), # unix timestamp [s since epoch]
        ])

    def __init__(self, **params):
        super(EventBuilder,self).__init__(**params)

        self.events_dset_name = params.get('events_dset_name')
        self.hits_dset_name = params.get('hits_dset_name')
        self.ext_trigs_dset_name = params.get('ext_trigs_dset_name')

    def init(self, source_name):
        # save all config info
        self.data_manager.set_attrs(self.events_dset_name,
            classname=self.classname,
            class_version=self.class_version,
            source_dset=source_name,
            hits_dset=self.hits_dset_name,
            ext_trigs_dset=self.ext_trigs_dset_name
            )

        # then set up new datasets
        self.data_manager.create_dset(self.events_dset_name, dtype=self.events_dtype)
        self.data_manager.create_ref(self.events_dset_name, self.hits_dset_name)
        self.data_manager.create_ref(self.events_dset_name, self.ext_trigs_dset_name)

    def run(self, source_name, source_slice, cache):
        raw_event_data = cache[source_name]
        hits_data = cache[self.hits_dset_name]
        ext_trigs_data = cache[self.ext_trigs_dset_name]

        # write event
        events_slice = self.data_manager.reserve_data(self.events_dset_name, source_slice)
        events_arr = np.zeros((len(raw_event_data,)), dtype=self.events_dtype)
        events_arr['id'] = raw_event_data['id']
        events_arr['unix_ts'] = raw_event_data['unix_ts']
        events_arr['nhit'] = [len(hits) for hits in hits_data]
        events_arr['q'] = [np.sum(hits['q']) for hits in hits_data]
        events_arr['ts_start'] = [np.min(np.r_[hits_data[i]['ts'], ext_trigs_data[i]['ts']]) for i in range(len(raw_event_data))]
        events_arr['ts_end'] = [np.max(np.r_[hits_data[i]['ts'], ext_trigs_data[i]['ts']]) for i in range(len(raw_event_data))]
        events_arr['n_ext_trigs'] = [len(ext_trigs) for ext_trigs in ext_trigs_data]
        self.data_manager.write_data(self.events_dset_name, events_slice, events_arr)

        # save references
        self.data_manager.reserve_ref(self.events_dset_name, self.hits_dset_name, source_slice)
        ref = [hits['id'] if len(hits) else slice(0,0) for hits in hits_data]
        self.data_manager.write_ref(self.events_dset_name, self.hits_dset_name, source_slice, ref)

        self.data_manager.reserve_ref(self.events_dset_name, self.ext_trigs_dset_name, source_slice)
        ref = [trigs['id'] if len(trigs) else slice(0,0) for trigs in ext_trigs_data]
        self.data_manager.write_ref(self.events_dset_name, self.ext_trigs_dset_name, source_slice, ref)

