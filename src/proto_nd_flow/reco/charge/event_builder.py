import numpy as np
import numpy.ma as ma
import numpy.lib.recfunctions as rfn
import numpy.typing as npt
from collections import defaultdict
import logging

from h5flow.core import H5FlowStage, resources


class EventBuilder(H5FlowStage):
    '''
        High-level event builder - converts raw_events with external trigger and
        hit associations into a high level event

        Parameters:
         - ``events_dset_name`` : ``str``, required, output dataset path
         - ``hits_dset_name`` : ``str``, required, input dataset path for hits
         - ``ext_trigs_dset_name`` : ``str``, required, input dataset path for external triggers
         - ``pps_delay_corrector_enabled`` : ``bool``, optional, whether to correct unix_ts_usec
             for the GPS-to-PPS delay (see ``pps_delay_extractor_enabled'' in RawEventGenerator)
         - ``pps_delay_corrector_config`` : ``dict``, optional, modify parameters of PPS corrector

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

        ``events`` datatype::

            id              u8, unique identifier per event
            nhit            u4, number of hits in event
            q               f8, total charge in event [mV]
            ts_start        f8, first external trigger or hit corrected PPS timestamp [ticks]
            ts_end          f8, last external trigger of hit corrected PPS timestamp [ticks]
            n_ext_trigs     u4, number of external triggers in event
            unix_ts         u8, unix timestamp (from PACMAN) of event [s since epoch]
            unix_ts_usec    u4, estimated microsecond component of event time

    '''
    class_version = '1.0.0'

    events_dtype = np.dtype([
        ('id', 'u8'),
        ('nhit', 'u4'),
        ('ADC', 'f8'),
        ('ts_start', 'f8'), ('ts_end', 'f8'),
        ('n_ext_trigs', 'u4'),
        ('unix_ts', 'u8'),
        ('unix_ts_usec', 'u4'),
    ])

    def __init__(self, **params):
        super(EventBuilder, self).__init__(**params)

        self.events_dset_name = params.get('events_dset_name')
        self.hits_dset_name = params.get('hits_dset_name')
        self.ext_trigs_dset_name = params.get('ext_trigs_dset_name')

        self.pps_delay_corrector_enabled = \
            params.get('pps_delay_corrector_enabled', False)
        self.pps_delay_corrector_config = \
            params.get('pps_delay_corrector_config', {})
        self.pps_delay_correction = 0

    def init(self, source_name):
        super(EventBuilder, self).init(source_name)

        # save all config info
        self.data_manager.set_attrs(self.events_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    source_dset=source_name,
                                    hits_dset=self.hits_dset_name,
                                    ext_trigs_dset=self.ext_trigs_dset_name,
                                    pps_delay_corrector_enabled=self.pps_delay_corrector_enabled)

        # then set up new datasets
        self.data_manager.create_dset(self.events_dset_name, dtype=self.events_dtype)
        self.data_manager.create_ref(source_name, self.events_dset_name)
        self.data_manager.create_ref(self.events_dset_name, self.hits_dset_name)
        self.data_manager.create_ref(self.events_dset_name, self.ext_trigs_dset_name)

        if self.pps_delay_corrector_enabled:
            self.pps_delay_correction = self.get_pps_delay_correction()

    def run(self, source_name, source_slice, cache):
        super(EventBuilder, self).run(source_name, source_slice, cache)

        raw_event_data = cache[source_name]
        hits_data = cache[self.hits_dset_name]
        ext_trigs_data = cache[self.ext_trigs_dset_name]

        hits_mask = ~rfn.structured_to_unstructured(hits_data.mask).any(axis=-1)
        ext_trigs_mask = ~rfn.structured_to_unstructured(ext_trigs_data.mask).any(axis=-1)

        # write event
        events_slice = self.data_manager.reserve_data(self.events_dset_name, source_slice)
        events_arr = np.zeros((len(raw_event_data,)), dtype=self.events_dtype)
        events_arr['id'] = raw_event_data['id']
        events_arr['unix_ts'] = raw_event_data['unix_ts']
        events_arr['nhit'] = np.count_nonzero(hits_mask, axis=-1)
        events_arr['ADC'] = hits_data['ADC'].sum(axis=-1)
        ts = ma.concatenate((hits_data['ts_pps'], ext_trigs_data['ts']), axis=-1)
        events_arr['ts_start'] = ts.min(axis=-1)
        events_arr['ts_end'] = ts.max(axis=-1)
        events_arr['n_ext_trigs'] = np.count_nonzero(ext_trigs_mask, axis=-1)
        events_arr['unix_ts_usec'] = self.get_unix_ts_usec(events_arr)

        self.data_manager.write_data(self.events_dset_name, events_slice, events_arr)

        # save references
        self.data_manager.write_ref(self.events_dset_name, source_name, np.c_[np.r_[events_slice], np.r_[source_slice]])

        ev_id = np.arange(source_slice.start, source_slice.stop, dtype=int).reshape(-1, 1)
        hits_ev_id = np.broadcast_to(ev_id, hits_data.shape)
        ref = np.c_[hits_ev_id[hits_mask], hits_data[hits_mask]['id']]
        self.data_manager.write_ref(self.events_dset_name, self.hits_dset_name, ref)

        trigs_ev_id = np.broadcast_to(ev_id, ext_trigs_data.shape)
        ref = np.c_[trigs_ev_id[ext_trigs_mask], ext_trigs_data[ext_trigs_mask]['id']]
        self.data_manager.write_ref(self.events_dset_name, self.ext_trigs_dset_name, ref)

    def get_pps_delay_correction(self) -> int:
        dset_name = self.pps_delay_corrector_config['packets_dset_name']
        # [(io_group, delay_ticks), ...]:
        delay_table: list[tuple[int, int]] \
            = self.data_manager.get_attrs(dset_name)['pps_delays']
        delays = [r[1] for r in delay_table]

        # It appears that sometimes a PACMAN's system clock falls out of sync
        # with Unix Time, resulting in an invalid measurement of the PPS delay.
        # (E.g., io_group 7 at the beginning of the 2x2 sandbox period.) To
        # mitigate this, we let the PACMEN "vote" for the delay by taking the
        # median measurement.
        return int(np.median(delays))

    def get_unix_ts_usec(self, events_arr: npt.NDArray[events_dtype]) \
            -> npt.NDArray[np.uint32]:
        ticks2usec = 0.1
        if self.pps_delay_corrector_enabled:
            rollover = resources['RunData'].rollover_ticks
            return (((events_arr['ts_start'] + self.pps_delay_correction)
                     % rollover) * ticks2usec).astype(np.uint32)
        else:
            return (events_arr['ts_start'] * ticks2usec).astype(np.uint32)
