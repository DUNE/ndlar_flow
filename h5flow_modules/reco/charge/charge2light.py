import numpy as np
import numpy.ma as ma
import numpy.lib.recfunctions as rfn
import logging

from h5flow.core import H5FlowStage, resources
import module0_flow.util.units as units


class Charge2LightAssociation(H5FlowStage):
    '''
    Generate references between charge events and light events. In general,
    matches a given light event to a given charge event if::

        |light_unix_ts_second - charge_unix_ts_second| <= unix_ts_window
        AND
        |light_ts_10MHz - charge_ts_10MHz| <= ts_window

    where ``*_unix_ts_second`` is the unix timestamp of the event in seconds and
    ``*_ts_10MHz`` is the timestamp in 10MHz ticks since SYNC / PPS. Creates
    references from both external triggers to light events as well as references
    from charge events to light events.

    Requires the ``ext_trigs_dset`` in the data cache as well as its indices
    (stored under the name ``ext_trigs_dset + '_idcs'``).

    Also requires RunData resource in workflow.

    Example config::

        charge_light_associator:
          classname: Charge2LightAssociation
          requires:
            - 'charge/ext_trigs'
            - name: 'charge/ext_trigs_idcs'
              path: 'charge/ext_trigs'
              index_only: True
          params:
            light_event_dset_name: 'light/events'
            ext_trigs_dset_name: 'charge/ext_trigs'
            unix_ts_window: 3
            ts_window: 10

    '''
    class_version = '0.0.1'

    default_unix_ts_window = 1  # how big of a symmetric window to use with unix timestamps (0=exact match, 1=±1 second, ...) [s]
    default_ts_window = 1000  # how big of a symmetric window to use with PPS timestamps (0=exact match, 10=±10 ticks, ...) [ticks]

    def __init__(self, **params):
        super(Charge2LightAssociation, self).__init__(**params)

        self.light_event_dset_name = params.get('light_event_dset_name')
        self.ext_trigs_dset_name = params.get('ext_trigs_dset_name')
        self.events_dset_name = None  # put off until init stage

        self.unix_ts_window = params.get('unix_ts_window', self.default_unix_ts_window)
        self.ts_window = params.get('ts_window', self.default_ts_window)

    def init(self, source_name):
        super(Charge2LightAssociation, self).init(source_name)

        # save all config info
        self.events_dset_name = source_name
        self.data_manager.set_attrs(self.events_dset_name,
                                    charge_to_light_assoc_classname=self.classname,
                                    charge_to_light_assoc_class_version=self.class_version,
                                    light_event_dset=self.light_event_dset_name,
                                    charge_to_light_assoc_unix_ts_window=self.unix_ts_window,
                                    charge_to_light_assoc_ts_window=self.ts_window
                                    )

        # then set up new datasets
        self.data_manager.create_ref(self.events_dset_name, self.light_event_dset_name)
        self.data_manager.create_ref(self.ext_trigs_dset_name, self.light_event_dset_name)

        # load in light system timestamps (use max to get non-null timestamp entries)
        self.light_event_id = self.data_manager.get_dset(self.light_event_dset_name)['id'][:]
        self.light_event_mask = self.data_manager.get_dset(self.light_event_dset_name)['wvfm_valid'][:].astype(bool)
        self.light_unix_ts = self.data_manager.get_dset(self.light_event_dset_name)['utime_ms'][:]
        self.light_unix_ts = ma.array(self.light_unix_ts, mask=~self.light_event_mask).mean(axis=-1).mean(axis=-1)
        self.light_unix_ts = self.light_unix_ts * (units.ms / units.s)  # convert ms -> s
        self.light_ts = self.data_manager.get_dset(self.light_event_dset_name)['tai_ns'][:]
        self.light_ts = ma.array(self.light_ts, mask=~self.light_event_mask).mean(axis=-1).mean(axis=-1)
        self.light_ts = self.light_ts * (units.ns / resources['RunData'].crs_ticks)  # convert ns -> larpix clock ticks

        self.light_unix_ts_start = self.light_unix_ts.min()
        self.light_unix_ts_end = self.light_unix_ts.max()

    def run(self, source_name, source_slice, cache):
        super(Charge2LightAssociation, self).run(source_name, source_slice, cache)

        event_data = cache[self.events_dset_name]
        ext_trigs_data = cache[self.ext_trigs_dset_name]
        ext_trigs_idcs = cache[self.ext_trigs_dset_name + '_idcs']
        ext_trigs_mask = ~rfn.structured_to_unstructured(ext_trigs_data.mask).any(axis=-1)

        nevents = len(event_data)

        lengths = np.count_nonzero(ext_trigs_mask, axis=-1)
        ext_trigs_all = ext_trigs_data.data[ext_trigs_mask]
        ext_trigs_idcs = ext_trigs_idcs.data[ext_trigs_mask]
        ext_trigs_unix_ts = np.broadcast_to(event_data['unix_ts'].reshape(-1, 1), ext_trigs_data.shape)[ext_trigs_mask]

        if nevents and len(ext_trigs_all):
            unix_ts_start = ext_trigs_unix_ts.min()
            unix_ts_end = ext_trigs_unix_ts.max()

            if self.light_unix_ts_start >= unix_ts_end + self.unix_ts_window or \
                    self.light_unix_ts_end <= unix_ts_start - self.unix_ts_window:
                # no overlap, short circuit
                idcs = np.empty((0, 2), dtype=int)

            else:
                # find relevant region of light array
                i_min = np.argmax((self.light_unix_ts >= unix_ts_start - self.unix_ts_window))
                i_max = len(self.light_unix_ts) - 1 - np.argmax((self.light_unix_ts <= unix_ts_end + self.unix_ts_window)[::-1])
                sl = slice(i_min, i_max)

                # perform matching
                charge_unix_ts = ext_trigs_unix_ts.astype(int)
                charge_ts = ext_trigs_all['ts']

                assoc_mat = \
                    (np.abs(self.light_unix_ts[sl].reshape(1, -1) - charge_unix_ts.reshape(-1, 1)) <= self.unix_ts_window) \
                    & (np.abs(self.light_ts[sl].reshape(1, -1) - charge_ts.reshape(-1, 1)) <= self.ts_window)
                idcs = np.argwhere(assoc_mat)

                if len(idcs):
                    idcs[:, 1] = self.light_event_id[sl][idcs[:, 1]]  # idcs now contains ext trigger index <-> global light event id
        else:
            idcs = np.empty((0, 2), dtype=int)

        ext_trig_ref = np.c_[ext_trigs_idcs[idcs[:, 0]], idcs[:, 1]]

        ev_id = np.arange(source_slice.start, source_slice.stop, dtype=int).reshape(-1, 1)
        ev_id = np.broadcast_to(ev_id, ext_trigs_data.shape)

        if len(idcs):
            ev_ref = np.unique(np.c_[ev_id[ext_trigs_mask][idcs[:, 0]], idcs[:, 1]], axis=0)
        else:
            ev_ref = np.empty((0, 2), dtype=int)

        # write references
        # ext trig -> light event
        self.data_manager.write_ref(self.ext_trigs_dset_name, self.light_event_dset_name, ext_trig_ref)

        # charge event -> light event
        self.data_manager.write_ref(self.events_dset_name, self.light_event_dset_name, ev_ref)
