import numpy as np
import numpy.ma as ma
import logging

from h5flow.core import H5FlowStage

class Charge2LightAssociation(H5FlowStage):
    '''


    '''
    class_version = '0.0.1'

    default_unix_ts_window = 1 # how big of a symmetric window to use with unix timestamps (0=exact match, 1=±1 second, ...) [s]
    default_ts_window = 1000 # how big of a symmetric window to use with PPS timestamps (0=exact match, 10=±10 ticks, ...) [ticks]

    def __init__(self, **params):
        super(Charge2LightAssociation,self).__init__(**params)

        self.light_event_dset_name = params.get('light_event_dset_name')
        self.ext_trigs_dset_name = params.get('ext_trigs_dset_name')
        self.events_dset_name = None # put off until init stage

        self.unix_ts_window = params.get('unix_ts_window', self.default_unix_ts_window)
        self.ts_window = params.get('ts_window', self.default_ts_window)

    def init(self, source_name):
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
        self.light_unix_ts = self.light_unix_ts / 1000. # convert ms -> s
        self.light_ts = self.data_manager.get_dset(self.light_event_dset_name)['tai_ns'][:]
        self.light_ts = ma.array(self.light_ts, mask=~self.light_event_mask).mean(axis=-1).mean(axis=-1)
        self.light_ts = self.light_ts / 100. # convert ns -> 0.1us

        self.light_unix_ts_start = self.light_unix_ts.min()
        self.light_unix_ts_end = self.light_unix_ts.max()

    def run(self, source_name, source_slice, cache):
        event_data = cache[self.events_dset_name]
        ext_trigs_data = cache[self.ext_trigs_dset_name]
        nevents = len(event_data)

        lengths = [len(ext_trigs) for ext_trigs in ext_trigs_data]
        ext_trigs_all = np.concatenate(ext_trigs_data, axis=0) if nevents else np.empty((0,))
        ext_trigs_unix_ts = np.concatenate([np.full(length, event['unix_ts']) for length,event in zip(lengths, event_data)]) if nevents else np.empty((0,))

        if nevents and len(ext_trigs_all):
            unix_ts_start = ext_trigs_unix_ts.min()
            unix_ts_end = ext_trigs_unix_ts.max()

            if self.light_unix_ts_start >= unix_ts_end+self.unix_ts_window or \
                self.light_unix_ts_end <= unix_ts_start-self.unix_ts_window:
                # no overlap, short circuit
                idcs = np.empty((0,2), dtype=int)

            else:
                # find relevant region of light array
                i_min = np.argmax((self.light_unix_ts >= unix_ts_start-self.unix_ts_window))
                i_max = len(self.light_unix_ts)-1 - np.argmax((self.light_unix_ts <= unix_ts_end+self.unix_ts_window)[::-1])
                sl = slice(i_min, i_max)

                # perform matching
                charge_unix_ts = ext_trigs_unix_ts.astype(int)
                charge_ts = ext_trigs_all['ts']

                assoc_mat = \
                    (np.abs(self.light_unix_ts[sl].reshape(1,-1) - charge_unix_ts.reshape(-1,1)) <= self.unix_ts_window) \
                    & (np.abs(self.light_ts[sl].reshape(1,-1) - charge_ts.reshape(-1,1)) <= self.ts_window)
                idcs = np.argwhere(assoc_mat)

                if len(idcs):
                    idcs[:,1] = self.light_event_id[sl][idcs[:,1]] # idcs now contains ext trigger index <-> global light event id
        else:
            idcs = np.empty((0,2), dtype=int)

        # collect by external trigger
        ext_trig_assoc = list()
        for i in range(len(ext_trigs_all)):
            mask = idcs[:,0] == i
            if np.any(mask):
                ext_trig_assoc.append(idcs[mask,1])
            else:
                ext_trig_assoc.append(list())

        # write references
        # ext trig -> light event
        spec = ext_trigs_all['id'] if len(ext_trigs_all) else np.empty((0,), dtype=int)
        self.data_manager.reserve_ref(self.ext_trigs_dset_name, self.light_event_dset_name, spec)
        self.data_manager.write_ref(self.ext_trigs_dset_name, self.light_event_dset_name, spec, ext_trig_assoc)

        # charge event -> light event
        self.data_manager.reserve_ref(self.events_dset_name, self.light_event_dset_name, source_slice)
        ref = []
        for i in range(len(event_data)):
            sl = slice(sum(lengths[:i]), sum(lengths[:i+1]))
            ref.append(self.data_manager.merge_region_specs(ext_trig_assoc[sl]))
        self.data_manager.write_ref(self.events_dset_name, self.light_event_dset_name, source_slice, ref)

