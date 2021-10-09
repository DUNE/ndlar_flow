import numpy as np

from h5flow.core import H5FlowStage, resources

import module0_flow.util.units as units


class T0Reconstruction(H5FlowStage):
    '''
        Reconstructs an event T0 using light signals or embedded timestamps.
        Depending on the available data for each event, one of the following
        algorithms is used:

         - ``EXT_TRIG``: only embedded timestamp is available, so use earliest embedded timestamp
         - ``LIGHT_MATCHED``: light trigger data is available, so uses the rising edge of the largest light signal
         - ``NONE``: no timing data is available, so just uses earliest hit timestamp

        Parameters:
         - ``t0_dset_name``: ``str``, path to output dataset
         - ``light_hits_dset_name``: ``str``, path to input light hits dataset, ``None`` to disable light hit lookup
         - ``ext_trigs_dset_name``: ``str``, path to input external trigger dataset, ``None`` to disable ext trig lookup

         Both ``light_hits_dset_name`` and ``ext_trigs_dset_name`` are required
         in the cache, if enabled.

         Requires RunData resource in workflow.

         ``t0`` datatype::

            id          u4,     unique identifier
            ts          f8,     PPS timestamp to be used for T0 [crs ticks]
            ts_err      f8,     estimated error on T0 [crs ticks]
            type        u1,     type indicator for T0 algorithm used, see attr. ``type_lookup`` for value definitions


    '''
    class_version = '0.0.1'

    t0_dtype = np.dtype([
        ('id', 'u4'),
        ('ts', 'f8'),
        ('ts_err', 'f8'),
        ('type', 'u1'),
    ])

    default_t0_dset_name = 'combined/t0'
    default_light_hits_dset_name = None
    default_ext_trigs_dset_name = None

    t0_type = dict(
        NONE=0,
        EXT_TRIG=1,
        LIGHT_MATCHED=2
    )

    def __init__(self, **params):
        super(T0Reconstruction, self).__init__(**params)
        self.t0_dset_name = params.get('t0_dset_name', self.default_t0_dset_name)
        self.light_hits_dset_name = params.get('light_hits_dset_name', self.default_light_hits_dset_name)
        self.ext_trigs_dset_name = params.get('ext_trigs_dset_name', self.default_ext_trigs_dset_name)

    def init(self, source_name):
        super(T0Reconstruction, self).init(source_name)

        # create t0 dset
        self.data_manager.create_dset(self.t0_dset_name, self.t0_dtype)

        # create event -> t0 refs
        self.data_manager.create_ref(source_name, self.t0_dset_name)

        # set metadata
        self.data_manager.set_attrs(self.t0_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    source_dset=source_name,
                                    light_hits_dset=self.light_hits_dset_name
                                    if self.light_hits_dset_name is not None else '',
                                    ext_trigs_dset=self.ext_trigs_dset_name
                                    if self.ext_trigs_dset_name is not None else '',
                                    type_lookup=np.array([tuple([t for t in self.t0_type.values()])],
                                                         dtype=np.dtype([(t, 'u1')
                                                                         for t in self.t0_type.keys()]))
                                    )

    def run(self, source_name, source_slice, cache):
        super(T0Reconstruction, self).run(source_name, source_slice, cache)

        events = cache[source_name]                       # shape: (N,)
        if self.ext_trigs_dset_name is not None:
            ext_trigs = cache[self.ext_trigs_dset_name]   # shape: (N,n)
        else:
            ext_trigs = None
        if self.light_hits_dset_name is not None:
            light_hits = cache[self.light_hits_dset_name]  # shape: (N,M,m)
        else:
            light_hits = None

        if len(events):
            t0_array = np.empty(events.shape, dtype=self.t0_dtype)
            t0_array['id'] = np.r_[source_slice]

            # if external trigger, use min(trigger_ts) - 2 as t0, type == EXT_TRIG
            if ext_trigs is not None:
                ext_trig_mask = events['n_ext_trigs'] > 0
                t0_array['ts'][ext_trig_mask] = ext_trigs['ts'].min(axis=-1)[ext_trig_mask] - 2
                t0_array['ts_err'][ext_trig_mask] = 1 / np.sqrt(12)  # uniform 1 tick
                t0_array['type'][ext_trig_mask] = self.t0_type['EXT_TRIG']
            else:
                ext_trig_mask = np.zeros(events.shape, dtype=bool)

            # if light hits present, use rising edge of largest signal, type == LIGHT_MATCHED
            if light_hits is not None:
                # light/t_ns + busy_ns + rising_spline
                ts = (light_hits['ns'] + light_hits['busy_ns']
                      + light_hits['rising_spline'])
                ts_err = light_hits['rising_err_spline']

                ts = ts * (units.ns
                           / resources['RunData'].crs_ticks)  # ns -> crs ticks
                ts_err = ts_err * (units.ns
                                   / resources['RunData'].crs_ticks)  # ns -> crs ticks

                ts = ts.reshape(events.shape + (-1,))
                ts_err = ts_err.reshape(events.shape + (-1,))

                largest_hit = light_hits['sum_spline'].reshape(events.shape + (-1,)).argmax(axis=-1)
                largest_hit = np.expand_dims(largest_hit, axis=-1)
                ts = np.take_along_axis(ts, largest_hit, axis=-1).ravel()
                ts_err = np.take_along_axis(ts_err, largest_hit, axis=-1).ravel()

                light_matched_mask = ~ts.mask
                t0_array['ts'][light_matched_mask] = ts[light_matched_mask]
                t0_array['ts_err'][light_matched_mask] = ts_err[light_matched_mask]
                t0_array['type'][light_matched_mask] = self.t0_type['LIGHT_MATCHED']
            else:
                light_matched_mask = np.zeros(events.shape, dtype=bool)

            # if no external trigger, use start timestamp, type == NONE
            none_mask = ~(ext_trig_mask | light_matched_mask)
            t0_array['ts'][none_mask] = events['ts_start'][none_mask]
            t0_array['ts_err'][none_mask] = 0
            t0_array['type'][none_mask] = self.t0_type['NONE']
        else:
            t0_array = np.empty((0,), dtype=self.t0_dtype)

        # save data
        t0_slice = self.data_manager.reserve_data(self.t0_dset_name, source_slice)
        self.data_manager.write_data(self.t0_dset_name, t0_slice, t0_array)

        # save refs
        ref = np.c_[source_slice, t0_slice]
        self.data_manager.write_ref(source_name, self.t0_dset_name, ref)
