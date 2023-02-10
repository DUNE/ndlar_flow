import numpy as np

from h5flow.core import H5FlowStage, resources


class DriftReconstruction(H5FlowStage):
    '''
        Reconstructs hit positions based on drift time and T0

        Parameters:
         - ``t0_dset_name``: ``str``, path to input T0 dataset
         - ``hits_dset_name``: ``str``, path to input hits dataset
         - ``drift_dset_name``: ``str``, path to output dataset

         Both ``hits_dset_name`` and ``t0_dset_name`` are required
         in the cache, if enabled.

         Requires RunData, LArData, and Geometry resource in workflow.

         ``drift`` datatype::

            id          u4,     unique identifier
            t_drift     f8,     relative PPS timestamp to be used for T0 [crs ticks]
            d_drift     f8,     drift distance [mm]
            z           f8,     z (drift) coordinate of hit [mm]


    '''
    class_version = '0.0.0'

    drift_dtype = np.dtype([
        ('id', 'u4'),
        ('t_drift', 'f8'),
        ('d_drift', 'f8'),
        ('z', 'f8'),
    ])

    default_t0_dset_name = 'combined/t0'
    default_hits_dset_name = 'charge/hits'
    default_drift_dset_name = 'combined/hits_drift'

    def __init__(self, **params):
        super(DriftReconstruction, self).__init__(**params)
        self.t0_dset_name = params.get('t0_dset_name', self.default_t0_dset_name)
        self.hits_dset_name = params.get('hits_dset_name', self.default_hits_dset_name)
        self.drift_dset_name = params.get('drift_dset_name', self.default_drift_dset_name)

    def init(self, source_name):
        super(DriftReconstruction, self).init(source_name)

        # create drift dset
        self.data_manager.create_dset(self.drift_dset_name, self.drift_dtype)

        # create hit -> drift refs
        self.data_manager.create_ref(self.hits_dset_name, self.drift_dset_name)

        # set metadata
        self.data_manager.set_attrs(self.t0_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    source_dset=source_name,
                                    hits_dset=self.hits_dset_name,
                                    t0_dset=self.t0_dset_name
                                    )

    def run(self, source_name, source_slice, cache):
        super(DriftReconstruction, self).run(source_name, source_slice, cache)

        t0 = cache[self.t0_dset_name]
        hits = cache[self.hits_dset_name]

        drift_t = hits['ts'] - t0['ts']
        drift_d = drift_t * (resources['LArData'].v_drift * resources['RunData'].crs_ticks)
        z = resources['Geometry'].get_z_coordinate(hits['iogroup'], hits['iochannel'], drift_d)

        drift_array = np.empty(hits['id'].compressed().shape, dtype=self.drift_dtype)
        drift_array['z'] = z.compressed()
        drift_array['t_drift'] = drift_t.compressed()
        drift_array['d_drift'] = drift_d.compressed()

        # save data
        drift_slice = self.data_manager.reserve_data(self.drift_dset_name, len(drift_array))
        drift_array['id'] = np.r_[drift_slice]
        self.data_manager.write_data(self.drift_dset_name, drift_slice, drift_array)

        # save refs
        ref = np.c_[hits['id'].compressed(), drift_slice]
        self.data_manager.write_ref(self.hits_dset_name, self.drift_dset_name, ref)
