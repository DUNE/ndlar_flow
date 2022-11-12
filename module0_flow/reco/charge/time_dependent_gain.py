import numpy as np
import scipy.interpolate as interpolate
import logging

from h5flow import H5FLOW_MPI
from h5flow.core import H5FlowStage, resources


class TimeDependentGain(H5FlowStage):
    '''
        Applies a global, time-dependent gain correction to each hit using a cubic spline.
        Assumes that the time-dependent gain variation is over a long enough time scale that
        the gain is constant over a single event.

        Uses an input .npz file containing
          - ``f``: 1D array of correction (``q_calib = q_raw * f``)
          - ``f_err``: 1D array of 1-sigma gain correction error
          - ``unix_s``: 1D array of unix timestamp in [s] for each gain correction term

        Parameters:
         - ``hits_dset_name``: ``str``, path to input hits dataset (only requirement is a ``q`` field)
         - ``gain_file``: ``str``, path to .npz file grab correction factors from
         - ``calib_dset_name``: ``str``, path to output dataset

        Both ``hits_dset_name`` and ``drift_dset`` are required
        in the cache.

        Requires RunData resource in workflow.

        Example config::

            time_dependent_gain:
                classname: TimeDependentGain
                path: module0_flow.reco.charge.time_dependent_gain
                requires:
                  - 'charge/hits'
                params:
                  hits_dset_name: 'charge/hits'
                  calib_dset_name: 'charge/q_calib_tdg'
                  gain_file: 'module0_time_dependent_gain.npz'

        ``calib`` datatype::

            id          u4,     unique identifier
            f           f4,     calibration factor (q_calib = f * q)
            q           f4,     resulting calibrated value [mV]

    '''
    class_version = '0.0.0'

    calib_dtype = np.dtype([
        ('id', 'u4'),
        ('f', 'f4'),
        ('q', 'f4')
    ])

    default_gain_file = 'h5flow_data/module0_time_dependent_gain.npz'
    default_hits_dset_name = 'charge/hits'
    default_calib_dset_name = 'combined/q_calib_tdg'

    def __init__(self, **params):
        super(TimeDependentGain, self).__init__(**params)
        # set up dataset names
        self.hits_dset_name = params.get('hits_dset_name', self.default_hits_dset_name)
        self.calib_dset_name = params.get('calib_dset_name', self.default_calib_dset_name)

        self.gain_file = params['gain_file']

    def init(self, source_name):
        super(TimeDependentGain, self).init(source_name)

        self.is_mc = resources['RunData'].is_mc

        # create calib dset            
        self.data_manager.create_dset(self.calib_dset_name, self.calib_dtype)

        # create hit -> calib refs
        self.data_manager.create_ref(self.hits_dset_name, self.calib_dset_name)

        # set metadata
        self.data_manager.set_attrs(self.calib_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    source_dset=source_name,
                                    hits_dset=self.hits_dset_name,
                                    mc_flag=self.is_mc)

        if not self.is_mc:
            logging.info(f'Loading time dependent gain calibration from {self.gain_file}')
            gain_data = np.load(self.gain_file)
            self.gain_spline = interpolate.CubicSpline(gain_data['unix_s'], gain_data['f'])

            # also copy the gain correction data (for later reference, if needed)
            self.data_manager.set_attrs(self.calib_dset_name,
                                        gain_file=self.gain_file,
                                        unix_s=gain_data['unix_s'],
                                        f=gain_data['f'],
                                        f_err=gain_data['f_err'])
        else:
            logging.info('Skipped loading time dependent gain calibration because file is simulation.')
            

    def run(self, source_name, source_slice, cache):
        super(TimeDependentGain, self).run(source_name, source_slice, cache)

        # fetch data
        events = cache[source_name]
        hits = cache[self.hits_dset_name]

        if self.is_mc:
            f = np.ones((1,))
        else:
            f = self.gain_spline(events['unix_ts'])[:,np.newaxis]
        q = f * hits['q']

        calib_array = np.empty(hits['id'].compressed().shape, dtype=self.calib_dtype)
        calib_array['q'] = q.compressed()
        calib_array['f'] = np.broadcast_to(f, q.shape)[~q.mask]

        # save data
        calib_slice = self.data_manager.reserve_data(self.calib_dset_name, len(calib_array))
        calib_array['id'] = np.r_[calib_slice]
        self.data_manager.write_data(self.calib_dset_name, calib_slice, calib_array)

        # save refs
        ref = np.c_[hits['id'].compressed(), calib_slice]
        self.data_manager.write_ref(self.hits_dset_name, self.calib_dset_name, ref)
