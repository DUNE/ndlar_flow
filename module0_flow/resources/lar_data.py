import numpy as np
import logging
import scipy.interpolate as interpolate
import os

from h5flow.core import H5FlowResource, resources

from module0_flow.util.compat import assert_compat_version
import module0_flow.util.units as units


class LArData(H5FlowResource):
    '''
        Provides helper functions for calculating properties of liquid argon.
        Values will be saved and/or loaded from metadata within the output file.

        Requires ``RunData`` resource within workflow.

        Parameters:
         - ``path``: ``str``, path to stored lar data within file
         - ``electron_mobility_params``: ``list``, electron mobility calculation parameters, see ``LArData.electron_mobility``

        Provides:
         - ``v_drift``: electron drift velocity in mm/us
         - ``ionization_w``: ionization W-value
         - ``density``: LAr density
         - ``ionization_recombination(dedx)``: helper function for calculating recombination factor
         - ``electron_lifetime(unix_ts)``: helper function for looking up electron lifetime at a given timestamp
         - ``A``: atomic mass number for atmospheric Argon
         - ``Z``: atomic number for Argon
         - ``radiation_length``: radiation length in argon

        Example usage::

            from h5flow.core import resources

            resources['LArData'].v_drift

        Example config::

            resources:
                - classname: LArData
                  params:
                    path: 'lar_info'

    '''
    class_version = '0.2.0'

    default_path = 'lar_info'
    default_electron_mobility_params = np.array([551.6, 7158.3, 4440.43, 4.29, 43.63, 0.2053])
    default_electron_lifetime = 2.2e3  # us
    default_electron_lifetime_file = None

    electron_lifetime_data_dtype = np.dtype([
        ('unix_s', 'f8'),
        ('lt_us', 'f8')
    ])

    def __init__(self, **params):
        super(LArData, self).__init__(**params)

        self.path = params.get('path', self.default_path)

        self.electron_mobility_params = np.array(params.get('electron_mobility_params', self.default_electron_mobility_params))
        self._electron_lifetime = params.get('electron_lifetime', self.default_electron_lifetime)
        self.electron_lifetime_file = params.get('electron_lifetime_file', self.default_electron_lifetime_file)

    def init(self, source_name):
        super(LArData, self).init(source_name)

        # create group (if not present)
        if not self.data_manager.attr_exists(self.path, 'classname'):
            # no data stored in file, generate it
            self.data = dict()

            self.v_drift
            self.density
            self.ionization_w
            self._init_electron_lifetime()
            self.data['classname'] = self.classname
            self.data['class_version'] = self.class_version
            self.data['electron_mobility_params'] = self.electron_mobility_params
            self.data_manager.set_attrs(self.path, **self.data)
        else:
            self.data = dict(self.data_manager.get_attrs(self.path))
            assert_compat_version(self.class_version, self.data['class_version'])
            self._init_electron_lifetime()            

        if self.rank == 0:
            logging.info(f'v_drift: {self.v_drift}')
            logging.info(f'density: {self.density}')
            logging.info(f'W(ionization): {self.ionization_w}')

    def _init_electron_lifetime(self):
        if 'electron_lifetime_central_value' in self.data:
            # handle case when electron lifetime is saved in file
            central_value_x = self.data['electron_lifetime_central_value']['unix_s']
            central_value_y = self.data['electron_lifetime_central_value']['lt_us']
            upper_bound_x = self.data['electron_lifetime_upper_bound']['unix_s']
            upper_bound_y = self.data['electron_lifetime_upper_bound']['lt_us']
            lower_bound_x = self.data['electron_lifetime_lower_bound']['unix_s']
            lower_bound_y = self.data['electron_lifetime_lower_bound']['lt_us']
        elif (self.electron_lifetime_file is not None
              and os.path.exists(self.electron_lifetime_file)
              and self.electron_lifetime_file[-5:] == '.root'
              and not resources['RunData'].is_mc):
            # handle case when electron lifetime root file is specified
            import ROOT
            f = ROOT.TFile(self.electron_lifetime_file, 'READ')
            central_value = f.Get('CentralValue')
            lower_bound = f.Get('LowerBound')
            upper_bound = f.Get('UpperBound')
            central_value_x = np.array(central_value.GetX())
            central_value_y = np.array(central_value.GetY()) * units.ms
            upper_bound_x = np.array(upper_bound.GetX())
            upper_bound_y = np.array(upper_bound.GetY()) * units.ms
            lower_bound_x = np.array(lower_bound.GetX())
            lower_bound_y = np.array(lower_bound.GetY()) * units.ms
        elif (self.electron_lifetime_file is not None
              and os.path.exists(self.electron_lifetime_file)
              and self.electron_lifetime_file[-4:] == '.npz'
              and not resources['RunData'].is_mc):
            # handle case when electron lifetime numpy file is specified
            d = np.load(self.electron_lifetime_file)
            central_value_x = d['electron_lifetime_central_value']['unix_s']
            central_value_y = d['electron_lifetime_central_value']['lt_us']
            upper_bound_x = d['electron_lifetime_upper_bound']['unix_s']
            upper_bound_y = d['electron_lifetime_upper_bound']['lt_us']
            lower_bound_x = d['electron_lifetime_lower_bound']['unix_s']
            lower_bound_y = d['electron_lifetime_lower_bound']['lt_us']
        else:
            central_value_x = np.array([0, 1])
            central_value_y = np.array([self._electron_lifetime] * 2)
            upper_bound_x = np.array([0, 1])
            upper_bound_y = np.array([self._electron_lifetime] * 2)
            lower_bound_x = np.array([0, 1])
            lower_bound_y = np.array([self._electron_lifetime] * 2)

        self._electron_lifetime_central_interp = interpolate.interp1d(
            central_value_x, central_value_y, bounds_error=False,
            fill_value=(central_value_y[0], central_value_y[-1]))
        self._electron_lifetime_upper_interp = interpolate.interp1d(
            upper_bound_x, upper_bound_y, bounds_error=False,
            fill_value=(upper_bound_y[0], upper_bound_y[-1]))
        self._electron_lifetime_lower_interp = interpolate.interp1d(
            lower_bound_x, lower_bound_y, bounds_error=False,
            fill_value=(lower_bound_y[0], lower_bound_y[-1]))

        self.data['electron_lifetime_central_value'] = np.empty(
            (len(central_value_x),), dtype=self.electron_lifetime_data_dtype)
        self.data['electron_lifetime_upper_bound'] = np.empty(
            (len(upper_bound_x),), dtype=self.electron_lifetime_data_dtype)
        self.data['electron_lifetime_lower_bound'] = np.empty(
            (len(lower_bound_x),), dtype=self.electron_lifetime_data_dtype)

        self.data['electron_lifetime_central_value']['unix_s'] = central_value_x
        self.data['electron_lifetime_central_value']['lt_us'] = central_value_y
        self.data['electron_lifetime_upper_bound']['unix_s'] = upper_bound_x
        self.data['electron_lifetime_upper_bound']['lt_us'] = upper_bound_y
        self.data['electron_lifetime_lower_bound']['unix_s'] = lower_bound_x
        self.data['electron_lifetime_lower_bound']['lt_us'] = lower_bound_y

    def electron_lifetime(self, unix_ts):
        '''
        Convert the run unix timestamp into an electron lifetime value

        :returns: ``lifetime, (lifetime_lower_bound, lifetime_upper_bound)``
        '''
        return self._electron_lifetime_central_interp(unix_ts), (
            self._electron_lifetime_lower_interp(unix_ts),
            self._electron_lifetime_upper_interp(unix_ts)
        )

    @property
    def ionization_w(self):
        ''' Ionization W-value in LAr in keV/e-. Fixed value of 0.0236 '''
        if 'ionization_w' in self.data:
            return self.data['ionization_w']

        self.data['ionization_w'] = 23.6 * units.eV / units.e
        return self.ionization_w

    @property
    def scintillation_w(self):
        ''' Scintillation W-value in LAr in keV/photon. Fixed value of 0.0195 '''
        if 'scintillation_w' in self.data:
            return self.data['scintillation_w']
        
        self.data['scintillation_w'] = 19.5 * units.eV
        return self.scintillation_w
        
    def ionization_recombination(self, dedx):
        '''
            Calculate charge recombination factor using Birks Model with parameters:

             - ``A = 0.8``
             - ``K = 0.0486`` (units = g/(MeV cm^2) kV/cm)

            Defined by::

                R_i = dQ * W_i / dE

        '''
        A = 0.8
        K = (0.0486 * units.kV * units.g / units.MeV / (units.cm)**3)
        eps = resources['RunData'].e_field * self.density

        rv = A / (1 + (K / eps) * dedx)
        return rv

    def scintillation_recombination(self, dedx):
        '''
            Calculate photon recombination factor using Birks Model (see ``ionization_recombination()``).
            Defined by::

                R_s = dQ * W_s / dE
        '''
        return 1 - self.ionization_recombination(self, dedx) * self.scintillation_w()/self.ionization_w()

    @property
    def A(self):
        ''' Fixed value of 39.948 '''
        return 39.948

    @property
    def Z(self):
        ''' Fixed value of 18 '''
        return 18

    @property
    def radiation_length(self):
        ''' 19.55 g cm^-2 / density'''
        return 19.55 * units.g / (units.cm)**2 / self.density

    @property
    def density(self):
        ''' Liquid argon density in g/mm^3. Fixed value of 0.00138 '''
        if 'density' in self.data:
            return self.data['density']

        # self.data['density'] = 1.3962 * units.g / (units.cm)**3
        self.data['density'] = 1.38 * units.g / (units.cm)**3
        return self.density

    @property
    def v_drift(self):
        ''' Electron drift velocity in kV/mm '''
        if 'v_drift' in self.data:
            return self.data['v_drift']

        # get electric field from run data
        e_field = resources['RunData'].e_field

        # calculate drift velocity
        self.data['v_drift'] = self.electron_mobility(e_field) * e_field

        return self.v_drift

    def electron_mobility(self, e, t=87.17):
        '''
            Calculation of the electron mobility w.r.t temperature and electric
            field.

            References:
             - https://lar.bnl.gov/properties/trans.html (summary)
             - https://doi.org/10.1016/j.nima.2016.01.073 (parameterization)

            :param e: electric field in kV/mm

            :param t: temperature in K

            :returns: electron mobility in mm^2/kV/us

        '''
        a0, a1, a2, a3, a4, a5 = self.electron_mobility_params

        e = e / (units.kV / units.cm)
        t = t / (units.K)

        num = a0 + a1 * e + a2 * np.power(e, 1.5) + a3 * np.power(e, 2.5)
        denom = 1 + (a1 / a0) * e + a4 * np.power(e, 2) + a5 * np.power(e, 3)
        temp_corr = np.power(t / 89, -1.5)

        mu = num / denom * temp_corr

        mu = mu * ((units.cm**2) / units.V / units.s)

        return mu
