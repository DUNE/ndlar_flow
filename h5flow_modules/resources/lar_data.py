import numpy as np
import logging

from h5flow.core import H5FlowResource, resources

class LArData(H5FlowResource):
    '''
        Provides helper functions for calculating properties of liquid argon.
        Some values will be saved to attributes in the

        Requires both Units and RunData resources within workflow.

    '''
    class_version = '0.0.0'

    default_path = 'lar_info'
    default_electron_mobility_params = np.array([551.6, 7158.3, 4440.43, 4.29, 43.63, 0.2053])

    def __init__(self, **params):
        super(LArData,self).__init__(self, **params)

        self.path = params.get('path', self.default_path)

        self.electron_mobility_params = np.array(params.get('electron_mobility_params', self.default_electron_mobility_params))

    def init(self, source_name):
        # create group (if not present)
        self.data_manager.set_attrs(self.path)
        # load data (if present)
        self.data = dict(self.data_manager.get_attrs(self.path))

        logging.info(f'v_drift: {self.v_drift}')

    @cached_property
    def v_drift(self):
        if 'v_drift' in self.data:
            return self.data['v_drift']

        # get electric field from run data
        e_field = resources['RunData'].e_field

        # calculate drift velocity
        self.data['v_drift'] = self.electron_mobility(e_field) * e_field

        return self.data['v_drift']

    def electron_mobility(self, e, t=89):
        '''
            Calculation of the electron mobility w.r.t temperature and electric
            field.

            Refs:
                [0]: https://lar.bnl.gov/properties/trans.html (summary)
                [1]: https://doi.org/10.1016/j.nima.2016.01.073

            Note for units:
                Accepts an electric field and temperature in "module0_flow" units

        '''
        a0 = self.electron_mobility_params[0]
        a1 = self.electron_mobility_params[1]
        a2 = self.electron_mobility_params[2]
        a3 = self.electron_mobility_params[3]
        a4 = self.electron_mobility_params[4]
        a5 = self.electron_mobility_params[5]

        e = e / (resources['Units'].kV / resources['Units'].cm)
        t = t / (resources['Units'].K)

        num = a0 + a1*e + a2*np.power(e,1.5) + a3*np.power(e,2.5)
        denom = 1 + (a1/a0)*e + a4*np.power(e,2) + a5*np.power(e,3)
        temp_corr = np.power(t/89,-1.5)

        mu = num / denom * temp_corr
        mu = mu * ((resources['Units'].cm**2) / resources['Units'].V / resources['Units'].s)

        return mu


    def finish(self, source_name):
        # write data (if present)
        self.data_manager.set_attrs(self.path, **self.data)
