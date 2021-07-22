import numpy as np
import logging

from h5flow.core import H5FlowResource, resources

class LArData(H5FlowResource):
    '''
        Provides helper functions for calculating properties of liquid argon.
        Values will be saved and/or loaded from metadata within the output file.

        Requires both ``Units`` and ``RunData`` resources within workflow.

        Parameters:
         - ``path``: ``str``, path to stored lar data within file
         - ``electron_mobility_params``: ``list``, electron mobility calculation parameters, see ``LArData.electron_mobility``

        Provides:
         - ``v_drift``: electron drift velocity in mm/us

        Example usage::

            from h5flow.core import resources

            resources['LArData'].v_drift

        Example config::

            resources:
                - classname: LArData
                  params:
                    path: 'lar_info'

    '''
    class_version = '0.0.0'

    default_path = 'lar_info'
    default_electron_mobility_params = np.array([551.6, 7158.3, 4440.43, 4.29, 43.63, 0.2053])

    def __init__(self, **params):
        super(LArData,self).__init__(**params)

        self.path = params.get('path', self.default_path)

        self.electron_mobility_params = np.array(params.get('electron_mobility_params', self.default_electron_mobility_params))

    def init(self, source_name):
        # create group (if not present)
        self.data_manager.set_attrs(self.path)
        # load data (if present)
        self.data = dict(self.data_manager.get_attrs(self.path))

        logging.info(f'v_drift: {self.v_drift}')

    @property
    def v_drift(self):
        ''' Electron drift velocity in kV/mm '''
        if 'v_drift' in self.data:
            return self.data['v_drift']

        # get electric field from run data
        e_field = resources['RunData'].e_field

        # calculate drift velocity
        self.data['v_drift'] = self.electron_mobility(e_field) * e_field

        return self.data['v_drift']

    def electron_mobility(self, e, t=85.3):
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
        a0,a1,a2,a3,a4,a5 = self.electron_mobility_params

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
