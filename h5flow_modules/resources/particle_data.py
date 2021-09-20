import numpy as np
import logging

from h5flow.core import H5FlowResource, resources

from module0_flow.util.compat import assert_compat_version
import module0_flow.util.units as units


class ParticleData(H5FlowResource):
    '''
        Provides helper functions for calculating and accessing particle
        properties. Range tables will be saved and/or loaded to/from metadata
        within the output file.

        Requires ``LArData`` resource within workflow.

        Parameters:
         - ``path``: ``str``, path to stored particle data within file
         - ``muon_range_table_path``: ``str``, path to PDG text file of muon range in LAr
         - ``proton_range_table_path``: ``str``, path to NIST text file of proton range in LAr

        Provides:
         - ``muon_range_table``: Range, kinetic energy, and <dE/dx> for muons in LAr
         - ``proton_range_table``: Range, kinetic energy, and <dE/dx> for protons in LAr
         - ``landau_width``: 1-sigma width of Landau dE/dx distribution in LAr
         - ``landau_peak``: MPV of Landau dE/dx distribution in LAr
         - ``{particle}_mass``: for proton (``p``), neutron (``n``), muon (``mu``), electron (``e``), pion (``pi``), pi0 (``pi0``)

        Example usage::

            from h5flow.core import resources

            resources['ParticleData'].muon_range_table['range']

        Example config::

            resources:
                - classname: ParticleData
                  params:
                    path: 'particle_info'

    '''
    class_version = '0.0.0'

    default_path = 'particle_info'
    default_muon_range_table_path = 'PDG_muon_range_table_Ar.txt'
    default_proton_range_table_path = 'NIST_proton_range_table_Ar.txt'

    _K = 0.307075 * units.MeV / (units.cm)**2

    #: electron mass
    e_mass = 510.9989461 * units.keV

    #: muon mass
    mu_mass = 105.6583745 * units.MeV

    #: proton mass
    p_mass = 938.2720813 * units.MeV

    #: neutron mass
    n_mass = 939.5654133 * units.MeV

    #: charged pion mass
    pi_mass = 139.57039 * units.MeV

    #: neutral pion mass
    pi0_mass = 134.9768 * units.MeV

    def __init__(self, **params):
        super(ParticleData, self).__init__(**params)

        self.path = params.get('path', self.default_path)
        self.muon_range_table_path = params.get('muon_range_table_path',
                                                self.default_muon_range_table_path)
        self.proton_range_table_path = params.get('proton_range_table_path',
                                                  self.default_proton_range_table_path)

    def init(self, source_name):
        if not self.data_manager.attr_exists(self.path, 'classname'):
            # no data stored in file, generate it
            muon_table = self.load_pdg_range_table(self.muon_range_table_path)
            proton_table = self.load_nist_range_table(self.proton_range_table_path)

            self.data = dict()

            # appropriate units from tables
            self.data['muon_range'] = muon_table['range'] * units.cm
            self.data['muon_t'] = muon_table['t'] * units.MeV
            self.data['muon_dedx'] = (muon_table['dedx'] * units.g
                                      * units.MeV / units.cm
                                      * resources['LArData'].density)
            self.data['proton_range'] = proton_table['range'] * units.cm
            self.data['proton_t'] = proton_table['t'] * units.MeV
            self.data['proton_dedx'] = (proton_table['dedx'] * units.g
                                        * units.MeV / units.cm
                                        * resources['LArData'].density)

            self.data['classname'] = self.classname
            self.data['class_version'] = self.class_version
            self.data_manager.set_attrs(self.path, **self.data)
        else:
            # data exists, check version compatibility
            assert_compat_version(self.class_version, self.data['class_version'])
            self.data = dict(self.data_manager.get_attrs(self.path))

    @property
    def muon_range_table(self):
        '''
            Range v. kinetic energy v. dE/dx for a muon in LAr. ``dict`` with
            keys: ``range``, ``t``, and ``dedx``

        '''
        return dict(range=self.data['muon_range'], t=self.data['muon_t'],
                    dedx=self.data['muon_dedx'])

    @property
    def proton_range_table(self):
        '''
            Range v. kinetic energy v. dE/dx for a proton in LAr. ``dict`` with
            keys: ``range``, ``t``, and ``dedx``

        '''
        return dict(range=self.data['proton_range'], t=self.data['proton_t'],
                    dedx=self.data['proton_dedx'])

    def landau_width(self, t, mass):
        ''' Moyal scale factor for Landau dE/dx width in LAr '''
        e = t + mass
        p = np.sqrt(e**2 - mass**2)
        beta = p / e

        Z = resources['LArData'].Z
        A = resources['LArData'].A
        ksi = self._K / 2 * (Z / A) * resources['LArData'].density / beta**2

        return (4 * ksi / 3.59)

    def landau_peak(self, t, mass):
        ''' Moyal peak location for Landau dE/dx distribution in LAr '''
        e = t + mass
        p = np.sqrt(e**2 - mass**2)
        beta = p / e
        gamma = e / mass

        Z = resources['LArData'].Z
        A = resources['LArData'].A
        ksi = self._K / 2 * (Z / A) * resources['LArData'].density / beta**2
        I = 188.0 * units.eV

        t0 = np.log(2 * self.e_mass * (beta * gamma)**2 / I)
        t1 = np.log(ksi / I)
        t2 = 0.200 - beta**2 + self._delta(beta * gamma)

        return ksi * (t0 + t1 + t2)

    @staticmethod
    def _delta(betagamma):
        #: values from PDG LAr data
        a = 0.1956
        x0 = 0.2
        x1 = 3.0
        cbar = 5.2146
        k = 3.00

        return (betagamma < x0) * (
            (betagamma < x1) * (2 * np.log(10) * betagamma - cbar + a * (x1 - betagamma)**k)
            + (betagamma > x1) * (2 * np.log(10) * betagamma - cbar))

    @staticmethod
    def load_nist_range_table(path):
        '''
            Loads particle range, kinetic energy, and dE/dx from a
            NIST text file [https://physics.nist.gov/PhysRefData/Star/Text/PSTAR-t.html].

            :param path: path to range table file

            :returns: ``dict`` with keys ``range``, ``t``, ``dedx``

        '''
        with open(path, 'r') as fi:
            _data = fi.readlines()[15:]
            _r = np.empty(len(_data))
            _ke = np.empty(len(_data))
            _dedx = np.empty(len(_data))
            for i, line in enumerate(_data):
                row_data = line.strip().split()
                if row_data:
                    _ke[i] = float(row_data[0])
                    _r[i] = float(row_data[4])
                    _dedx[i] = float(row_data[3])

        _table = dict(range=_r,
                      t=_ke,
                      dedx=_dedx)

        return _table

    @staticmethod
    def load_pdg_range_table(path):
        '''
            Loads particle range, kinetic energy, and dE/dx from a
            PDG text file [https://pdg.lbl.gov/2021/AtomicNuclearProperties/].

            :param path: path to range table file

            :returns: ``dict`` with keys ``range``, ``t``, ``dedx``

        '''
        with open(path, 'r') as fi:
            _data = fi.readlines()[10:]
            _r = np.empty(len(_data))
            _ke = np.empty(len(_data))
            _dedx = np.empty(len(_data))
            for i, line in enumerate(_data):
                row_data = line.strip().split()
                if row_data:
                    _ke[i] = float(row_data[0])
                    _r[i] = float(row_data[8])
                    _dedx[i] = float(row_data[7])

        _table = dict(range=_r,
                      t=_ke,
                      dedx=_dedx)

        return _table
