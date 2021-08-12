import numpy as np
import numpy.lib.recfunctions as rfn
from collections import defaultdict
import logging
import h5py

from h5flow.core import H5FlowResource, resources, H5FLOW_MPI

from module0_flow.util.compat import assert_compat_version


class RunData(H5FlowResource):
    '''
        Provides access to run-level data:

         - ``charge_data_file``: charge raw data file source name
         - ``light_data_file``: light raw data file source name
         - ``e_field``: TPC electric field in kV/mm
         - ``light_nsamples``: light system number of samples
         - ``charge_threshold``: charge system global thresholds (either ``high`` or ``medm``)
         - ``is_mc``: boolean flag, ``True`` if file was produced by simulation

        Requires ``Units`` resource in workflow.

        Parameters:
         - ``path``: ``str``, path to run data within file
         - ``runlist_file``: ``str``, path to runlist file containing run meta data
         - ``defaults``: ``dict``, key value pairs of ``data_name: data_value`` to use if lookup fails

        To access data, use the corresponding ``RunData`` property, e.g.::

            resources['RunData'].e_field

        A runlist file is required the first time the resource is included in
        a workflow. For subsequent workflows, data is stored and loaded directly
        from the hdf5 file.

        Example config::

            resources:
                - classname: RunData
                  params:
                    path: 'run_info'
                    runlist_file: 'runlist.txt'

        Run list file specification:

         1. Whitespace-delimited text file
         2. First line of the text file contains the column names: ``e_field``, ``charge_filename``, ``light_filename``, ``charge_thresholds``, ``light_samples``, in any order.
         3. The remainder of the file consists of whitespace separated data corresponding to the column names

        ``e_field`` run list file units are V/cm.

    '''
    class_version = '0.1.1'

    default_path = 'run_info'
    default_runlist_file = 'runlist.txt'

    source_filename_columns = ('charge_filename', 'light_filename')
    required_attr = ('charge_filename', 'light_filename', 'e_field',
                     'light_samples', 'charge_thresholds', 'is_mc', 'crs_ticks', 'lrs_ticks')

    def __init__(self, **params):
        super(RunData, self).__init__(**params)

        self.path = params.get('path', self.default_path)
        self.runlist_file = params.get('runlist_file', self.default_runlist_file)
        self.defaults = params.get('defaults', dict())

    def init(self, source_name):
        self.source_name = source_name
        self.data_manager.set_attrs(self.path)
        self.data = dict(self.data_manager.get_attrs(self.path))

        if not len(self.data.keys()):
            # run data does not exist, get it from input run list file
            self._update_data()
            self.data['classname'] = self.classname
            self.data['class_version'] = self.class_version
            self.data['runlist_file'] = self.runlist_file
            for key, val in self.defaults.items():
                self.data[f'{key}_default'] = val
            self.data_manager.set_attrs(self.path, **self.data)
        else:
            assert_compat_version(self.class_version, self.data['class_version'])

        for attr in self.required_attr:
            logging.info(f'{attr}: {getattr(self,attr)}')

    def _lookup_row_in_runlist(self):
        '''
            Load the run list file and check the charge or light data files against:

             1. ``source_name`` attribute input file for a match
             2. input filename passed along by the ``-i`` command line argument

            in that order.

        '''
        input_filenames = list()

        try:
            input_filenames.append(self.data_manager.get_attrs(self.source_name)['input_filename'])
        except (RuntimeError, KeyError):
            logging.warning(f'Source dataset {self.source_name} has no input file in metadata stored under \'input_filename\', using {self.input_filename} for RunData lookup')
            input_filenames.append(self.input_filename)

        row = dict()
        try:
            with open(self.runlist_file, 'r') as fi:
                lines = fi.readlines()
                column_names = lines[0].strip().split()
                logging.info(f'Loading from {self.runlist_file}')
                logging.info(lines[0].strip())

                for line in lines[1:]:
                    row_data = dict([(n, v) for n, v in zip(column_names, line.strip().split())])
                    if not row_data:
                        continue
                    if any([row_data[key] in f for key in self.source_filename_columns for f in input_filenames]):
                        row = row_data
                        break
                if row:
                    logging.info(line.strip())
                else:
                    logging.warning(f'Could not find row matching {input_filenames} in {self.runlist_file}')
        except Exception as e:
            logging.warning(f'Failed to load {self.runlist_file}: {e}')

        self.data.update(row)

    def _lookup_mc_info(self):
        '''
            Check if input file is a larnd-sim output file with MC truth
            information, and set ``is_mc`` flag.

        '''
        if self.data.get('is_mc', None) is not None:
            # mc info has already exists, return
            return

        if self.input_filename[-3:] == '.h5':
            if H5FLOW_MPI:
                with h5py.File(self.input_filename, 'r', driver='mpio', comm=self.comm) as f:
                    is_mc = 'mc_packets_assn' in f
            else:
                with h5py.File(self.input_filename, 'r') as f:
                    is_mc = 'mc_packets_assn' in f

            self.data['is_mc'] = is_mc
        else:
            self.data['is_mc'] = False

    def _update_data(self):
        # check input file for MC info to set mc flag
        self._lookup_mc_info()

        # read in run list file and update run data
        self._lookup_row_in_runlist()

        # fill in from defaults
        for key, val in self.defaults.items():
            if key not in self.data:
                self.data[key] = val

        for key in self.required_attr:
            assert key in self.data, f'missing {key} from RunData'

        # convert data types that might be incorrect
        self.data['e_field'] = float(self.data['e_field']) * (resources['Units'].V / resources['Units'].cm)
        self.data['light_samples'] = int(self.data['light_samples'])

    @property
    def charge_filename(self):
        ''' Base string for run file with charge data '''
        return self.data['charge_filename']

    @property
    def light_filename(self):
        ''' Base string for run file with light data '''
        return self.data['light_filename']

    @property
    def e_field(self):
        ''' TPC electric field in kV/mm '''
        return self.data['e_field']

    @property
    def light_samples(self):
        ''' Number of light waveform samples per trigger '''
        return self.data['light_samples']

    @property
    def charge_thresholds(self):
        ''' Charge threshold setting, either ``'high'`` or ``'medm'`` '''
        return self.data['charge_thresholds']

    @property
    def is_mc(self):
        ''' Simulation flag, ``True`` if file comes from simulation '''
        return self.data['is_mc']

    @property
    def cds_ticks(self):
        ''' Charge readout system clock cycle (us) '''
        return self.data['cds_ticks']

    @property
    def crs_ticks(self):
        ''' Charge readout system clock cycle (us) '''
        return self.data['crs_ticks']

    @property
    def lrs_ticks(self):
        ''' Light readout system clock cycle (us) '''
        return self.data['lrs_ticks']
