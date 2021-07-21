import numpy as np
import numpy.lib.recfunctions as rfn
from collections import defaultdict
import logging

from h5flow.core import H5FlowResource, resources

class RunData(H5FlowResource):
    '''
        Provides access to run-level data including:

         - ``charge_data_file``: charge raw data file source name
         - ``light_data_file``: light raw data file source name
         - ``e_field``: TPC electric field
         - ``light_nsamples``: light system number of samples
         - ``charge_threshold``: charge system global thresholds (either ``high`` or ``medm``)

        Requires Units resource in workflow.

        To access data, use the corresponding RunData property, e.g.::

            resources['RunData'].e_field

        this will load the data from the runlist file if the value is not already
        present as metadata in the file.

        Runlist file specification:
         1. The file is a simple, whitespace-delimited text file
         2. First line of the text file contains the column names:
         ``e_field``, ``charge_filename``, ``light_filename``, ``charge_thresholds``,
         ``light_samples``, in any order.
         3. The remainder of the file consists of whitespace separated data
         corresponding to the column names



    '''
    class_version = '0.0.0'

    default_path = 'run_info'
    default_runlist_file = 'runlist.txt'

    source_filename_columns = ('charge_filename', 'light_filename')
    valid_attr = ('charge_filename', 'light_filename', 'e_field', 'light_samples', 'charge_thresholds')

    def __init__(self, **params):
        super(RunData,self).__init__(self, **params)

        self.path = params.get('path', self.default_path)
        self.runlist_file = params.get('runlist_file', self.default_runlist_file)

    def init(self, source_name):
        self.source_name = source_name
        self.data_manager.set_attrs(self.path)
        self.data = dict(self.data_manager.get_attrs(self.path))

        if not len(self.data.keys()):
            self.update_data()

        if 'source_name' not in self.data:
            self.data['source_name'] = source_name

        for attr in self.valid_attr:
            logging.info(f'{attr}: {getattr(self,attr)}')

    def find_row(self):
        '''
            Check two places for a match in either the charge or light data
            file:

             1. check ``source_name`` attribute input file for a match
             2. check input filename passed along by the `-i` command line argument

            in that order.

                :returns: a ``dict`` of the parsed row of ``column_name: column_val``, None if file can't be found
        '''
        input_filenames = list()

        try:
            input_filenames.append(self.data_manager.get_attrs(self.source_name)['input_filename'])
        except KeyError:
            logging.warning(f'Source dataset {source_name} has no input file in metadata stored under input_filename, only using {self.input_filename} for RunData lookup')
            input_filenames.append(self.input_filename)

        with open(self.runlist_file,'r') as fi:
            lines = fi.readlines()
            column_names = lines[0].strip().split()
            logging.info(lines[0].strip())

            row = None
            for line in lines[1:]:
                row_data = dict([(n,v) for n,v in zip(column_names, line.strip().split())])
                if not row_data:
                    continue
                if any([row_data[key] in input_filenames for key in self.source_filename_columns]):
                    row = row_data
                    break

        return row

    def update_data(self):
        row = self.find_row()

        for key in row:
            self.data[key] = row[key]

        self.data['e_field'] = float(self.data['e_field']) * (resources['Units'].kV / resources['Units'].cm)
        self.data['light_samples'] = int(self.data['light_samples'])

    @property
    def charge_filename(self):
        return self.data['charge_filename']

    @property
    def light_filename(self):
        return self.data['light_filename']

    @property
    def e_field(self):
        return self.data['e_field']

    @property
    def light_samples(self):
        return self.data['light_samples']

    @property
    def charge_thresholds(self):
        return self.data['charge_thresholds']

    def finish(self, source_name):
        self.data_manager.set_attrs(self.path, **self.data)

