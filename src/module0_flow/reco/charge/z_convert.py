import numpy as np
import h5py
import pandas as pd
import logging
import sys

from h5flow.core import H5FlowStage, resources

class ZConvert(H5FlowStage):

    defaults = dict(
        hits_dset_name='charge/hits',
        events_dset_name='charge/events',
        ext_trigs_dset_name='charge/ext_trigs',
        hough_i_dset_name='charge/hough_i',
        )

    class_version = '0.0.0'

    hough_i_dtype = np.dtype([('iogroup', 'u1'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('q', 'f8'), ('evid', 'u1')])

    def __init__(self, **params):
        super(ZConvert, self).__init__(**params)
        for key in self.defaults:
            setattr(self, key, params.get(key, self.defaults[key]))

    def init(self, source_name):
        super(ZConvert, self).init(source_name)

        # then set up new datasets
        self.data_manager.create_dset(self.hough_i_dset_name, dtype=self.hough_i_dtype)

    def run(self, source_name, source_slice, cache):

        super(ZConvert, self).run(source_name, source_slice, cache)

        hits_data = cache[self.hits_dset_name]
        events_data = cache[self.events_dset_name]

        print(self.hough_i_dset_name)
        module = 'mod0'
        if module == 'mod0' or module == 'mod2':

            evid = [value[0] for value, count in zip(events_data['id'], events_data['nhit'])]
            print(evid)
            ts = [value[0] for value, count in zip(events_data['ts_start'], events_data['nhit'])]
            next = [value[0] for value, count in zip(events_data['n_ext_trigs'], events_data['nhit'])]

            z = (hits_data['ts']-ts) * 1.648e-1
            z = z - 304.31
            z *= -1

            hough_i_array = np.empty(hits_data['id'].shape, dtype=self.hough_i_dtype)
            hough_i_array['iogroup'] = hits_data['iogroup']
            hough_i_array['x'] = hits_data['px']
            hough_i_array['y'] = hits_data['py']
            hough_i_array['z'] = z
            hough_i_array['q'] = hits_data['q']
            hough_i_array['evid'] = evid
            print(hough_i_array['evid'])
#            hough_i_slice = self.data_manager.reserve_data(self.hough_i_dset_name, len(hough_i_array))
            hough_i_slice = self.data_manager.reserve_data(self.hough_i_dset_name, len(evid))

#            self.data_manager.write_data(self.hough_i_dset_name, [hits_data['iogroup'], hits_data['px'], hits_data['py'], z, hits_data['q'], evid], [hits_data['iogroup'], hits_data['px'], hits_data['py'], z, hits_data['q'], evid])
            self.data_manager.write_data(self.hough_i_dset_name, hough_i_slice, hough_i_array)


        else:

            hits_data = pd.DataFrame(np.array(h5_cov['charge/hits/data']))
            events_data = pd.DataFrame(np.array(h5_cov['charge/events/data']))
            z = pd.DataFrame(np.array(h5_cov['combined/hit_drift/data']))['z']

            hits_data['z'] = z

            evid = [value for value, count in zip(events_data['id'], events_data['nhit']) for _ in range(count)]
            next = [value for value, count in zip(events_data['n_ext_trigs'], events_data['nhit']) for _ in range(count)]

            hits_data['evid'] = evid
            hits_data['next'] = next
            hits_data.loc[hits_data['iogroup'] == 1, 'z'] *= -1

#        hits_data = hits_data[hits_data['next']>0]

#        throw_list = hits_data['evid'][hits_data['z']==0]
#        hits_data = hits_data[~hits_data['evid'].isin(throw_list)]

#        hits_data = hits_data[(np.abs(hits_data['z']) < 315) & (hits_data['z'] > 0)]

#        value_counts = hits_data['evid'].value_counts()
#        hits_data = hits_data[hits_data['evid'].isin(value_counts[value_counts >= 100].index)]

#        hough_i_slice = self.data_manager.reserve_data(self.hough_i_dset_name, len(hough_i_array))

#            self.data_manager.write_data(self.hough_i_dset_name, [hits_data['iogroup'], hits_data['px'], hits_data['py'], z, hits_data['q'], evid], [hits_data['iogroup'], hits_data['px'], hits_data['py'], z, hits_data['q'], evid])
#        self.data_manager.write_data(self.hough_i_dset_name, hough_i_slice, hough_i_array)
