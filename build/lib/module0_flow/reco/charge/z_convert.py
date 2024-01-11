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
        module = 'mod0',
        )

    class_version = '0.0.0'

    hough_i_dtype = np.dtype([('iogroup', 'u1'), ('next', 'u2'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('Q', 'f8'), ('E', 'f8'), ('id', 'u2')])

    def __init__(self, **params):
        super(ZConvert, self).__init__(**params)
        for key in self.defaults:
            setattr(self, key, params.get(key, self.defaults[key]))

    def init(self, source_name):
        super(ZConvert, self).init(source_name)

        # then set up new datasets
        self.data_manager.create_dset(self.hough_i_dset_name, dtype=self.hough_i_dtype)

        self.data_manager.create_ref(self.events_dset_name, self.hough_i_dset_name)


    def run(self, source_name, source_slice, cache):

        super(ZConvert, self).run(source_name, source_slice, cache)

        hits_data = cache[self.hits_dset_name]
        events_data = cache[self.events_dset_name]

        if self.module == 'mod0' or self.module == 'mod2':

            ts = [value for value, count in zip(events_data['ts_start'], events_data['nhit'])]
            next = [value for value, count in zip(events_data['n_ext_trigs'], events_data['nhit'])]

            z = (hits_data['ts']-ts) * 1.648e-1
            z = z - 304.31
            z *= -1

            hough_i_array = np.empty(hits_data['id'].shape, dtype=self.hough_i_dtype)
            hough_i_array['iogroup'] = hits_data['iogroup']
            hough_i_array['x'] = hits_data['px']
            hough_i_array['y'] = hits_data['py']
            hough_i_array['z'] = z
            hough_i_array['Q'] = hits_data['q']
            hough_i_array['E'] = hits_data['q']
            hough_i_array['id'] = hits_data['id']
            hough_i_array['next'] = next

        else:

            z = pd.DataFrame(np.array(h5_cov['combined/hit_drift/data']))['z']

            next = [value for value, count in zip(events_data['n_ext_trigs'], events_data['nhit']) for _ in range(count)]

            hough_i_array = np.empty(hits_data['id'].shape, dtype=self.hough_i_dtype)
            hough_i_array['iogroup'] = hits_data['iogroup']
            hough_i_array['x'] = hits_data['px']
            hough_i_array['y'] = hits_data['py']
            hough_i_array['z'] = z
            hough_i_array['Q'] = hits_data['q']
            hough_i_array['E'] = hits_data['q']
            hough_i_array['id'] = hits_data['id']
            hough_i_array['next'] = next

            hough_i_array.loc[hough_i_array['iogroup'] == 1, 'z'] *= -1



        hough_i_slice = self.data_manager.reserve_data(self.hough_i_dset_name, len(hough_i_array[0]['id']))


        self.data_manager.write_data(self.hough_i_dset_name, hough_i_slice, hough_i_array[0])

        ev_id = np.arange(source_slice.start, source_slice.stop, dtype=int).reshape(-1, 1)
        hits_ev_id = np.broadcast_to(ev_id, (1, len(hough_i_array[0])))
        ref = np.c_[hits_ev_id[0], hough_i_array[0]['id']]

        self.data_manager.write_ref(self.events_dset_name, self.hough_i_dset_name, ref)


#        hits_data = hits_data[hits_data['next']>0]
#        throw_list = hits_data['evid'][hits_data['z']==0]
#        hits_data = hits_data[~hits_data['evid'].isin(throw_list)]

#        hits_data = hits_data[(np.abs(hits_data['z']) < 315) & (hits_data['z'] > 0)]

#        value_counts = hits_data['evid'].value_counts()
#        hits_data = hits_data[hits_data['evid'].isin(value_counts[value_counts >= 100].index)]

