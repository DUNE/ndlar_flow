import numpy as np
import numpy.ma as ma
import logging
import scipy.stats as stats

from h5flow.core import H5FlowStage, resources


class Dropout(H5FlowStage):
    '''
    Masks the specified cached hits with a given uniform (aka Bernoulli) probability for downstream components.

    Only runs on simulation.

    Requires ``RunData`` resource.

    Example config::

      dropout:
        classname: Dropout
        path: module0_flow.analysis.dropout
        requires:
          - 'charge/hits' # should mimic description used by downstream stages
        params:
          mask: # multiple datasets can be masked as long as they are broadcastable, the first dataset is used to determine the baseline shape and generate the drop out mask
            - 'charge/hits'
          p: 0.03 # probability of dropping an item

    '''
    class_version = '0.0.0'
    defaults = dict(p=0.05)

    def __init__(self, **params):
        super(Dropout, self).__init__(**params)
        self.is_mc = False
        for key in self.defaults:
            setattr(self, key, params.get(key, self.defaults[key]))
        self.mask = params['mask']

    def init(self, source_name):
        super(Dropout, self).init(source_name)
        self.is_mc = resources['RunData'].is_mc
        if not self.is_mc:
            return

    def run(self, source_name, source_slice, cache):
        super(Dropout, self).run(source_name, source_slice, cache)
        if not self.is_mc:
            logging.info('skipping drop out on data file')
            return

        # fetch datasets from file
        dsets = []
        for name in self.mask:
            dsets.append(cache[name])

        # extract number of valid entries
        n_unmasked = dsets[0].size
        if hasattr(dsets[0], 'recordmask'):
            n_unmasked = (~(dsets[0].recordmask)).sum()
        logging.info(f'using {n_unmasked} unmasked elements in {self.mask[0]} {dsets[0].shape} for dropout base mask')

        # generate random mask
        rvs = stats.bernoulli.rvs(p=self.p, size=n_unmasked).astype(bool)
        logging.info(f'dropping {rvs.sum()} / {n_unmasked}')

        # create a new mask using random mask
        if hasattr(dsets[0], 'recordmask'):
            new_mask = dsets[0].recordmask.copy()
        else:
            # default behavior is to use all elements
            new_mask = np.zeros(dsets[0].shape, dtype=bool)
        if n_unmasked > 0:
            np.place(new_mask, ~new_mask, rvs)

        # update cache with new masked arrays
        for i, name in enumerate(self.mask):
            new_mask_i = new_mask.copy()
            while len(new_mask_i.shape) < len(cache[name].shape):
                new_mask_i = np.expand_dims(new_mask_i, axis=-1)
            new_mask_i = np.broadcast_to(new_mask_i, cache[name].shape)
            cache[name] = ma.array(cache[name], mask=new_mask_i.copy())
