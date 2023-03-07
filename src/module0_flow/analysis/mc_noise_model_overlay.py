import numpy as np
import numpy.ma as ma
import logging

from h5flow.core import H5FlowStage, resources


class MCNoiseModelOverlay(H5FlowStage):
    '''
    Modifies the specified cached hits with a random overall scale factor and an additional binomial noise 
    contribution.

    Only runs on simulation.

    Note that ``save_result == False`` only data in the cache is modified, so changes are not propogated to the output file.

    Requires ``RunData`` resource.

    Example config::

      noise_overlay:
        classname: MCNoiseModelOverlay
        path: module0_flow.analysis.mc_noise_model_overlay
        requires:
          - 'charge/hits' # should mimic description used by downstream stages
          - name: 'charge/hits_idx' # optional, only required if save_result == True
            path: ['charge/hits']
            index_only: True
        params:
          hits_name: 'charge/hits'
          hit_charge_name: 'charge/hits' # dataset to grab 'q' from
          hits_idx_name: 'charge/hits_idx' # optional, only required if save_result == True
          smear_name: 'charge/hits_q_smear' # optional, only required if save_result == True
          save_result: False
          model_params:
            medm:
              type: 'scale_plus_binomial_noise'
              scale_factor: 1.0
              scale_width: 0.068
              binom_noise_scale: 20.74 # mV
              binom_noise_width: 3.447 # mV
              binom_noise_prob: 0.2577
              binom_noise_asymm: 0.50

    '''
    class_version = '0.0.0'
    defaults = dict(
        save_result = False,
        hits_name = 'charge/hits',
        hit_charge_name = 'charge/hits',
        hits_idx_name = 'charge/hits_idx',
        smear_name = 'charge/hit_q_smear',
        model_params = dict(
            type = 'scale_plus_binomial_noise',
            scale_factor = 1.,
            scale_width = 0.,
            binom_noise_scale = 0,
            binom_noise_width = 0.,
            binom_noise_prob = 0.,
            binom_noise_asymm = 0.5
            )
        )

    smear_dtype = np.dtype('f4')

    def __init__(self, **params):
        super(MCNoiseModelOverlay, self).__init__(**params)
        self.is_mc = False
        for key in self.defaults:
            setattr(self, key, params.get(key, self.defaults[key]))

    def init(self, source_name):
        super(MCNoiseModelOverlay, self).init(source_name)
        self.is_mc = resources['RunData'].is_mc

        if self.save_result:
            self.data_manager.create_dset(self.smear_name, dtype=self.smear_dtype)
            self.data_manager.create_ref(self.hits_name, self.smear_name)
        
        if not self.is_mc:
            return

        threshold_mode = resources['RunData'].charge_thresholds
        self.model_params = self.model_params[threshold_mode]
        logging.info(f'using {threshold_mode} noise overlay model: {str(self.model_params)}')

    @staticmethod
    def scale_plus_binomial_noise(q, unique_id, scale_factor, scale_width, binom_noise_scale, binom_noise_width, binom_noise_prob, binom_noise_asymm, **kwargs):
        '''
        Modify the given charge values [in mV] by a random scale variation (mean=scale_factor, std=scale_width) on each channel
        and then add a +/- offset with a (mean=binom_noise_scale, std=binom_noise_width) on each channel with probability binom_noise_prob. The likelihood of a +/- application is controlled via the binom_noise_asymm parameter (0.5 == equal probability, 1.0 == 100% positive noise shift, 0.0 == 100% negative noise shift)
        for each hit

        :param q: 1D array of hit charge to apply noise overlay [mV]

        :param unique_id: 1D integer array, same shape as q indicating the unique channel id for each hit

        :param scale_factor: mean value of channel-to-channel gain correction

        :param scale_width: std of channel-to-channel gain correction

        :param binom_noise_scale: mean value of binomial noise scale

        :param binom_noise_wdith: channel-to-channel std of binomial noise scale

        :param binom_noise_prob: probability of observing binomial noise on each hit

        :param binom_noise_asymm: probability of observing + binomial noise (vs -)

        :returns: 1D array of hit charges with noise effect applied [mV]
        '''
        # sort by unique identifier
        order = np.argsort(unique_id)
        unique_id = np.take_along_axis(unique_id, order, axis=-1)

        # generate single channel mask (to properly simulate smearing effects correlated to the channel, rather than hit)
        unique_channel_mask = np.r_[True, np.diff(unique_id) != 0]
        unique_channel_mask = unique_channel_mask & ~unique_id.mask

        q = np.take_along_axis(q.filled(0), order, axis=0)
        
        n_channels = int(unique_channel_mask.sum())
        
        # first generate gain smearing
        gain_smearing = np.random.normal(loc=scale_factor, scale=scale_width, size=n_channels)
        gain_smearing = np.take(gain_smearing, (np.cumsum(unique_channel_mask, dtype=int) - 1))

        # then generate binomial noise scale
        binomial_scale = np.random.normal(loc=binom_noise_scale, scale=binom_noise_width, size=n_channels)
        binomial_scale = np.take(binomial_scale, (np.cumsum(unique_channel_mask, dtype=int) - 1))

        # finally apply smearing effects
        smeared_q = q * gain_smearing
        smeared_q += binomial_scale * np.random.binomial(1, binom_noise_prob, size=q.shape) * (np.random.binomial(1, binom_noise_asymm, size=q.shape) * 2 - 1)

        # put back into original order
        rv = np.empty_like(q)
        np.put(rv, order, smeared_q)

        if hasattr(q, 'mask'):
            rv = ma.array(rv, mask=q.mask)
        return rv

    def run(self, source_name, source_slice, cache):
        super(MCNoiseModelOverlay, self).run(source_name, source_slice, cache)
        if not self.is_mc and not self.save_result:
            return

        hits = cache[self.hits_name]
        hit_q = cache[self.hit_charge_name]

        # get unique identifier
        unique_id = (((hits['iogroup'].astype(int) * 100
                       + hits['iochannel'].astype(int)) * 1000
                      + hits['chipid'].astype(int)) * 100
                     + hits['channelid'].astype(int))

        if (~hits.mask['id']).sum() > 0:
            smeared_q = getattr(self, self.model_params['type'])(hit_q['q'].ravel(), unique_id.ravel(), **self.model_params)
        else:
            smeared_q = hit_q['q'].ravel()

        if self.is_mc:
            hit_q['q'] = smeared_q.reshape(hit_q.shape)
            cache[self.hit_charge_name] = hit_q

        if self.save_result:
            hits_idx = cache[self.hits_idx_name].reshape(hits.shape)

            self.data_manager.reserve_data(self.smear_name, hits_idx.compressed())
            self.data_manager.write_data(self.smear_name, hits_idx.compressed(), smeared_q.ravel()[~hits_idx.mask])
            self.data_manager.write_ref(self.hits_name, self.smear_name, np.c_[(hits_idx.compressed(),)*2])
