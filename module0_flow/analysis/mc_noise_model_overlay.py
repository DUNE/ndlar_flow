import numpy as np
import logging

from h5flow.core import H5FlowStage, resources


class MCNoiseModelOverlay(H5FlowStage):
    '''
    Modifies the specified cached hits with a random overall scale factor and an additional binomial noise 
    contribution.

    Only runs on simulation.

    Note that only data in the cache is modified, so changes are not propogated to the output file.

    '''
    class_version = '0.0.0'
    defaults = dict(
        hits_name = 'charge/hits',
        model_params = dict(
            type = 'scale_plus_binomial_noise',
            scale_factor = 1.,
            scale_width = 0.,
            binom_noise_scale = 0,
            binom_noise_width = 0.,
            binom_noise_prob = 0.
            )
        )

    def __init__(self, **params):
        super(MCNoiseModelOverlay, self).__init__(**params)
        self.is_mc = False
        for key in self.defaults:
            setattr(self, key, params.get(key, self.defaults[key]))

    def init(self, source_name):
        super(MCNoiseModelOverlay, self).init(source_name)
        self.is_mc = resources['RunData'].is_mc
        if not self.is_mc:
            return

    @staticmethod
    def scale_plus_binomial_noise(q, unique_channel_mask, scale_factor, scale_width, binom_noise_scale, binom_noise_width, binom_noise_prob, **kwargs):
        '''
        Modify the given charge values [in mV] by a random scale variation (mean=scale_factor, std=scale_width) on each channel
        and then add a +/- offset with a (mean=binom_noise_scale, std=binom_noise_width) on each channel with probability binom_noise_prob
        for each hit
        '''
        n_channels = int(unique_channel_mask.sum())
        
        # first generate gain smearing
        gain_smearing = np.random.normal(loc=scale_factor, scale=scale_width, size=n_channels)
        gain_smearing = np.take(gain_smearing, (np.cumsum(unique_channel_mask, dtype=int) - 1))

        # then generate binomial noise scale
        binomial_scale = np.random.normal(loc=binom_noise_scale, scale=binom_noise_width, size=n_channels)
        binomial_scale = np.take(binomial_scale, (np.cumsum(unique_channel_mask, dtype=int) - 1))

        # finally apply smearing effects
        rv = q * gain_smearing
        rv += binomial_scale * np.random.binomial(1, binom_noise_prob, size=q.shape) * (np.random.binomial(1, 0.5, size=q.shape) * 2 - 1)

        return rv

    def run(self, source_name, source_slice, cache):
        super(MCNoiseModelOverlay, self).run(source_name, source_slice, cache)
        if not self.is_mc:
            return
        
        hits = cache[self.hits_name]

        # get unique identifier
        unique_id = (((hits['iogroup'].astype(int) * 100
                       + hits['iochannel'].astype(int)) * 1000
                      + hits['chipid'].astype(int)) * 100
                     + hits['channelid'].astype(int))

        # sort by unique identifier
        order = np.argsort(unique_id.ravel())

        # generate single channel mask (to properly simulate smearing effects correlated to the channel, rather than hit)
        channel_mask = np.r_[True, np.diff(np.take_along_axis(unique_id.ravel(), order, axis=-1)) != 0]
        channel_mask = channel_mask & ~unique_id.mask.ravel()

        smeared_q = getattr(self, self.model_params['type'])(
            np.take_along_axis(hits['q'].filled(0).ravel(), order, axis=0),
            channel_mask,
            **self.model_params)
        np.put(hits['q'], order, smeared_q)

        cache[self.hits_name] = hits
