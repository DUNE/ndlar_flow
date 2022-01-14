import numpy as np
import logging

from h5flow.core import H5FlowStage


class WaveformSummary(H5FlowStage):
    '''
        Extracts summary parameters from light waveforms

        Parameters:
         - ``pretrigger_window`` : ``tuple`` of first sample and last sample to use for calculating pre-trigger values
         - ``wvfm_dset_name`` : ``str`` dataset path to waveforms to process
         - ``wvfm_summ_dset_name`` : ``str``, optional, output dataset name, defaults to ``{wvfm_dset_name}_summ``

         ``{}_summ`` datatype::

            id              u8, unique identifier per waveform
            event_id        i8, unique identifier for event
            pre_std         f8, std of pretrigger samples
            pre_mean        f8, mean of pretrigger samples
            post_sum        f8, sum of posttrigger samples
            post_max        f8, max of posttrigger samples
            post_rising     f8, sample index of max derivative
            ch              u4, channel id
            sn              u4, serial number of adc
            adc             u4, adc index 1:1 w/ serial number

    '''
    class_version = '0.0.0'

    default_pretrigger_window = (0, 80)
    default_wvfm_dset_name = 'light/wvfm'
    default_wvfm_summ_dset_fmt = '{}_summ'  # uses wvfm_dset + _summ as default

    dtype = np.dtype([
        ('id', 'u8'),
        ('event_id', 'i8'),
        ('pre_std', 'f8'),
        ('pre_mean', 'f8'),
        ('post_sum', 'f8'),
        ('post_max', 'f8'),
        ('post_rising', 'f8'),
        ('ch', 'u4'),
        ('sn', 'u4'),
        ('adc', 'u4'),
    ])

    def __init__(self, **params):
        super(WaveformSummary, self).__init__(**params)

        self.wvfm_dset_name = params.get('wvfm_dset_name')
        self.wvfm_summ_dset_name = params.get('wvfm_summ_dset_name', self.default_wvfm_summ_dset_fmt.format(self.wvfm_dset_name))
        self.pretrigger_window = params.get('pretrigger_window', self.default_pretrigger_window)

    def init(self, source_name):
        super(WaveformSummary, self).init(source_name)

        # save all config info
        self.data_manager.set_attrs(self.wvfm_summ_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    source_dset=source_name,
                                    wvfm_dset=self.wvfm_dset_name,
                                    pretrigger_window=self.pretrigger_window
                                    )

        # then set up new datasets
        self.data_manager.create_dset(self.wvfm_summ_dset_name, dtype=self.dtype)

    def run(self, source_name, source_slice, cache):
        super(WaveformSummary, self).run(source_name, source_slice, cache)

        event_data = cache[source_name]
        wvfm_data = cache[self.wvfm_dset_name].data

        if len(wvfm_data):
            pre_wvfm = np.take(wvfm_data['samples'].astype('f8'), np.arange(self.pretrigger_window[0], self.pretrigger_window[-1]), axis=-1)
            post_wvfm = np.take(wvfm_data['samples'].astype('f8'), np.arange(self.pretrigger_window[-1], wvfm_data['samples'].shape[-1]), axis=-1)
            pre_std = np.std(pre_wvfm, axis=-1)
            pre_mean = np.mean(pre_wvfm, axis=-1)

            post_sum = np.sum(post_wvfm - np.expand_dims(pre_mean, -1), axis=-1)
            post_max = np.max(post_wvfm - np.expand_dims(pre_mean, -1), axis=-1)
            post_rising = np.argmax(np.diff(post_wvfm - np.expand_dims(pre_mean, -1), axis=-1), axis=-1)

            ch = event_data['ch']
            sn = np.expand_dims(event_data['sn'], axis=-1)
            sn = np.broadcast_to(sn, ch.shape)
            adc = np.arange(sn.shape[1]).reshape(1, -1, 1)
            adc = np.broadcast_to(adc, ch.shape)

            event_id = np.broadcast_to(event_data['id'].reshape(-1, 1, 1), ch.shape)

        valid = event_data['wvfm_valid'].astype(bool)

        summ_arr = np.empty(np.count_nonzero(valid), dtype=self.dtype)
        summ_slice = self.data_manager.reserve_data(self.wvfm_summ_dset_name, len(summ_arr))
        if len(wvfm_data):
            mask = valid.flatten()
            summ_arr['id'] = np.arange(summ_slice.start, summ_slice.stop)
            summ_arr['event_id'] = event_id.flat[mask]
            summ_arr['pre_std'] = pre_std.flat[mask]
            summ_arr['pre_mean'] = pre_mean.flat[mask]
            summ_arr['post_sum'] = post_sum.flat[mask]
            summ_arr['post_max'] = post_max.flat[mask]
            summ_arr['post_rising'] = post_rising.flat[mask]
            summ_arr['ch'] = ch.flat[mask]
            summ_arr['sn'] = sn.flat[mask]
            summ_arr['adc'] = adc.flat[mask]
        self.data_manager.write_data(self.wvfm_summ_dset_name, summ_slice, summ_arr)
