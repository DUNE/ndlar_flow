import numpy as np
import numpy.ma as ma

from h5flow.core import H5FlowStage


class WaveformAlign(H5FlowStage):
    '''
        Calculates the relative aligment of each ADC trigger within the event.

        Parameters:
         - ``wvfm_dset_name`` : ``str``, required, input dataset path
         - ``busy_channel``: ``dict`` of ``int`` of ``<adc #>: <channel number>

        ``wvfm_dset_name`` is required in the data cache.

        Example config::

            wvfm_align:
                classname: WaveformAlign
                requires:
                    - 'light/deconv'
                params:
                    wvfm_dset_name: 'light/deconv'
                    busy_channel:
                        All: 0

        Saves the alignment data to ``{wvfm_dset_name}/alignment``

        ``alignment`` datatype::

            offset  f4(n_adc,),   sample offset relative to the first trigger in event

    '''
    class_version = '0.0.0'

    default_busy_channel = dict(All=0)


    def align_dtype(self, nadc):
        return np.dtype([('ns', 'f8'),
            ('sample_idx', 'f4', (nadc,))])


    def __init__(self, **params):
        super(WaveformAlign, self).__init__(**params)

        self.wvfm_dset_name = params.get('wvfm_dset_name')
        self.align_dset_name = f'{self.wvfm_dset_name}/alignment'
        self.busy_channel = params.get('busy_channel', self.default_busy_channel)


    def init(self, source_name):
        super(WaveformAlign, self).init(source_name)

        wvfm_dset = self.data_manager.get_dset(self.wvfm_dset_name)
        self.nadc, self.nchannel = wvfm_dset.dtype['samples'].shape[:-1]
        self.align_dtype = self.align_dtype(self.nadc)
        busy_channel = np.full(self.nadc, -1, dtype=int)
        for adc in range(self.nadc):
            if 'All' in self.busy_channel:
                busy_channel[adc] = self.busy_channel['All']
            elif adc in self.busy_channel:
                busy_channel[adc] = self.busy_channel[adc]

        # save all config info
        self.data_manager.set_attrs(self.align_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    source_dset=source_name,
                                    wvfm_dset=self.wvfm_dset_name,
                                    busy_channel=busy_channel
                                    )

        # then set up new datasets
        self.data_manager.create_dset(self.align_dset_name, dtype=self.align_dtype)
        self.data_manager.create_ref(source_name, self.align_dset_name)


    def run(self, source_name, source_slice, cache):
        super(WaveformAlign, self).run(source_name, source_slice, cache)

        event_data = cache[source_name]
        wvfm_data = cache[self.wvfm_dset_name].reshape(event_data.shape)

        align_data = np.zeros(event_data.shape, dtype=self.align_dtype)
        for adc in range(wvfm_data['samples'].shape[1]):
            if 'All' in self.busy_channel:
                busy_channel = self.busy_channel['All']
            elif adc in self.busy_channel:
                busy_channel
            else:
                continue
            busy_wvfm = wvfm_data['samples'][..., adc, busy_channel, :]
            busy_d = np.diff(busy_wvfm, axis=-1)
            rising_edge = np.expand_dims(np.argmax(busy_d, axis=-1), axis=-1)
            # project to 0-crossing for sub-sample resolution
            rising_edge = rising_edge - np.take_along_axis(
                busy_wvfm, rising_edge, axis=-1) / np.take_along_axis(busy_d, rising_edge, axis=-1)

            align_data['sample_idx'][:,adc] = rising_edge.ravel()

        i_event_offset = np.argmax(align_data['sample_idx'], axis=-1)[...,np.newaxis]
        align_data['ns'] = np.take_along_axis(
            ma.median(ma.array(event_data['tai_ns'],
                mask=~event_data['wvfm_valid'].astype(bool)), axis=-1),
            i_event_offset, axis=-1).ravel()

        # reserve new data
        align_slice = self.data_manager.reserve_data(self.align_dset_name, source_slice)
        self.data_manager.write_data(self.align_dset_name, align_slice, align_data)

        # save references
        ref = np.c_[source_slice, align_slice]
        self.data_manager.write_ref(source_name, self.align_dset_name, ref)

