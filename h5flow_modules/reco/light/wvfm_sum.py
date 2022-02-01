import numpy as np

from h5flow.core import H5FlowStage


class WaveformSum(H5FlowStage):
    '''
        Sums the signal on a collection of light detector SiPM channels.

        Parameters:
         - ``wvfm_dset_name`` : ``str``, required, input dataset path
         - ``swvfm_dset_name`` : ``str``, required, output dataset path
         - ``detector_channels`` : ``dict`` of ``int : list`` of ``list`` pairs, optional, ``int`` represents ADC number and ``list`` indicates which channels on the ADC should be summed for each detector channel, use special keyword ``All`` to apply the same summation across all ADCs

        ``wvfm_dset_name`` is required in the data cache.

        Example config::

            wvfm_sum:
                classname: WaveformSum
                requires:
                    - 'light/events'
                    - 'light/deconv_wvfm'
                params:
                    wvfm_dset_name: 'light/deconv_wvfm'
                    swvfm_dset_name: 'light/swvfm'
                    detector_channels:
                        All:
                            # ArcLights have 6 SiPMs each
                            - [ 2,  3,  4,  5,  6,  7, ] # det 0
                            # LCMs have 2 SiPMs each
                            - [ 9, 10,] # det 1
                            - [11, 12,] # det 2
                            - [13, 14,] # det 3
                            - [18, 19, 20, 21, 22, 23, ] # det 4
                            - [25, 26,] # det 5
                            - [27, 28,] # det 6
                            - [29, 30,] # det 7
                            - [34, 35, 36, 37, 38, 39, ] # det 8
                            - [41, 42,] # det 9
                            - [43, 44,] # det 10
                            - [45, 46,] # det 11
                            - [50, 51, 52, 53, 54, 55, ] # det 12
                            - [57, 58,] # det 13
                            - [59, 60,] # det 14
                            - [61, 62,] # det 15

        Uses the same dtype as the input waveform dataset except with
        ``'samples'`` resized to be ``(nadc, ndet)``.

    '''
    class_version = '0.0.0'

    default_detector_channels = [list(range(64))]

    def swvfm_dtype(self, nadc, ndet, nsamples): return np.dtype([('samples', 'f4', (nadc, ndet, nsamples))])

    def __init__(self, **params):
        super(WaveformSum, self).__init__(**params)

        self.wvfm_dset_name = params.get('wvfm_dset_name')
        self.swvfm_dset_name = params.get('swvfm_dset_name')
        self.detector_channels = params.get('detector_channels', {'All': self.default_detector_channels})
        self.ndet = max([len(det) for det in self.detector_channels.values()])

    def init(self, source_name):
        super(WaveformSum, self).init(source_name)

        # save all config info
        self.data_manager.set_attrs(self.swvfm_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    source_dset=source_name,
                                    wvfm_dset=self.wvfm_dset_name
                                    )

        # then set up new datasets
        wvfm_dset = self.data_manager.get_dset(self.wvfm_dset_name)

        self.swvfm_dtype = self.swvfm_dtype(wvfm_dset.dtype['samples'].shape[0],
            self.ndet, wvfm_dset.dtype['samples'].shape[2])
        self.data_manager.create_dset(self.swvfm_dset_name, dtype=self.swvfm_dtype)
        self.data_manager.create_ref(source_name, self.swvfm_dset_name)

    def run(self, source_name, source_slice, cache):
        super(WaveformSum, self).run(source_name, source_slice, cache)

        event_data = cache[source_name]
        wvfm_data = cache[self.wvfm_dset_name].reshape(event_data.shape).data

        swvfm_data = np.empty(event_data.shape, dtype=self.swvfm_dtype)
        for i_adc in range(swvfm_data['samples'].shape[1]):
            for i_det in range(wvfm_data['samples'].shape[2]):
                if 'All' in self.detector_channels:
                    key = 'All'
                elif i_adc in self.detector_channels:
                    key = i_adc
                else:
                    raise KeyError(f'ADC #{i_adc} not found in detector_channels')

                if i_det < len(self.detector_channels[key]):
                    channels = np.array(self.detector_channels[key][i_det])
                    swvfm_data['samples'][:,i_adc,i_det,:] = np.sum(np.expand_dims(event_data['wvfm_valid'][:,i_adc,channels].astype(float),-1)
                                                                    * wvfm_data['samples'][:,i_adc,channels,:], axis=1)

        # reserve new data
        swvfm_slice = self.data_manager.reserve_data(self.swvfm_dset_name, source_slice)
        self.data_manager.write_data(self.swvfm_dset_name, source_slice, swvfm_data)

        # save references
        ref = np.c_[source_slice, swvfm_slice]
        self.data_manager.write_ref(source_name, self.swvfm_dset_name, ref)

