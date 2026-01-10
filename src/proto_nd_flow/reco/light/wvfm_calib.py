import numpy as np
from collections import defaultdict

from h5flow.core import H5FlowStage, resources


class WaveformCalib(H5FlowStage):
    '''
        Sums the signal across light detector SiPM channels, while applying
        a gain correction to each SiPM.

        Parameters:
         - ``wvfm_dset_name`` : ``str``, required, input dataset path
         - ``wvfm_calib_dset_name`` : ``str``, required, output calibrated wvfm dataset path
         - ``crms_dset_name`` : ``str``, required, output calibrated rms dataset path
         - ``gain``: ``dict`` of ``dict`` of ``<adc #>: <channel #>: <gain correction>`` where each gain correction converts the ADC value to visible energy
         - ``gain_mc``: same as ``gain``, but only applied if datafile is simulation

        ``wvfm_dset_name`` is required in the data cache.
        RMS data is expected to be at ``<source_name>/rms`` (created by WaveformNoiseFilter).

        The Geometry resource is required in the workflow.

        Example config::

            wvfm_calib:
                classname: WaveformCalib
                requires:
                    - 'light/events'
                    - 'light/deconv'
                params:
                    wvfm_dset_name: 'light/deconv'
                    wvfm_calib_dset_name: 'light/cwvfm'
                    crms_dset_name: 'light/cwvfm_rms'
                    gain:
                        default: 1.0

    '''
    class_version = '1.0.0'

    default_detector_channels = [list(range(64))]

    def cwvfm_dtype(self, nadc, nchannels, nsamples):
        return np.dtype([('samples', 'f4', (nadc, nchannels, nsamples))])

    def crms_dtype(self, nadc, nchannels):
        return np.dtype([('rms', 'f4', (nadc, nchannels))])

    def align_dtype(self, nadc, nchannels):
        return np.dtype([('ns', 'f8'), ('sample_idx', 'f4', (nadc, nchannels))])

    def __init__(self, **params):
        super(WaveformCalib, self).__init__(**params)

        self.wvfm_dset_name = params.get('wvfm_dset_name')
        self.wvfm_align_dset_name = f'{self.wvfm_dset_name}/alignment'
        self.cwvfm_dset_name = params.get('cwvfm_dset_name')
        self.align_dset_name = f'{self.cwvfm_dset_name}/alignment'

        # RMS data is a subdataset of the source, will be set in init()
        self.rms_dset_name = None
        self.crms_dset_name = params.get('crms_dset_name')

        self.gain = params.get('gain',{'default': 1.0})
        self.gain_mc = params.get('gain_mc',{'default': 1.0})


    def _load_gain_data(self, gain_data):
        self.gain = defaultdict(lambda : defaultdict((lambda : gain_data.get('default',1.0))))
        for adc in gain_data:
            if adc == 'default':
                continue
            for chan in gain_data[adc]:
                self.gain[adc][chan] = gain_data[adc][chan]


    def init(self, source_name):
        super(WaveformCalib, self).init(source_name)

        # Set RMS dataset path to subdataset of source
        self.rms_dset_name = f'{source_name}/rms'

        # use appropriate gain data
        if resources['RunData'].is_mc:
            self._load_gain_data(self.gain_mc)
        else:
            self._load_gain_data(self.gain)

        # save all config info
        gain = np.array([(int(adc), int(chan), self.gain[adc][chan])
                for adc in self.gain for chan in self.gain[adc]],
                dtype=np.dtype([('adc','i4'), ('chan','i4'), ('gain','f8')]))
        self.data_manager.set_attrs(self.cwvfm_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    source_dset=source_name,
                                    wvfm_dset=self.wvfm_dset_name,
                                    gain=gain
                                    )

        # then set up new datasets
        wvfm_dset = self.data_manager.get_dset(self.wvfm_dset_name)

        self.cwvfm_dtype = self.cwvfm_dtype(*wvfm_dset.dtype['samples'].shape)
        self.data_manager.create_dset(self.cwvfm_dset_name, dtype=self.cwvfm_dtype)
        self.data_manager.create_ref(source_name, self.cwvfm_dset_name)

        # crms_dtype only needs nadc, nchannels - not nsamples
        nadc, nchannels, nsamples = wvfm_dset.dtype['samples'].shape
        self.crms_dtype = self.crms_dtype(nadc, nchannels)
        self.data_manager.create_dset(self.crms_dset_name, dtype=self.crms_dtype)
        self.data_manager.create_ref(source_name, self.crms_dset_name)

        if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
            self.align_dtype = self.align_dtype(wvfm_dset.dtype['samples'].shape[-3], wvfm_dset.dtype['samples'].shape[-2])
            self.data_manager.create_dset(self.align_dset_name, dtype=self.align_dtype)
            self.data_manager.create_ref(source_name, self.align_dset_name)

    def run(self, source_name, source_slice, cache):
        super(WaveformCalib, self).run(source_name, source_slice, cache)

        event_data = cache[source_name]
        wvfm_data = cache[self.wvfm_dset_name].reshape(event_data.shape)
        cwvfm_data = np.zeros(event_data.shape, dtype=self.cwvfm_dtype)
        rms_data = cache[self.rms_dset_name].reshape(event_data.shape)
        crms_data = np.zeros(event_data.shape, dtype=self.crms_dtype)

        if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
            wvfm_align_data = cache[self.wvfm_align_dset_name].reshape(event_data.shape)
            align_data = np.zeros(event_data.shape, dtype=self.align_dtype)

        if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
            for adc in range(wvfm_data['samples'].shape[1]):
                for chan in range(wvfm_data['samples'].shape[2]):
                    mask = event_data['wvfm_valid'][:,adc,chan].astype(bool)
                    if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
                        align_data['sample_idx'][mask,adc,chan] = wvfm_align_data['sample_idx'][mask,adc]
                        align_data['ns'][mask] = wvfm_align_data['ns'][mask]

        for adc in range(wvfm_data['samples'].shape[1]):
            for chan in range(wvfm_data['samples'].shape[2]):
                mask = event_data['wvfm_valid'][:,adc,chan].astype(bool)
                cwvfm_data['samples'][mask,adc,chan,:] = (
                    wvfm_data['samples'][mask,adc,chan].filled(0)
                    * self.gain[adc][chan])
                crms_data['rms'][mask,adc,chan] = (
                    rms_data['rms'][mask,adc,chan]
                    * self.gain[adc][chan])

        # reserve new data
        cwvfm_slice = self.data_manager.reserve_data(self.cwvfm_dset_name, source_slice)
        self.data_manager.write_data(self.cwvfm_dset_name, source_slice, cwvfm_data)

        if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
            align_slice = self.data_manager.reserve_data(self.align_dset_name, source_slice)
            self.data_manager.write_data(self.align_dset_name, align_slice, align_data)

        crms_slice = self.data_manager.reserve_data(self.crms_dset_name, source_slice)
        self.data_manager.write_data(self.crms_dset_name, source_slice, crms_data)

        # save references
        ref = np.c_[source_slice, cwvfm_slice]
        self.data_manager.write_ref(source_name, self.cwvfm_dset_name, ref)

        if(self.data_manager.dset_exists(self.wvfm_align_dset_name)):
            ref = np.c_[source_slice, align_slice]
            self.data_manager.write_ref(source_name, self.align_dset_name, ref)

        ref = np.c_[source_slice, crms_slice]
        self.data_manager.write_ref(source_name, self.crms_dset_name, ref)

