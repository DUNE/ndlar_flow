import numpy as np
import numpy.ma as ma
from collections import defaultdict
import scipy.interpolate
from scipy.signal import butter, filtfilt

from h5flow.core import H5FlowStage, resources

import module0_flow.util.units as units
import time

class WaveformHitFinder(H5FlowStage):
    '''
        Extracts "hits" from light events using a simple threshold. So if a maximum value in a waveform is above
        a threshold, then it is considered a hit. This class currently only uses the sum waveforms dataset, not the
        SiPM waveforms.

        Parameters:
         - ``wvfm_dset_name``: ``str``, path to input waveforms
         - ``t_ns_dset_name``: ``str``, path to corrected light PPS timestamps
         - ``hits_dset_name``: ``str``, path to output hits dataset
         - ``threshold``: ``dict`` of ``dict`` containing sets of ``tpc_index: {channel_index: threshold, ...}`` used for hit finding. A fixed global value can also be specified with a single ``float`` value
         - ``module_override``: ``int``, allows for overriding the module (for 2x2) for saving the right tpc values in the sum hits dataset. Relevant for the low threshold runs where the code may find the wrong tpc values. 

         Both ``wvfm_dset_name``, ``{wvfm_dset_name}/alignment``, and ``t_ns_dset_name`` are required in the cache.

         Requires RunData resource in workflow.

         ``hits`` datatype::

            id                  u4,             unique identifier
            tpc                 u1,             tpc (for sum_hit)
            det                 u1,             detector (for sum_hit/sipm_hit)
            boundary            f4(3),          (x,y,z) center of det
            samples             f4(nsamples,),  waveform adc values
            samples_filtered    f4(nsamples,),  filtered waveform adc values
            amplitude           f4,             peak adc value
    '''
    class_version = '2.0.0'

    default_hits_dset_name = 'light/hits'
    default_global_threshold = 1000
    default_module_override = None
    default_real_ADC = None
    default_rel_ADC = None
    default_mask = []

    def default_threshold(self, global_threshold):
        return defaultdict(lambda: defaultdict(lambda: global_threshold))

    def hits_dtype(self, nsamples):
        if self.hit_level=="sum":
            return np.dtype([
                    ('id', 'u4'),
                    ('tpc', 'u1'),
                    ('det', 'u1'),
                    ('pos', 'f4', (3,)),
                    ('samples', 'f4', (nsamples,)),
                    ('samples_filtered', 'f4', (nsamples,)),
                    ('amplitude', 'f4'),
                    ('ts_pps', 'f8'),
                    ('unix', 'i8')
                ])
        elif self.hit_level=='sipm':
            return np.dtype([
                    ('id', 'u4'),
                    ('adc', 'u1'),
                    ('chan', 'u1'),
                    ('pos', 'f4', (3,)),
                    ('samples', 'f4', (nsamples,)),
                    ('samples_filtered', 'f4', (nsamples,)),
                    ('amplitude', 'f4'),
                    ('ts_pps', 'f8'),
                    ('unix', 'i8')
                ])
        else:
            raise RuntimeError(f'Invalid hit level {self.hit_level}')
    
    def apply_filter(self, waveform, cutoff_freq, order=2):
        """
        Apply a low pass filter to the input waveform.

        Parameters:
            waveform (array_like): Input waveform array.
            cutoff_freq (float): Cutoff frequency of the low-pass filter in Hz.
            order (int): Order of the low-pass filter.

        Returns:
            array_like: Filtered waveform.
        """
        sampling_freq = 62.5e6
        normalized_cutoff_freq = cutoff_freq / (sampling_freq / 2)
        b, a = butter(order, normalized_cutoff_freq, btype='lowpass', analog=False)
        filtered_waveform = filtfilt(b, a, waveform)

        return filtered_waveform
    
    def __init__(self, **params):
        super(WaveformHitFinder, self).__init__(**params)
        self.wvfm_dset_name = params.get('wvfm_dset_name')
        self.t_ns_dset_name = params.get('t_ns_dset_name')
        self.hits_dset_name = params.get('hits_dset_name', self.default_hits_dset_name)
        self.module_override = params.get('module_override', self.default_module_override)
        self.real_ADC = params.get('real_ADC', self.default_real_ADC)
        self.rel_ADC = params.get('rel_ADC', self.default_rel_ADC)
        self.hit_level = params.get('hit_level')
        
        # set hit finding thresholds (will be converted to an array later in init())
        self.threshold = params.get('threshold', self.default_global_threshold)
        
        if isinstance(self.threshold, int) or isinstance(self.threshold, float):
            # if a global threshold is specified, use the default generator
            self.threshold = self.default_threshold(
                self.threshold)
        elif isinstance(self.threshold, dict):
            # otherwise convert to a defaultdict
            new_dict = self.default_threshold(
                self.default_global_threshold)
            for key, subdict in self.threshold.items():
                for subkey, subval in subdict.items():
                    new_dict[key][subkey] = subval
            self.threshold = new_dict
        print('threshold = ', self.threshold)
            
    def init(self, source_name):
        super(WaveformHitFinder, self).init(source_name)

        wvfm_dset = self.data_manager.get_dset(self.wvfm_dset_name)

        # get convert sample rate to ns
        self.sample_rate = (resources['RunData'].lrs_ticks
                            / units.ns)

        # get waveform shape information
        self.ntpc = wvfm_dset.dtype['samples'].shape[0]
        self.ndet = wvfm_dset.dtype['samples'].shape[1]
        print(f'ndet = {self.ndet}')
        self.nsamples = wvfm_dset.dtype['samples'].shape[2]
        self.hits_dtype = self.hits_dtype(self.nsamples)
        
        # convert channel thresholds into an array
        #threshold_array = np.zeros((self.ntpc, self.ndet, 1))
        #for tpc in range(self.ntpc):
        #    for det in range(self.ndet):
        #        threshold_array[tpc,
        #                        det] = self.threshold[tpc][det]
        #self.threshold = threshold_array
        self.threshold = 1500
        # create datasets and references
        self.data_manager.create_dset(self.hits_dset_name,
                                      dtype=self.hits_dtype)
        #self.data_manager.create_ref(source_name, self.hits_dset_name)
        self.data_manager.create_ref(self.hits_dset_name, source_name)
        self.data_manager.create_ref(self.hits_dset_name, self.wvfm_dset_name)
        self.data_manager.set_attrs(self.hits_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    wvfm_dset=self.wvfm_dset_name,
                                    thresholds=self.threshold,
                                    ntpc=self.ntpc,
                                    ndet=self.ndet,
                                    nsamples=self.nsamples,
                                    hit_level=self.hit_level
                                    )

    def run(self, source_name, source_slice, cache):
        super(WaveformHitFinder, self).run(source_name, source_slice, cache)
        wvfms = np.array(cache[self.wvfm_dset_name].reshape(cache[source_name].shape)[
            'samples'])  # 1:1 relationship
        events = np.array(cache[source_name])
        
        event_id = []
        hits_data = np.zeros((0,), dtype=self.hits_dtype)
        #adc_mask = np.all(wvfms[0][:,-1,:] != 0, axis=1)
        #print('adc_mask = ', adc_mask)
        #adcs = np.where(adc_mask)[0]
        #if self.module_override is not None:
        #    if self.module_override == 0:
        #        tpcs = [0,1]
        #        adcs = [1,0]
        #    elif self.module_override == 1:
        #        tpcs = [2,3]
        #        adcs = [2,3]
        #    elif self.module_override == 2:
        #        tpcs = [4,5]
        #        adcs = [5,4]
        #    elif self.module_override == 3:
        #        tpcs = [6,7]
        #        adcs = [7,6]
        #    else:
        #        raise Exception('module_override value not supported')
        #    self.ntpc = 2
        self.ntpc = 2
        
        #else:
        #    tpcs = [tpc for tpc in range(np.sum(adc_mask))]
        #    self.ntpc = np.sum(adc_mask)
        
        adc_indices_rel, det_indices = np.where(np.any(wvfms[0] != 0, axis=2))
        #if self.module_override is not None:
        #    adc_indices = np.copy(adc_indices_rel)
        #    for i, unique_adc in enumerate(np.unique(adc_indices)):
        #        adc_indices[adc_indices == unique_adc] = adcs[i]
        #print('adc_indices = ', adc_indices)
        #print('det_indices = ', det_indices)
        #utime_ms_all = events['utime_ms'][:,0]
        #tai_ns_all = events['tai_ns'][:,0]

        avg_time1=[]
        avg_time2=[]
        avg_time3=[]
        
        wvfms = wvfms - np.mean(wvfms, axis=3)[:, :, :, np.newaxis]

        for i in range(len(events)):
            #print(f'event {i}')
            start = time.time()
            event = events[i]
            
            tai_ns = np.unique(event['tai_ns'])
            tai_ns = tai_ns[tai_ns != 0][0]*1e-3
            utime_ms = np.unique(event['utime_ms'])
            utime_ms = int(utime_ms[utime_ms != 0][0]*1e-3)
            
            #tai_ns = tai_ns_all[i]*1e-3
            #utime_ms = int(utime_ms_all[i]*1e-3)
            #wvfms_event = wvfms[i]#[adc_mask]
            avg_time1.append(time.time()-start)
            total_hits = 0
            for j in range(len(adc_indices_rel)):
                adc_rel = adc_indices_rel[j]
                if self.rel_ADC is not None and adc_rel != self.rel_ADC:
                    continue
                if self.real_ADC is not None:
                    adc = self.real_ADC #adc_indices[j]
                else:
                    adc = adc_rel
                det_id = det_indices[j]
                
                #tpc = tpcs[j]
                #adc = adcs[j]
                #if not np.all(wvfms_event[j,0,:] != 0):
                #    continue # skip tpcs without data
                #for det_id in range(self.ndet):
                hit_data = np.zeros((1,), self.hits_dtype)
                #wvfm = np.array(wvfms_event[adc, det_id, :]).astype('int')
                #wvfm -= int(np.mean(wvfm))
                wvfm = wvfms[i, adc_rel, det_id, :]
                #wvfm_filtered = self.apply_filter(wvfm, 10e6, 2)
                #print(np.max(np.abs(wvfm)), np.max(np.abs(wvfm_filtered)))
                #if np.any(wvfm > self.threshold[adc][det_id][0]):
                wvfm_max = np.max(wvfm)
                
                if wvfm_max > self.threshold:
                    #print('det_id = ', det_id)
                    total_hits+=1
                    event_id.append(events[i]['id'])
                    hit_data['samples'] = wvfm
                    #hit_data['samples_filtered'] = wvfm_filtered
                    hit_data['amplitude'] = wvfm_max
                    if self.hit_level=="sum":
                        hit_data['tpc'] = adc // 2
                        hit_data['det'] = det_id
                        boundary = np.array(resources['Geometry'].det_bounds[(adc // 2,det_id)][0])
                        hit_data['pos'] = (boundary[1]+boundary[0])/2
                    elif self.hit_level=='sipm':
                        #det_id += 32
                        hit_data['adc'] = adc
                        hit_data['chan'] = det_id
                        hit_data['pos'] = np.array(resources['Geometry'].sipm_abs_pos[(adc,det_id)][0])
                        
                        #print('adc, det_id = ', adc, ' , ', det_id)
                        #print(hit_data['pos'])
                    hit_data['ts_pps'] = tai_ns
                    hit_data['unix'] = utime_ms
                    hits_data = np.concatenate((hits_data, hit_data))
                #else:
                #    print('adc, det_id = ', adc, ' , ', det_id)
                #    print('wvfm_max = ', np.min(wvfm))
                #    print('wvfm samples = ', wvfm[0:100])
            #print('total hits in event = ', total_hits)
            #print('avg time 1 = ', sum(avg_time1)/len(avg_time1))
            #print('avg time 2 = ', sum(avg_time2)/len(avg_time2))
            #print('avg time 3 = ', sum(avg_time3)/len(avg_time3))
                    
        # save data
        hit_slice = self.data_manager.reserve_data(
            self.hits_dset_name, len(hits_data))
        #if len(hit_data):
        #    hit_data['id'] = np.r_[hit_slice]
        hits_data['id'] = hit_slice.start + np.arange(len(hits_data), dtype=int)
        self.data_manager.write_data(self.hits_dset_name, hit_slice, hits_data)

        if len(hits_data):
            ref = np.c_[hits_data['id'], event_id]
        else:
            ref = np.empty((0, 2))
        self.data_manager.write_ref(self.hits_dset_name, source_name, ref)
        self.data_manager.write_ref(
            self.hits_dset_name, self.wvfm_dset_name, ref)
        
        