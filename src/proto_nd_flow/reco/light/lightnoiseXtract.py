import numpy as np
import numpy.ma as ma
from scipy.fft import rfft, rfftfreq
import os

from h5flow.core import H5FlowStage, resources
from h5flow import H5FLOW_MPI
import proto_nd_flow.util.units as units


class LightNoiseExtraction(H5FlowStage):
    '''
    Generate noise arrays for each optical channel. 
    Computes an average fft for each channel across all events in a file.

    Doesn't need to create references: only needs to output a file.

    Requires ``light/wvfm`` in the data cache.

    Also requires RunData resource in workflow.

    Example config::

        light_noise_extractor:
          classname: LightNoiseExtraction
          params:
            light_event_dset_name: 'light/events'
            light_wvfm_dset_name: 'light/wvfm'

    '''
    class_version = '0.0.1'

    SAMPLE_RATE = 0.016 # us
    BIT = 14  # factor from unused ADC bits on LRS: would be nice to have in a resource .yaml

    def __init__(self, **params):
        super(LightNoiseExtractor, self).__init__(**params)

        self.light_event_dset_name = params.get('light_evnt_dset_name')
        self.light_wvfm_dset_name = params.get('ligt_wvfm_dset_name')
        self.n_file = params.get('n_file', self.n_file)
        
        self.BIT = 2**(16 - resources['RunData'].lrs_bit)

    def init(self, source_name):
        super(LightNoiseExtraction, self).init(source_name)

        # save all config info
        self.light_wvfm_dset_name = source_name
        self.data_manager.set_attrs(self.events_dset_name,
                                    charge_to_light_assoc_classname=self.classname,
                                    charge_to_light_assoc_class_version=self.class_version,
                                    light_event_dset=self.light_event_dset_name,
                                    charge_to_light_assoc_unix_ts_window=self.unix_ts_window,
                                    charge_to_light_assoc_ts_window=self.ts_window
                                    )

        # load in light system waveforms (only take 1000, since they take a lot of space)
        _, ADC, CHAN, SAMPLES = self.data_manager.get_dset(self.light_wvfm_dset_name).dtype['samples'].shape
        self.light_event_mask = self.data_manager.get_dset(self.light_event_dset_name)['wvfm_valid']

        # only keep the positive fft frequencies
        #self.fft_freq = np.fft.fftfreq(self.SAMPLES, self.SAMPLE_RATE)[:self.SAMPLES//2]      

    def finish(self, source_name):
        super(LightNoiseExtraction, self).finish(source_name)
        
        if H5FLOW_MPI:
            self.total_charge_events = self.comm.reduce(self.total_charge_events, root=0)
            self.total_charge_triggers = self.comm.reduce(self.total_charge_triggers, root=0)
            self.total_matched_triggers = self.comm.reduce(self.total_matched_triggers, root=0)
            self.total_matched_events = self.comm.reduce(self.total_matched_events, root=0)
            self.matched_light = self.comm.reduce(self.matched_light, root=0)

        if self.rank == 0:
            self.total_matched_light = self.matched_light.clip(0,1).sum()
            trigger_eff = self.total_matched_triggers/max(self.total_charge_triggers, 1)
            event_eff = self.total_matched_events/max(self.total_charge_events, 1)
            light_eff = self.total_matched_light/max(self.total_light_events, 1)
            print(f'Total charge trigger matching: {self.total_matched_triggers}/{self.total_charge_triggers} ({trigger_eff:0.04f})')
            print(f'Total charge event matching: {self.total_matched_events}/{self.total_charge_events} ({event_eff:0.04f})')
            print(f'Total light event matching: {self.total_matched_light}/{self.total_light_events} ({light_eff:0.04f})') 

    def fast_fourier(self, adc):
        # output is an array shape (64, 1000) for each adc
        spectra_array = []
        adc_matrix = self.light_event_dset_name[:,adc,:,:]
        #valid_wvfm = self.light_event_mask[:,adc,:]

        #channel_mask = valid_wvfm[0,:]
        #t_valid_wvfm = np.transpose(valid_wvfm, (1,0))[channel_mask==1]
        t_adc_matrix = np.transpose(adc_matrix, (1, 0, 2))#[channel_mask==1]

        for i in range(64):
            #valid_chan_wvfm = t_adc_matrix[i][t_valid_wvfm[i]==1]/self.BIT
            valid_chan_wvfm = t_adc_matrix[i]/self.BIT
            
            # choose first 45 samples as signal-free
            # calculate mean and standard deviation to define signal vs. no sigal
            stdev = np.std(valid_chan_wvfm[:,0:45], axis=-1)
            mean = np.mean(valid_chan_wvfm[:,0:45],axis=-1)
            thd = (stdev*3.5) + mean
            no_signal_mask = (valid_chan_wvfm.max(axis=1) < thd)

            # remove the pedestal and select "signal-free" waveforms
            valid_chan_nped = (valid_chan_wvfm.astype(float) - (valid_chan_wvfm[:,0:45]).mean(axis=-1, keepdims=True))
            noise_wvfms = valid_chan_nped[no_signal_mask==1][:,:self.SAMPLES]
            try:
                spectrum = np.fft.fft(noise_wvfms)
                normalized_spectrum = np.abs(spectrum[:,:self.SAMPLES//2) / (1e3*self.SAMPLE_RATE)
                # remove the DC component
                normalized_spectrum[:,0] = 0
                # calculate an average fft
                spectrum_average = (np.sum(normalized_spectrum, axis=0))/len(normalized_spectrum)
                spectra_array.append(np.nan_to_num(spec_sum, nan=0))
            except:
                pass
        return np.array(spectra_array)

    def flow2sim(self, adc_lcm, adc_acl):
        f2s_idx = [63,62,61,60,59,58,\
                    57,56,55,54,53,52,\
                    47,46,45,44,43,42,\
                    41,40,39,38,37,36,\
                    31,30,29,28,27,26,\
                    25,24,23,22,21,20,\
                    15,14,13,12,11,10,\
                    9,8,7,6,5,4]
        mod_array = np.concatenate((adc_lcm[f2s_idx[0:6]],adc_acl[f2s_idx[0:6]],\
                                    adc_lcm[f2s_idx[6:12]],adc_acl[f2s_idx[6:12]],\
                                    adc_lcm[f2s_idx[12:18]],adc_acl[f2s_idx[12:18]],\
                                    adc_lcm[f2s_idx[18:24]],adc_acl[f2s_idx[18:24]],\
                                    adc_lcm[f2s_idx[24:30]],adc_acl[f2s_idx[24:30]],\
                                    adc_lcm[f2s_idx[30:36]],adc_acl[f2s_idx[30:36]],\
                                    adc_lcm[f2s_idx[36:42]],adc_acl[f2s_idx[36:42]],\
                                    adc_lcm[f2s_idx[42:48]],adc_acl[f2s_idx[42:48]]))
        
    def run(self, source_name, source_slice, cache):
        super(LightNoiseExtraction, self).run(source_name, source_slice, cache)

        adc_list = np.arange(cache[self.nadc])

    

    def run(self, source_name, source_slice, cache):
        super(LightNoiseExtraction, self).run(source_name, source_slice, cache)

        event_data = cache[self.events_dset_name]
        ext_trigs_data = cache[self.ext_trigs_dset_name]
        ext_trigs_idcs = cache[self.ext_trigs_dset_name + '_idcs']
        ext_trigs_mask = ~rfn.structured_to_unstructured(ext_trigs_data.mask).any(axis=-1)

        nevents = len(event_data)
        ev_id = np.arange(source_slice.start, source_slice.stop, dtype=int)
        ext_trig_ref = np.empty((0, 2), dtype=int)
        ev_ref = np.empty((0, 2), dtype=int)        

        # check match on external triggers
        if nevents:
            ext_trigs_mask = ~rfn.structured_to_unstructured(ext_trigs_data.mask).any(axis=-1)
            if np.any(ext_trigs_mask):
                ext_trigs_all = ext_trigs_data.data[ext_trigs_mask]
                ext_trigs_idcs = ext_trigs_idcs.data[ext_trigs_mask]
                ext_trigs_unix_ts = np.broadcast_to(event_data['unix_ts'].reshape(-1, 1), ext_trigs_data.shape)[ext_trigs_mask]
                ext_trigs_ts = ext_trigs_all['ts']
                idcs = self.match_on_timestamp(ext_trigs_unix_ts, ext_trigs_ts)

                if len(idcs):
                    ext_trig_ref = np.append(ext_trig_ref, np.c_[ext_trigs_idcs[idcs[:, 0]], idcs[:, 1]], axis=0)
                    ev_id_bcast = np.broadcast_to(ev_id[:,np.newaxis], ext_trigs_mask.shape)
                    ev_ref = np.unique(np.append(ev_ref, np.c_[ev_id_bcast[ext_trigs_mask][idcs[:, 0]], idcs[:, 1]], axis=0), axis=0)

                logging.info(f'found charge/light match on {len(ext_trig_ref)}/{ext_trigs_mask.sum()} triggers')
                logging.info(f'found charge/light match on {len(ev_ref)}/{len(event_data)} events')
                self.total_charge_triggers += ext_trigs_mask.sum()
                self.total_matched_triggers += len(np.unique(ext_trig_ref[:,0]))
                self.total_matched_events += len(np.unique(ev_ref[:,0]))
                self.matched_light[np.unique(ext_trig_ref[:,1])] = True

        # write references
        # ext trig -> light event
        self.data_manager.write_ref(self.ext_trigs_dset_name, self.light_event_dset_name, ext_trig_ref)

        # charge event -> light event
        self.data_manager.write_ref(self.events_dset_name, self.light_event_dset_name, ev_ref)

        self.total_charge_events += len(event_data)
