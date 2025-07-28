import pandas as pd
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import os
import h5py
import glob
import numpy as np
import yaml
import argparse
from scipy.ndimage import uniform_filter1d

def thd_correct(array):

    # Define start and end indices
    indices = np.arange(0, 25) * 25  # (39,)
    start_indices, end_indices = indices[:-1], indices[1:]  # (39,)

    segment_range = np.arange(25)  # Shape: (25,)
    index_array = start_indices[:, None] + segment_range  # Shape: (39, 25)

    # Extract data using advanced indexing
    sliced_data = array[..., index_array]
    
    ranges = np.ptp(sliced_data, axis=-1)  # Compute range (n, 8, 64, 39)
    means = np.mean(sliced_data, axis=-1)  # Compute mean (n, 8, 64, 39)

    # Find ordering based on the smallest range
    smallest_ordering = np.argsort(ranges, axis=-1)  # Shape (n, 8, 64, 39)

    # Sort means using the ordering
    sorted_means = np.take_along_axis(means, smallest_ordering, axis=-1)  # Shape (n, 8, 64, 39)
    sorted_range = np.take_along_axis(ranges, smallest_ordering, axis=-1)
    # Compute average of 2nd, 3rd, and 4th smallest means
    average_mean = np.mean(sorted_means[..., 1:4], axis=-1)  # Shape (n, 8, 64)
    expanded_mean = average_mean[..., None] 
    broadcasted_mean = np.tile(expanded_mean, (1, 600))  
    filtered_wvfm = array - broadcasted_mean

    return filtered_wvfm

def pedestal(array):

    # Define start and end indices
    indices = np.arange(0, 13) * 50  # (39,)
    start_indices, end_indices = indices[:-1], indices[1:]  # (39,)

    segment_range = np.arange(50)  # Shape: (25,)
    index_array = start_indices[:, None] + segment_range  # Shape: (39, 25)

    # Extract data using advanced indexing
    sliced_data = array[..., index_array]
    
    ranges = np.ptp(sliced_data, axis=-1)  # Compute range (n, 8, 64, 39)

    # Find ordering based on the smallest range
    smallest_region = np.argmin(ranges, axis=-1) * 50 # Shape (n, 8, 64, 39)
    pedestal_region = array[smallest_region:smallest_region+50]

    return pedestal_region

def kill_weirdos(array):
    good_mask = (array[:,:,:,-1] > (array[:,:,:,0] - 500))
    #broadcasted_mask = np.tile(good_mask, (1, 1000)) 
    filtered_array = array * good_mask[:, :, :, np.newaxis]

    return filtered_array

def peak_finder(wvfm,
                    n_noise_factor,
                    n_bins_rolled,
                    n_sqrt_rt_factor,
                    pe_weight,
                    use_rising_edge=True):
        # height = flat threshold over noise (n*sigma)
        height = n_noise_factor[..., np.newaxis, np.newaxis] * np.ones(wvfm.shape[-1]) #* noise[..., np.newaxis] * np.ones(wvfm.shape[-1])
        # dynamic_threshold = rolling threshold of previous 5 bins + n*sqrt(rolling threshold)
        wvfm_rolled = np.roll(wvfm, n_bins_rolled)
        rolling_average = uniform_filter1d(wvfm_rolled, size=n_bins_rolled)
        sqrt_rolling_average = np.sqrt(np.abs(rolling_average) * pe_weight**2)
        sqrt_rolling_average[sqrt_rolling_average == 0] = 1
        dynamic_threshold = rolling_average + n_sqrt_rt_factor*sqrt_rolling_average
        # find bins over dynamic threshold and noise floor
        bins_over_dynamic_threshold = (wvfm > dynamic_threshold) & (wvfm > height)
        # Find first bins over threshold (rising edge)
        first_bins_over = bins_over_dynamic_threshold.copy()
        first_bins_over[..., 1:] &= ~bins_over_dynamic_threshold[..., :-1]
        if use_rising_edge:
            return first_bins_over
        
def tag_dark_counts(N_min, N_max):

     file_num = 0
     dark_count_wvfm = np.zeros((6000,4,64,200), dtype=np.int16)
     count = np.zeros((4,64))
     sipm_channels = ([0,1,2,3,4,5,6,7,8,9] + \
                 [10,11,12,13,14,15,16,17,18,19] + \
                 [20,21,22,23,24,25,26,27,28,29] + \
                 [32,33,34,35,36,37,38,39,40,41] + \
                 [42,43,44,45,46,47,48,49,50,51] + \
                 [52,53,54,55,56,57,58,59,60,61])
     print('aaa')
     Num_list = np.arange(N_min,N_max,1)
     #file_list = glob.glob("/global/cfs/cdirs/dune/www/data/2x2/nearline/flowed_light/data_bin004/*.FLOW.hdf5")
     #file_list = glob.glob("/global/cfs/cdirs/dune/users/ajwhite/2x2_Data/2x2_Filtered/LRS_FLOW/mpd_run_hvramp_rctl_105_p1*")
     #for file in file_list:
     for Nf in Num_list:
          #file = f'/global/cfs/cdirs/dune/www/data/FSD/nearline/flowed_light/calib_runs/mpd_run_calib_rctl_{Nf}.FLOW.hdf5'
          file = f'/global/cfs/cdirs/dune/www/data/FSD/nearline/flowed_light/data_bin_04/mpd_run_data_rctl_165_p{Nf}.FLOW.hdf5'
          if file_num < len(Num_list): #508:
               print('file number:', file_num)
               file_num += 1
               if not os.path.isfile(file):
                    continue
               else:
                    with h5py.File(file, 'r') as h5:
                         light_wvfms = h5['light/wvfm/data']['samples']
                         #print('a')
                         offbeam_wvfm_v2 =  thd_correct(light_wvfms)
                         del light_wvfms
                         #print('b')
                         #offbeam_wvfm_v2 = kill_weirdos(offbeam_wvfm_v1)
                         #print('c')
                         #del offbeam_wvfm_v1
                         #del offbeam_wvfm_v1
                         offbeam_wvfm_v3 =  offbeam_wvfm_v2[:, :, sipm_channels, :60] #* gain_array[:, :, np.newaxis]
                         ped_wvfm_v3 =  offbeam_wvfm_v2[:, :, sipm_channels, -50:]
                         #print('d')
                         del offbeam_wvfm_v2

                         #noise_thresholds = np.load('/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/15Files_noise_thd.npz')['data']
                         n_noise_factor=np.array([200, 200, 200, 200])
                         first_bins = peak_finder(wvfm=offbeam_wvfm_v3, n_noise_factor=n_noise_factor, n_bins_rolled=1, n_sqrt_rt_factor=0, pe_weight=0, use_rising_edge=True)
                         #print('Shape of first_bins:', np.shape(first_bins))
                         #print(first_bins)
                         #del noise_thresholds

                         #print(np.shape(np.sum(np.sum(np.sum(first_bins, axis=-1), axis=-1), axis=-1)))
                         #print(np.sum(np.sum(np.sum(first_bins, axis=-1), axis=-1), axis=-1))
                         #per_event_hits = np.sum(np.sum(np.sum(first_bins, axis=-1), axis=-1), axis=-1)
                         #print(np.sum(per_event_hits > 0))

                         for adc in range(4):
                              for channel in range(60):
                                   adc_channel = sipm_channels[channel]
                                   tester=0
                                   for event in range(np.shape(offbeam_wvfm_v3)[0]):
                                        hit_idx = np.where(first_bins[event, adc, channel]==1)[0]
                                        if len(hit_idx) > 0:
                                             cut_1 = (count[adc, adc_channel] < 5999)
                                             cut_2 = (hit_idx[0] >= 3 )
                                             cut_3 = (hit_idx[0] <= 33)
                                             combo_cut = cut_2*cut_3*cut_1
                                             if combo_cut==1:
                                                  dark_count_form = offbeam_wvfm_v3[event, adc, channel, hit_idx[0]-3:hit_idx[0]+27]
                                                  #dark_count_int = (dark_count_form / gain_array[adc, channel]).astype(np.int16)
                                                  cut_4 = (dark_count_form[-1] < 150)
                                                  cut_5 = (np.min(dark_count_form) > -150)
                                                  cut_6 = (np.max(ped_wvfm_v3[event, adc, channel,:]) < np.max(dark_count_form))
                                                  combo_cut_2 = cut_4*cut_5*cut_6
                                                  if combo_cut_2==1:                                                  
                                                       dark_count_int = (dark_count_form).astype(np.int16)
                                                       pedestal_int = (ped_wvfm_v3[event, adc, channel, :]).astype(np.int16)
                                                       del dark_count_form
                                                       #try:
                                                       next_wvfm_idx = np.where(dark_count_wvfm[:,adc,adc_channel, 63] == 0)[0][0]
                                                       dark_count_wvfm[next_wvfm_idx,adc,adc_channel, 60:90] += dark_count_int
                                                       dark_count_wvfm[next_wvfm_idx,adc,adc_channel, 0:50] += pedestal_int
                                                       del next_wvfm_idx
                                                       del dark_count_int
                                                       del pedestal_int
                                                       count[adc, adc_channel] += 1
                                                  else: 
                                                       del dark_count_form
                                                  #except: 
                                             if (cut_1+tester) == 0:
                                                  print(f'ADC {adc}, Channel {channel}, Event {event}')
                                                  tester += 1
                                        #else:
                                        #     if count[adc, adc_channel] < 500:
                                        #          dark_count_form = offbeam_wvfm_v4[event, adc, channel, 0:25]
                                        #          dark_count_int = (dark_count_form / gain_array[adc, channel]).astype(np.int16)
                                        #          del dark_count_form
                                        #          next_wvfm_idx = np.where(dark_count_wvfm[:,adc,adc_channel, 63] == 0)[0][0]
                                        #          dark_count_wvfm[next_wvfm_idx,adc,adc_channel, 60:85] += dark_count_int
                                        #          del dark_count_int
                                        #          del next_wvfm_idx
                                        #          count[adc, adc_channel] += 1
                                        del hit_idx
                                   del adc_channel
                                   del tester
                         del offbeam_wvfm_v3
                         del ped_wvfm_v3
                         del first_bins   
     print(count)
     del count

     return dark_count_wvfm

def main(output_file):
    
    new_wvfms = tag_dark_counts(N_min=0, N_max=300)
    np.savez(output_file, data=new_wvfms)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_file', default=None, required=True, type=str, \
                        help='''string corresponding to desired output file path and name''')
    args = parser.parse_args()
    main(**vars(args))