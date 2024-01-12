################################################################################
##                                                                            ##
##    CONTAINS: Script to plot contents in output file from proton selection  ##
##              being run over Bern Module Data.                              ##
##                                                                            ##
################################################################################

import h5py, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import file_parsing
from plot_hit_level_metrics import plot_event_hit_summ_metrics, plot_channel_metrics 

def main(file_dir, is_sim, hits_dset):

    is_sim = bool(is_sim == 'True')
    # initialize plotting datasets
    event_hit_summ_dict = dict()
    channel_metric_dict = dict()
    print("Is MC?:", is_sim)
    if is_sim:
        sample_type = 'MC'
    else:
        sample_type = 'data'

    count = 0

    for file in glob.glob(file_dir+'/*.h5'): # Loop over files files

        if count > 10: break
        count+=1
        f = h5py.File(file,'r')

        # Prepare datasets for plotting
        events = f['charge/events/data']
        tracks = f['combined/tracklets/data']
        tracks_ref = f['charge/events/ref/combined/tracklets/ref']
        tracks_region = f['charge/events/ref/combined/tracklets/ref_region']
        hits_trk_ref = f['combined/tracklets/ref/charge/'+hits_dset+'/ref']
        hits_trk_region = f['combined/tracklets/ref/charge/'+hits_dset+'/ref_region']
        hits_drift = f['combined/hit_drift/data']
        hits = f['charge/'+hits_dset+'/data']
        hits_ref = f['charge/events/ref/charge/'+hits_dset+'/ref']
        hits_region = f['charge/events/ref/charge/'+hits_dset+'/ref_region']
        if not is_sim:
            charge_hits = hits#f['combined/q_calib_el/data']
            charge_hits_ref = hits_ref#f['charge/events/ref/combined/q_calib_el/ref']
            charge_hits_region = hits_region#f['charge/events/ref/combined/q_calib_el/ref_region']
        else:
            charge_hits = hits
            charge_hits_ref = hits_ref
            charge_hits_region = hits_region
        ext_trigs = f['charge/ext_trigs/data']
        ext_trigs_ref = f['charge/events/ref/charge/ext_trigs/ref']
        ext_trigs_region = f['charge/events/ref/charge/ext_trigs/ref_region']
        print("Available datasets:",f.keys(),'\n')
        sel_reco = f['high_purity_sel']['hips']['sel_reco']['data']
        if is_sim:
            sel_truth = f['high_purity_sel']['hips']['sel_truth']['data']
            mc_truth_events = f['mc_truth/events/data']
        
        print("File:", file)
        sel_mask = (sel_reco['sel'] == True)
        sel_event_ids = sel_reco[sel_mask]['event_id']
        print("Selected Event Ids:", sel_event_ids)
        if is_sim==True:
            sel_truth_mask = (sel_truth['sel'] == True)
            sel_truth_protons = sel_truth[sel_mask]['hips']
            sel_truth_sel = sel_truth[sel_truth_mask]['event_id']
            sel_pdg_mask = (sel_truth[sel_truth_mask]['pdg_id'] != 0)
            sel_truth_pdg = sel_truth[sel_truth_mask]['pdg_id'][sel_pdg_mask]
            print("Selected Proton?:", sel_truth_protons)
            print("Selected True?:", sel_truth_sel)
            print("Selected PDG IDs:", sel_truth_pdg)
            for event in sel_event_ids:
                event_sel_mask = f['high_purity_sel']['hips']['sel_truth']['data']['event_id'] == event
                zero_mask = f['high_purity_sel']['hips']['sel_truth']['data'][event_sel_mask]['pdg_id'] != 0.
                print('Selected event true PID:', f['high_purity_sel']['hips']['sel_truth']['data'][event_sel_mask]['pdg_id'][zero_mask], "| Event ID:", event)

        ### partition file by selected events
        sel_event_mask = np.isin(events['id'], sel_event_ids)
        #print("Events:", events[sel_event_mask])
        for event_id in sel_event_ids:

            # Get hit information related to given event_id
            charge_hit_ref = charge_hits_ref[charge_hits_region[int(event_id),'start']:charge_hits_region[int(event_id),'stop']]
            charge_hit_ref = np.sort(charge_hit_ref[charge_hit_ref[:,0] == event_id, 1])
            
            # Event-level hit metrics
            charge_hits_data = charge_hits[charge_hit_ref]['Q']
            ts_hits_data = charge_hits[charge_hit_ref]['ts_pps']
            num_charge_hits = len(charge_hits_data)

            # Channel-level hit metrics
            iogroup_hits = charge_hits[charge_hit_ref]['io_group']
            iochannel_hits = charge_hits[charge_hit_ref]['io_channel']
            chipid_hits = charge_hits[charge_hit_ref]['chip_id']
            channelid_hits = charge_hits[charge_hit_ref]['channel_id']

            channel_id = np.array([int(str(iogroup_hits[i])+str(iochannel_hits[i])+str(chipid_hits[i])+str(channelid_hits[i])) for i in range(num_charge_hits)])
            unique_channels, unique_channel_hit_counts = np.unique(channel_id, return_counts=True)
            num_channels = len(unique_channels)

            #print("String of channels:", channel_id)
            #print("Number of unique channels:", num_channels)
            #print("Hits per channel:", unique_channel_hit_counts)
            #print("Length of hits per channel:", len(unique_channel_hit_counts))
            for i in range(num_channels):

                channel = unique_channels[i]
                hits_per_channel = unique_channel_hit_counts[i]
                channel_mask = np.argwhere(channel_id == channel).flatten()
                channel_hit_amps = charge_hits_data[channel_mask]
                channel_hit_ts = ts_hits_data[channel_mask] / 10. # convert to us
                
                max_hit_amp = max(channel_hit_amps)
                min_hit_amp = min(channel_hit_amps)

                first_hit_idx = np.argmin(channel_hit_ts)
                last_hit_idx = np.argmax(channel_hit_ts)
                first_hit_amp = channel_hit_amps[first_hit_idx]
                last_hit_amp = channel_hit_amps[last_hit_idx]
                first_last_hit_delta_t = abs(channel_hit_ts[last_hit_idx] - channel_hit_ts[first_hit_idx])

                #print("Channel hit amplitudes:", channel_hit_amps)
                #print("Channel hit timestamps:", channel_hit_ts)
                #print("Maximum hit amplitude:", max_hit_amp)
                #print("Minimum hit amplitude:", min_hit_amp)
                #print("First hit amplitude:", first_hit_amp)
                #print("Last hit amplitude:", last_hit_amp)
                #print("First/Last hit delta t:", first_last_hit_delta_t)



                channel_metric_dict[(file, event_id, channel)]=dict(
                    hit_mult = int(hits_per_channel), 
                    max_hit_amp = float(max_hit_amp),
                    min_hit_amp = float(min_hit_amp),
                    first_hit_amp = float(first_hit_amp),
                    last_hit_amp = float(last_hit_amp),
                    first_last_hit_delta_t = float(first_last_hit_delta_t)
                )

            event_hit_summ_dict[(file, event_id)]=dict(
                total_charge=float(sum(charge_hits_data)),
                num_hits=int(num_charge_hits),
                num_channels=int(num_channels)
            )
        
    ## Save all Python dictionaries to JSON files
    file_parsing.save_dict_to_json(event_hit_summ_dict, sample_type+"event_hit_summ_dict", True)
    file_parsing.save_dict_to_json(channel_metric_dict, sample_type+"channel_metric_dict", True)


    # PLOT: Signal Event Info      
    plot_event_hit_summ_metrics(event_hit_summ_dict, is_sim==True)
    plot_channel_metrics(channel_metric_dict, is_sim==True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--file_dir', default=None, required=True, type=str, \
                        help='''string corresponding to the path of the directory containing processed files for plotting''')
    parser.add_argument('-mc', '--is_sim', default=False, required=True, type=str, \
                        help='''str corresponding to bool whether files are simulation (MC) or data''')
    parser.add_argument('-hd', '--hits_dset', default='calib_final_hits', required=True, type=str,\
                        help='''str corresponding to bool of hits dataset name''')
    args = parser.parse_args()
    main(**vars(args))