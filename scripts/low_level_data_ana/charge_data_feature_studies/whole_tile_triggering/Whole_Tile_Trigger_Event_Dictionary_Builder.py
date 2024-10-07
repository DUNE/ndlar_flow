#!/usr/bin/env python
# NOTE: Change name of output dictionary in line 437 to avoid overwriting previous dictionaries

# This script is used to identify potential full tile fire events in 2x2 data
# It uses flow-level files to identify events with many hits on a single tile and
# saves information about those events to a JSON dictionary. Previous versions of this
# script also saved plots related to time distributions of hits on tiles and saved event
# displays for full tile fire events. These features have been commented out for now.
# For questions, please contact Elise Hinkle (ehinkle@uchicago.edu)

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import h5py
import argparse
from datetime import datetime
import glob
import json
import cmasher as cmr
import math
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os
#sys.path.append('/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/2x2_sim/run-ndlar-flow/ndlar_flow/event_display/LAr_evd/')
#from lar2x2_evd import *

from matplotlib.axes import Axes

packet_types = {
    0: "Data",
    1: "Test",
    2: "Write",
    3: "Read",
    4: "Timestamp",
    5: "Message",
    6: "Sync",
    7: "Trigger"
}

# Helper function to go from IO Channel to Tile ID
def io_channel_to_tile(io_channel):
    return int(np.floor((io_channel-1-((io_channel-1)%4))/4+1))

# Rasterize plots 
_old_axes_init = Axes.__init__
def _new_axes_init(self, *a, **kw):
    _old_axes_init(self, *a, **kw)
    # https://matplotlib.org/stable/gallery/misc/zorder_demo.html
    # 3 => leave text and legends vectorized
    self.set_rasterization_zorder(3)
def rasterize_plots():
    Axes.__init__ = _new_axes_init
def vectorize_plots():
    Axes.__init__ = _old_axes_init
rasterize_plots()

# Helper functions to save dictionary to JSON file
def tuple_key_to_string(d):
    out={}
    for key in d.keys():
        string_key=""
        max_length=len(key)
        for i in range(max_length):
            if i<len(key)-1: string_key+=str(key[i])+"-"
            else: string_key+=str(key[i])
        out[string_key]=d[key]
    return out  

def save_dict_to_json(d, name, if_tuple):
    with open(name+".json", "w") as outfile:
        if if_tuple==True:
            updated_d = tuple_key_to_string(d)
            json.dump(updated_d, outfile, indent=4)
        else:
            json.dump(d, outfile, indent=4)  

def main(datadir=None, simulation=False):

    full_tile_trigger_dict = dict() # Initialize dictionary

    h5_files = []

    if simulation: 
        h5_files = glob.glob('000*000/*.FLOW.hdf5', root_dir=datadir, recursive=True)
    else:
        h5_files_8 = glob.glob('july8_2024/nominal_hv/packet-0*.hdf5', root_dir=datadir, recursive=True)
        h5_files_10 = glob.glob('july10_2024/nominal_hv/packet-0*.hdf5', root_dir=datadir, recursive=True)
        #h5_files_12 = glob.glob('july12_2024/lrs_low_threshold/packet-0*.hdf5', root_dir=datadir, recursive=True)
        h5_files.extend(h5_files_8)
        h5_files.extend(h5_files_10)
        #h5_files.extend(h5_files_12)
        #h5_files_self_trigger = glob.glob('july[8,10,12]_2024/**/packet-self*.hdf5', root_dir=datadir, recursive=True)
        #h5_files.extend(h5_files_self_trigger)

    print("Number of H5 Files:", len(h5_files))
    #print(h5_files)

    total_events = 0
    total_beam_events = 0
    selected_events = 0
    full_tile_spike_events = 0
    selected_beam_events = 0 
    selected_full_tile_spike_beam_events = 0

    
    for i in range(len(h5_files)):

        #if i>50: continue   
        if i%10==0: print("File number:", i)
        print("Opening file:", h5_files[i])
        
        file = datadir+h5_files[i]
        directory = os.path.dirname(file)
        filename = file.split("/")[-1]
        subdirs = directory.split(datadir)[-1]

        output_pdf_name = 'plots/'+subdirs+'/'+filename.split(".FLOW.hdf5")[0]+'_full_tile_fire.pdf'
        os.makedirs(os.path.dirname(output_pdf_name), exist_ok=True)

        # put file in this directory for now
        with PdfPages(output_pdf_name, keep_empty=False) as output:

            f = h5py.File(file,'r')
            plt.rcParams["figure.figsize"] = (10,8)

            # Set thresholds for what is considered a full tile fire event
            min_triggers_per_event = 2000
            min_triggers_per_io_group = 1000
            min_triggers_per_tile = 1000
            min_chips_per_tile_with_triggers = 95
            max_ratio_tile_hist_peak_to_adjacent_bins = 0.5
            bin_size_us = 10

            # The above settings catch events which do not feature a "true" full tile fire event
            # The following setting is used for now to (hopefully) better identify bona fide full tile fire events
            # However, we still keep events not meeting this min_difference_between_peak_and_adjacent_bins threshold
            # in case our definition using this tag is too strict
            min_difference_between_peak_and_adjacent_bins = 500 # in Hz --> 500 hits in 10 us 

            # Make sure you can open the file
            # If you can open the file, get events and hit information
            try:
                events = f['charge/events/data']
                #print("Events: ", len(events))
                hits_dset = 'calib_prompt_hits'
                hits_full = f['charge/'+hits_dset+'/data']
            except:
                print("Not including file ", file)
                continue
            hits_ref = f['charge/events/ref/charge/'+hits_dset+'/ref']
            hits_region = f['charge/events/ref/charge/'+hits_dset+'/ref_region']
            packets_full = f['charge/packets/data']
            packets_ref = f['charge/'+hits_dset+'/ref/charge/packets/ref']
            packets_region = f['charge/'+hits_dset+'/ref/charge/packets/ref_region']
            full_tile_events = []
            total_events += len(events)

            # Identify beam trigger events for data:
            if not simulation:
                try:
                    # Load external triggers dataset
                    exttrigs_full = f['charge/ext_trigs/data']
                    exttrigs_ref = f['charge/events/ref/charge/ext_trigs/ref']
                    exttrigs_region = f['charge/events/ref/charge/ext_trigs/ref_region']
                    exttrigs_beam = np.where(exttrigs_full['iogroup'] == 5)
                    beam_events_ref = np.sort(exttrigs_ref[:,0][exttrigs_beam])
                    beam_events = events[beam_events_ref]
                    events = beam_events
                except:
                    print("No beam trigger events found in file ", file)
                    beam_events = []
            total_beam_events += len(beam_events)
            # Filter out events with fewer than the minimum number of hits
            events = events[events['nhit'] > min_triggers_per_event] # HITS PER EVENT THRESHOLD SET
            beam_events = beam_events[beam_events['nhit'] > min_triggers_per_event] # HITS PER EVENT THRESHOLD SET IN BEAM EVENTS AS WELL

            # Loop over events in file which have met the threshold for minimum number of hits per event
            for ev_id in events['id']:
                event_mask = events['id'] == ev_id
                event = events[event_mask]
                #print(event['unix_ts'])
                event_datetime = datetime.utcfromtimestamp(event['unix_ts'][0]).strftime('%Y-%m-%d %H:%M:%S')
                #print(event_datetime)
                #.strftime('%Y-%m-%d %H:%M:%S')
                # Load hits and packets for event
                hit_ref = hits_ref[hits_region[ev_id,'start']:hits_region[ev_id,'stop']]
                hit_ref = np.sort(hit_ref[hit_ref[:,0] == ev_id, 1])
                hits = hits_full[hit_ref]
                packet_ref = packets_ref[hit_ref]
                packet_ref = np.sort(packet_ref[:, 1])
                packets = packets_full[packet_ref]
                if len(hits) != len(packets):
                    print("Number of hits and packets do not match")
                    print("Number of hits: ", len(hits))
                    print("Number of packets: ", len(packets))
                    continue

                # Check if event is a beam event
                if simulation:
                    is_beam_event = True
                if not simulation: #and file not in h5_files_12:
                    is_beam_event = True
                    #if event in beam_events:
                    #    is_beam_event = True
                    #else:
                    #    is_beam_event = False
                else:
                    is_beam_event = False

                # Count number of packets per IO Group (TPC)
                io_groups, iog_counts = np.unique(packets['io_group'], return_counts=True)
                iog_count_over_threshold = iog_counts > min_triggers_per_io_group # IO GROUP PACKET THRESHOLD SET

                # Initialize lists to store information about full tile fire events
                event_iogs = []
                event_tiles = []
                event_full_tile_spike = []
                io_group_tile_for_event = []
                chips_per_event = []
                fraction_of_negative_charge_hits_on_tile = []
                dt_5th_to_95th_percentile_of_packets_on_tile = []
                energy_per_iog_positive = []
                energy_per_iog_negative = []
                energy_per_iog_total = []
                charge_per_iog_positive = []
                charge_per_iog_negative = []
                charge_per_iog_total = []


                # Loop over IO Groups with more than the threshold number of packets
                for iog in io_groups[iog_count_over_threshold]:
                    iog_mask = packets['io_group'] == iog
                    single_iog_packets = packets[iog_mask]
                    # Convert IO Channel to Tile ID for network agnostic identifier
                    single_iog_packets_tile_id = [io_channel_to_tile(io_channel) for io_channel in single_iog_packets['io_channel']]
                    tile_ids, tile_counts = np.unique(single_iog_packets_tile_id, return_counts=True)
                    tile_count_over_threshold = tile_counts > min_triggers_per_tile # TILE PACKET THRESHOLD SET
                    num_tiles_over_threshold = len(tile_ids[tile_count_over_threshold])

                    # Loop over tiles with more than the threshold number of packets
                    for tile_idx in range(num_tiles_over_threshold):
                        tile = tile_ids[tile_count_over_threshold][tile_idx]
                        single_tile_packets_mask = np.where(single_iog_packets_tile_id == tile)
                        single_tile_packets = single_iog_packets[single_tile_packets_mask]
                        chip_ids= np.unique(single_tile_packets['chip_id'])

                        # Skip tile if it does not have enough chips with triggers -- CHIP PACKET THRESHOLD SET
                        if len(chip_ids) < min_chips_per_tile_with_triggers:
                            continue

                        else:
                            # These lines help to identify the full tile events by checking the time difference
                            # between the 5th and 95th percentile of packet timestamps for this tile
                            single_tile_timestamps = (single_tile_packets['timestamp'] - min(single_tile_packets['timestamp']))/10
                            dt_5th_to_95th_percentile_of_packets_on_tile.append(np.percentile(single_tile_timestamps, 95) - np.percentile(single_tile_timestamps, 5))

                            # Create a histogram of packet times (relative to first packet on tile) to study packet frequency on tile
                            max_timestamp = math.ceil(max(single_tile_timestamps)/100)*100
                            tile_hist, bin_edges = np.histogram(single_tile_timestamps, bins=np.linspace(0, max_timestamp, int(max_timestamp/bin_size_us+1)))
                            tile_hist_bin_ratios_high = (tile_hist[1:]+1)/(tile_hist[:-1]+1) # deal w/ divide by zero
                            tile_hist_bin_ratios_low = (tile_hist[:-1]+1)/(tile_hist[1:]+1)  # deal w/ divide by zero
                            tile_hist_bin_diff = np.diff(tile_hist)
                            # Max tile hist is a number of hits in a 5 us window. We convert the threshold frequency to a number of hits in a 5 us window 
                            # for comparison. TO DO: Make this less clunky code-wise (changing settings requires change here and where min_difference_between_peak_and_adjacent_bins is defined)
                            #if max(tile_hist) < min_difference_between_peak_and_adjacent_bins*5e-6:
                            if (min(tile_hist_bin_ratios_low) < max_ratio_tile_hist_peak_to_adjacent_bins or min(tile_hist_bin_ratios_high) < max_ratio_tile_hist_peak_to_adjacent_bins) and max(tile_hist_bin_diff) > min_difference_between_peak_and_adjacent_bins:
                                event_full_tile_spike.append(True)
                            else:
                                event_full_tile_spike.append(False)
                            # Check if this event has already been recorded as a full tile event before recording as full tile event
                            if (len(full_tile_events) == 0) or (len(full_tile_events)>0 and full_tile_events[-1] != ev_id):
                                ## Print Event Display
                                #evd = LArEventDisplay(filedir=directory+'/', filename=filename, nhits=min_triggers_per_event, ntrigs=0, show_light=False, show_colorbars=True)
                                #evd.display_event(ev_id)
                                #output.savefig(bbox_inches='tight')
                                #plt.close()
                                ## Get hit time distribution for full event
                                #fig = plt.figure(figsize=(10,10))
                                #fig.tight_layout()
                                #max_t_drift = math.ceil(max(hits['t_drift'])/1000)*100
                                #for iogr in io_groups:
                                #    plt.hist(hits[hits['io_group'] == iogr]['t_drift']/10, bins=np.linspace(0, max_t_drift, int(max_t_drift/5+1)), histtype='step', linewidth=1, label='IO Group '+str(iogr))
                                #plt.legend()
                                #plt.xlabel("Flow Calculated Drift Time [us]")
                                #plt.ylabel("Number of Hits / 5 us")
                                #plt.title("Event "+str(ev_id)+'-'+event_datetime)
                                #plt.yscale('log')
                                #output.savefig()
                                #plt.close()
                                selected_events += 1
                                if is_beam_event:
                                    selected_beam_events += 1
                                full_tile_events.append(ev_id)

                            # Get fraction of negative charge hits on tile
                            iog_mask_hits = hits['io_group'] == iog
                            single_iog_hits = hits[iog_mask_hits]
                            single_iog_hits_tile_id = [io_channel_to_tile(io_channel) for io_channel in single_iog_hits['io_channel']]
                            single_tile_hits_mask = np.where(single_iog_hits_tile_id == tile)
                            single_tile_hits = single_iog_hits[single_tile_hits_mask]
                            negative_charge_hit_fraction = len(single_tile_hits[single_tile_hits['Q'] < 0])/len(single_tile_hits)

                            # Sum hits charge, energy on IOG for positive and negative charge hits separately
                            negative_charge_iog_hits = single_iog_hits[single_iog_hits['Q'] <= 0]
                            positive_charge_iog_hits = single_iog_hits[single_iog_hits['Q'] > 0]
                            total_iog_positive_charge = sum(positive_charge_iog_hits['Q'])
                            total_iog_negative_charge = sum(negative_charge_iog_hits['Q'])
                            total_iog_positive_energy = sum(positive_charge_iog_hits['E'])
                            total_iog_negative_energy = sum(negative_charge_iog_hits['E'])


                            # Record full tile event
                            chips_per_event.append(len(chip_ids))
                            io_group_tile_for_event.append((iog, tile))
                            event_iogs.append(iog)
                            event_tiles.append(tile)
                            fraction_of_negative_charge_hits_on_tile.append(negative_charge_hit_fraction)
                            energy_per_iog_positive.append(total_iog_positive_energy)
                            energy_per_iog_negative.append(total_iog_negative_energy)
                            energy_per_iog_total.append(total_iog_positive_energy + total_iog_negative_energy)
                            charge_per_iog_positive.append(total_iog_positive_charge)
                            charge_per_iog_negative.append(total_iog_negative_charge)
                            charge_per_iog_total.append(total_iog_positive_charge + total_iog_negative_charge)
                    
                num_iogs = len(event_iogs)
                if num_iogs > 0:
                    #fig2 = plt.figure(figsize=(num_iogs*10,10))
                    #fig2.tight_layout()
                    #fig3 = plt.figure(figsize=(num_iogs*10,10))
                    #fig3.tight_layout()
                    #fig4 = plt.figure(figsize=(num_iogs*10,10))
                    #fig4.tight_layout()
                    #ax2=[]
                    #ax3=[]
                    #ax4=[]
                    #for j in range(num_iogs):
                    #    iog = event_iogs[j]
                    #    tile = event_tiles[j]
                    #    # Get packets for problem tiles
                    #    iog_mask_packets = packets['io_group'] == iog
                    #    single_iog_packets = packets[iog_mask_packets]
                    #    single_iog_packets_tile_id = [io_channel_to_tile(io_channel) for io_channel in single_iog_packets['io_channel']]
                    #    single_tile_packets_mask = np.where(single_iog_packets_tile_id == tile)
                    #    single_tile_packets = single_iog_packets[single_tile_packets_mask]
                    #    # Get hits for problem tiles
                    #    iog_mask_hits = hits['io_group'] == iog
                    #    single_iog_hits = hits[iog_mask_hits]
                    #    single_iog_hits_tile_id = [io_channel_to_tile(io_channel) for io_channel in single_iog_hits['io_channel']]
                    #    single_tile_hits_mask = np.where(single_iog_hits_tile_id == tile)
                    #    single_tile_hits = single_iog_hits[single_tile_hits_mask]
                    #    # Get packet time distribution for problem tiles
                    #    single_tile_timestamps = (single_tile_packets['timestamp'] - min(single_tile_packets['timestamp']))/10
                    #    max_timestamp = math.ceil(max(single_tile_timestamps)/100)*100
                    #    timestamp_bins = np.linspace(0, max_timestamp, int(max_timestamp/5+1))
                    #    dw_bins = np.linspace(0, 256, 65)
                    #    hit_charge_max = max(max(single_tile_hits['Q']), 1.)
                    #    hit_charge_min = min(single_tile_hits['Q'])
                    #    hit_charge_diff = hit_charge_max - hit_charge_min
                    #    hit_bins = np.linspace(hit_charge_min, hit_charge_max, math.ceil(hit_charge_diff/5)+1)
                    #    
                    #    ax2.append(fig2.add_subplot(1,num_iogs,j+1))
                    #    ax2[j].hist(single_tile_timestamps, bins=timestamp_bins, histtype='step', linewidth=1)
                    #    ax2[j].set_xlabel("Packet Timestamp Distribution (0 = First Hit on Tile) [us]")
                    #    ax2[j].set_ylabel("Number of Packets / 5 us")
                    #    ax2[j].set_title("Event "+str(ev_id)+'-'+event_datetime+'- IO Group '+str(iog)+'- Tile '+str(tile))
                    #
                    #    # Set color maps
                    #    if j==0:
                    #        print("Setting color maps")
                    #        values3, counts3 = np.unique(single_tile_packets['dataword'], return_counts=True)
                    #        norm3 = mpl.colors.Normalize(vmin=0,vmax=max(counts3)/2)
                    #        cmap3 = plt.colormaps.get_cmap('viridis')
                    #        mcharge3 = plt.cm.ScalarMappable(norm=norm3, cmap=cmap3)
                    #        values4, counts4 = np.unique(single_tile_hits['Q'], return_counts=True)
                    #        norm4 = mpl.colors.Normalize(vmin=0,vmax=max(counts4)/2)
                    #        cmap4 = plt.colormaps.get_cmap('magma')
                    #        mcharge4 = plt.cm.ScalarMappable(norm=norm4, cmap=cmap4)
                    #    # Look at time distribution vs. dataword
                    #    ax3.append(fig3.add_subplot(1,num_iogs,j+1))
                    #    ax3[j].hist2d(single_tile_timestamps, single_tile_packets['dataword'], bins=[timestamp_bins, dw_bins], cmap=cmap3, norm=norm3)
                    #    ax3[j].set_xlabel("Packet Timestamp Distribution (0 = First Hit on Tile) [us]")
                    #    ax3[j].set_ylabel("Dataword")
                    #    ax3[j].set_title("Event "+str(ev_id)+'-'+event_datetime+'- IO Group '+str(iog)+'- Tile '+str(tile))
                    #    # Look at time distribution vs. hit charge
                    #    ax4.append(fig4.add_subplot(1,num_iogs,j+1))
                    #    ax4[j].hist2d(single_tile_hits['t_drift']/10, single_tile_hits['Q'], bins=[timestamp_bins, hit_bins], cmap=cmap4, norm=norm4)
                    #    ax4[j].set_xlabel("Flow Calculated Drift Time [us]")
                    #    ax4[j].set_ylabel(r'Hit Charge [$\mathbf{10^3}$ e]')
                    #    ax4[j].set_title("Event "+str(ev_id)+'-'+event_datetime+'- IO Group '+str(iog)+'- Tile '+str(tile))
                    #
                    #fig3.subplots_adjust(right=0.88)
                    #cbar_ax3 = fig3.add_axes([0.92, 0.12, 0.02, 0.75])
                    #cbar3 = fig3.colorbar(mcharge3, cax=cbar_ax3, label='Number of Packets')
                    #cbar3.set_label(r'Number of Packets', size=16)
                    #cbar_ax3.tick_params(labelsize=14)
                    #fig4.subplots_adjust(right=0.88)
                    #cbar_ax4 = fig4.add_axes([0.92, 0.12, 0.02, 0.75])
                    #cbar4 = fig4.colorbar(mcharge4, cax=cbar_ax4, label='Number of Hits')
                    #cbar4.set_label(r'Number of Hits', size=16)
                    #cbar_ax4.tick_params(labelsize=14)
                    #output.savefig(fig2)
                    #output.savefig(fig3)
                    #output.savefig(fig4)
                    #plt.close()
                    if np.any(event_full_tile_spike):
                        full_tile_spike_events += 1
                        if is_beam_event:
                            selected_full_tile_spike_beam_events += 1

                    # Save information about full tile events to dictionary for additional analysis
                    full_tile_trigger_dict[(str(file), int(ev_id))]=dict(event_id = int(ev_id),
                                                                         filepath=str(file),
                                                                         timestamp=str(event_datetime), 
                                                                         is_beam = bool(is_beam_event),
                                                                         chips_triggered_on_tile=[int(k) for k in chips_per_event],
                                                                         io_group_and_tile_id=[(int(iog), int(tile)) for iog, tile in io_group_tile_for_event],
                                                                         io_group=[int(iog) for iog in event_iogs], 
                                                                         fraction_of_negative_charge_hits_on_tile=[float(round(k,4)) for k in fraction_of_negative_charge_hits_on_tile],
                                                                         iog_negative_charge=[float(round(k,4)) for k in charge_per_iog_negative],
                                                                         iog_positive_charge=[float(round(k,4)) for k in charge_per_iog_positive],
                                                                         iog_total_charge=[float(round(k,4)) for k in charge_per_iog_total],
                                                                         iog_negative_energy=[float(round(k,4)) for k in energy_per_iog_negative],
                                                                         iog_positive_energy=[float(round(k,4)) for k in energy_per_iog_positive],
                                                                         iog_total_energy=[float(round(k,4)) for k in energy_per_iog_total],                                                                         
                                                                         dt_5th_to_95th_percentile_of_packets_on_tile=[float(round(k,4)) for k in dt_5th_to_95th_percentile_of_packets_on_tile],
                                                                         event_full_tile_spike=[bool(k) for k in event_full_tile_spike], 
                                                                         min_triggers_per_event=int(min_triggers_per_event),
                                                                         min_triggers_per_io_group=int(min_triggers_per_io_group),
                                                                         min_triggers_per_tile=int(min_triggers_per_tile),
                                                                         min_chips_per_tile_with_triggers=int(min_chips_per_tile_with_triggers),
                                                                         min_difference_between_peak_and_adjacent_bins=float(min_difference_between_peak_and_adjacent_bins),
                                                                         max_ratio_tile_hist_peak_to_adjacent_bins=float(max_ratio_tile_hist_peak_to_adjacent_bins), 
                                                                         bin_size_us=int(bin_size_us))
                
            output.close()

    save_dict_to_json(full_tile_trigger_dict, "full_tile_trigger_dict_V5_beam_only_test", True) # Save dictionary to JSON file; currently needs to be updated to save to a different file name each time
    print("Total number of events:", total_events)
    print("Total number of beam events:", total_beam_events)
    print("Number of selected events:", selected_events, "(", selected_events/total_events*100, "%)")
    print("Number of selected beam events:", selected_beam_events, "(", selected_beam_events/total_beam_events*100, "%)")
    print("Number of full tile spike events:", full_tile_spike_events, "(", full_tile_spike_events/total_events*100, "%)")
    print("Number of selected full tile spike beam events:", selected_full_tile_spike_beam_events, "(", selected_full_tile_spike_beam_events/total_beam_events*100, "%)")

                


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--datadir', default=None,type=str,help='''string of data directory where data are stored.''')
    parser.add_argument('-mc','--simulation', default=False, help='''Flag to indicate if input files are simulation or data.''')
    args = parser.parse_args()
    main(**vars(args))