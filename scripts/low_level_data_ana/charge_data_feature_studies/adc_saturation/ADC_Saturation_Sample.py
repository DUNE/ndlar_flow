#!/usr/bin/env python
# NOTE: Change lines 352-356 to change the name of the output JSON dictionary file. 
# ADDITIONAL NOTE: Change the noisy_channels array to include the noisy channels for the specific run you are analyzing.
#                 This may need to be modified in the future if networks are changed midrun (i.e. change array to use tile_id vs. iochannel to be network agnostic)

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

noisy_channels = np.array([[ 2, 29 ,31 , 0], 
                  [ 2 ,29 ,31 , 1],
                  [ 2 ,29, 31,  2],
                  [ 2 ,29, 31,  3],
                  [ 2 ,29, 31,  4],
                  [ 2 ,29, 31,  5],
                  [ 2 ,29, 31, 10],
                  [ 2 ,29, 31, 11],
                  [ 2 ,29, 31, 12],
                  [ 2 ,29, 31, 13],
                  [ 2 ,29, 31, 14],
                  [ 2 ,29, 31, 15],
                  [ 2 ,29, 31, 16],
                  [ 2 ,29, 31, 17],
                  [ 2 ,29, 31, 18],
                  [ 2 ,29, 31, 19],
                  [ 2 ,29, 31, 20],
                  [ 2 ,29, 31, 21],
                  [ 2 ,29, 31, 26],
                  [ 2 ,29, 31, 27],
                  [ 2 ,29, 31, 29],
                  [ 2 ,29, 31, 30],
                  [ 2 ,29, 31, 31],
                  [ 2 ,29, 31, 32],
                  [ 2 ,29, 31, 33],
                  [ 2 ,29, 31, 34],
                  [ 2 ,29, 31, 35],
                  [ 2 ,29, 31, 36],
                  [ 2 ,29, 31, 41],
                  [ 2 ,29, 31, 42],
                  [ 2 ,29, 31, 43],
                  [ 2 ,29, 31, 45],
                  [ 2 ,29, 31, 48],
                  [ 2 ,29, 31, 49],
                  [ 2 ,29, 31, 50],
                  [ 2 ,29, 31, 51],
                  [ 2 ,29, 31, 52],
                  [ 2 ,29, 31, 53],
                  [ 2 ,29, 31, 58],
                  [ 2 ,29, 31, 59],
                  [ 2 ,29, 31, 60],
                  [ 2 ,29, 31, 61],
                  [ 2 ,29, 31, 62],
                  [ 2 ,29, 31, 63],
                  [ 6 , 7, 99,  9],
                  [ 6 , 7, 99, 12],
                  [ 6 , 7, 99, 59],
                  [ 6 , 7, 99, 60],
                  [ 6 , 7, 99, 62],
                  [ 8 ,21, 13,  1],
                  [ 8 ,21, 13,  4],
                  [ 8 ,21, 13, 10],
                  [ 8 ,21, 13, 31],
                  [ 8 ,21, 13, 32],
                  [ 8 ,21, 13, 48],
                  [ 2 ,17 ,21 , 4],
                  [ 2 ,17, 21 ,13]])

def main(directory=None, simulation=False):

    event_hits_dict = dict() # Initialize dictionary

    h5_files = []

    if simulation: 
        h5_files = glob.glob('000*000/*.FLOW.hdf5', root_dir=directory, recursive=True)
    else:
        h5_files_8 = glob.glob('july8_2024/nominal_hv/packet-0*.hdf5', root_dir=directory, recursive=True)
        h5_files_10 = glob.glob('july10_2024/nominal_hv/packet-0*.hdf5', root_dir=directory, recursive=True)
        #h5_files_12 = glob.glob('july12_2024/lrs_low_threshold/packet-0*.hdf5', root_dir=directory, recursive=True)
        h5_files.extend(h5_files_8)
        h5_files.extend(h5_files_10)
        #h5_files.extend(h5_files_12)
        #h5_files_self_trigger = glob.glob('july[8,10,12]_2024/**/packet-self*.hdf5', root_dir=directory, recursive=True)
        #h5_files.extend(h5_files_self_trigger)

    print("Number of H5 Files:", len(h5_files))
    #print(h5_files)

    total_events = 0
    total_beam_events = 0
    selected_events = 0
    selected_beam_events = 0
    noise_only_events = 0

    
    for i in range(len(h5_files)):

        #if i<300: continue   
        #if i>310: continue
        if i%10==0: print("File number:", i)
        print("Opening file:", h5_files[i])
        
        file = directory+h5_files[i]
        
        f = h5py.File(file,'r')

        # Load hits and packet information
        #total_events += events_in_this_file

        # Get events for data:
        if not simulation:
            try:
                events = f['charge/events/data']
                events_in_this_file = len(events)
                # Load external triggers dataset
                exttrigs_full = f['charge/ext_trigs/data']
                exttrigs_ref = f['charge/events/ref/charge/ext_trigs/ref']
                exttrigs_region = f['charge/events/ref/charge/ext_trigs/ref_region']
                exttrigs_beam = np.where(exttrigs_full['iogroup'] == 5)
                beam_events_ref = np.sort(exttrigs_ref[:,0][exttrigs_beam])
                beam_events = events[beam_events_ref]
                events = beam_events
                #total_events += len(beam_events)
            except:
                print("No beam trigger events found in file ", file)
                continue
                #total_events -= events_in_this_file
        hits_dset = 'calib_prompt_hits'
        #print(f['charge'].keys())
        try:
            hits_full = f['charge/'+hits_dset+'/data']
        except:
            print("Not including file ", file)
            continue
        hits_ref = f['charge/events/ref/charge/'+hits_dset+'/ref']
        hits_region = f['charge/events/ref/charge/'+hits_dset+'/ref_region']
        total_events += events_in_this_file
        total_beam_events += len(beam_events)
        if not simulation:
            packets_full = f['charge/packets/data']
            packets_ref = f['charge/'+hits_dset+'/ref/charge/packets/ref']
            packets_region = f['charge/'+hits_dset+'/ref/charge/packets/ref_region']
            packets_hit_ref = f['charge/'+hits_dset+'/ref/charge/packets/ref']

        for ev_id in events['id']:
            event_mask = events['id'] == ev_id
            event = events[event_mask]
            #print(event['unix_ts'])
            event_datetime = datetime.utcfromtimestamp(event['unix_ts'][0]).strftime('%Y-%m-%d %H:%M:%S')
            #print(event_datetime)
            #.strftime('%Y-%m-%d %H:%M:%S')
            hit_ref = hits_ref[hits_region[ev_id,'start']:hits_region[ev_id,'stop']]
            hit_ref = np.sort(hit_ref[hit_ref[:,0] == ev_id, 1])
            hits = hits_full[hit_ref]

            # Check if event is a beam event
            if simulation:
                is_beam = True
            if not simulation:# and file not in h5_files_12:
                if event in beam_events:
                    is_beam = True
                else:
                    is_beam = False
            else:
                is_beam = False
            if not simulation:
                packet_ref = packets_ref[hit_ref]
                packet_ref = np.sort(packet_ref[:, 1])
                packets = packets_full[packet_ref]
                if len(hits) != len(packets):
                    print("Number of hits and packets do not match")
                    print("Number of hits: ", len(hits))
                    print("Number of packets: ", len(packets))
                    continue

                data_packets = packets[packets['packet_type']==0]
                #unique_datawords, dw_counts = np.unique(data_packets['dataword'], return_counts=True)
                adc_sat_packets = data_packets[packets['dataword']==255]

                if len(adc_sat_packets) > 0:
                    #print("Number of ADC saturation packets: ", len(adc_sat_packets))
                    #adc_sat_idx = np.where(packets['packet_type']==0) and np.where(packets['dataword']==255)
                    #adc_sat_hits_ref = np.sort(packets_hit_ref[:,0][adc_sat_idx])
                    #adc_sat_hits = hits[adc_sat_hits_ref]
                    #adc_sat_hits_xyz = np.array([[hit['x'], hit['y'], hit['z']] for hit in adc_sat_hits])
                    #print("Number of ADC saturation packets: ", len(adc_sat_packets))
                    #print("ADC saturation packets: ", adc_sat_packets)
                    adc_sat_packets_iog = np.array(adc_sat_packets['io_group'])
                    adc_sat_packets_io_channels = np.array(adc_sat_packets['io_channel'])
                    #adc_sat_packets_tiles = np.array([io_channel_to_tile(j) for j in adc_sat_packets['io_channel']])
                    adc_sat_packets_chip_ids = np.array(adc_sat_packets['chip_id'])
                    adc_sat_packets_channel_ids = np.array(adc_sat_packets['channel_id'])
                    #channels_impacted = np.array([[iog, tile, chipid, channel] for iog in adc_sat_packets_iog for tile in adc_sat_packets_tiles for chipid in adc_sat_packets_chip_ids for channel in adc_sat_packets_channel_ids])
                    #unique_channels_impacted = np.unique(channels_impacted, axis=0)
                    channels_impacted_with_iochannel = np.array([[adc_sat_packets_iog[k], adc_sat_packets_io_channels[k], adc_sat_packets_chip_ids[k], adc_sat_packets_channel_ids[k]] for k in range(len(adc_sat_packets))])
                    unique_channels_impacted_with_iochannel, unique_channels_counts = np.unique(channels_impacted_with_iochannel, axis=0, return_counts=True)
                    #print("Unique channels impacted: ", unique_channels_impacted_with_iochannel)
                    #print("Number of unique channels impacted: ", len(unique_channels_impacted_with_iochannel))
                    delete_channels = []
                    orig_len = len(unique_channels_impacted_with_iochannel)
                    orig_channels = unique_channels_impacted_with_iochannel
                    #if orig_len > 100:
                    #    print(orig_channels)
                    for ch in range(len(unique_channels_impacted_with_iochannel)):
                        #print("Channel with 0: ", unique_channels_impacted_with_iochannel[0][ch])
                        #print("Channel with no 0: ", unique_channels_impacted_with_iochannel[ch])
                        for noisy_ch in noisy_channels:
                            if np.array_equal(unique_channels_impacted_with_iochannel[ch], noisy_ch):
                                #print("Event ID: ", ev_id)
                                #print("Ch: ", ch)
                                #print("Noisy channel: ", noisy_ch)
                                #print("Noisy channel impacted: ", unique_channels_impacted_with_iochannel[ch])
                                delete_channels.append(ch)
                    unique_channels_impacted_with_iochannel = np.delete(unique_channels_impacted_with_iochannel, delete_channels, axis=0)
                    unique_channels_counts = np.delete(unique_channels_counts, delete_channels)
                    #print("Unique channels impacted after noisy channel removal: ", unique_channels_impacted_with_iochannel)
                    if len(unique_channels_impacted_with_iochannel) == 0:
                        #print("Noise Only Event")
                        noise_only_events += 1
                        continue
                    #if len(unique_channels_impacted_with_iochannel) < orig_len:
                    #    print("Number of channels before noise removal: ", orig_len)
                    #    print("Original channels: ", orig_channels)
                    #    print("Number of channels after noise removal: ", len(unique_channels_impacted_with_iochannel))
                    #print("Channels impacted: ", channels_impacted)
                    #print("Channels impacted with IO channel: ", channels_impacted_with_iochannel)
                    #if len(unique_channels_counts) < len(adc_sat_packets):
                    #    print("Event ID: ", ev_id)
                    #    print("Unique channels impacted: ", unique_channels_impacted_with_iochannel)
                    #    print("Unique channels impacted counts: ", unique_channels_counts)
                    #print("ADC saturation packets: ", adc_sat_packets)
                    energy_per_iog_positive = np.zeros(8)
                    energy_per_iog_negative = np.zeros(8)
                    charge_per_iog_positive = np.zeros(8)
                    charge_per_iog_negative = np.zeros(8)

                    
                    # Sum hits charge, energy in 2x2 for positive and negative charge hits separately
                    for iog in np.unique(adc_sat_packets_iog):
                        iog_mask = hits['io_group'] == iog
                        single_iog_hits = hits[iog_mask]
                        negative_charge_hits_iog = single_iog_hits[single_iog_hits['Q'] <= 0]
                        positive_charge_hits_iog = single_iog_hits[single_iog_hits['Q'] > 0]
                        charge_per_iog_negative[iog-1] = sum(negative_charge_hits_iog['Q'])
                        charge_per_iog_positive[iog-1] = sum(positive_charge_hits_iog['Q'])
                        energy_per_iog_negative[iog-1] = sum(negative_charge_hits_iog['E'])
                        energy_per_iog_positive[iog-1] = sum(positive_charge_hits_iog['E'])

                    positive_charge_hits = hits[hits['Q'] > 0]
                    negative_charge_hits = hits[hits['Q'] <= 0]
                    total_positive_charge = sum(positive_charge_hits['Q'])
                    total_negative_charge = sum(negative_charge_hits['Q'])
                    total_positive_energy = sum(positive_charge_hits['E'])
                    total_negative_energy = sum(negative_charge_hits['E'])


                    selected_events += 1
                    if is_beam:
                        selected_beam_events += 1
                    if selected_events%100==0:
                        print("Selected events: ", selected_events)
                        print("Selected beam events: ", selected_beam_events)
                    event_hits_dict[(str(file), int(ev_id))]=dict(event_id = int(ev_id),
                                                                  filepath=str(file),
                                                                  #datawords= [int(k) for k in unique_datawords],
                                                                  #dw_counts = [int(k) for k in dw_counts])
                                                                  timestamp=str(event_datetime), 
                                                                  nhits=int(event['nhit'][0]),
                                                                  num_adc_saturation_packets=int(sum(unique_channels_counts)), 
                                                                  #adc_sat_hits_xyz=adc_sat_hits_xyz.tolist(),
                                                                  #adc_sat_packets_iog=[int(k) for k in np.unique(adc_sat_packets_iog)],
                                                                  unique_channels_impacted=[[int(k) for k in channels] for channels in unique_channels_impacted_with_iochannel],
                                                                  total_sat_packets_per_unique_channel=[int(k) for k in unique_channels_counts],
                                                                  iog_negative_charge=[float(round(k,4)) for k in charge_per_iog_negative],
                                                                  iog_positive_charge=[float(round(k,4)) for k in charge_per_iog_positive],
                                                                  iog_negative_energy=[float(round(k,4)) for k in energy_per_iog_negative],
                                                                  iog_positive_energy=[float(round(k,4)) for k in energy_per_iog_positive],
                                                                  total_negative_charge=float(round(total_negative_charge,4)),
                                                                  total_positive_charge=float(round(total_positive_charge,4)),
                                                                  total_negative_energy=float(round(total_negative_energy,4)),
                                                                  total_positive_energy=float(round(total_positive_energy,4)),
                                                                  #unique_channels_impacted_with_iochannel=[[int(k) for k in channels] for channels in unique_channels_impacted_with_iochannel],
                                                                  #num_unique_channels_impacted=int(len(unique_channels_impacted)), 
                                                                  #adc_sat_packets_tiles=[int(k) for k in adc_sat_packets_tiles],
                                                                  #adc_sat_packets_chip_ids=[int(k) for k in adc_sat_packets_chip_ids], 
                                                                  #adc_sat_packets_channel_ids=[int(k) for k in adc_sat_packets_channel_ids],
                                                                  is_beam=bool(is_beam))

    print("Made it past event loop.")       
    if not simulation:
        save_dict_to_json(event_hits_dict, "adc_saturation_beam_events_V5_FINAL_REMOVAL_dict", True)
        #save_dict_to_json(event_hits_dict, "all_datawords_dict", True)
    else:
        save_dict_to_json(event_hits_dict, "adc_saturation_events_dict_sim", True)

    print("Noise only events: ", noise_only_events)
    print("Total events: ", total_events)
    print("Total beam events: ", total_beam_events)
    print("Selected events: ", selected_events)
    print("Fraction of events with ADC saturation: ", 100*round((selected_events/total_events), 4), "%")
    print("Selected beam events: ", selected_beam_events)
    print("Fraction of beam events with ADC saturation: ", 100*round((selected_beam_events/total_beam_events), 4), "%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--directory', default=None,type=str,help='''string of data directory where data are stored.''')
    parser.add_argument('-mc','--simulation', default=False, help='''Flag to indicate if input files are simulation or data.''')
    args = parser.parse_args()
    main(**vars(args))