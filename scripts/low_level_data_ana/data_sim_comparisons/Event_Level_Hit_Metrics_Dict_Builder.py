#!/usr/bin/env python
# NOTE: Please change lines 74-77 (data version or sim version) to the correct version of the data or simulation you are analyzing
#       before running the script. This will ensure that the output JSON file is named correctly.
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

def main(directory=None, simulation=False):

    event_hits_dict = dict() # Initialize dictionary
    if simulation:
        sim_version = "MR6"
    else:
        flowed_data_version = "V5"

    h5_files = []
    #h5_files = glob.glob('000*000/*.FLOW.hdf5', root_dir=directory, recursive=True)

    #h5_files_8 = glob.glob('packet-0*.hdf5', root_dir=directory, recursive=True)
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
    selected_events = 0

    
    for i in range(len(h5_files)):

        #if i<300: continue   
        #if i>310: continue
        if i%10==0: print("File number:", i)
        print("Opening file:", h5_files[i])
        
        file = directory+h5_files[i]
        dir_only = os.path.dirname(file)
        filename = file.split("/")[-1]
        subdirs = dir_only.split(directory)[-1]

        output_pdf_name = 'plots/'+subdirs+'/'+filename.split(".FLOW.hdf5")[0]+'_full_tile_fire.pdf'
        os.makedirs(os.path.dirname(output_pdf_name), exist_ok=True)

        # put file in this directory for now
        with PdfPages(output_pdf_name, keep_empty=False) as output:

            f = h5py.File(file,'r')
            plt.rcParams["figure.figsize"] = (10,8)
            #events_in_this_file = len(events)
            #total_events += events_in_this_file

            # Get beam trigger events for data:
            if not simulation:
                try:
                    # Load external triggers dataset
                    # Load hits and packet information
                    events = f['charge/events/data']
                    exttrigs_full = f['charge/ext_trigs/data']
                    exttrigs_ref = f['charge/events/ref/charge/ext_trigs/ref']
                    exttrigs_region = f['charge/events/ref/charge/ext_trigs/ref_region']
                    exttrigs_beam = np.where(exttrigs_full['iogroup'] == 5)
                    beam_events_ref = np.sort(exttrigs_ref[:,0][exttrigs_beam])
                    beam_events = events[beam_events_ref]
                    events = beam_events
                except:
                    print("No beam trigger events found in file ", file)
                    #total_events -= events_in_this_file
                    continue
            hits_dset = 'calib_prompt_hits'
            #print(f['charge'].keys())
            try:
                hits_full = f['charge/'+hits_dset+'/data']
            except:
                print("Not including file ", file)
                #total_events -= events_in_this_file
                continue
            hits_ref = f['charge/events/ref/charge/'+hits_dset+'/ref']
            hits_region = f['charge/events/ref/charge/'+hits_dset+'/ref_region']
            if not simulation:
                packets_full = f['charge/packets/data']
                packets_ref = f['charge/'+hits_dset+'/ref/charge/packets/ref']
                packets_region = f['charge/'+hits_dset+'/ref/charge/packets/ref_region']

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
                if not simulation:
                    packet_ref = packets_ref[hit_ref]
                    packet_ref = np.sort(packet_ref[:, 1])
                    packets = packets_full[packet_ref]
                    if len(hits) != len(packets):
                        print("Number of hits and packets do not match")
                        print("Number of hits: ", len(hits))
                        print("Number of packets: ", len(packets))
                        continue
                #packet_timestamps = (packets['timestamp'] - min(packets['timestamp']))/10
                #dt_5th_to_95th_percentile_of_packets = (np.percentile(packet_timestamps, 95) - np.percentile(packet_timestamps, 5))

                hits_per_iog = np.zeros(8)
                min_hit_charge_per_iog = np.zeros(8)
                max_hit_charge_per_iog = np.zeros(8)
                mean_hit_charge_per_iog = np.zeros(8)
                std_hit_charge_per_iog = np.zeros(8)
                max_hit_drift_time_per_iog = np.zeros(8)
                for j in range(8):
                    iog_mask = hits['io_group'] == j+1
                    hits_per_iog[j] = len(hits[iog_mask])
                    if hits_per_iog[j] > 0:
                        min_hit_charge_per_iog[j] = min(hits['Q'][iog_mask])
                        max_hit_charge_per_iog[j] = max(hits['Q'][iog_mask])
                        mean_hit_charge_per_iog[j] = np.mean(hits['Q'][iog_mask])
                        std_hit_charge_per_iog[j] = np.std(hits['Q'][iog_mask])
                        max_hit_drift_time_per_iog[j] = max(hits['t_drift'][iog_mask])
                    else:
                        min_hit_charge_per_iog[j] = 0
                        max_hit_charge_per_iog[j] = 0
                        mean_hit_charge_per_iog[j] = 0
                        std_hit_charge_per_iog[j] = 0
                        max_hit_drift_time_per_iog[j] = 0

                #if max(packet_timestamps) > 205:
                #    selected_events += 1
                #    long_events.append(ev_id)
                #    print("Event: ", ev_id, " has a long time distribution")
                #    print("Max timestamp: ", max(packet_timestamps))
                #    print("Percentile difference: ", dt_5th_to_95th_percentile_of_packets)
                    #print("File: ", file)
                    
                #print("Event nhits: ", event['nhit'][0])
                if event['nhit'][0] > 0:
                    min_hit_charge = min(hits['Q'])
                    max_hit_charge = max(hits['Q'])
                    mean_hit_charge = np.mean(hits['Q'])
                    std_hit_charge = np.std(hits['Q'])
                    max_hit_drift_time = max(hits['t_drift'])
                else:
                    min_hit_charge = 0
                    max_hit_charge = 0
                    mean_hit_charge = 0
                    std_hit_charge = 0
                    max_hit_drift_time = 0

                event_hits_dict[(str(file), int(ev_id))]=dict(event_id = int(ev_id),
                                                              filepath=str(file),
                                                              timestamp=str(event_datetime), 
                                                              nhits=int(event['nhit'][0]), 
                                                              max_hit_charge=float(round(max_hit_charge,4)),
                                                              min_hit_charge=float(round(min_hit_charge,4)),  
                                                              mean_hit_charge=float(round(mean_hit_charge,4)),
                                                              std_hit_charge=float(round(std_hit_charge,4)),
                                                              max_hit_drift_time=float(round(max_hit_drift_time,4)),
                                                              hits_per_iog=[int(k) for k in hits_per_iog],
                                                              min_hit_charge_per_iog=[float(round(k, 4)) for k in min_hit_charge_per_iog],
                                                              max_hit_charge_per_iog=[float(round(k, 4)) for k in max_hit_charge_per_iog],
                                                              mean_hit_charge_per_iog=[float(round(k, 4)) for k in mean_hit_charge_per_iog],
                                                              std_hit_charge_per_iog=[float(round(k, 4)) for k in std_hit_charge_per_iog],
                                                              max_hit_drift_time_per_iog=[float(round(k, 4)) for k in max_hit_drift_time_per_iog])

                
            output.close()
    if not simulation:
        save_dict_to_json(event_hits_dict, "event_hits_dict_"+flowed_data_version, True)
    else:
        save_dict_to_json(event_hits_dict, "event_hits_dict_"+sim_version, True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--directory', default=None,type=str,help='''string of data directory where data are stored.''')
    parser.add_argument('-mc','--simulation', default=False, help='''Flag to indicate if input files are simulation or data.''')
    args = parser.parse_args()
    main(**vars(args))