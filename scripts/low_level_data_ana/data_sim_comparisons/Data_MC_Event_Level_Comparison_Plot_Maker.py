#!/usr/bin/env python
# NOTE: Please change lines 73-78 (data version or sim version) to the correct version of the data or simulation you are analyzing
#       before running the script. This will ensure that the plots are correctly named and that the output file is named correctly.
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

def main(data_dict, sim_dict, log_scale=False):
    
    if log_scale==True:
        output_pdf_name = 'Data_MC_Comparison_Log_Scale_COMPARE_V5_MR6.pdf'
    elif log_scale==False:
        output_pdf_name = 'Data_MC_Comparison_Linear_Scale_COMPARE_V5_MR6.pdf'
    data_sample_name = 'V5 Beam Data' # 'Beam Data'
    sim_sample_name = 'MiniRun6 Sim'
    data_file = open(data_dict)
    data=json.load(data_file)
    sim_file = open(sim_dict)
    sim=json.load(sim_file)
    print("Log scale:", log_scale)
    print("output_pdf_name:", output_pdf_name)
    filtered_data = {key: value for key, value in data.items() if 'low_threshold' not in key}
    data = filtered_data

    # put file in this directory for now
    with PdfPages(output_pdf_name, keep_empty=False) as output:

        # Plot number of hits per event
        plt.rcParams["figure.figsize"] = (10,8)
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_axes([0.1, 0.25, 0.8, 0.6])
        data_nhits = np.array([data[key]['nhits'] for key in data.keys()])
        sim_nhits = np.array([sim[key]['nhits'] for key in sim.keys()])
        total_num_data_events = len(data_nhits)
        total_num_sim_events = len(sim_nhits)
        max_nhits = math.ceil(max(np.max(data_nhits), np.max(sim_nhits))/10)*10
        data_nhits_counts, data_nhits_bins = np.histogram(data_nhits, bins=int(max_nhits/50), range=(0,max_nhits))
        sim_nhits_counts, sim_nhits_bins = np.histogram(sim_nhits, bins=int(max_nhits/50), range=(0,max_nhits))
        ax.hist(data_nhits_bins[:-1], bins=data_nhits_bins, weights=data_nhits_counts/total_num_data_events,\
                histtype='stepfilled', label=data_sample_name, color=(0,0,1,0.2),edgecolor=(0,0,1,0.8), linestyle='-')
        ax.hist(sim_nhits_bins[:-1], bins=sim_nhits_bins, weights=sim_nhits_counts/total_num_sim_events,\
                histtype='stepfilled', label=sim_sample_name, color=(1,0,0,0.2),edgecolor=(1,0,0,0.8),linestyle='--')
        ax.set_xlabel('Number of Hits per Event')
        ax.set_ylabel('Fraction of Total Events in Sample / 50 Hits')
        if log_scale==True: ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()
        ax_residual = fig.add_axes([0.1, 0.1, 0.8, 0.15], sharex=ax)
        residuals = 100*(((data_nhits_counts / total_num_data_events) - (sim_nhits_counts / total_num_sim_events)) / (sim_nhits_counts / total_num_sim_events))
        residuals = np.nan_to_num(residuals, nan=0, posinf=1000, neginf=-1000)
        ax_residual.hist(data_nhits_bins[:-1], bins=data_nhits_bins, weights=residuals ,color='black', alpha=0.8)
        ax_residual.axhline(50, color='grey', linestyle='--', linewidth=0.75)
        ax_residual.axhline(-50, color='grey', linestyle='--', linewidth=0.75)
        ax_residual.set_xlabel('Number of Hits per Event')
        ax_residual.set_ylabel('Residuals [%] \n(Data-Sim)/Sim')
        ax_residual.set_xscale('log')
        ax_residual.set_ylim(-100, 150)
        output.savefig(fig)
        plt.close(fig)

        # Mask events with no hits
        data_nhits_mask = data_nhits > 0
        sim_nhits_mask = sim_nhits > 0
        num_data_events_with_hits = np.sum(data_nhits_mask)
        num_sim_events_with_hits = np.sum(sim_nhits_mask)

        # Plot max charge per event
        plt.rcParams["figure.figsize"] = (10,8)
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_axes([0.1, 0.25, 0.8, 0.6])
        data_max_hit_charge = np.array([data[key]['max_hit_charge'] for key in data.keys()])[data_nhits_mask]
        sim_max_hit_charge = np.array([sim[key]['max_hit_charge'] for key in sim.keys()])[sim_nhits_mask]
        max_max_hit_charge = math.ceil(max(np.max(data_max_hit_charge), np.max(sim_max_hit_charge))/10)*10
        min_max_hit_charge = math.floor(min(np.min(data_max_hit_charge), np.min(sim_max_hit_charge))/10)*10
        data_max_hit_charge_counts, data_max_hit_charge_bins = np.histogram(data_max_hit_charge, \
                                                                            bins=int((max_max_hit_charge-min_max_hit_charge)/5), \
                                                                            range=(min_max_hit_charge,max_max_hit_charge))
        sim_max_hit_charge_counts, sim_max_hit_charge_bins = np.histogram(sim_max_hit_charge, \
                                                                            bins=int((max_max_hit_charge-min_max_hit_charge)/5), \
                                                                            range=(min_max_hit_charge,max_max_hit_charge))
        ax.hist(data_max_hit_charge_bins[:-1], bins=data_max_hit_charge_bins, weights=data_max_hit_charge_counts/num_data_events_with_hits, \
                histtype='stepfilled', label=data_sample_name, color=(0,0,1,0.2),edgecolor=(0,0,1,0.8), linestyle='-')
        ax.hist(sim_max_hit_charge_bins[:-1], bins=sim_max_hit_charge_bins, weights=sim_max_hit_charge_counts/num_sim_events_with_hits, \
                histtype='stepfilled', label=sim_sample_name, color=(1,0,0,0.2),edgecolor=(1,0,0,0.8),linestyle='--')
        ax.set_ylabel('Fraction of Events in Sample with Hits / 5 ke')
        if log_scale==True: ax.set_yscale('log')
        ax.legend()
        ax_residual = fig.add_axes([0.1, 0.1, 0.8, 0.15], sharex=ax)
        residuals = 100*(((data_max_hit_charge_counts / num_data_events_with_hits) - (sim_max_hit_charge_counts / num_sim_events_with_hits)) / (sim_max_hit_charge_counts / num_sim_events_with_hits))
        residuals = np.nan_to_num(residuals, nan=0, posinf=1000, neginf=-1000)
        ax_residual.hist(data_max_hit_charge_bins[:-1], bins=data_max_hit_charge_bins, weights=residuals ,color='black', alpha=0.8)
        ax_residual.axhline(50, color='grey', linestyle='--', linewidth=0.75)
        ax_residual.axhline(-50, color='grey', linestyle='--', linewidth=0.75)
        ax_residual.set_ylabel('Residuals [%] \n(Data-Sim)/Sim')
        ax_residual.set_xlabel('Max Hit Charge per Event with at least one Hit [ke]')
        ax_residual.set_xlim(min_max_hit_charge,max_max_hit_charge+5)
        ax_residual.set_ylim(-100, 150)
        output.savefig(fig)
        plt.close(fig)

        # Plot min charge per event
        plt.rcParams["figure.figsize"] = (10,8)
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_axes([0.1, 0.25, 0.8, 0.6])
        data_min_hit_charge = np.array([data[key]['min_hit_charge'] for key in data.keys()])[data_nhits_mask]
        sim_min_hit_charge = np.array([sim[key]['min_hit_charge'] for key in sim.keys()])[sim_nhits_mask]
        max_min_hit_charge = math.ceil(max(np.max(data_min_hit_charge), np.max(sim_min_hit_charge))/10)*10
        min_min_hit_charge = math.floor(min(np.min(data_min_hit_charge), np.min(sim_min_hit_charge))/10)*10
        data_min_hit_charge_counts, data_min_hit_charge_bins = np.histogram(data_min_hit_charge, \
                                                                            bins=int((max_min_hit_charge-min_min_hit_charge)/5), \
                                                                            range=(min_min_hit_charge,max_min_hit_charge))
        sim_min_hit_charge_counts, sim_min_hit_charge_bins = np.histogram(sim_min_hit_charge, \
                                                                            bins=int((max_min_hit_charge-min_min_hit_charge)/5), \
                                                                            range=(min_min_hit_charge,max_min_hit_charge))
        ax.hist(data_min_hit_charge_bins[:-1], bins=data_min_hit_charge_bins, weights=data_min_hit_charge_counts/num_data_events_with_hits, \
                histtype='stepfilled', label=data_sample_name, color=(0,0,1,0.2),edgecolor=(0,0,1,0.8), linestyle='-')
        ax.hist(sim_min_hit_charge_bins[:-1], bins=sim_min_hit_charge_bins, weights=sim_min_hit_charge_counts/num_sim_events_with_hits, \
                histtype='stepfilled', label=sim_sample_name, color=(1,0,0,0.2),edgecolor=(1,0,0,0.8),linestyle='--')
        ax.set_xlabel('Min Hit Charge per Event with at least one Hit [ke]')
        ax.set_ylabel('Fraction of Events in Sample with Hits / 5 ke')
        if log_scale==True: ax.set_yscale('log')
        ax.set_xlim(min_min_hit_charge,max_min_hit_charge)
        ax_residual = fig.add_axes([0.1, 0.1, 0.8, 0.15], sharex=ax)
        residuals = 100*(((data_min_hit_charge_counts / num_data_events_with_hits) - (sim_min_hit_charge_counts / num_sim_events_with_hits)) / (sim_min_hit_charge_counts / num_sim_events_with_hits))
        residuals = np.nan_to_num(residuals, nan=0, posinf=1000, neginf=-1000)
        ax_residual.hist(data_min_hit_charge_bins[:-1], bins=data_min_hit_charge_bins, weights=residuals ,color='black', alpha=0.8)
        ax_residual.axhline(50, color='grey', linestyle='--', linewidth=0.75)
        ax_residual.axhline(-50, color='grey', linestyle='--', linewidth=0.75)
        ax_residual.set_ylabel('Residuals [%] \n(Data-Sim)/Sim')
        ax_residual.set_xlabel('Min Hit Charge per Event with at least one Hit [ke]')
        ax_residual.set_xlim(min_min_hit_charge,max_min_hit_charge+5)
        ax_residual.set_ylim(-100, 150)
        ax.legend()
        output.savefig(fig)
        plt.close(fig)

        # Plot mean hit charge per event
        plt.rcParams["figure.figsize"] = (10,8)
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_axes([0.1, 0.25, 0.8, 0.6])
        data_mean_hit_charge = np.array([data[key]['mean_hit_charge'] for key in data.keys()])[data_nhits_mask]
        sim_mean_hit_charge = np.array([sim[key]['mean_hit_charge'] for key in sim.keys()])[sim_nhits_mask]
        max_mean_hit_charge = math.ceil(max(np.max(data_mean_hit_charge), np.max(sim_mean_hit_charge))/10)*10
        min_mean_hit_charge = math.floor(min(np.min(data_mean_hit_charge), np.min(sim_mean_hit_charge))/10)*10
        data_mean_hit_charge_counts, data_mean_hit_charge_bins = np.histogram(data_mean_hit_charge, \
                                                                            bins=int((max_mean_hit_charge-min_mean_hit_charge)/5), \
                                                                            range=(min_mean_hit_charge,max_mean_hit_charge))
        sim_mean_hit_charge_counts, sim_mean_hit_charge_bins = np.histogram(sim_mean_hit_charge, \
                                                                            bins=int((max_mean_hit_charge-min_mean_hit_charge)/5), \
                                                                            range=(min_mean_hit_charge,max_mean_hit_charge))
        ax.hist(data_mean_hit_charge_bins[:-1], bins=data_mean_hit_charge_bins, weights=data_mean_hit_charge_counts/num_data_events_with_hits, \
                histtype='stepfilled', label=data_sample_name, color=(0,0,1,0.2),edgecolor=(0,0,1,0.8), linestyle='-')
        ax.hist(sim_mean_hit_charge_bins[:-1], bins=sim_mean_hit_charge_bins, weights=sim_mean_hit_charge_counts/num_sim_events_with_hits, \
                histtype='stepfilled', label=sim_sample_name, color=(1,0,0,0.2),edgecolor=(1,0,0,0.8),linestyle='--')
        ax.set_xlabel('Mean Hit Charge per Event with at least one Hit [ke]')
        ax.set_ylabel('Fraction of Events in Sample with Hits / 5 ke')
        if log_scale==True: ax.set_yscale('log')
        ax.set_xlim(min_mean_hit_charge,max_mean_hit_charge)
        ax.legend()
        ax_residual = fig.add_axes([0.1, 0.1, 0.8, 0.15], sharex=ax)
        residuals = 100*(((data_mean_hit_charge_counts / num_data_events_with_hits) - (sim_mean_hit_charge_counts / num_sim_events_with_hits)) / (sim_mean_hit_charge_counts / num_sim_events_with_hits))
        residuals = np.nan_to_num(residuals, nan=0, posinf=1000, neginf=-1000)
        ax_residual.hist(data_mean_hit_charge_bins[:-1], bins=data_mean_hit_charge_bins, weights=residuals ,color='black', alpha=0.8)
        ax_residual.axhline(50, color='grey', linestyle='--', linewidth=0.75)
        ax_residual.axhline(-50, color='grey', linestyle='--', linewidth=0.75)
        ax_residual.set_ylabel('Residuals [%] \n(Data-Sim)/Sim')
        ax_residual.set_xlabel('Mean Hit Charge per Event with at least one Hit [ke]')
        ax_residual.set_xlim(min_mean_hit_charge,max_mean_hit_charge+5)
        ax_residual.set_ylim(-100, 150)
        output.savefig(fig)
        plt.close(fig)

        # Plot std hit charge per event
        plt.rcParams["figure.figsize"] = (10,8)
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_axes([0.1, 0.25, 0.8, 0.6])
        data_std_hit_charge = np.array([data[key]['std_hit_charge'] for key in data.keys()])[data_nhits_mask]
        sim_std_hit_charge = np.array([sim[key]['std_hit_charge'] for key in sim.keys()])[sim_nhits_mask]
        max_std_hit_charge = math.ceil(max(np.max(data_std_hit_charge), np.max(sim_std_hit_charge))/10)*10
        min_std_hit_charge = math.floor(min(np.min(data_std_hit_charge), np.min(sim_std_hit_charge))/10)*10
        data_std_hit_charge_counts, data_std_hit_charge_bins = np.histogram(data_std_hit_charge, \
                                                                            bins=int((max_std_hit_charge-min_std_hit_charge)/5), \
                                                                            range=(min_std_hit_charge,max_std_hit_charge))
        sim_std_hit_charge_counts, sim_std_hit_charge_bins = np.histogram(sim_std_hit_charge, \
                                                                            bins=int((max_std_hit_charge-min_std_hit_charge)/5), \
                                                                            range=(min_std_hit_charge,max_std_hit_charge))
        ax.hist(data_std_hit_charge_bins[:-1], bins=data_std_hit_charge_bins, weights=data_std_hit_charge_counts/num_data_events_with_hits, \
                histtype='stepfilled', label=data_sample_name, color=(0,0,1,0.2),edgecolor=(0,0,1,0.8), linestyle='-')
        ax.hist(sim_std_hit_charge_bins[:-1], bins=sim_std_hit_charge_bins, weights=sim_std_hit_charge_counts/num_sim_events_with_hits, \
                histtype='stepfilled', label=sim_sample_name, color=(1,0,0,0.2),edgecolor=(1,0,0,0.8),linestyle='--')
        ax.set_xlabel('Standard Deviation of Hit Charge per Event with at least one Hit [ke]')
        ax.set_ylabel('Fraction of Events in Sample with Hits / 5 ke')
        if log_scale==True: ax.set_yscale('log')
        ax.set_xlim(min_std_hit_charge,max_std_hit_charge)
        ax.legend()
        ax_residual = fig.add_axes([0.1, 0.1, 0.8, 0.15], sharex=ax)
        residuals = 100*(((data_std_hit_charge_counts / num_data_events_with_hits) - (sim_std_hit_charge_counts / num_sim_events_with_hits)) / (sim_std_hit_charge_counts / num_sim_events_with_hits))
        residuals = np.nan_to_num(residuals, nan=0, posinf=1000, neginf=-1000)
        ax_residual.hist(data_std_hit_charge_bins[:-1], bins=data_std_hit_charge_bins, weights=residuals ,color='black', alpha=0.8)
        ax_residual.axhline(50, color='grey', linestyle='--', linewidth=0.75)
        ax_residual.axhline(-50, color='grey', linestyle='--', linewidth=0.75)
        ax_residual.set_ylabel('Residuals [%] \n(Data-Sim)/Sim')
        ax_residual.set_xlabel('Standard Deviation of Hit Charge per Event with at least one Hit [ke]')
        ax_residual.set_xlim(min_std_hit_charge,max_std_hit_charge+5)
        ax_residual.set_ylim(-100, 150)
        output.savefig(fig)
        plt.close(fig)

        # Plot max hit drift time per event with at least one hit
        plt.rcParams["figure.figsize"] = (10,8)
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_axes([0.1, 0.25, 0.8, 0.6])
        data_max_hit_drift_time = np.array([data[key]['max_hit_drift_time']/10 for key in data.keys()])[data_nhits_mask]
        sim_max_hit_drift_time = np.array([sim[key]['max_hit_drift_time']/10 for key in sim.keys()])[sim_nhits_mask]
        max_max_hit_drift_time = math.ceil(max(np.max(data_max_hit_drift_time), np.max(sim_max_hit_drift_time))/100)*100
        min_max_hit_drift_time = math.floor(min(np.min(data_max_hit_drift_time), np.min(sim_max_hit_drift_time))/100)*100
        data_max_hit_drift_time_counts, data_max_hit_drift_time_bins = np.histogram(data_max_hit_drift_time, \
                                                                            bins=int((max_max_hit_drift_time-min_max_hit_drift_time)/10), \
                                                                            range=(min_max_hit_drift_time,max_max_hit_drift_time))
        sim_max_hit_drift_time_counts, sim_max_hit_drift_time_bins = np.histogram(sim_max_hit_drift_time, \
                                                                            bins=int((max_max_hit_drift_time-min_max_hit_drift_time)/10), \
                                                                            range=(min_max_hit_drift_time,max_max_hit_drift_time))
        ax.hist(data_max_hit_drift_time_bins[:-1], bins=data_max_hit_drift_time_bins, weights=data_max_hit_drift_time_counts/num_data_events_with_hits, \
                histtype='stepfilled', label=data_sample_name, color=(0,0,1,0.2),edgecolor=(0,0,1,0.8), linestyle='-')
        ax.hist(sim_max_hit_drift_time_bins[:-1], bins=sim_max_hit_drift_time_bins, weights=sim_max_hit_drift_time_counts/num_sim_events_with_hits, \
                histtype='stepfilled', label=sim_sample_name, color=(1,0,0,0.2),edgecolor=(1,0,0,0.8),linestyle='--')
        ax.set_xlabel('Max Hit Drift Time per Event with at least one Hit [us]')
        ax.set_ylabel('Fraction of Events in Sample with Hits / 10 us')
        if log_scale==True: ax.set_yscale('log')
        ax.set_xlim(min_max_hit_drift_time,max_max_hit_drift_time)
        ax.legend()
        ax_residual = fig.add_axes([0.1, 0.1, 0.8, 0.15], sharex=ax)
        residuals = 100*(((data_max_hit_drift_time_counts / num_data_events_with_hits) - (sim_max_hit_drift_time_counts / num_sim_events_with_hits)) / (sim_max_hit_drift_time_counts / num_sim_events_with_hits))
        residuals = np.nan_to_num(residuals, nan=0, posinf=1000, neginf=-1000)
        ax_residual.hist(data_max_hit_drift_time_bins[:-1], bins=data_max_hit_drift_time_bins, weights=residuals ,color='black', alpha=0.8)
        ax_residual.axhline(50, color='grey', linestyle='--', linewidth=0.75)
        ax_residual.axhline(-50, color='grey', linestyle='--', linewidth=0.75)
        ax_residual.set_ylabel('Residuals [%] \n(Data-Sim)/Sim')
        ax_residual.set_xlabel('Max Hit Drift Time per Event with at least one Hit [us]')
        ax_residual.set_xlim(min_max_hit_drift_time,max_max_hit_drift_time+5)
        ax_residual.set_ylim(-100, 150)
        output.savefig(fig)
        plt.close(fig)

        # Plot max hit drift time per event with at least one hit Zoomed in
        plt.rcParams["figure.figsize"] = (10,8)
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_axes([0.1, 0.25, 0.8, 0.6])
        ax.hist(data_max_hit_drift_time_bins[:-1], bins=data_max_hit_drift_time_bins, weights=data_max_hit_drift_time_counts/num_data_events_with_hits, \
                histtype='stepfilled', label=data_sample_name, color=(0,0,1,0.2),edgecolor=(0,0,1,0.8), linestyle='-')
        ax.hist(sim_max_hit_drift_time_bins[:-1], bins=sim_max_hit_drift_time_bins, weights=sim_max_hit_drift_time_counts/num_sim_events_with_hits, \
                histtype='stepfilled', label=sim_sample_name, color=(1,0,0,0.2),edgecolor=(1,0,0,0.8),linestyle='--')
        ax.set_xlabel('Max Hit Drift Time per Event with at least one Hit\n and Max Hit Drift Time <210 us [us]')
        ax.set_ylabel('Fraction of Events in Sample with Hits / 10 us')
        if log_scale==True: ax.set_yscale('log')
        ax.set_xlim(-10,210)
        #ax.set_xscale('log')
        ax.legend()
        ax_residual = fig.add_axes([0.1, 0.1, 0.8, 0.15], sharex=ax)
        ax_residual.hist(data_max_hit_drift_time_bins[:-1], bins=data_max_hit_drift_time_bins, weights=residuals ,color='black', alpha=0.8)
        ax_residual.axhline(50, color='grey', linestyle='--', linewidth=0.75)
        ax_residual.axhline(-50, color='grey', linestyle='--', linewidth=0.75)
        ax_residual.set_ylabel('Residuals [%] \n(Data-Sim)/Sim')
        ax_residual.set_xlabel('Max Hit Drift Time per Event with at least one Hit\n and Max Hit Drift Time <210 us [us]')
        ax_residual.set_ylim(-100, 150)
        output.savefig(fig)
        plt.close(fig)

        # Plot max hit drift time per event with at least one hit Zoomed in
        plt.rcParams["figure.figsize"] = (10,8)
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_axes([0.1, 0.25, 0.8, 0.6])
        ax.hist(data_max_hit_drift_time_bins[:-1], bins=data_max_hit_drift_time_bins, weights=data_max_hit_drift_time_counts/num_data_events_with_hits, \
                histtype='stepfilled', label=data_sample_name, color=(0,0,1,0.2),edgecolor=(0,0,1,0.8), linestyle='-')
        ax.hist(sim_max_hit_drift_time_bins[:-1], bins=sim_max_hit_drift_time_bins, weights=sim_max_hit_drift_time_counts/num_sim_events_with_hits, \
                histtype='stepfilled', label=sim_sample_name, color=(1,0,0,0.2),edgecolor=(1,0,0,0.8),linestyle='--')
        ax.set_xlabel('Max Hit Drift Time per Event with at least one Hit\n and Max Hit Drift Time >10000 us [us]')
        ax.set_ylabel('Fraction of Events in Sample with Hits / 10 us')
        if log_scale==True: ax.set_yscale('log')
        ax.set_xlim(999000,max_max_hit_drift_time+20)
        ax.set_ylim(0.000001,0.0001)
        #ax.set_xscale('log')
        ax.legend()
        ax_residual = fig.add_axes([0.1, 0.1, 0.8, 0.15], sharex=ax)
        ax_residual.hist(data_max_hit_drift_time_bins[:-1], bins=data_max_hit_drift_time_bins, weights=residuals ,color='black', alpha=0.8)
        ax_residual.axhline(50, color='grey', linestyle='--', linewidth=0.75)
        ax_residual.axhline(-50, color='grey', linestyle='--', linewidth=0.75)
        ax_residual.set_ylabel('Residuals [%] \n(Data-Sim)/Sim')
        ax_residual.set_xlabel('Max Hit Drift Time per Event with at least one Hit\n and Max Hit Drift Time >10000 us [us]')
        ax_residual.set_ylim(-100, 150)
        output.savefig(fig)
        plt.close(fig)

        # Max vs. Min Hit Charge per Event with at least one hit
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
        if log_scale==True: 
            norm = mpl.colors.LogNorm(vmin=0.000001,vmax=1)
        else:
            norm = mpl.colors.Normalize(vmin=0,vmax=1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
        sm.set_array([])
        ax[0].hist2d(data_max_hit_charge, data_min_hit_charge, bins=[data_max_hit_charge_bins, data_min_hit_charge_bins], \
                     weights=np.full_like(data_max_hit_charge,1/num_data_events_with_hits), cmap='viridis')
        ax[1].hist2d(sim_max_hit_charge, sim_min_hit_charge, bins=[sim_max_hit_charge_bins, sim_min_hit_charge_bins], \
                     weights=np.full_like(sim_max_hit_charge,1/num_sim_events_with_hits), cmap='viridis')
        ax[0].set_xlabel('Max Hit Charge per Event with at least one Hit [ke]')
        ax[0].set_ylabel('Min Hit Charge per Event with at least one Hit [ke]')
        ax[1].set_xlabel('Max Hit Charge per Event with at least one Hit [ke]')
        ax[1].set_ylabel('Min Hit Charge per Event with at least one Hit [ke]')
        ax[0].set_title(data_sample_name)
        ax[1].set_title(sim_sample_name)
        fig.subplots_adjust(right=0.9)  
        cbar_ax = fig.add_axes([0.935, 0.125, 0.015, 0.74])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label(r'Fraction of Events in Sample with Hits / 5 ke, 5 ke')
        output.savefig(fig)
        plt.close(fig)

        # Number of hits vs. Min Hit Charge per Event with at least one hit
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
        if log_scale==True: 
            norm = mpl.colors.LogNorm(vmin=0.000001,vmax=1)
        else:
            norm = mpl.colors.Normalize(vmin=0,vmax=1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
        sm.set_array([])
        ax[0].hist2d(data_nhits[data_nhits_mask], data_min_hit_charge, bins=[data_nhits_bins, data_min_hit_charge_bins],\
                     weights=np.full_like(data_min_hit_charge,1/num_data_events_with_hits), cmap='viridis')
        ax[1].hist2d(sim_nhits[sim_nhits_mask], sim_min_hit_charge, bins=[sim_nhits_bins, sim_min_hit_charge_bins], \
                     weights=np.full_like(sim_min_hit_charge,1/num_sim_events_with_hits), cmap='viridis')
        ax[0].set_xlabel('Number of Hits per Event')
        ax[0].set_ylabel('Min Hit Charge per Event with at least one Hit [ke]')
        ax[1].set_xlabel('Number of Hits per Event')
        ax[1].set_ylabel('Min Hit Charge per Event with at least one Hit [ke]')
        ax[0].set_title(data_sample_name)
        ax[1].set_title(sim_sample_name)
        fig.subplots_adjust(right=0.9)  
        cbar_ax = fig.add_axes([0.935, 0.125, 0.015, 0.74])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label(r'Fraction of Events in Sample with Hits / 50 hits, 5 ke')
        ax[0].set_xlim(0,3000)
        ax[1].set_xlim(0,3000)
        ax[0].set_ylim(-25,50)
        ax[1].set_ylim(-25,50)
        output.savefig(fig)
        plt.close(fig)

        # Number of hits vs. Max Hit Charge per Event with at least one hit
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
        if log_scale==True: 
            norm = mpl.colors.LogNorm(vmin=0.000001,vmax=1)
        else:
            norm = mpl.colors.Normalize(vmin=0,vmax=1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
        sm.set_array([])
        ax[0].hist2d(data_nhits[data_nhits_mask], data_max_hit_charge, bins=[data_nhits_bins, data_max_hit_charge_bins],\
                     weights=np.full_like(data_max_hit_charge,1/num_data_events_with_hits), cmap='viridis')
        ax[1].hist2d(sim_nhits[sim_nhits_mask], sim_max_hit_charge, bins=[sim_nhits_bins, sim_max_hit_charge_bins], \
                     weights=np.full_like(sim_max_hit_charge,1/num_sim_events_with_hits), cmap='viridis')
        ax[0].set_xlabel('Number of Hits per Event')
        ax[0].set_ylabel('Max Hit Charge per Event with at least one Hit [ke]')
        ax[1].set_xlabel('Number of Hits per Event')
        ax[1].set_ylabel('Max Hit Charge per Event with at least one Hit [ke]')
        ax[0].set_title(data_sample_name)
        ax[1].set_title(sim_sample_name)
        fig.subplots_adjust(right=0.9)  
        cbar_ax = fig.add_axes([0.935, 0.125, 0.015, 0.74])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label(r'Fraction of Events in Sample with Hits / 50 hits, 5 ke')
        ax[0].set_xlim(0,3000)
        ax[1].set_xlim(0,3000)
        ax[0].set_ylim(0,200)
        ax[1].set_ylim(0,200)
        output.savefig(fig)
        plt.close(fig)

        # Number of hits vs. Mean Hit Charge per Event with at least one hit
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
        if log_scale==True: 
            norm = mpl.colors.LogNorm(vmin=0.000001,vmax=1)
        else:
            norm = mpl.colors.Normalize(vmin=0,vmax=1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
        sm.set_array([])
        ax[0].hist2d(data_nhits[data_nhits_mask], data_mean_hit_charge, bins=[data_nhits_bins, data_mean_hit_charge_bins],\
                     weights=np.full_like(data_mean_hit_charge,1/num_data_events_with_hits), cmap='viridis')
        ax[1].hist2d(sim_nhits[sim_nhits_mask], sim_mean_hit_charge, bins=[sim_nhits_bins, sim_mean_hit_charge_bins], \
                     weights=np.full_like(sim_mean_hit_charge,1/num_sim_events_with_hits), cmap='viridis')
        ax[0].set_xlabel('Number of Hits per Event')
        ax[0].set_ylabel('Mean Hit Charge per Event with at least one Hit [ke]')
        ax[1].set_xlabel('Number of Hits per Event')
        ax[1].set_ylabel('Mean Hit Charge per Event with at least one Hit [ke]')
        ax[0].set_title(data_sample_name)
        ax[1].set_title(sim_sample_name)
        fig.subplots_adjust(right=0.9)  
        cbar_ax = fig.add_axes([0.935, 0.125, 0.015, 0.74])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label(r'Fraction of Events in Sample with Hits / 50 hits, 5 ke')
        ax[0].set_xlim(0,3000)
        ax[1].set_xlim(0,3000)
        ax[0].set_ylim(0,50)
        ax[1].set_ylim(0,50)
        output.savefig(fig)
        plt.close(fig)

        # Number of hits vs. RMS Hit Charge per Event with at least one hit
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
        if log_scale==True: 
            norm = mpl.colors.LogNorm(vmin=0.000001,vmax=1)
        else:
            norm = mpl.colors.Normalize(vmin=0,vmax=1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
        sm.set_array([])
        ax[0].hist2d(data_nhits[data_nhits_mask], data_std_hit_charge, bins=[data_nhits_bins, data_std_hit_charge_bins],\
                     weights=np.full_like(data_std_hit_charge,1/num_data_events_with_hits), cmap='viridis')
        ax[1].hist2d(sim_nhits[sim_nhits_mask], sim_std_hit_charge, bins=[sim_nhits_bins, sim_std_hit_charge_bins], \
                     weights=np.full_like(sim_std_hit_charge,1/num_sim_events_with_hits), cmap='viridis')
        ax[0].set_xlabel('Number of Hits per Event')
        ax[0].set_ylabel('RMS Hit Charge per Event with at least one Hit [ke]')
        ax[1].set_xlabel('Number of Hits per Event')
        ax[1].set_ylabel('RMS Hit Charge per Event with at least one Hit [ke]')
        ax[0].set_title(data_sample_name)
        ax[1].set_title(sim_sample_name)
        fig.subplots_adjust(right=0.9)  
        cbar_ax = fig.add_axes([0.935, 0.125, 0.015, 0.74])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label(r'Fraction of Events in Sample with Hits / 50 hits, 5 ke')
        ax[0].set_xlim(0,3000)
        ax[1].set_xlim(0,3000)
        ax[0].set_ylim(0,25)
        ax[1].set_ylim(0,25)
        output.savefig(fig)
        plt.close(fig)

        # Mean vs. RMS Hit Charge per Event with at least one hit
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
        if log_scale==True: 
            norm = mpl.colors.LogNorm(vmin=0.000001,vmax=1)
        else:
            norm = mpl.colors.Normalize(vmin=0,vmax=1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
        sm.set_array([])
        ax[0].hist2d(data_mean_hit_charge, data_std_hit_charge, bins=[data_mean_hit_charge_bins, data_std_hit_charge_bins], \
                     weights=np.full_like(data_mean_hit_charge,1/num_data_events_with_hits), cmap='viridis')
        ax[1].hist2d(sim_mean_hit_charge, sim_std_hit_charge, bins=[sim_mean_hit_charge_bins, sim_std_hit_charge_bins], \
                     weights=np.full_like(sim_mean_hit_charge,1/num_sim_events_with_hits), cmap='viridis')
        ax[0].set_xlabel('Mean Hit Charge per Event with at least one Hit [ke]')
        ax[0].set_ylabel('RMS Hit Charge per Event with at least one Hit [ke]')
        ax[1].set_xlabel('Mean Hit Charge per Event with at least one Hit [ke]')
        ax[1].set_ylabel('RMS Hit Charge per Event with at least one Hit [ke]')
        ax[0].set_title(data_sample_name)
        ax[1].set_title(sim_sample_name)
        fig.subplots_adjust(right=0.9)  
        cbar_ax = fig.add_axes([0.935, 0.125, 0.015, 0.74])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label(r'Fraction of Events in Sample with Hits / 5 ke, 5 ke')
        ax[0].set_xlim(0,50)
        ax[1].set_xlim(0,50)
        ax[0].set_ylim(0,50)
        ax[1].set_ylim(0,50)
        output.savefig(fig)
        plt.close(fig)

        # Number of hits vs. Max Hit Drift Time per Event with at least one hit
        data_max_hit_drift_time_LIMITED_counts, data_max_hit_drift_time_LIMITED_bins = np.histogram(data_max_hit_drift_time, \
                                                                            bins=23, \
                                                                            range=(-10,220))
        sim_max_hit_drift_time_LIMITED_counts, sim_max_hit_drift_time_LIMITED_bins = np.histogram(sim_max_hit_drift_time, \
                                                                            bins=23, \
                                                                            range=(-10,220))
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
        if log_scale==True: 
            norm = mpl.colors.LogNorm(vmin=0.000001,vmax=1)
        else:
            norm = mpl.colors.Normalize(vmin=0,vmax=1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
        sm.set_array([])#[data_nhits_bins, data_min_hit_charge_bins], \
        ax[0].hist2d(data_nhits[data_nhits_mask], data_max_hit_drift_time, bins=[data_nhits_bins, data_max_hit_drift_time_LIMITED_bins],\
                     weights=np.full_like(data_max_hit_drift_time,1/num_data_events_with_hits), cmap='viridis')
        ax[1].hist2d(sim_nhits[sim_nhits_mask], sim_max_hit_drift_time, bins=[sim_nhits_bins, sim_max_hit_drift_time_LIMITED_bins], \
                     weights=np.full_like(sim_max_hit_drift_time,1/num_sim_events_with_hits), cmap='viridis')
        ax[0].set_xlabel('Number of Hits per Event')
        ax[0].set_ylabel('Max Hit Drift Time per Event with at least one Hit [us]')
        ax[1].set_xlabel('Number of Hits per Event')
        ax[1].set_ylabel('Max Hit Drift Time per Event with at least one Hit [us]')
        ax[0].set_title(data_sample_name)
        ax[1].set_title(sim_sample_name)
        fig.subplots_adjust(right=0.9)  
        cbar_ax = fig.add_axes([0.935, 0.125, 0.015, 0.74])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label(r'Fraction of Events in Sample with Hits / 50 hits, 10 us')
        ax[0].set_xlim(0,3500)
        ax[1].set_xlim(0,3500)
        output.savefig(fig)
        plt.close(fig)

        ################ IOG PLOTS ################
        # Plot number of hits per iog
        plt.rcParams["figure.figsize"] = (10,8)
        data_nhits_per_iog = np.array([data[key]['hits_per_iog'] for key in data.keys()])
        sim_nhits_per_iog = np.array([sim[key]['hits_per_iog'] for key in sim.keys()])
        total_num_data_events = len(data_nhits)
        total_num_sim_events = len(sim_nhits)
        iog_colors = [
            (31/255, 119/255, 180/255, 0.1),
            (255/255, 127/255, 14/255, 0.1),
            (44/255, 160/255, 44/255, 0.1),
            (214/255, 39/255, 40/255, 0.1),
            (148/255, 103/255, 189/255, 0.1),
            (140/255, 86/255, 75/255, 0.1),
            (227/255, 119/255, 194/255, 0.1),
            (127/255, 127/255, 127/255, 0.1)
        ]
        iog_edgecolors = [
            (31/255, 119/255, 180/255, 0.9),
            (255/255, 127/255, 14/255, 0.9),
            (44/255, 160/255, 44/255, 0.9),
            (214/255, 39/255, 40/255, 0.9),
            (148/255, 103/255, 189/255, 0.9),
            (140/255, 86/255, 75/255, 0.9),
            (227/255, 119/255, 194/255, 0.9),
            (127/255, 127/255, 127/255, 0.9)
        ]
        iog_linestyles = ['-', '-', '--', '--', ':', ':', '-.',  '-.']

        # Create a figure with a grid of subplots
        fig = plt.figure(figsize=(20, 10))
        fig2 = plt.figure(figsize=(20, 8)) # Adjust the number of rows and columns as needed
        ax2_data = fig2.add_axes([0.05, 0.25, 0.4, 0.6])
        ax2_sim = fig2.add_axes([0.55, 0.25, 0.4, 0.6])
        ax2_data_r= fig2.add_axes([0.05, 0.1, 0.4, 0.15], sharex=ax2_data)
        ax2_sim_r = fig2.add_axes([0.55, 0.1, 0.4, 0.15], sharex=ax2_sim)

        max_nhits = math.ceil(max(np.max(data_nhits), np.max(sim_nhits))/10)*10
        # Loop over each iog and create the histograms
        for i in range(8):
            iog = i + 1
            residual_height= 0.08
            main_height = 0.32
            main_bottom = (1- (i//4)) * 0.5 + residual_height +0.06  # Adjust based on number of plots
            residual_bottom = main_bottom - residual_height  # Adjust space between plots

            main_left = (i % 4) / 4 + 0.045
            # Main plot
            ax_main = fig.add_axes([main_left, main_bottom, 0.2, main_height])

            # Extract data and simulation hits for the current iog
            data_nhits_iog = data_nhits_per_iog[:, i]
            sim_nhits_iog = sim_nhits_per_iog[:, i]


            # Determine histogram bins and ranges
            data_nhits_iog_counts, data_nhits_iog_bins = np.histogram(data_nhits_iog, bins=int(max_nhits / 50), range=(0, max_nhits))
            sim_nhits_iog_counts, sim_nhits_iog_bins = np.histogram(sim_nhits_iog, bins=int(max_nhits / 50), range=(0, max_nhits))

            # Plot histograms
            ax_main.hist(data_nhits_iog_bins[:-1], bins=data_nhits_iog_bins, weights=data_nhits_iog_counts / total_num_data_events,
                    histtype='stepfilled', label=data_sample_name, color=(0, 0, 1, 0.2), edgecolor=(0, 0, 1, 0.8), linestyle='-')
            ax_main.hist(sim_nhits_iog_bins[:-1], bins=sim_nhits_iog_bins, weights=sim_nhits_iog_counts / total_num_sim_events,
                    histtype='stepfilled', label=sim_sample_name, color=(1, 0, 0, 0.2), edgecolor=(1, 0, 0, 0.8), linestyle='--')

            # Set labels and scales
            ax_main.set_ylabel('Fraction of Total Events \nin Sample / 50 Hits')
            if log_scale==True: ax_main.set_yscale('log')
            ax_main.set_xscale('log')
            ax_main.set_xticks([])
            ax_main.set_xlabel('')
            ax_main.set_xlim(20, max_nhits)

            ax_main.legend()

            # Set the title for each subplot
            ax_main.set_title(f'IOG '+str(iog), weight='bold')

            # Residual plot
            ax_residual = fig.add_axes([main_left, residual_bottom, 0.2, residual_height], sharex=ax_main)

            # Calculate residuals
            residuals = 100*((data_nhits_iog_counts / total_num_data_events) - (sim_nhits_iog_counts / total_num_sim_events)) / (sim_nhits_iog_counts / total_num_sim_events)
            residuals = np.nan_to_num(residuals, nan=0, posinf=1000, neginf=-1000)

            # Plot residuals
            ax_residual.hist(data_nhits_iog_bins[:-1], bins=data_nhits_iog_bins, weights=residuals ,color='black', alpha=0.8)

            # Set labels
            ax_residual.set_xlabel('Number of Hits in IO Group '+str(i+1)+' per Event')
            ax_residual.set_ylabel('Residuals [%] \n(Data-Sim)/Sim')
            ax_residual.set_xscale('log')
            ax_residual.set_ylim(-100, 150)
            ax_residual.set_xlim(20, max_nhits)

                # Plot histograms
            ax2_data.hist(data_nhits_iog_bins[:-1], bins=data_nhits_iog_bins, weights=data_nhits_iog_counts / total_num_data_events,
                    histtype='stepfilled', label=f'IOG '+str(iog), color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])
            ax2_sim.hist(sim_nhits_iog_bins[:-1], bins=sim_nhits_iog_bins, weights=sim_nhits_iog_counts / total_num_sim_events,
                    histtype='stepfilled', label=f'IOG '+str(iog), color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])

            # Set labels and scales
            ax2_data.set_xlabel('Number of Hits per Event')
            ax2_data.set_ylabel('Fraction of Total Events in Sample / 50 Hits')
            if log_scale==True: ax2_data.set_yscale('log')
            ax2_data.set_xscale('log')
            ax2_data.legend()

            ax2_sim.set_xlabel('Number of Hits per Event')
            ax2_sim.set_ylabel('Fraction of Total Events in Sample / 50 Hits')
            if log_scale==True: ax2_sim.set_yscale('log')
            ax2_sim.set_xscale('log')
            ax2_sim.legend()

            # Set the title for each subplot
            ax2_data.set_title(data_sample_name, weight='bold')
            ax2_sim.set_title(sim_sample_name, weight='bold')

            # Calculate residuals
            residuals_data = 100*((data_nhits_iog_counts / total_num_data_events) - (data_nhits_counts / total_num_data_events)) / (data_nhits_counts / total_num_data_events)
            residuals_data = np.nan_to_num(residuals_data, nan=0, posinf=1000, neginf=-1000)

            # Plot residuals
            ax2_data_r.hist(data_nhits_iog_bins[:-1], bins=data_nhits_iog_bins, weights=residuals_data, histtype='stepfilled', \
                            color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])

            # Set labels
            ax2_data_r.set_xlabel('Number of Hits per Event')
            ax2_data_r.set_ylabel('Residuals [%] \n(IOG-Total)/Total')
            ax2_data_r.set_xscale('log')
            ax2_data_r.set_ylim(-100, 150)
            ax2_data_r.set_xlim(20, max_nhits)

            # Calculate residuals
            residuals_sim = 100*((sim_nhits_iog_counts / total_num_sim_events) - (sim_nhits_counts / total_num_sim_events)) / (sim_nhits_counts / total_num_sim_events)
            residuals_sim = np.nan_to_num(residuals_sim, nan=0, posinf=1000, neginf=-1000)

            # Plot residuals
            ax2_sim_r.hist(sim_nhits_iog_bins[:-1], bins=sim_nhits_iog_bins, weights=residuals_sim, histtype='stepfilled', \
                            color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])

            # Set labels
            ax2_sim_r.set_xlabel('Number of Hits per Event')
            ax2_sim_r.set_ylabel('Residuals [%] \n(IOG-Total)/Total')
            ax2_sim_r.set_xscale('log')
            ax2_sim_r.set_ylim(-100, 150)
            ax2_sim_r.set_xlim(20, max_nhits)
        output.savefig(fig)
        output.savefig(fig2)
        plt.close(fig)
        plt.close(fig2)


        # Plot max hit charge per iog
        data_max_hit_charge_per_iog = np.array([data[key]['max_hit_charge_per_iog'] for key in data.keys()])
        sim_max_hit_charge_per_iog = np.array([sim[key]['max_hit_charge_per_iog'] for key in sim.keys()])

        # Create a figure with a grid of subplots
        fig = plt.figure(figsize=(20, 10))
        fig2 = plt.figure(figsize=(20, 8)) # Adjust the number of rows and columns as needed
        ax2_data = fig2.add_axes([0.05, 0.25, 0.4, 0.6])
        ax2_sim = fig2.add_axes([0.55, 0.25, 0.4, 0.6])
        ax2_data_r= fig2.add_axes([0.05, 0.1, 0.4, 0.15], sharex=ax2_data)
        ax2_sim_r = fig2.add_axes([0.55, 0.1, 0.4, 0.15], sharex=ax2_sim)

        max_max_hit_charge = math.ceil(max(np.max(data_max_hit_charge), np.max(sim_max_hit_charge))/10)*10
        min_max_hit_charge = math.floor(min(np.min(data_max_hit_charge), np.min(sim_max_hit_charge))/10)*10
        # Loop over each iog and create the histograms
        for i in range(8):
            iog = i + 1
            residual_height= 0.08
            main_height = 0.32
            main_bottom = (1- (i//4)) * 0.5 + residual_height +0.06  # Adjust based on number of plots
            residual_bottom = main_bottom - residual_height  # Adjust space between plots

            main_left = (i % 4) / 4 + 0.045
            # Main plot
            ax_main = fig.add_axes([main_left, main_bottom, 0.2, main_height])

            # Extract data and simulation hits for the current iog
            data_max_hit_charge_iog = data_max_hit_charge_per_iog[:, i]
            sim_max_hit_charge_iog = sim_max_hit_charge_per_iog[:, i]

            # Determine histogram bins and ranges
            data_max_hit_charge_iog_counts, data_max_hit_charge_iog_bins = np.histogram(data_max_hit_charge_iog, bins=int((max_max_hit_charge-min_max_hit_charge) / 5), range=(min_max_hit_charge, max_max_hit_charge))
            sim_max_hit_charge_iog_counts, sim_max_hit_charge_iog_bins = np.histogram(sim_max_hit_charge_iog, bins=int((max_max_hit_charge-min_max_hit_charge) / 5), range=(min_max_hit_charge, max_max_hit_charge))

            # Plot histograms
            ax_main.hist(data_max_hit_charge_iog_bins[:-1], bins=data_max_hit_charge_iog_bins, weights=data_max_hit_charge_iog_counts / num_data_events_with_hits,
                    histtype='stepfilled', label=data_sample_name, color=(0, 0, 1, 0.2), edgecolor=(0, 0, 1, 0.8), linestyle='-')
            ax_main.hist(sim_max_hit_charge_iog_bins[:-1], bins=sim_max_hit_charge_iog_bins, weights=sim_max_hit_charge_iog_counts / num_sim_events_with_hits,
                    histtype='stepfilled', label=sim_sample_name, color=(1, 0, 0, 0.2), edgecolor=(1, 0, 0, 0.8), linestyle='--')

            # Set labels and scales
            ax_main.set_ylabel('Fraction of Total Events \nin Sample / 5 ke')
            if log_scale==True: ax_main.set_yscale('log')
            ax_main.set_xlim(min_max_hit_charge, max_max_hit_charge)

            ax_main.legend()

            # Set the title for each subplot
            ax_main.set_title(f'IOG '+str(iog), weight='bold')

            # Residual plot
            ax_residual = fig.add_axes([main_left, residual_bottom, 0.2, residual_height], sharex=ax_main)


            # Calculate residuals
            residuals = 100*((data_max_hit_charge_iog_counts / num_data_events_with_hits) - (sim_max_hit_charge_iog_counts / num_sim_events_with_hits)) / (sim_max_hit_charge_iog_counts / num_sim_events_with_hits)
            residuals = np.nan_to_num(residuals, nan=0, posinf=1000, neginf=-1000)

            # Plot residuals
            ax_residual.hist(data_max_hit_charge_iog_bins[:-1], bins=data_max_hit_charge_iog_bins, weights=residuals ,color='black', alpha=0.8)

            # Set labels
            ax_residual.set_xlabel('Max Hit Charge on IO Group '+str(i+1)+' \nper Event with at least one Hit [ke]')
            ax_residual.set_ylabel('Residuals [%] \n(Data-Sim)/Sim')
            #ax_residual.set_yscale('log')
            ax_residual.set_ylim(-100, 150)

                # Plot histograms
            ax2_data.hist(data_max_hit_charge_iog_bins[:-1], bins=data_max_hit_charge_iog_bins, weights=data_max_hit_charge_iog_counts / num_data_events_with_hits,
                    histtype='stepfilled', label=f'IOG '+str(iog), color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])
            ax2_sim.hist(sim_max_hit_charge_iog_bins[:-1], bins=sim_max_hit_charge_iog_bins, weights=sim_max_hit_charge_iog_counts / num_sim_events_with_hits,
                    histtype='stepfilled', label=f'IOG '+str(iog), color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])

            # Set labels and scales
            ax2_data.set_xlabel('Max Hit Charge per Event with at least one Hit [ke]')
            ax2_data.set_ylabel('Fraction of Total Events in Sample / 5 ke')
            if log_scale==True: ax2_data.set_yscale('log')
            ax2_data.legend()

            ax2_sim.set_xlabel('Max Hit Charge per Event with at least one Hit [ke]')
            ax2_sim.set_ylabel('Fraction of Total Events in Sample / 5 ke')
            if log_scale==True: ax2_sim.set_yscale('log')
            ax2_sim.legend()

            # Set the title for each subplot
            ax2_data.set_title(data_sample_name, weight='bold')
            ax2_sim.set_title(sim_sample_name, weight='bold')

            # Calculate residuals
            residuals_data = 100*((data_max_hit_charge_iog_counts / num_data_events_with_hits) - (data_max_hit_charge_counts / num_data_events_with_hits)) / (data_max_hit_charge_counts / num_data_events_with_hits)
            residuals_data = np.nan_to_num(residuals_data, nan=0, posinf=1000, neginf=-1000)

            # Plot residuals
            ax2_data_r.hist(data_max_hit_charge_iog_bins[:-1], bins=data_max_hit_charge_iog_bins, weights=residuals_data, histtype='stepfilled', \
                            color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])

            # Set labels
            ax2_data_r.set_xlabel('Max Hit Charge per Event with at least one Hit [ke]')
            ax2_data_r.set_ylabel('Residuals [%] \n(IOG-Total)/Total')
            ax2_data_r.set_ylim(-100, 150)
            ax2_data_r.set_xlim(min_max_hit_charge, max_max_hit_charge)

            # Calculate residuals
            residuals_sim = 100*((sim_max_hit_charge_iog_counts / num_sim_events_with_hits) - (sim_max_hit_charge_counts / num_sim_events_with_hits)) / (sim_max_hit_charge_counts / num_sim_events_with_hits)
            residuals_sim = np.nan_to_num(residuals_sim, nan=0, posinf=1000, neginf=-1000)

            # Plot residuals
            ax2_sim_r.hist(sim_max_hit_charge_iog_bins[:-1], bins=sim_max_hit_charge_iog_bins, weights=residuals_sim, histtype='stepfilled', \
                            color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])

            # Set labels
            ax2_sim_r.set_xlabel('Max Hit Charge per Event with at least one Hit [ke]')
            ax2_sim_r.set_ylabel('Residuals [%] \n(IOG-Total)/Total')
            ax2_sim_r.set_ylim(-100, 150)
            ax2_sim_r.set_xlim(min_max_hit_charge, max_max_hit_charge)
        output.savefig(fig)
        output.savefig(fig2)
        plt.close(fig)
        plt.close(fig2)

        data_min_hit_charge_per_iog = np.array([data[key]['min_hit_charge_per_iog'] for key in data.keys()])
        sim_min_hit_charge_per_iog = np.array([sim[key]['min_hit_charge_per_iog'] for key in sim.keys()])
        # Create a figure with a grid of subplots
        fig = plt.figure(figsize=(20, 10))
        fig2 = plt.figure(figsize=(20, 8)) # Adjust the number of rows and columns as needed
        ax2_data = fig2.add_axes([0.05, 0.25, 0.4, 0.6])
        ax2_sim = fig2.add_axes([0.55, 0.25, 0.4, 0.6])
        ax2_data_r= fig2.add_axes([0.05, 0.1, 0.4, 0.15], sharex=ax2_data)
        ax2_sim_r = fig2.add_axes([0.55, 0.1, 0.4, 0.15], sharex=ax2_sim)
        max_min_hit_charge = math.ceil(max(np.max(data_min_hit_charge), np.max(sim_min_hit_charge))/10)*10
        min_min_hit_charge = math.floor(min(np.min(data_min_hit_charge), np.min(sim_min_hit_charge))/10)*10
        # Loop over each iog and create the histograms
        for i in range(8):
            iog = i + 1
            residual_height= 0.08
            main_height = 0.32
            main_bottom = (1- (i//4)) * 0.5 + residual_height +0.06  # Adjust based on number of plots
            residual_bottom = main_bottom - residual_height  # Adjust space between plots
            main_left = (i % 4) / 4 + 0.045
            # Main plot
            ax_main = fig.add_axes([main_left, main_bottom, 0.2, main_height])
            # Extract data and simulation hits for the current iog
            data_min_hit_charge_iog = data_min_hit_charge_per_iog[:, i]
            sim_min_hit_charge_iog = sim_min_hit_charge_per_iog[:, i]
            # Determine histogram bins and ranges
            data_min_hit_charge_iog_counts, data_min_hit_charge_iog_bins = np.histogram(data_min_hit_charge_iog, bins=int((max_min_hit_charge-min_min_hit_charge) / 5), range=(min_min_hit_charge, max_min_hit_charge))
            sim_min_hit_charge_iog_counts, sim_min_hit_charge_iog_bins = np.histogram(sim_min_hit_charge_iog, bins=int((max_min_hit_charge-min_min_hit_charge) / 5), range=(min_min_hit_charge, max_min_hit_charge))
            # Plot histograms
            ax_main.hist(data_min_hit_charge_iog_bins[:-1], bins=data_min_hit_charge_iog_bins, weights=data_min_hit_charge_iog_counts / num_data_events_with_hits,
                    histtype='stepfilled', label=data_sample_name, color=(0, 0, 1, 0.2), edgecolor=(0, 0, 1, 0.8), linestyle='-')
            ax_main.hist(sim_min_hit_charge_iog_bins[:-1], bins=sim_min_hit_charge_iog_bins, weights=sim_min_hit_charge_iog_counts / num_sim_events_with_hits,
                    histtype='stepfilled', label=sim_sample_name, color=(1, 0, 0, 0.2), edgecolor=(1, 0, 0, 0.8), linestyle='--')
            # Set labels and scales
            ax_main.set_ylabel('Fraction of Total Events \nin Sample / 5 ke')
            if log_scale==True: ax_main.set_yscale('log')
            ax_main.set_xlim(min_min_hit_charge, max_min_hit_charge)
            ax_main.legend()
            # Set the title for each subplot
            ax_main.set_title(f'IOG '+str(iog), weight='bold')
            # Residual plot
            ax_residual = fig.add_axes([main_left, residual_bottom, 0.2, residual_height], sharex=ax_main)
            # Calculate residuals
            residuals = 100*((data_min_hit_charge_iog_counts / num_data_events_with_hits) - (sim_min_hit_charge_iog_counts / num_sim_events_with_hits)) / (sim_min_hit_charge_iog_counts / num_sim_events_with_hits)
            residuals = np.nan_to_num(residuals, nan=0, posinf=1000, neginf=-1000)
            # Plot residuals
            ax_residual.hist(data_min_hit_charge_iog_bins[:-1], bins=data_min_hit_charge_iog_bins, weights=residuals ,color='black', alpha=0.8)
            # Set labels
            ax_residual.set_xlabel('Min Hit Charge on IO Group '+str(i+1)+' \nper Event with at least one Hit [ke]')
            ax_residual.set_ylabel('Residuals [%] \n(Data-Sim)/Sim')
            #ax_residual.set_yscale('log')
            ax_residual.set_ylim(-100, 150)
                # Plot histograms
            ax2_data.hist(data_min_hit_charge_iog_bins[:-1], bins=data_min_hit_charge_iog_bins, weights=data_min_hit_charge_iog_counts / num_data_events_with_hits,
                    histtype='stepfilled', label=f'IOG '+str(iog), color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])
            ax2_sim.hist(sim_min_hit_charge_iog_bins[:-1], bins=sim_min_hit_charge_iog_bins, weights=sim_min_hit_charge_iog_counts / num_sim_events_with_hits,
                    histtype='stepfilled', label=f'IOG '+str(iog), color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])
            # Set labels and scales
            ax2_data.set_xlabel('Min Hit Charge per Event with at least one Hit [ke]')
            ax2_data.set_ylabel('Fraction of Total Events in Sample / 5 ke')
            if log_scale==True: ax2_data.set_yscale('log')
            ax2_data.legend()
            ax2_sim.set_xlabel('Min Hit Charge per Event with at least one Hit [ke]')
            ax2_sim.set_ylabel('Fraction of Total Events in Sample / 5 ke')
            if log_scale==True: ax2_sim.set_yscale('log')
            ax2_sim.legend()
            # Set the title for each subplot
            ax2_data.set_title(data_sample_name, weight='bold')
            ax2_sim.set_title(sim_sample_name, weight='bold')
            # Calculate residuals
            residuals_data = 100*((data_min_hit_charge_iog_counts / num_data_events_with_hits) - (data_min_hit_charge_counts / num_data_events_with_hits)) / (data_min_hit_charge_counts / num_data_events_with_hits)
            residuals_data = np.nan_to_num(residuals_data, nan=0, posinf=1000, neginf=-1000)
            # Plot residuals
            ax2_data_r.hist(data_min_hit_charge_iog_bins[:-1], bins=data_min_hit_charge_iog_bins, weights=residuals_data, histtype='stepfilled', \
                            color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])
            # Set labels
            ax2_data_r.set_xlabel('Min Hit Charge per Event with at least one Hit [ke]')
            ax2_data_r.set_ylabel('Residuals [%] \n(IOG-Total)/Total')
            ax2_data_r.set_ylim(-100, 150)
            ax2_data_r.set_xlim(min_min_hit_charge, max_min_hit_charge)
            # Calculate residuals
            residuals_sim = 100*((sim_min_hit_charge_iog_counts / num_sim_events_with_hits) - (sim_min_hit_charge_counts / num_sim_events_with_hits)) / (sim_min_hit_charge_counts / num_sim_events_with_hits)
            residuals_sim = np.nan_to_num(residuals_sim, nan=0, posinf=1000, neginf=-1000)
            # Plot residuals
            ax2_sim_r.hist(sim_min_hit_charge_iog_bins[:-1], bins=sim_min_hit_charge_iog_bins, weights=residuals_sim, histtype='stepfilled', \
                            color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])
            # Set labels
            ax2_sim_r.set_xlabel('Min Hit Charge per Event with at least one Hit [ke]')
            ax2_sim_r.set_ylabel('Residuals [%] \n(IOG-Total)/Total')
            ax2_sim_r.set_ylim(-100, 150)
            ax2_sim_r.set_xlim(min_min_hit_charge, max_min_hit_charge)
        output.savefig(fig)
        output.savefig(fig2)
        plt.close(fig)
        plt.close(fig2)

        data_mean_hit_charge_per_iog = np.array([data[key]['mean_hit_charge_per_iog'] for key in data.keys()])
        sim_mean_hit_charge_per_iog = np.array([sim[key]['mean_hit_charge_per_iog'] for key in sim.keys()])
        # Create a figure with a grid of subplots
        fig = plt.figure(figsize=(20, 10))
        fig2 = plt.figure(figsize=(20, 8)) # Adjust the number of rows and columns as needed
        ax2_data = fig2.add_axes([0.05, 0.25, 0.4, 0.6])
        ax2_sim = fig2.add_axes([0.55, 0.25, 0.4, 0.6])
        ax2_data_r= fig2.add_axes([0.05, 0.1, 0.4, 0.15], sharex=ax2_data)
        ax2_sim_r = fig2.add_axes([0.55, 0.1, 0.4, 0.15], sharex=ax2_sim)
        max_mean_hit_charge = math.ceil(max(np.max(data_mean_hit_charge), np.max(sim_mean_hit_charge))/10)*10
        min_mean_hit_charge = math.floor(min(np.min(data_mean_hit_charge), np.min(sim_mean_hit_charge))/10)*10
        # Loop over each iog and create the histograms
        for i in range(8):
            iog = i + 1
            residual_height= 0.08
            main_height = 0.32
            main_bottom = (1- (i//4)) * 0.5 + residual_height +0.06  # Adjust based on number of plots
            residual_bottom = main_bottom - residual_height  # Adjust space between plots
            main_left = (i % 4) / 4 + 0.045
            # Main plot
            ax_main = fig.add_axes([main_left, main_bottom, 0.2, main_height])
            # Extract data and simulation hits for the current iog
            data_mean_hit_charge_iog = data_mean_hit_charge_per_iog[:, i]
            sim_mean_hit_charge_iog = sim_mean_hit_charge_per_iog[:, i]
            # Determine histogram bins and ranges
            data_mean_hit_charge_iog_counts, data_mean_hit_charge_iog_bins = np.histogram(data_mean_hit_charge_iog, bins=int((max_mean_hit_charge-min_mean_hit_charge) / 5), range=(min_mean_hit_charge, max_mean_hit_charge))
            sim_mean_hit_charge_iog_counts, sim_mean_hit_charge_iog_bins = np.histogram(sim_mean_hit_charge_iog, bins=int((max_mean_hit_charge-min_mean_hit_charge) / 5), range=(min_mean_hit_charge, max_mean_hit_charge))
            # Plot histograms
            ax_main.hist(data_mean_hit_charge_iog_bins[:-1], bins=data_mean_hit_charge_iog_bins, weights=data_mean_hit_charge_iog_counts / num_data_events_with_hits,
                    histtype='stepfilled', label=data_sample_name, color=(0, 0, 1, 0.2), edgecolor=(0, 0, 1, 0.8), linestyle='-')
            ax_main.hist(sim_mean_hit_charge_iog_bins[:-1], bins=sim_mean_hit_charge_iog_bins, weights=sim_mean_hit_charge_iog_counts / num_sim_events_with_hits,
                    histtype='stepfilled', label=sim_sample_name, color=(1, 0, 0, 0.2), edgecolor=(1, 0, 0, 0.8), linestyle='--')
            # Set labels and scales
            ax_main.set_ylabel('Fraction of Total Events \nin Sample / 5 ke')
            if log_scale==True: ax_main.set_yscale('log')
            ax_main.set_xlim(min_mean_hit_charge, max_mean_hit_charge)
            ax_main.legend()
            # Set the title for each subplot
            ax_main.set_title(f'IOG '+str(iog), weight='bold')
            # Residual plot
            ax_residual = fig.add_axes([main_left, residual_bottom, 0.2, residual_height], sharex=ax_main)
            # Calculate residuals
            residuals = 100*((data_mean_hit_charge_iog_counts / num_data_events_with_hits) - (sim_mean_hit_charge_iog_counts / num_sim_events_with_hits)) / (sim_mean_hit_charge_iog_counts / num_sim_events_with_hits)
            residuals = np.nan_to_num(residuals, nan=0, posinf=1000, neginf=-1000)
            # Plot residuals
            ax_residual.hist(data_mean_hit_charge_iog_bins[:-1], bins=data_mean_hit_charge_iog_bins, weights=residuals ,color='black', alpha=0.8)
            # Set labels
            ax_residual.set_xlabel('Mean Hit Charge on IO Group '+str(i+1)+' \nper Event with at least one Hit [ke]')
            ax_residual.set_ylabel('Residuals [%] \n(Data-Sim)/Sim')
            #ax_residual.set_yscale('log')
            ax_residual.set_ylim(-100, 150)
                # Plot histograms
            ax2_data.hist(data_mean_hit_charge_iog_bins[:-1], bins=data_mean_hit_charge_iog_bins, weights=data_mean_hit_charge_iog_counts / num_data_events_with_hits,
                    histtype='stepfilled', label=f'IOG '+str(iog), color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])
            ax2_sim.hist(sim_mean_hit_charge_iog_bins[:-1], bins=sim_mean_hit_charge_iog_bins, weights=sim_mean_hit_charge_iog_counts / num_sim_events_with_hits,
                    histtype='stepfilled', label=f'IOG '+str(iog), color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])
            # Set labels and scales
            ax2_data.set_xlabel('Mean Hit Charge per Event with at least one Hit [ke]')
            ax2_data.set_ylabel('Fraction of Total Events in Sample / 5 ke')
            if log_scale==True: ax2_data.set_yscale('log')
            ax2_data.legend()
            ax2_sim.set_xlabel('Mean Hit Charge per Event with at least one Hit [ke]')
            ax2_sim.set_ylabel('Fraction of Total Events in Sample / 5 ke')
            if log_scale==True: ax2_sim.set_yscale('log')
            ax2_sim.legend()
            # Set the title for each subplot
            ax2_data.set_title(data_sample_name, weight='bold')
            ax2_sim.set_title(sim_sample_name, weight='bold')
            # Calculate residuals
            residuals_data = 100*((data_mean_hit_charge_iog_counts / num_data_events_with_hits) - (data_mean_hit_charge_counts / num_data_events_with_hits)) / (data_mean_hit_charge_counts / num_data_events_with_hits)
            residuals_data = np.nan_to_num(residuals_data, nan=0, posinf=1000, neginf=-1000)
            # Plot residuals
            ax2_data_r.hist(data_mean_hit_charge_iog_bins[:-1], bins=data_mean_hit_charge_iog_bins, weights=residuals_data, histtype='stepfilled', \
                            color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])
            # Set labels
            ax2_data_r.set_xlabel('Mean Hit Charge per Event with at least one Hit [ke]')
            ax2_data_r.set_ylabel('Residuals [%] \n(IOG-Total)/Total')
            ax2_data_r.set_ylim(-100, 150)
            ax2_data_r.set_xlim(min_mean_hit_charge, max_mean_hit_charge)
            # Calculate residuals
            residuals_sim = 100*((sim_mean_hit_charge_iog_counts / num_sim_events_with_hits) - (sim_mean_hit_charge_counts / num_sim_events_with_hits)) / (sim_mean_hit_charge_counts / num_sim_events_with_hits)
            residuals_sim = np.nan_to_num(residuals_sim, nan=0, posinf=1000, neginf=-1000)
            # Plot residuals
            ax2_sim_r.hist(sim_mean_hit_charge_iog_bins[:-1], bins=sim_mean_hit_charge_iog_bins, weights=residuals_sim, histtype='stepfilled', \
                            color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])
            # Set labels
            ax2_sim_r.set_xlabel('Mean Hit Charge per Event with at least one Hit [ke]')
            ax2_sim_r.set_ylabel('Residuals [%] \n(IOG-Total)/Total')
            ax2_sim_r.set_ylim(-100, 150)
            ax2_sim_r.set_xlim(min_mean_hit_charge, max_mean_hit_charge)
        output.savefig(fig)
        output.savefig(fig2)
        plt.close(fig)
        plt.close(fig2)

        data_std_hit_charge_per_iog = np.array([data[key]['std_hit_charge_per_iog'] for key in data.keys()])
        sim_std_hit_charge_per_iog = np.array([sim[key]['std_hit_charge_per_iog'] for key in sim.keys()])
        # Create a figure with a grid of subplots
        fig = plt.figure(figsize=(20, 10))
        fig2 = plt.figure(figsize=(20, 8)) # Adjust the number of rows and columns as needed
        ax2_data = fig2.add_axes([0.05, 0.25, 0.4, 0.6])
        ax2_sim = fig2.add_axes([0.55, 0.25, 0.4, 0.6])
        ax2_data_r= fig2.add_axes([0.05, 0.1, 0.4, 0.15], sharex=ax2_data)
        ax2_sim_r = fig2.add_axes([0.55, 0.1, 0.4, 0.15], sharex=ax2_sim)
        max_std_hit_charge = math.ceil(max(np.max(data_std_hit_charge), np.max(sim_std_hit_charge))/10)*10
        min_std_hit_charge = math.floor(min(np.min(data_std_hit_charge), np.min(sim_std_hit_charge))/10)*10
        # Loop over each iog and create the histograms
        for i in range(8):
            iog = i + 1
            residual_height= 0.08
            main_height = 0.32
            main_bottom = (1- (i//4)) * 0.5 + residual_height +0.06  # Adjust based on number of plots
            residual_bottom = main_bottom - residual_height  # Adjust space between plots
            main_left = (i % 4) / 4 + 0.045
            # Main plot
            ax_main = fig.add_axes([main_left, main_bottom, 0.2, main_height])
            # Extract data and simulation hits for the current iog
            data_std_hit_charge_iog = data_std_hit_charge_per_iog[:, i]
            sim_std_hit_charge_iog = sim_std_hit_charge_per_iog[:, i]
            # Determine histogram bins and ranges
            data_std_hit_charge_iog_counts, data_std_hit_charge_iog_bins = np.histogram(data_std_hit_charge_iog, bins=int((max_std_hit_charge-min_std_hit_charge) / 5), range=(min_std_hit_charge, max_std_hit_charge))
            sim_std_hit_charge_iog_counts, sim_std_hit_charge_iog_bins = np.histogram(sim_std_hit_charge_iog, bins=int((max_std_hit_charge-min_std_hit_charge) / 5), range=(min_std_hit_charge, max_std_hit_charge))
            # Plot histograms
            ax_main.hist(data_std_hit_charge_iog_bins[:-1], bins=data_std_hit_charge_iog_bins, weights=data_std_hit_charge_iog_counts / num_data_events_with_hits,
                    histtype='stepfilled', label=data_sample_name, color=(0, 0, 1, 0.2), edgecolor=(0, 0, 1, 0.8), linestyle='-')
            ax_main.hist(sim_std_hit_charge_iog_bins[:-1], bins=sim_std_hit_charge_iog_bins, weights=sim_std_hit_charge_iog_counts / num_sim_events_with_hits,
                    histtype='stepfilled', label=sim_sample_name, color=(1, 0, 0, 0.2), edgecolor=(1, 0, 0, 0.8), linestyle='--')
            # Set labels and scales
            ax_main.set_ylabel('Fraction of Total Events \nin Sample / 5 ke')
            if log_scale==True: ax_main.set_yscale('log')
            ax_main.set_xlim(min_std_hit_charge, max_std_hit_charge+5)
            ax_main.legend()
            # Set the title for each subplot
            ax_main.set_title(f'IOG '+str(iog), weight='bold')
            # Residual plot
            ax_residual = fig.add_axes([main_left, residual_bottom, 0.2, residual_height], sharex=ax_main)
            # Calculate residuals
            residuals = 100*((data_std_hit_charge_iog_counts / num_data_events_with_hits) - (sim_std_hit_charge_iog_counts / num_sim_events_with_hits)) / (sim_std_hit_charge_iog_counts / num_sim_events_with_hits)
            residuals = np.nan_to_num(residuals, nan=0, posinf=1000, neginf=-1000)
            # Plot residuals
            ax_residual.hist(data_std_hit_charge_iog_bins[:-1], bins=data_std_hit_charge_iog_bins, weights=residuals ,color='black', alpha=0.8)
            # Set labels
            ax_residual.set_xlabel('Standard Deviation of Hit Charge on IO Group '+str(i+1)+' \nper Event with at least one Hit [ke]')
            ax_residual.set_ylabel('Residuals [%] \n(Data-Sim)/Sim')
            #ax_residual.set_yscale('log')
            ax_residual.set_ylim(-100, 150)
                # Plot histograms
            ax2_data.hist(data_std_hit_charge_iog_bins[:-1], bins=data_std_hit_charge_iog_bins, weights=data_std_hit_charge_iog_counts / num_data_events_with_hits,
                    histtype='stepfilled', label=f'IOG '+str(iog), color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])
            ax2_sim.hist(sim_std_hit_charge_iog_bins[:-1], bins=sim_std_hit_charge_iog_bins, weights=sim_std_hit_charge_iog_counts / num_sim_events_with_hits,
                    histtype='stepfilled', label=f'IOG '+str(iog), color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])
            # Set labels and scales
            ax2_data.set_xlabel('Standard Deviation of Hit Charge per Event with at least one Hit [ke]')
            ax2_data.set_ylabel('Fraction of Total Events in Sample / 5 ke')
            if log_scale==True: ax2_data.set_yscale('log')
            ax2_data.legend()
            ax2_sim.set_xlabel('Standard Deviation of Hit Charge per Event with at least one Hit [ke]')
            ax2_sim.set_ylabel('Fraction of Total Events in Sample / 5 ke')
            if log_scale==True: ax2_sim.set_yscale('log')
            ax2_sim.legend()
            # Set the title for each subplot
            ax2_data.set_title(data_sample_name, weight='bold')
            ax2_sim.set_title(sim_sample_name, weight='bold')
            # Calculate residuals
            residuals_data = 100*((data_std_hit_charge_iog_counts / num_data_events_with_hits) - (data_std_hit_charge_counts / num_data_events_with_hits)) / (data_std_hit_charge_counts / num_data_events_with_hits)
            residuals_data = np.nan_to_num(residuals_data, nan=0, posinf=1000, neginf=-1000)
            # Plot residuals
            ax2_data_r.hist(data_std_hit_charge_iog_bins[:-1], bins=data_std_hit_charge_iog_bins, weights=residuals_data, histtype='stepfilled', \
                            color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])
            # Set labels
            ax2_data_r.set_xlabel('Standard Deviation of Hit Charge per Event with at least one Hit [ke]')
            ax2_data_r.set_ylabel('Residuals [%] \n(IOG-Total)/Total')
            ax2_data_r.set_ylim(-100, 150)
            ax2_data_r.set_xlim(min_std_hit_charge, max_std_hit_charge+5)
            # Calculate residuals
            residuals_sim = 100*((sim_std_hit_charge_iog_counts / num_sim_events_with_hits) - (sim_std_hit_charge_counts / num_sim_events_with_hits)) / (sim_std_hit_charge_counts / num_sim_events_with_hits)
            residuals_sim = np.nan_to_num(residuals_sim, nan=0, posinf=1000, neginf=-1000)
            # Plot residuals
            ax2_sim_r.hist(sim_std_hit_charge_iog_bins[:-1], bins=sim_std_hit_charge_iog_bins, weights=residuals_sim, histtype='stepfilled', \
                            color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])
            # Set labels
            ax2_sim_r.set_xlabel('Standard Deviation of Hit Charge per Event with at least one Hit [ke]')
            ax2_sim_r.set_ylabel('Residuals [%] \n(IOG-Total)/Total')
            ax2_sim_r.set_ylim(-100, 150)
            ax2_sim_r.set_xlim(min_std_hit_charge, max_std_hit_charge+5)
        output.savefig(fig)
        output.savefig(fig2)
        plt.close(fig)
        plt.close(fig2)


        data_max_hit_drift_time = np.array([data[key]['max_hit_drift_time']/10 for key in data.keys()])[data_nhits_mask]
        sim_max_hit_drift_time = np.array([sim[key]['max_hit_drift_time']/10 for key in sim.keys()])[sim_nhits_mask]
        max_max_hit_drift_time = 210
        min_max_hit_drift_time = -10
        data_max_hit_drift_time_counts, data_max_hit_drift_time_bins = np.histogram(data_max_hit_drift_time, \
                                                                            bins=int((max_max_hit_drift_time-min_max_hit_drift_time)/10), \
                                                                            range=(min_max_hit_drift_time,max_max_hit_drift_time))
        sim_max_hit_drift_time_counts, sim_max_hit_drift_time_bins = np.histogram(sim_max_hit_drift_time, \
                                                                            bins=int((max_max_hit_drift_time-min_max_hit_drift_time)/10), \
                                                                            range=(min_max_hit_drift_time,max_max_hit_drift_time))

        data_max_hit_drift_time_per_iog = np.array([data[key]['max_hit_drift_time_per_iog'] for key in data.keys()])/10
        sim_max_hit_drift_time_per_iog = np.array([sim[key]['max_hit_drift_time_per_iog'] for key in sim.keys()])/10
        # Create a figure with a grid of subplots
        fig = plt.figure(figsize=(20, 10))
        fig2 = plt.figure(figsize=(20, 8)) # Adjust the number of rows and columns as needed
        ax2_data = fig2.add_axes([0.05, 0.25, 0.4, 0.6])
        ax2_sim = fig2.add_axes([0.55, 0.25, 0.4, 0.6])
        ax2_data_r= fig2.add_axes([0.05, 0.1, 0.4, 0.15], sharex=ax2_data)
        ax2_sim_r = fig2.add_axes([0.55, 0.1, 0.4, 0.15], sharex=ax2_sim)
        # Loop over each iog and create the histograms
        for i in range(8):
            iog = i + 1
            residual_height= 0.08
            main_height = 0.32
            main_bottom = (1- (i//4)) * 0.5 + residual_height +0.06  # Adjust based on number of plots
            residual_bottom = main_bottom - residual_height  # Adjust space between plots
            main_left = (i % 4) / 4 + 0.045
            # Main plot
            ax_main = fig.add_axes([main_left, main_bottom, 0.2, main_height])
            # Extract data and simulation hits for the current iog
            data_max_hit_drift_time_iog = data_max_hit_drift_time_per_iog[:, i]
            sim_max_hit_drift_time_iog = sim_max_hit_drift_time_per_iog[:, i]
            # Determine histogram bins and ranges
            data_max_hit_drift_time_iog_counts, data_max_hit_drift_time_iog_bins = np.histogram(data_max_hit_drift_time_iog, bins=int((max_max_hit_drift_time-min_max_hit_drift_time) / 10), range=(min_max_hit_drift_time, max_max_hit_drift_time))
            sim_max_hit_drift_time_iog_counts, sim_max_hit_drift_time_iog_bins = np.histogram(sim_max_hit_drift_time_iog, bins=int((max_max_hit_drift_time-min_max_hit_drift_time) / 10), range=(min_max_hit_drift_time, max_max_hit_drift_time))
            # Plot histograms
            ax_main.hist(data_max_hit_drift_time_iog_bins[:-1], bins=data_max_hit_drift_time_iog_bins, weights=data_max_hit_drift_time_iog_counts / num_data_events_with_hits,
                    histtype='stepfilled', label=data_sample_name, color=(0, 0, 1, 0.2), edgecolor=(0, 0, 1, 0.8), linestyle='-')
            ax_main.hist(sim_max_hit_drift_time_iog_bins[:-1], bins=sim_max_hit_drift_time_iog_bins, weights=sim_max_hit_drift_time_iog_counts / num_sim_events_with_hits,
                    histtype='stepfilled', label=sim_sample_name, color=(1, 0, 0, 0.2), edgecolor=(1, 0, 0, 0.8), linestyle='--')
            # Set labels and scales
            ax_main.set_ylabel('Fraction of Total Events \nin Sample / 10 us')
            if log_scale==True: ax_main.set_yscale('log')
            ax_main.set_xlim(min_max_hit_drift_time, max_max_hit_drift_time+5)
            ax_main.legend()
            # Set the title for each subplot
            ax_main.set_title(f'IOG '+str(iog), weight='bold')
            # Residual plot
            ax_residual = fig.add_axes([main_left, residual_bottom, 0.2, residual_height], sharex=ax_main)
            # Calculate residuals
            residuals = 100*((data_max_hit_drift_time_iog_counts / num_data_events_with_hits) - (sim_max_hit_drift_time_iog_counts / num_sim_events_with_hits)) / (sim_max_hit_drift_time_iog_counts / num_sim_events_with_hits)
            residuals = np.nan_to_num(residuals, nan=0, posinf=1000, neginf=-1000)
            # Plot residuals
            ax_residual.hist(data_max_hit_drift_time_iog_bins[:-1], bins=data_max_hit_drift_time_iog_bins, weights=residuals ,color='black', alpha=0.8)
            # Set labels
            ax_residual.set_xlabel('Max Hit Drift Time on IO Group '+str(i+1)+' \nper Event with at least one Hit [us]')
            ax_residual.set_ylabel('Residuals [%] \n(Data-Sim)/Sim')
            #ax_residual.set_yscale('log')
            ax_residual.set_ylim(-100, 150)
                # Plot histograms
            ax2_data.hist(data_max_hit_drift_time_iog_bins[:-1], bins=data_max_hit_drift_time_iog_bins, weights=data_max_hit_drift_time_iog_counts / num_data_events_with_hits,
                    histtype='stepfilled', label=f'IOG '+str(iog), color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])
            ax2_sim.hist(sim_max_hit_drift_time_iog_bins[:-1], bins=sim_max_hit_drift_time_iog_bins, weights=sim_max_hit_drift_time_iog_counts / num_sim_events_with_hits,
                    histtype='stepfilled', label=f'IOG '+str(iog), color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])
            # Set labels and scales
            ax2_data.set_xlabel('Max Hit Drift Time per Event with at least one Hit [us]')
            ax2_data.set_ylabel('Fraction of Total Events in Sample / 10 us')
            if log_scale==True: ax2_data.set_yscale('log')
            ax2_data.legend()
            ax2_sim.set_xlabel('Max Hit Drift Time per Event with at least one Hit [us]')
            ax2_sim.set_ylabel('Fraction of Total Events in Sample / 10 us')
            if log_scale==True: ax2_sim.set_yscale('log')
            ax2_sim.legend()
            # Set the title for each subplot
            ax2_data.set_title(data_sample_name, weight='bold')
            ax2_sim.set_title(sim_sample_name, weight='bold')
            # Calculate residuals
            residuals_data = 100*((data_max_hit_drift_time_iog_counts / num_data_events_with_hits) - (data_max_hit_drift_time_counts / num_data_events_with_hits)) / (data_max_hit_drift_time_counts / num_data_events_with_hits)
            residuals_data = np.nan_to_num(residuals_data, nan=0, posinf=1000, neginf=-1000)
            # Plot residuals
            ax2_data_r.hist(data_max_hit_drift_time_iog_bins[:-1], bins=data_max_hit_drift_time_iog_bins, weights=residuals_data, histtype='stepfilled', \
                            color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])
            # Set labels
            ax2_data_r.set_xlabel('Max Hit Drift Time per Event with at least one Hit [us]')
            ax2_data_r.set_ylabel('Residuals [%] \n(IOG-Total)/Total')
            ax2_data_r.set_ylim(-100, 150)
            ax2_data_r.set_xlim(min_max_hit_drift_time, max_max_hit_drift_time+5)
            # Calculate residuals
            residuals_sim = 100*((sim_max_hit_drift_time_iog_counts / num_sim_events_with_hits) - (sim_max_hit_drift_time_counts / num_sim_events_with_hits)) / (sim_max_hit_drift_time_counts / num_sim_events_with_hits)
            residuals_sim = np.nan_to_num(residuals_sim, nan=0, posinf=1000, neginf=-1000)
            # Plot residuals
            ax2_sim_r.hist(sim_max_hit_drift_time_iog_bins[:-1], bins=sim_max_hit_drift_time_iog_bins, weights=residuals_sim, histtype='stepfilled', \
                            color=iog_colors[i], edgecolor=iog_edgecolors[i], linestyle=iog_linestyles[i])
            # Set labels
            ax2_sim_r.set_xlabel('Max Hit Drift Time per Event with at least one Hit [us]')
            ax2_sim_r.set_ylabel('Residuals [%] \n(IOG-Total)/Total')
            ax2_sim_r.set_ylim(-100, 150)
            ax2_sim_r.set_xlim(min_max_hit_drift_time, max_max_hit_drift_time+5)
        output.savefig(fig)
        output.savefig(fig2)
        plt.close(fig)
        plt.close(fig2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--data_dict', default=None,type=str,help='''string of data json dictionary file.''')
    parser.add_argument('-mc','--sim_dict', default=None,type=str,help='''string of simulation json dictionary file.''')
    parser.add_argument('-log','--log_scale', action='store_true', help='''bool telling whether or not to plot on log scale.''')
    args = parser.parse_args()
    main(**vars(args))