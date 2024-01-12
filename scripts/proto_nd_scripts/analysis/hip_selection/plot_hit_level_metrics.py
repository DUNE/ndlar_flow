################################################################################
##                                                                            ##
##    CONTAINS: Script to create plots describing data/MC metrics             ##
##              events using a dictionary                                     ##
##                                                                            ##
################################################################################

import matplotlib.pyplot as plt
import numpy as np

def plot_event_hit_summ_metrics(d, is_mc):

    if is_mc:
        mc_title = '[Simulation]'
        sample_type = "MC"
    else:
        mc_title = '[Data]'
        sample_type = "Data"

    # PLOT: total charge in an event
    fig0, ax0 = plt.subplots(figsize=(6,4))
    data0tot = np.array([d[key]['total_charge'] for key in d.keys()])
    counts0tot, bins0tot = np.histogram(data0tot, bins=np.linspace(0,20,21))
    ax0.hist(bins0tot[:-1], bins=bins0tot, weights = counts0tot)
    ax0.set_xlabel('Total Charge [ke-]')
    ax0.set_ylabel('Count / ke-')
    ax0.set_title(r'Total Charge Per Selected Event '+mc_title)
    plt.savefig(sample_type+"_selected_events_total_charge.png")
    plt.close(fig0)

    # PLOT: number of hits in an event
    fig1, ax1 = plt.subplots(figsize=(6,4))
    data1tot = np.array([d[key]['num_hits'] for key in d.keys()])
    counts1tot, bins1tot = np.histogram(data1tot, bins=np.linspace(50,5000,101))
    ax1.hist(bins1tot[:-1], bins=bins1tot, weights = counts1tot)
    ax1.set_xlabel('Number of Hits')
    ax1.set_ylabel('Event Count / 50 Hits')
    ax1.set_title(r'Number of Hits Per Selected Event '+mc_title)
    plt.savefig(sample_type+"_selected_events_total_hits_per_event.png")
    plt.close(fig1)


    # PLOT: number of separate pixels triggered in an event
    fig2, ax2 = plt.subplots(figsize=(6,4))
    data2tot = np.array([d[key]['num_channels'] for key in d.keys()])
    counts2tot, bins2tot = np.histogram(data2tot, bins=np.linspace(0,5000,251))
    ax2.hist(bins2tot[:-1], bins=bins2tot, weights = counts2tot)
    ax2.set_xlabel('Number of Unique Channels Triggered')
    ax2.set_ylabel('Event Count / 20 Channels')
    ax2.set_title("Number of Unique Channels Triggered \nPer Selected Event "+mc_title)
    plt.savefig(sample_type+"_selected_events_total_unique_channels_per_event.png")
    plt.close(fig2)

    return

def plot_channel_metrics(d, is_mc):

    if is_mc:
        mc_title = '[Simulation]'
        sample_type = "MC"
    else:
        mc_title = '[Data]'
        sample_type = "Data"

    # PLOT: hits per channel per event
    fig0, ax0 = plt.subplots(figsize=(6,4))
    data0tot = np.array([d[key]['hit_mult'] for key in d.keys()])
    counts0tot, bins0tot = np.histogram(data0tot, bins=np.linspace(0,20,21))
    ax0.hist(bins0tot[:-1], bins=bins0tot, weights = counts0tot)
    ax0.set_xlabel('Hit Multiplicity / Channel / Event')
    ax0.set_ylabel('Channel Count / Hit')
    ax0.set_title(r'Hit Multiplicity Per Channel in Selected Events '+mc_title)
    plt.savefig(sample_type+"_selected_events_hits_per_channel_per_event.png")
    plt.close(fig0)

    # PLOT: max hit amplitude per channel per event
    fig1, ax1 = plt.subplots(figsize=(6,4))
    data1tot = np.array([d[key]['max_hit_amp'] for key in d.keys()])
    counts1tot, bins1tot = np.histogram(data1tot, bins=np.linspace(0,500,26))
    ax1.hist(bins1tot[:-1], bins=bins1tot, weights = counts1tot)
    ax1.set_xlabel('Max Hit Amplitude / Channel / Event [ke-]')
    ax1.set_ylabel('Channel Count / 20 ke-')
    ax1.set_title(r'Maximum Hit Amplitiude Per Channel in Selected Events '+mc_title)
    plt.savefig(sample_type+"_selected_events_max_hit_amp_per_channel_per_event.png")
    plt.close(fig1)

    # PLOT: min hit amplitude per channel per event
    fig2, ax2 = plt.subplots(figsize=(6,4))
    data2tot = np.array([d[key]['min_hit_amp'] for key in d.keys()])
    counts2tot, bins2tot = np.histogram(data2tot, bins=np.linspace(0,500,26))
    ax2.hist(bins2tot[:-1], bins=bins2tot, weights = counts2tot)
    ax2.set_xlabel('Min Hit Amplitude / Channel / Event [ke-]')
    ax2.set_ylabel('Channel Count / 20 ke-')
    ax2.set_title(r'Minimum Hit Amplitiude Per Channel in Selected Events '+mc_title)
    plt.savefig(sample_type+"_selected_events_min_hit_amp_per_channel_per_event.png")
    plt.close(fig2)

    # PLOT: first hit amplitude per channel per event
    fig3, ax3 = plt.subplots(figsize=(6,4))
    data3tot = np.array([d[key]['first_hit_amp'] for key in d.keys()])
    counts3tot, bins3tot = np.histogram(data3tot, bins=np.linspace(0,500,26))
    ax3.hist(bins3tot[:-1], bins=bins3tot, weights = counts3tot)
    ax3.set_xlabel('First Hit Amplitude / Channel / Event [ke-]')
    ax3.set_ylabel('Channel Count / 20 ke-')
    ax3.set_title(r'First Hit Amplitiude Per Channel in Selected Events '+mc_title)
    plt.savefig(sample_type+"_selected_events_first_hit_amp_per_channel_per_event.png")
    plt.close(fig3)

    # PLOT: last hit amplitude per channel per event
    fig4, ax4 = plt.subplots(figsize=(6,4))
    data4tot = np.array([d[key]['last_hit_amp'] for key in d.keys()])
    counts4tot, bins4tot = np.histogram(data4tot, bins=np.linspace(0,500,26))
    ax4.hist(bins4tot[:-1], bins=bins4tot, weights = counts4tot)
    ax4.set_xlabel('Last Hit Amplitude / Channel / Event [ke-]')
    ax4.set_ylabel('Channel Count / 20 ke-')
    ax4.set_title(r'Last Hit Amplitiude Per Channel in Selected Events '+mc_title)
    plt.savefig(sample_type+"_selected_events_last_hit_amp_per_channel_per_event.png")
    plt.close(fig4)
    
    # PLOT: first/last hit delta(t) per channel per event
    fig4, ax4 = plt.subplots(figsize=(6,4))
    data4tot = np.array([d[key]['first_last_hit_delta_t'] for key in d.keys()])
    counts4tot, bins4tot = np.histogram(data4tot, bins=np.linspace(0,20,41))
    ax4.hist(bins4tot[:-1], bins=bins4tot, weights = counts4tot)
    ax4.set_xlabel(r'First/Last Hit $\Delta$t / Channel / Event [$\mu$s]')
    ax4.set_ylabel(r'Channel Count / 0.5 $\mu$s')
    ax4.set_title("Difference in Time between First and Last Hit\nPer Channel in Selected Events "+mc_title)
    plt.savefig(sample_type+"_selected_events_first_last_hit_deltat_per_channel_per_event.png")
    plt.close(fig4)

    return