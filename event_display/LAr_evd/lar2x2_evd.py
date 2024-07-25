# Import packages to check for and install missing packages
import sys
import subprocess

# Function to install missing packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure setuptools is installed to use pkg_resources
try:
    import pkg_resources
except ImportError:
    install('setuptools')
    import pkg_resources

# Ensure all non-standard packages are installed
required_packages = [
    'numpy', 'h5py', 'cmasher', 'IPython', 'PyMuPDF', 'matplotlib', 'pillow', 'uproot', 'h5flow', 'ipywidgets'
]

installed_packages = {pkg.key for pkg in pkg_resources.working_set}
missing_packages = [pkg for pkg in required_packages if pkg not in installed_packages]

if missing_packages:
    for package in missing_packages:
        install(package)

# Import modules
import warnings
import fitz
import numpy as np
from datetime import datetime
import ipywidgets as widgets
from io import BytesIO
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.proto_nd_flow.util.lut import LUT
from h5flow.core import resources
import itertools
import math
import h5py
import cmasher as cmr
from IPython.display import display, clear_output
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.image as mpimg
from PIL import Image
from math import fabs
import uproot


class LArEventDisplay:

    ''' 
        Class to set up interactive 2x2 display for files run through proto_nd_flow.

        Inputs are as follows:

            - filedir         (str): path to input file
            - filename        (str): name of flow file
            - nhits           (int): number of hits (threshold) for events to be made available (default: 1)
            - ntrigs          (int): number of external triggers (threshold)  threshold for events to be made available (default: 0)
            - show_light      (bool): whether to show light information in display (default: True)
            - public          (bool): whether to display for public i.e. include extra charge/light thresholds (default: False)
            - filepath_mx2    (str): path to Mx2 file if using Mx2 data (default: None)
            
        In order to run the display, set up a Jupyter Notebook, import everything in this file,
        and execute the run() method, e.g.:

        from lar_only_evd import *
        plt.ion()

        d = '/path/to/file/'
        f = 'filename'
        dune = '/path/to/dune_logo.png'
        twobytwo = '/path/to/2x2_logo.png'
        evd = LArEventDisplay(filedir=d, filename=f, dune_logo=dune, subexp_logo=twobytwo, nhits=1, show_light=False, public=False)
        evd.run()

        Alternatively, you can a display for a specific event by calling the display_event() method with the event number as an argument, e.g.:
        evd.display_event(0). This can also be done within another python script. 

    '''

    def __init__(self, filedir,filename, nhits=1, ntrigs=0, show_light=True, filepath_mx2=None, public=False):
        
        f = h5py.File(filedir+filename, 'r')
        if not filepath_mx2==None:
            f_mx2 = uproot.open(filepath_mx2)
            self.show_mx2 = True
        else:
            self.show_mx2 = False
        self.show_event_mx2 = self.show_mx2
        self.filename = filename
        self.filepath_mx2 = filepath_mx2
        self.show_light = show_light
        self.show_event_light = show_light
        self.public = public
        self.lar_evd_dir = os.path.dirname(__file__)
        
        dune_logo = os.path.join(self.lar_evd_dir, 'DUNElogo.pdf')
        self.dune_logo_pdf = fitz.open(dune_logo)#mpimg.imread(dune_logo)
        subexp_logo=os.path.join(self.lar_evd_dir, '2x2logo.png')
        self.subexp_logo = mpimg.imread(subexp_logo)

        # Resize DUNE logo image to fit in display
        dune_logo_page = self.dune_logo_pdf.load_page(0)
        dune_logo_pixmap = dune_logo_page.get_pixmap(matrix=fitz.Matrix(5, 5), dpi=600)
        dune_logo_image = Image.frombytes("RGB", [dune_logo_pixmap.width, dune_logo_pixmap.height], dune_logo_pixmap.samples)
        dune_logo_buf = BytesIO()
        dune_logo_image.save(dune_logo_buf, format='png')
        dune_logo_buf.seek(0)
        self.dune_logo_png = mpimg.imread(dune_logo_buf, format='png')

        # Load events dataset
        events = f['charge/events/data']
        self.events = events[events['nhit'] > nhits]
        self.events = self.events[self.events['n_ext_trigs'] >= ntrigs]


        # Load charge hits dataset
        self.hits_dset = 'calib_prompt_hits'
        self.hits_full = f['charge/'+self.hits_dset+'/data']
        self.hits_ref = f['charge/events/ref/charge/'+self.hits_dset+'/ref']
        self.hits_region = f['charge/events/ref/charge/'+self.hits_dset+'/ref_region']
        self.charge_threshold = 10.
        self.light_threshold = 150000#3e5
        if self.show_light:
            # Load light event and waveform datasets
            self.light_events = f['light/events/data']
            self.charge_light_ref = f['charge/events/ref/light/events/ref']
            self.charge_light_region = f['charge/events/ref/light/events/ref_region']
            self.light_wvfms = f['light/wvfm/data']
            self.light_event_wvfm_ref = f['light/events/ref']['light/wvfm']['ref']
            self.light_event_wvfm_region = f['light/events/ref']['light/wvfm']['ref_region']
            # ARTIFICIALLY ADDING LIGHT INFO:
            #self.light_events = lf['light/events/data']
            #self.light_wvfms = lf['light/wvfm/data']
            #self.light_event_wvfm_ref = lf['light/events/ref']['light/wvfm']['ref']
            #self.light_event_wvfm_region = lf['light/events/ref']['light/wvfm']['ref_region']
        
        # Load Mx2 data if using
        if self.show_mx2:
            self.minerva_hits_x_offset = f_mx2["minerva"]["offsetX"].array(library="np")
            self.minerva_hits_y_offset = f_mx2["minerva"]["offsetY"].array(library="np")
            self.minerva_hits_z_offset = f_mx2["minerva"]["offsetZ"].array(library="np")

            self.minerva_hits_x = f_mx2["minerva"]["trk_node_X"].array(library="np")
            self.minerva_hits_y = f_mx2["minerva"]["trk_node_Y"].array(library="np")
            self.minerva_hits_z = f_mx2["minerva"]["trk_node_Z"].array(library="np")

            self.minerva_trk_index = f_mx2["minerva"]["trk_index"].array(library="np")
            self.minerva_trk_nodes = f_mx2["minerva"]["trk_nodes"].array(library="np")
            self.minerva_trk_node_energy = f_mx2["minerva"]["clus_id_energy"].array(
                library="np"
            )

            self.minerva_times = (
                        f_mx2["minerva"]["ev_gps_time_sec"].array(library="np")
                        + f_mx2["minerva"]["ev_gps_time_usec"].array(library="np") / 1e6
                    )

        
        all_beam_triggers = []
        for ev_idx, iogroup in enumerate(f["charge/ext_trigs/data"]["iogroup"]):
            if iogroup == 5:
                all_beam_triggers.append(ev_idx)
        self.beam_triggers = all_beam_triggers

        # Load geometry and other info
        self.geometry = f['geometry_info']
        #print("Run info:", list(f['run_info'].dtype.names))
        #self.is_mc = f['run_info/is_mc/data']
        self.info = {
            'vdrift': f['lar_info'].attrs['v_drift'],
            'clock_period': 0.1,
        }
        if self.show_light:
          
            self.sipm_abs_pos = LUT.from_array(f["geometry_info/sipm_abs_pos"].attrs["meta"],f["geometry_info/sipm_abs_pos/data"])
            self.sipm_rel_pos = LUT.from_array(f["geometry_info/sipm_rel_pos"].attrs["meta"],f["geometry_info/sipm_rel_pos/data"])
            self.light_det_id = LUT.from_array(f["geometry_info/det_id"].attrs["meta"],f["geometry_info/det_id/data"])

            self.all_sipm_pos = f["geometry_info/sipm_abs_pos/data"]["data"][1:]
            self.sipm_unique_x = np.unique([pos[0] for pos in self.all_sipm_pos])
            self.sipm_unique_z = np.unique([pos[2] for pos in self.all_sipm_pos])
            self.sipm_unique_y = np.unique([pos[1] for pos in self.all_sipm_pos])
            # ARTIFICIALLY ADDING LIGHT INFO:
            #self.sipm_abs_pos = LUT.from_array(lf["geometry_info/sipm_abs_pos"].attrs["meta"],lf["geometry_info/sipm_abs_pos/data"])
            #self.sipm_rel_pos = LUT.from_array(lf["geometry_info/sipm_rel_pos"].attrs["meta"],lf["geometry_info/sipm_rel_pos/data"])
        #
            #self.all_sipm_pos = lf["geometry_info/sipm_abs_pos/data"]["data"][1:]
            #self.sipm_unique_x = np.unique([pos[0] for pos in self.all_sipm_pos])
            #self.sipm_unique_z = np.unique([pos[2] for pos in self.all_sipm_pos])
            #self.sipm_unique_y = np.unique([pos[1] for pos in self.all_sipm_pos])

        # Set up figure and subplots
        # NOTE: This is very different if Mx2 is shown
        if self.show_mx2: #TODO: go back to 8x8 array vs. 4x4 (whoops)
            
            if not self.public:
                self.fig = plt.figure(constrained_layout=False, figsize=(27, 27))
            else:
                self.fig = plt.figure(constrained_layout=False, figsize=(15, 15))
            self.axes_mosaic = [["ax_bd", "ax_bd",  "ax_subexp_logo", "ax_subexp_logo", "ax_bdv", "ax_bdv", "ax_bdv", "ax_bdv"],\
                                ["ax_bd", "ax_bd",  "ax_subexp_logo", "ax_subexp_logo", "ax_bdv", "ax_bdv", "ax_bdv", "ax_bdv"],\
                                ["ax_bv", "ax_bv", "ax_dv", "ax_dv", "ax_bdv", "ax_bdv", "ax_bdv", "ax_bdv"],\
                                ["ax_bv", "ax_bv", "ax_dv", "ax_dv", "ax_bdv", "ax_bdv", "ax_bdv", "ax_bdv"],\
                                ["ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2"],\
                                ["ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2"],\
                                ["ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2"],\
                                ["ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2"],\
                                ["ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2", "ax_mx2"]]
            self.axes_dict = self.fig.subplot_mosaic(self.axes_mosaic, \
                                                    per_subplot_kw={"ax_bdv": {"projection": "3d"}, 
                                                                    "ax_mx2": {"projection": "3d"}})
            self.ax_mx2 = self.axes_dict["ax_mx2"]
            if not self.public:
                cbar_ax = self.fig.add_axes([0.95, 0.025, 0.015, 0.88])
                self.fig.subplots_adjust(right=0.92)
                if self.show_light:
                    light_cbar_ax = self.fig.add_axes([0.125, 0.47, 0.8, 0.015])
            #else:
                #self.fig.subplots_adjust(right=0.7)
                #self.fig.subplots_adjust(top=0.8,bottom=0.4)
            self.fig.subplots_adjust(top=0.94,bottom=0.001)
            self.fig.subplots_adjust(wspace=0.02, hspace=0.02)
            current_mx2_pos = self.ax_mx2.get_position()
            #print("Current Mx2 position:", current_mx2_pos)
            #padding = 0.02
            if self.public:
                x0_shift = 0.13
                y0_shift = 0.04
            else: 
                x0_shift = 0.055
                y0_shift = 0.01
            new_mx2_pos = [current_mx2_pos.x0-x0_shift, current_mx2_pos.y0-y0_shift, \
                           current_mx2_pos.width*1.2, current_mx2_pos.height*1]
            self.ax_mx2.set_position(new_mx2_pos)

        else:
            self.fig = plt.figure(constrained_layout=False, figsize=(15, 8))
            self.axes_mosaic = [["ax_bd", "ax_subexp_logo", "ax_bdv", "ax_bdv"],["ax_bv", "ax_dv", "ax_bdv", "ax_bdv"],]
            self.axes_dict = self.fig.subplot_mosaic(self.axes_mosaic, \
                                                    per_subplot_kw={"ax_bdv": {"projection": "3d"}})
            if not self.public:
                cbar_ax = self.fig.add_axes([0.95, 0.12, 0.015, 0.75])
                self.fig.subplots_adjust(right=0.92)
                if self.show_light:
                    light_cbar_ax = self.fig.add_axes([0.15, 0.005, 0.75, 0.03])
            self.fig.subplots_adjust(bottom=0.1)
            self.fig.subplots_adjust(wspace=0.02, hspace=0.02)

        if self.show_mx2:
            ax_dune_logo = self.fig.add_axes([0.59, 0.942, 0.37, 0.057])
        else:
            ax_dune_logo = self.fig.add_axes([0.56, 0.895, 0.43, 0.099])
        self.ax_dune_logo = ax_dune_logo
        self.ax_bdv = self.axes_dict["ax_bdv"]
        self.ax_bd = self.axes_dict["ax_bd"]
        self.ax_bv = self.axes_dict["ax_bv"]
        self.ax_dv = self.axes_dict["ax_dv"]
        self.ax_subexp_logo = self.axes_dict["ax_subexp_logo"]
        if not self.public:
            self.cbar_ax = cbar_ax
            if self.show_light:
                self.light_cbar_ax = light_cbar_ax

        # Setup 3D view angles for GIFs
        self.base_angle = list(range(-180,180,2))
        self.azimuths = [ba for ba in self.base_angle]
        self.zeniths = [fabs(ba*0.25) for ba in self.base_angle]
        # Shift angles relative to improve perspectives
        self.offset = int(len(self.base_angle)/6)
        self.azimuths = self.azimuths[self.offset:] + self.azimuths[:self.offset]
        self.zeniths = self.zeniths[self.offset:] + self.zeniths[:self.offset]
        # Set view zoom
        self.zooms = [10,]*len(self.azimuths)

        # Create sliders for elevation and azimuthal angles # TO DO: FIX SLIDER WIDGET
        #self.elev_slider = widgets.FloatSlider(value=30, min=0, max=90, step=1, description='Elevation:')
        #self.azim_slider = widgets.FloatSlider(value=45, min=0, max=360, step=1, description='Azimuth:')
        

    ## Create the interactive widget # TO DO: FIX SLIDER WIDGET
    #def update_plot(self, elev, azim):
    #    self.ax_bdv.view_init(elev=elev, azim=azim)
    #    display(plt.gcf(), self.elev_slider, self.azim_slider)
    def save_to_pdf(self, ev_id):
        save_dir = self.lar_evd_dir
        filename = self.filename.split('.')[0]+'_Event_'+str(ev_id)+'.pdf'
        savepath = os.path.join(save_dir, filename)
        print("Saving to", savepath.split('.')[0]+'_final.pdf')
        self.ax_dune_logo.clear()
        self.ax_dune_logo.axis('off')
        self.fig.savefig(savepath, bbox_inches='tight')
        saved_pdf = fitz.open(savepath)
        saved_pdf_page = saved_pdf[0]
        rect_max_x = saved_pdf_page.rect[2]
        rect_max_y = saved_pdf_page.rect[3]
        include_dune_logo_rect = fitz.Rect(rect_max_x-300, 2, rect_max_x-5, 65)
        saved_pdf_page.show_pdf_page(include_dune_logo_rect, self.dune_logo_pdf, 0)
        saved_pdf.save(savepath.split('.')[0]+'_final.pdf')
        os.remove(savepath)
        
    def run(self):

        ## Link sliders to the update function
        #widgets.interactive(self.update_plot, elev=self.elev_slider, azim=self.azim_slider) # TO DO: FIX SLIDER WIDGET
  
        event_ids = [ev['id'] for ev in self.events]
        ev_idx = 0
        ev_id = event_ids[ev_idx]

        self.display_event(ev_id)
        # Displays event until user input determines next action
        # User can quit display (q), save current display to PDF (s), skip to next event (enter),
        # mkae a GIF of an event (g), or skip to a specific event number (type number)
        while True:

            #clear_output(wait=True)
            #self.display_event(ev_id)
            display(plt.gcf()) #, self.elev_slider, self.azim_slider) # TO DO: FIX SLIDER WIDGET
            # Display sliders and plot
            user_input = input(
                'Next event (q to exit/s to save to pdf/g to create gif/enter for next/number to skip to event)?\n')
            if not user_input:
                clear_output(wait=True)
                ev_idx += 1
                ev_id = event_ids[ev_idx]
                self.display_event(ev_id)
            elif user_input[0].lower() == 'q':
                sys.exit()
            elif user_input[0].lower() == 's':
                self.save_to_pdf(ev_id)
            elif user_input[0].lower() == 'g':
                print("Creating GIF of Event Display")
                # Loop over 3D views
                gif_dir = self.lar_evd_dir
                frame_num = 0
                for (azi,zen,zoom) in zip(self.azimuths,self.zeniths,self.zooms):
                    self.ax_bdv.view_init(zen, azi)   # see mpl_toolkits.mplot3d.axes3d.Axes3D.view_init
                    self.ax_bdv.dist = zoom
                    self.ax_bdv.set_box_aspect([1,1,1])
                    figname = gif_dir+'frame_%04d_%04d.png' % (ev_id, frame_num)
                    self.fig.savefig(figname)
                    frame_num += 1
                os.system("convert -delay 10 "+gif_dir+"frame*.png "+gif_dir+"animated_"+str(ev_id)+"no_axes.gif")
            else:
                try:
                    clear_output(wait=True)
                    ev_id = int(user_input)
                    ev_idx = event_ids.index(ev_id)
                    self.display_event(ev_id)
                except:
                    clear_output(wait=True)
                    print("Event number %s not valid" % user_input)
                    print("Proceeded to next available event instead")
                    ev_idx += 1
                    ev_id = event_ids[ev_idx]
                    self.display_event(ev_id)                 
            if ev_id >= event_ids[-1]:
                print("End of file")
                sys.exit()
    
    def clear_axes(self):

        self.ax_bdv.cla()
        self.ax_bd.cla()
        self.ax_bv.cla()
        self.ax_dv.cla()
        if not self.public:
            self.cbar_ax.cla()
            if self.show_light:
                self.light_cbar_ax.cla()
        if self.show_mx2:
            self.ax_mx2.cla()


    def get_event(self, ev_id):
        self.show_event_mx2 = self.show_mx2
        # Get event charge information
        ev_idx = np.where(self.events['id'] == ev_id)[0][0]
        print("Number of available events:", len(self.events))
        print("For fast-forwarding purposes, here is every 10th event number in your sample:", [ev for ev in self.events['id'][9::10]])

        #if not (ev_idx in self.beam_triggers):
        #    self.show_event_mx2 = False
        event = self.events[ev_idx]
        event_datetime = datetime.utcfromtimestamp(
                event['unix_ts']).strftime('%Y-%m-%d %H:%M:%S')
        # DEBUGGING TIMESTAMPS
        #print("Charge Unix TS:", event['unix_ts'])
        #print("Charge TS Start:", event['ts_start'])
        #print("Charge TS End:", event['ts_end'])
        print("Number of external triggers in this event:", event['n_ext_trigs'])
        hit_ref = self.hits_ref[self.hits_region[ev_id,'start']:self.hits_region[ev_id,'stop']]
        hit_ref = np.sort(hit_ref[hit_ref[:,0] == ev_id, 1])
        hits = self.hits_full[hit_ref]

        if self.show_light:
            # Get event light information
            self.show_event_light = True
            light_matches = self.charge_light_ref[self.charge_light_region[ev_id,'start']:self.charge_light_region[ev_id,'stop']]
            light_matches = np.sort(light_matches[light_matches[:,0] == ev_id, 1])

            # ARTIFICIALLY ADDING LIGHT INFO:
            #light_matches = self.light_events['id'] == 694
            light = self.light_events[light_matches]
            if len(light) == 0:
                print("No light information for event", ev_id)
                self.show_event_light = False

            if len(light) > 0:
                print("Light matches:", light_matches)
                print("Light unix timestamp:", light['utime_ms'])
                print("Light time since PPS:", light['tai_ns'])
                light_idx = light[0][0]

                light_wvfm_ref = self.light_event_wvfm_ref[self.light_event_wvfm_region[light_idx,'start']:self.light_event_wvfm_region[light_idx,'stop']]
                light_wvfm_ref = np.sort(light_wvfm_ref[light_wvfm_ref[:,0] == light_idx, 1])
                # Subtract pedestals for data:
                light_wvfm_get_peds = np.mean(self.light_wvfms[light_wvfm_ref]["samples"][:, :, :, 0:50], axis=-1)
                light_wvfm_peds_exp = np.expand_dims(light_wvfm_get_peds, axis=-1)
                light_wvfm_peds = light_wvfm_peds_exp * np.ones((1, 1, 1, 1000))
                light_wvfms = self.light_wvfms[light_wvfm_ref]["samples"] - light_wvfm_peds

        if self.show_event_mx2:
            
            charge_time = (event["unix_ts"] + event["ts_start"]/ 1e7)
            #print("Charge Time:", charge_time)
            #print("Minerva time 1:", self.minerva_times[0])
            #print("Minerva times:", len(self.minerva_times))
            # find the index of the minerva_times that matches the charge_time
            mx2_charge_time_diffs = np.abs(self.minerva_times - np.full_like(self.minerva_times, charge_time))
            trigger = np.argwhere(mx2_charge_time_diffs < 0.5).reshape(1,-1)[0]
            #print("Time Differences:", mx2_charge_time_diffs[trigger])
            #print("Trigger:", trigger)
            xs = []
            ys = []
            zs = []
            qs = []
            #print("All Minerva tracks:", self.minerva_trk_index[trigger][0])
            for trig in trigger:
                for idx in self.minerva_trk_index[trig]:
                    #print("Minerva time:", self.minerva_times[trig])
                    n_nodes = self.minerva_trk_nodes[trig][idx]
                    if n_nodes > 0:
                        x_nodes = (
                            self.minerva_hits_x[trig][idx][:n_nodes]
                            # - minerva_hits_x_offset[trig]
                        )
                        y_nodes = (
                            self.minerva_hits_y[trig][idx][:n_nodes]
                            # - minerva_hits_y_offset[trig]
                        )
                        z_nodes = self.minerva_hits_z[trig][idx][
                            :n_nodes
                        ]  # - minerva_hits_z_offset[trig]
                        q_nodes = self.minerva_trk_node_energy[trig][:n_nodes]
                    xs.extend((x_nodes / 10).tolist())
                    ys.extend((y_nodes / 10 - 21.8338).tolist())
                    zs.extend((z_nodes / 10 - 691.3).tolist())
                    qs.extend((q_nodes).tolist())
            mx2 = {'mx': xs, 'my':ys, 'mz':zs, 'mq':qs}
            if len(mx2['mq']) == 0:
                print("No matched Mx2 events for 2x2 Event ", ev_id)
                self.show_event_mx2 = False
            else:
                mx2_norm = mpl.colors.LogNorm(vmin=min(mx2['mq']),vmax=max(mx2['mq']))
                mx2_cmap = cmr.get_sub_cmap('cmr.torch_r', 0.13,0.95) # 0.03, 0.13 for torch_r
                #cmap_zero = cmr.get_sub_cmap('cmr.torch_r', 0.03, 0.95) #cosmic okay, toxic_r okay, sapphire_r okay (ember_r light?) emerald_r
                mmx2 = plt.cm.ScalarMappable(norm=mx2_norm, cmap=mx2_cmap)

        # Prepare color map for charge
        #print("Min charge:", min(hits['Q']), "Max charge:", max(hits['Q']))
        if self.public:
            min_charge = self.charge_threshold
        else:
            min_charge = min(hits['Q'])
        if max(hits['Q']) > min_charge:
            max_charge = max(hits['Q'])
        else: 
            max_charge = min_charge + 1
        charge_norm = mpl.colors.Normalize(vmin=min_charge,vmax=max_charge)
        cmap = cmr.get_sub_cmap('cmr.torch_r', 0.13,0.95) # 0.03, 0.13 for torch_r
        cmap_zero = cmr.get_sub_cmap('cmr.torch_r', 0.03, 0.95) #cosmic okay, toxic_r okay, sapphire_r okay (ember_r light?) emerald_r
        mcharge = plt.cm.ScalarMappable(norm=charge_norm, cmap=cmap)


        if self.show_event_light:
            # Prepare color map for light
            light_cmap=cmr.get_sub_cmap(cmr.sunburst_r, 0.0, 0.55) #'cmr.ember', 0.35,0.95) #.9 for torch .35 for ember .1 ember_r
            light_cmap_zero=cmr.get_sub_cmap(cmr.sunburst_r, 0.0, 0.55)#'cmr.ember', 0.05, 0.95)# .9 for torch .05 for ember 0.0 ember_r
            if self.public:
                min_light = self.light_threshold
            else:
                min_light = light_wvfms[0].sum(axis=-1).min()
            if light_wvfms[0].sum(axis=-1).max() > min_light:
                max_light = light_wvfms[0].sum(axis=-1).max()*2
            else:
                max_light = min_light*10
            light_norm = colors.LogNorm(min_light,max_light)
            mlight = plt.cm.ScalarMappable(norm=light_norm, cmap=light_cmap)

        # Set figure title
        if self.show_mx2:
            top_adjust=''
            bottom_adjust="\n\n\n\n\n\n\n\n\n\n\n\n"
        else:
            top_adjust="\n"
            bottom_adjust=""
        self.fig.suptitle(top_adjust+" Event %i: %s UTC" %
                          (ev_id, event_datetime)+bottom_adjust, x=0.05, size=28, weight='bold', ha='left', linespacing=0.15)
        
        if self.show_event_mx2 and self.show_event_light:
            return hits, mx2, light_wvfms, mcharge, mmx2, mlight, cmap, mx2_cmap, light_cmap, charge_norm, mx2_norm, light_norm, cmap_zero, light_cmap_zero
        elif self.show_event_mx2 and not self.show_event_light:
            return hits, mx2, mcharge, mmx2, cmap, mx2_cmap, charge_norm, mx2_norm, cmap_zero
        elif self.show_event_light and not self.show_event_mx2:
            return hits, light_wvfms, mcharge, mlight, cmap, light_cmap, charge_norm, light_norm, cmap_zero, light_cmap_zero
        else:
            return hits, mcharge, cmap, charge_norm, cmap_zero
    

    def set_axes(self, cmap, mcharge, cmap_zero, mlight=None):

        '''
            cmap: charge color map
            mcharge: charge color map scale
            mlight: light color map scale
        '''

        # Show 2x2 and DUNE logos

        self.ax_subexp_logo.axis('off')
        self.ax_subexp_logo.imshow(self.subexp_logo)
        self.ax_dune_logo.axis('off')
        self.ax_dune_logo.imshow(self.dune_logo_png)

        # Set axes for 3D canvas (Beam, Drift, Vertical)

        self.ax_bdv.set_xlabel('\nBeam Axis [cm]', fontsize=14, weight='bold', linespacing=2) #z
        self.ax_bdv.set_ylabel('\nDrift Axis [cm]', fontsize=14, weight='bold', linespacing=2) #x
        self.ax_bdv.set_zlabel('\nVertical Axis [cm]', fontsize=14, weight='bold', linespacing=2) #y
        self.ax_bdv.set_xlim(self.geometry.attrs['lar_detector_bounds'][0][2], \
            self.geometry.attrs['lar_detector_bounds'][1][2])
        self.ax_bdv.set_ylim(self.geometry.attrs['lar_detector_bounds'][0][0], \
            self.geometry.attrs['lar_detector_bounds'][1][0])
        self.ax_bdv.set_zlim(self.geometry.attrs['lar_detector_bounds'][0][1], \
            self.geometry.attrs['lar_detector_bounds'][1][1])
        self.ax_bdv.grid(False)
        self.ax_bdv.xaxis.pane.fill = True
        self.ax_bdv.yaxis.pane.fill = True
        self.ax_bdv.zaxis.pane.fill = True
        self.ax_bdv.xaxis.pane.set_facecolor('white')#(cmap_zero(0))
        self.ax_bdv.yaxis.pane.set_facecolor('white')#(cmap_zero(0))
        self.ax_bdv.zaxis.pane.set_facecolor('white')#(cmap_zero(0))
        self.ax_bdv.tick_params(axis='both', which='major', labelsize=12.5)
        self.ax_bdv.set_box_aspect([1,1,1], zoom=0.985)
        self.ax_bdv.view_init(azim=-75, elev=17)

        # Set axes for Beam vs Drift (ZX) canvas
        #self.ax_bd.set_xlabel('Beam Axis [cm]', fontsize=14)
        self.ax_bd.set_ylabel('Drift Axis [cm]', fontsize=14, weight='bold')
        self.ax_bd.set_ylim(self.geometry.attrs['lar_detector_bounds'][0][0]-2.95, \
            self.geometry.attrs['lar_detector_bounds'][1][0]+2.8)
        self.ax_bd.set_xlim(self.geometry.attrs['lar_detector_bounds'][0][2]-3.15,\
            self.geometry.attrs['lar_detector_bounds'][1][2]+3)
        self.ax_bd.tick_params(axis='y', which='major', labelsize=12.5)
        self.ax_bd.set_xticks([])
        self.ax_bd.set_yticks(np.arange(-60,61,20))

        # Set axes for Beam vs Vertical (ZY) canvas
        self.ax_bv.set_xlabel('Beam Axis [cm]', fontsize=14, weight='bold')
        self.ax_bv.set_ylabel('Vertical Axis [cm]', fontsize=14, weight='bold')
        self.ax_bv.set_xlim(self.geometry.attrs['lar_detector_bounds'][0][2]-3.15,\
            self.geometry.attrs['lar_detector_bounds'][1][2]+3)
        self.ax_bv.set_ylim(self.geometry.attrs['lar_detector_bounds'][0][1]-0.5, \
            self.geometry.attrs['lar_detector_bounds'][1][1]+0.5)
        self.ax_bv.tick_params(axis='both', which='major', labelsize=12.5)
        self.ax_bv.set_xticks(np.arange(-60,61,20))
        self.ax_bv.set_yticks(np.arange(-60,61,20))

        # Set axes for Drift vs Vertical (XY) canvas
        self.ax_dv.set_xlabel('Drift Axis [cm]', fontsize=14, weight='bold')
        #self.ax_dv.set_ylabel('Vertical Axis [cm]', fontsize=14)
        self.ax_dv.set_xlim(self.geometry.attrs['lar_detector_bounds'][0][0]-2.95,\
            self.geometry.attrs['lar_detector_bounds'][1][0]+2.8)
        self.ax_dv.set_ylim(self.geometry.attrs['lar_detector_bounds'][0][1]-0.5, \
            self.geometry.attrs['lar_detector_bounds'][1][1]+0.5)
        self.ax_dv.tick_params(axis='x', which='major', labelsize=12.5)
        self.ax_dv.set_xticks(np.arange(-60,61,20))
        self.ax_dv.set_yticks([])

        # Set Mx2 axis if using
        if self.show_mx2:
            self.ax_mx2.set_xlabel('\nBeam Axis [cm]', fontsize=14, weight='bold', linespacing=2) #z
            self.ax_mx2.set_ylabel('\nDrift Axis [cm]', fontsize=14, weight='bold', linespacing=2) #x
            self.ax_mx2.set_zlabel('\nVertical Axis [cm]', fontsize=14, weight='bold', linespacing=2) #y
            self.ax_mx2.tick_params(axis='both', which='major', labelsize=12.5)
            self.ax_mx2.set_xlim(self.geometry.attrs['lar_detector_bounds'][0][2] - 230, \
                self.geometry.attrs['lar_detector_bounds'][1][2] + 310) # beam
            self.ax_mx2.set_ylim(self.geometry.attrs['lar_detector_bounds'][0][2] - 110, \
                self.geometry.attrs['lar_detector_bounds'][1][2] + 120) # drift
            self.ax_mx2.set_zlim(self.geometry.attrs['lar_detector_bounds'][0][2] - 80, \
                self.geometry.attrs['lar_detector_bounds'][1][2] + 50) # vertical
            self.ax_mx2.grid(False)
            self.ax_mx2.xaxis.pane.fill = True
            self.ax_mx2.yaxis.pane.fill = True
            self.ax_mx2.zaxis.pane.fill = True
            self.ax_mx2.xaxis.pane.set_facecolor('white')#cmap_zero(0))
            self.ax_mx2.yaxis.pane.set_facecolor('white')#cmap_zero(0))
            self.ax_mx2.zaxis.pane.set_facecolor('white')#cmap_zero(0))
            if self.public:
                mx2_zoom = 1.35
            else:
                mx2_zoom = 1.38
            self.ax_mx2.set_box_aspect([7.75,3.5,3], zoom=mx2_zoom)
            self.ax_mx2.view_init(azim=-75, elev=17) # default -75, 17
            #self.ax_mx2.set_aspect('auto')

    
            # Plot Mx2
            x_base = [0, 108.0, 108.0, 0, -108.0, -108.0]
            shift = 245.0
            y_base = [
                -390.0 + shift,
                -330.0 + shift,
                -204.0 + shift,
                -145.0 + shift,
                -206.0 + shift,
                -330.0 + shift,
            ]

            z_base = {}
            z_base["ds"] = [164.0, 310.0]
            z_base["us"] = [-240.0, -190.0]
            for j in ["ds", "us"]:
                for i in range(len(x_base)):
                    self.ax_mx2.plot([z_base[j][0], z_base[j][0]], 
                                    [x_base[i], x_base[(i + 1) % len(x_base)]], 
                                    [y_base[i], y_base[(i + 1) % len(x_base)]], color="grey")

                    self.ax_mx2.plot([z_base[j][1], z_base[j][1]], 
                                    [x_base[i], x_base[(i + 1) % len(x_base)]], 
                                    [y_base[i], y_base[(i + 1) % len(x_base)]], color="grey")

                    self.ax_mx2.plot([z_base[j][0], z_base[j][1]], 
                                    [x_base[i], x_base[i]], 
                                    [y_base[i], y_base[i]], color="blue")

                    self.ax_mx2.plot([z_base[j][0], z_base[j][1]], 
                                    [x_base[i], x_base[i]], 
                                    [y_base[i], y_base[i]], color="grey")
                    
        for i in range(len(self.geometry.attrs['module_RO_bounds'])):

            # Plot cathodes for XYZ (beam, drift, vertical) 3D view:
            X_cathode, Y_cathode, Z_cathode = make_x_plane(self.geometry.attrs['module_RO_bounds'][i][0][1], self.geometry.attrs['module_RO_bounds'][i][1][1], \
                                                           self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2], 
                                                           self.geometry.attrs['module_RO_bounds'][i][0][0]+self.geometry.attrs['max_drift_distance']+self.geometry.attrs['cathode_thickness']/2)
            self.ax_bdv.plot_surface(Z_cathode,X_cathode,Y_cathode, color='gainsboro', alpha=0.1)
            if self.show_mx2:
                self.ax_mx2.plot_surface(Z_cathode,X_cathode,Y_cathode, color='gainsboro', alpha=0.1)
            
            for j in range(2):
                for k in range(2):
                    # Plot outlines of modules for XYZ (beam, drift, vertical) 3D view:
                    self.ax_bdv.plot([self.geometry.attrs['module_RO_bounds'][i][j][2], self.geometry.attrs['module_RO_bounds'][i][j][2]], \
                            [self.geometry.attrs['module_RO_bounds'][i][0][0], self.geometry.attrs['module_RO_bounds'][i][1][0]], \
                            [self.geometry.attrs['module_RO_bounds'][i][k][1], self.geometry.attrs['module_RO_bounds'][i][k][1]], color='black', alpha=0.35)

                    self.ax_bdv.plot([self.geometry.attrs['module_RO_bounds'][i][j][2], self.geometry.attrs['module_RO_bounds'][i][j][2]], \
                            [self.geometry.attrs['module_RO_bounds'][i][k][0], self.geometry.attrs['module_RO_bounds'][i][k][0]], \
                            [self.geometry.attrs['module_RO_bounds'][i][0][1], self.geometry.attrs['module_RO_bounds'][i][1][1]], color='black', alpha=0.35)

                    self.ax_bdv.plot([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                            [self.geometry.attrs['module_RO_bounds'][i][j][0], self.geometry.attrs['module_RO_bounds'][i][j][0]], \
                            [self.geometry.attrs['module_RO_bounds'][i][k][1], self.geometry.attrs['module_RO_bounds'][i][k][1]], color='black', alpha=0.35)
                    if self.show_mx2:
                        # Plot outlines of modules for XYZ (beam, drift, vertical) 3D view WITH Mx2:
                        self.ax_mx2.plot([self.geometry.attrs['module_RO_bounds'][i][j][2], self.geometry.attrs['module_RO_bounds'][i][j][2]], \
                                [self.geometry.attrs['module_RO_bounds'][i][0][0], self.geometry.attrs['module_RO_bounds'][i][1][0]], \
                                [self.geometry.attrs['module_RO_bounds'][i][k][1], self.geometry.attrs['module_RO_bounds'][i][k][1]], color='black', alpha=0.35)

                        self.ax_mx2.plot([self.geometry.attrs['module_RO_bounds'][i][j][2], self.geometry.attrs['module_RO_bounds'][i][j][2]], \
                                [self.geometry.attrs['module_RO_bounds'][i][k][0], self.geometry.attrs['module_RO_bounds'][i][k][0]], \
                                [self.geometry.attrs['module_RO_bounds'][i][0][1], self.geometry.attrs['module_RO_bounds'][i][1][1]], color='black', alpha=0.35)

                        self.ax_mx2.plot([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                                [self.geometry.attrs['module_RO_bounds'][i][j][0], self.geometry.attrs['module_RO_bounds'][i][j][0]], \
                                [self.geometry.attrs['module_RO_bounds'][i][k][1], self.geometry.attrs['module_RO_bounds'][i][k][1]], color='black', alpha=0.35)

                # Plot outlines of modules for ZX (beam, drift) projections:
                self.ax_bd.plot([self.geometry.attrs['module_RO_bounds'][i][j][2], self.geometry.attrs['module_RO_bounds'][i][j][2]], \
                         [self.geometry.attrs['module_RO_bounds'][i][0][0], self.geometry.attrs['module_RO_bounds'][i][1][0]],\
                         color=cmap_zero(0), alpha=1)
                self.ax_bd.plot([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                         [self.geometry.attrs['module_RO_bounds'][i][j][0], self.geometry.attrs['module_RO_bounds'][i][j][0]],\
                         color=cmap_zero(0), alpha=1)

                # Only two modules represented in ZY and XY projections
                if i >= 2: continue

                # Plot outlines of modules for ZY (beam, vertical) projections:
                self.ax_bv.plot([self.geometry.attrs['module_RO_bounds'][i][j][2], self.geometry.attrs['module_RO_bounds'][i][j][2]], \
                         [self.geometry.attrs['module_RO_bounds'][i][0][1], self.geometry.attrs['module_RO_bounds'][i][1][1]],\
                         color=cmap_zero(0), alpha=1)
                self.ax_bv.plot([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                         [self.geometry.attrs['module_RO_bounds'][i][j][1], self.geometry.attrs['module_RO_bounds'][i][j][1]],\
                         color=cmap_zero(0), alpha=1)

                # Plot outlines of modules for XY (drift, vertical) projections:
                # X footprints are the same for i=0 and i=1, but different for i=1 and i=2
                self.ax_dv.plot([self.geometry.attrs['module_RO_bounds'][i+1][j][0], self.geometry.attrs['module_RO_bounds'][i+1][j][0]], \
                         [self.geometry.attrs['module_RO_bounds'][i+1][0][1], self.geometry.attrs['module_RO_bounds'][i+1][1][1]],\
                         color=cmap_zero(0), alpha=1)
                self.ax_dv.plot([self.geometry.attrs['module_RO_bounds'][i+1][0][0], self.geometry.attrs['module_RO_bounds'][i+1][1][0]], \
                         [self.geometry.attrs['module_RO_bounds'][i+1][j][1], self.geometry.attrs['module_RO_bounds'][i+1][j][1]],\
                         color=cmap_zero(0), alpha=1)


            # Fill modules for ZX (beam, drift) projections:
            self.ax_bd.fill([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][0][2], \
                      self.geometry.attrs['module_RO_bounds'][i][1][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                     [self.geometry.attrs['module_RO_bounds'][i][0][0], self.geometry.attrs['module_RO_bounds'][i][1][0], \
                      self.geometry.attrs['module_RO_bounds'][i][1][0], self.geometry.attrs['module_RO_bounds'][i][0][0]],\
                     color=cmap_zero(0), alpha=0.85)

            # Only two modules represented in ZY and XY projections
            if i >= 2: continue

            # Fill modules for ZY (beam, vertical) projections:
            self.ax_bv.fill([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][0][2], \
                      self.geometry.attrs['module_RO_bounds'][i][1][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                     [self.geometry.attrs['module_RO_bounds'][i][0][1], self.geometry.attrs['module_RO_bounds'][i][1][1], \
                      self.geometry.attrs['module_RO_bounds'][i][1][1], self.geometry.attrs['module_RO_bounds'][i][0][1]],\
                     color=cmap_zero(0), alpha=0.85)

            # Fill modules for XY (drift, vertical) projections:
            self.ax_dv.fill([self.geometry.attrs['module_RO_bounds'][i+1][0][0], self.geometry.attrs['module_RO_bounds'][i+1][0][0], \
                      self.geometry.attrs['module_RO_bounds'][i+1][1][0], self.geometry.attrs['module_RO_bounds'][i+1][1][0]], \
                     [self.geometry.attrs['module_RO_bounds'][i+1][0][1], self.geometry.attrs['module_RO_bounds'][i+1][1][1], \
                      self.geometry.attrs['module_RO_bounds'][i+1][1][1], self.geometry.attrs['module_RO_bounds'][i+1][0][1]],\
                     color=cmap_zero(0), alpha=0.85)

            # Plot cathodes for XY (drift, vertical) projections:
            self.ax_dv.plot([self.geometry.attrs['module_RO_bounds'][i+1][0][0]+self.geometry.attrs['max_drift_distance']+self.geometry.attrs['cathode_thickness']/2, \
                      self.geometry.attrs['module_RO_bounds'][i+1][0][0]+self.geometry.attrs['max_drift_distance']+self.geometry.attrs['cathode_thickness']/2],\
                     [self.geometry.attrs['module_RO_bounds'][i+1][0][1], self.geometry.attrs['module_RO_bounds'][i+1][1][1]], \
                      color='gainsboro', alpha=0.9)

            if not self.show_event_light:

                # Plot cathodes for ZX (beam, drift) projections:
                for i in range(len(self.geometry.attrs['module_RO_bounds'])):
                    self.ax_bd.plot([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                             [self.geometry.attrs['module_RO_bounds'][i][0][0]+self.geometry.attrs['max_drift_distance']+self.geometry.attrs['cathode_thickness']/2, \
                              self.geometry.attrs['module_RO_bounds'][i][0][0]+self.geometry.attrs['max_drift_distance']+self.geometry.attrs['cathode_thickness']/2],\
                              color='gainsboro', alpha=0.9, linewidth=2,solid_capstyle='butt')
        if not self.public:
            # Set charge colorbar
            cbar = self.fig.colorbar(mcharge, cax=self.cbar_ax, label=r'Charge [$10^3$ e]')
            cbar.set_label(r'Charge [$\mathbf{10^3}$ e]', size=20, weight='bold')
            self.cbar_ax.tick_params(labelsize=18)

            if self.show_event_light:
                # Set light colorbar
                light_cbar = self.fig.colorbar(mlight, cax=self.light_cbar_ax, label=r'Light [ADC Counts]', orientation = 'horizontal')
                light_cbar.set_label(r'Light [ADC Counts]', size=20, weight='bold')
                self.light_cbar_ax.tick_params(labelsize=18)


    def display_event(self, ev_id):

        self.clear_axes()
        hits, *event_info = self.get_event(ev_id)
        if self.show_event_mx2 and self.show_event_light:
            mx2, light_wvfms, mcharge, mmx2, mlight, cmap, mx2_cmap, light_cmap, charge_norm, mx2_norm, light_norm, cmap_zero, light_cmap_zero = event_info
        elif self.show_event_mx2 and not self.show_event_light:
            mx2, mcharge, mmx2, cmap, mx2_cmap, charge_norm, mx2_norm, cmap_zero = event_info
        elif self.show_event_light and not self.show_event_mx2:
            light_wvfms, mcharge, mlight, cmap, light_cmap, charge_norm, light_norm, cmap_zero, light_cmap_zero = event_info
        else:
            mcharge, cmap, charge_norm, cmap_zero = event_info

        ev_idx = np.argwhere(self.events['id'] == ev_id)[0][0]

        # DEBUGGING:
        if ev_idx in self.beam_triggers:
            print("Event " + str(ev_id) + " is a beam trigger event")
        else:
            print("Event " + str(ev_id) + " is NOT a beam trigger event")

        if self.public:
            hits = hits[hits['Q'] > self.charge_threshold]
        # Plot hits in 3D view first so that cathodes/anodes go over the hits
        self.ax_bdv.scatter(hits['z'], hits['x'], hits['y'], lw=0, ec='C0', \
                            c=cmap(charge_norm(hits['Q'])), s=0.5, alpha=1, marker="s")

        #self.ax_bdv.scatter(hits[hits['io_group'] == 1]['z'], hits[hits['io_group'] == 1]['x'], hits[hits['io_group'] == 1]['y'],\
        #                    lw=0, s=5, alpha=0.9)
        #self.ax_bdv.scatter(hits[hits['io_group'] == 2]['z'], hits[hits['io_group'] == 2]['x'], hits[hits['io_group'] == 2]['y'],\
        #            lw=0, s=5, alpha=0.9)
        #self.ax_bdv.scatter(hits[hits['io_group'] == 3]['z'], hits[hits['io_group'] == 3]['x'], hits[hits['io_group'] == 3]['y'],\
        #            lw=0, s=5, alpha=0.9)
        #self.ax_bdv.scatter(hits[hits['io_group'] == 4]['z'], hits[hits['io_group'] == 4]['x'], hits[hits['io_group'] == 4]['y'],\
        #                    lw=0, s=5, alpha=0.9)
        #self.ax_bdv.scatter(hits[hits['io_group'] == 5]['z'], hits[hits['io_group'] == 5]['x'], hits[hits['io_group'] == 5]['y'],\
        #                    lw=0, s=5, alpha=0.9)
        #self.ax_bdv.scatter(hits[hits['io_group'] == 6]['z'], hits[hits['io_group'] == 6]['x'], hits[hits['io_group'] == 6]['y'],\
        #                    lw=0, s=5, alpha=0.9)
        #self.ax_bdv.scatter(hits[hits['io_group'] == 7]['z'], hits[hits['io_group'] == 7]['x'], hits[hits['io_group'] == 7]['y'],\
        #                    lw=0, s=5, alpha=0.9)
        #self.ax_bdv.scatter(hits[hits['io_group'] == 8]['z'], hits[hits['io_group'] == 8]['x'], hits[hits['io_group'] == 8]['y'],\
        #                    lw=0, s=5, alpha=0.9)



        if self.show_event_mx2:
            self.ax_mx2.scatter(hits['z'], hits['x'], hits['y'], lw=0, ec='C0', \
                    c=cmap(charge_norm(hits['Q'])), s=0.5, alpha=1, marker="s")
            #self.ax_mx2.scatter(mx2['mz'], mx2['mx'], mx2['my'], lw=0, ec='C0', \
            #                c=mx2_cmap(mx2_norm(mx2['mq'])), s=15, alpha=1)
            self.ax_mx2.scatter(mx2['mz'], mx2['mx'], mx2['my'], lw=0, ec='C0', \
                            c='red', s=4, alpha=1)            
        if self.show_event_light:
            self.set_axes(cmap, mcharge, cmap_zero, mlight)
        else:
            self.set_axes(cmap, mcharge, cmap_zero)

        if self.show_event_light:
            self.plot_light(light_wvfms, light_cmap, light_norm, light_cmap_zero)

        # Plot 2D charge hits
        self.ax_bd.scatter(hits['z'], hits['x'], lw=0, ec='C0', c=cmap(
                charge_norm(hits['Q'])), s=0.5, alpha=1, marker="s")
        self.ax_bv.scatter(hits['z'], hits['y'], lw=0, ec='C0', c=cmap(
                charge_norm(hits['Q'])), s=0.5, alpha=1, marker="s")
        self.ax_dv.scatter(hits['x'], hits['y'], lw=0, ec='C0', c=cmap(
                charge_norm(hits['Q'])), s=0.5, alpha=1, marker="s")
        # Lines below are for geometry debugging purposes
        #self.ax_bd.scatter(hits[hits['io_group'] == 8]['z'], hits[hits['io_group'] == 8]['x'], lw=0, ec='C0', c=cmap(
        #        charge_norm(hits[hits['io_group'] == 8]['Q'])), s=5, alpha=1)
        

        #self.ax_bd.scatter(hits[hits['io_group'] == 1]['z'], hits[hits['io_group'] == 1]['x'],\
        #                    lw=0, s=5, alpha=0.9)
        #self.ax_bd.scatter(hits[hits['io_group'] == 2]['z'], hits[hits['io_group'] == 2]['x'],\
        #            lw=0, s=5, alpha=0.9)
        #self.ax_bd.scatter(hits[hits['io_group'] == 3]['z'], hits[hits['io_group'] == 3]['x'],\
        #            lw=0, s=5, alpha=0.9)
        #self.ax_bd.scatter(hits[hits['io_group'] == 4]['z'], hits[hits['io_group'] == 4]['x'],\
        #                    lw=0, s=5, alpha=0.9)
        #self.ax_bd.scatter(hits[hits['io_group'] == 5]['z'], hits[hits['io_group'] == 5]['x'],\
        #                    lw=0, s=5, alpha=0.9)
        #self.ax_bd.scatter(hits[hits['io_group'] == 6]['z'], hits[hits['io_group'] == 6]['x'],\
        #                    lw=0, s=5, alpha=0.9)
        #self.ax_bd.scatter(hits[hits['io_group'] == 7]['z'], hits[hits['io_group'] == 7]['x'],\
        #                    lw=0, s=5, alpha=0.9)
        #self.ax_bd.scatter(hits[hits['io_group'] == 8]['z'], hits[hits['io_group'] == 8]['x'],\
        #                    lw=0, s=5, alpha=0.9)
#
        #self.ax_bv.scatter(hits[hits['io_group'] == 1]['z'], hits[hits['io_group'] == 1]['y'],\
        #                    lw=0, s=5, alpha=0.9)
        #self.ax_bv.scatter(hits[hits['io_group'] == 2]['z'], hits[hits['io_group'] == 2]['y'],\
        #            lw=0, s=5, alpha=0.9)
        #self.ax_bv.scatter(hits[hits['io_group'] == 3]['z'], hits[hits['io_group'] == 3]['y'],\
        #            lw=0, s=5, alpha=0.9)
        #self.ax_bv.scatter(hits[hits['io_group'] == 4]['z'], hits[hits['io_group'] == 4]['y'],\
        #                    lw=0, s=5, alpha=0.9)
        #self.ax_bv.scatter(hits[hits['io_group'] == 5]['z'], hits[hits['io_group'] == 5]['y'],\
        #                    lw=0, s=5, alpha=0.9)
        #self.ax_bv.scatter(hits[hits['io_group'] == 6]['z'], hits[hits['io_group'] == 6]['y'],\
        #                    lw=0, s=5, alpha=0.9)
        #self.ax_bv.scatter(hits[hits['io_group'] == 7]['z'], hits[hits['io_group'] == 7]['y'],\
        #                    lw=0, s=5, alpha=0.9)
        #self.ax_bv.scatter(hits[hits['io_group'] == 8]['z'], hits[hits['io_group'] == 8]['y'],\
        #                    lw=0, s=5, alpha=0.9)
        #
        #self.ax_dv.scatter(hits[hits['io_group'] == 1]['x'], hits[hits['io_group'] == 1]['y'],\
        #                    lw=0, s=5, alpha=0.9)
        #self.ax_dv.scatter(hits[hits['io_group'] == 2]['x'], hits[hits['io_group'] == 2]['y'],\
        #            lw=0, s=5, alpha=0.9)
        #self.ax_dv.scatter(hits[hits['io_group'] == 3]['x'], hits[hits['io_group'] == 3]['y'],\
        #            lw=0, s=5, alpha=0.9)
        #self.ax_dv.scatter(hits[hits['io_group'] == 4]['x'], hits[hits['io_group'] == 4]['y'],\
        #                    lw=0, s=5, alpha=0.9)
        #self.ax_dv.scatter(hits[hits['io_group'] == 5]['x'], hits[hits['io_group'] == 5]['y'],\
        #                    lw=0, s=5, alpha=0.9)
        #self.ax_dv.scatter(hits[hits['io_group'] == 6]['x'], hits[hits['io_group'] == 6]['y'],\
        #                    lw=0, s=5, alpha=0.9)
        #self.ax_dv.scatter(hits[hits['io_group'] == 7]['x'], hits[hits['io_group'] == 7]['y'],\
        #                    lw=0, s=5, alpha=0.9)
        #self.ax_dv.scatter(hits[hits['io_group'] == 8]['x'], hits[hits['io_group'] == 8]['y'],\
        #                    lw=0, s=5, alpha=0.9)


    def plot_light(self, light_wvfms, light_cmap, light_norm, light_cmap_zero):
        
        acl_det_ids = [0,4,8,12]

        # Plot light in XZ (drift, beam) projection
        for x,z in itertools.product(self.sipm_unique_x,self.sipm_unique_z):
            if x==-1: continue
            if z==-1: continue
            this_xz_sum = 0
            for i,j in itertools.product(range(light_wvfms[0].shape[0]),range(light_wvfms[0].shape[1])):
                #if i!=0: continue
                det_id = self.light_det_id[(i,j)][0]
                wvfm_factor = 1.
                if det_id in acl_det_ids:
                    wvfm_factor = 2.
                pos=self.sipm_abs_pos[(i,j)][0]
                rel_pos = self.sipm_rel_pos[(i,j)][0]
                if pos[0]==-1:
                    continue
                if (abs(pos[0]-x) < 0.5) and (pos[2]==z):
                    this_xz_sum += wvfm_factor*light_wvfms[0][i,j].sum()
                    #print("SIPM:",i,j,"Abs pos:",pos,"Rel pos:",rel_pos,"Light sum:",light_wvfms[0][i,j].sum())
            for i in range(len(self.geometry.attrs['module_RO_bounds'])):
                for j in range(2):
                    #if count != chosen_count: continue
                    z_offset = ((-1)**(j+1))*1.25
                    if (abs(z-self.geometry.attrs['module_RO_bounds'][i][j][2]) < 1):
                        if this_xz_sum==0:
                            cmap_value = light_cmap_zero(0)
                        elif this_xz_sum > self.light_threshold:
                            cmap_value = light_cmap(light_norm(this_xz_sum))
                            if (abs(x-self.geometry.attrs['module_RO_bounds'][i][0][0]) < 1):
                                self.ax_bd.plot([self.geometry.attrs['module_RO_bounds'][i][j][2]+z_offset, self.geometry.attrs['module_RO_bounds'][i][j][2]+z_offset], \
                                         [self.geometry.attrs['module_RO_bounds'][i][0][0], self.geometry.attrs['module_RO_bounds'][i][0][0]+self.geometry.attrs['max_drift_distance']],\
                                          color=cmap_value, alpha=1, linewidth=3.5, solid_capstyle='butt')
                            elif (abs(x-self.geometry.attrs['module_RO_bounds'][i][1][0]) < 1):
                                self.ax_bd.plot([self.geometry.attrs['module_RO_bounds'][i][j][2]+z_offset, self.geometry.attrs['module_RO_bounds'][i][j][2]+z_offset], \
                                         [self.geometry.attrs['module_RO_bounds'][i][1][0]-self.geometry.attrs['max_drift_distance'], self.geometry.attrs['module_RO_bounds'][i][1][0]] ,\
                                          color=cmap_value, alpha=1, linewidth=3.5, solid_capstyle='butt')

        # Plot cathodes for ZX (beam, drift) projections (done for charge only case in set_axes method):
        for i in range(len(self.geometry.attrs['module_RO_bounds'])):
            self.ax_bd.plot([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                     [self.geometry.attrs['module_RO_bounds'][i][0][0]+self.geometry.attrs['max_drift_distance']+self.geometry.attrs['cathode_thickness']/2, \
                      self.geometry.attrs['module_RO_bounds'][i][0][0]+self.geometry.attrs['max_drift_distance']+self.geometry.attrs['cathode_thickness']/2],\
                      color='gainsboro', alpha=0.9, linewidth=2,solid_capstyle='butt')

        # Plot light in ZY (beam, vertical) projection
        for z,y in itertools.product(self.sipm_unique_z,self.sipm_unique_y):
            if z==-1: continue
            if y==-1: continue
            this_zy_sum = 0
            for i,j in itertools.product(range(light_wvfms[0].shape[0]),range(light_wvfms[0].shape[1])):
                pos=self.sipm_abs_pos[(i,j)][0]
                det_id = self.light_det_id[(i,j)][0]
                wvfm_factor = 1.
                if det_id in acl_det_ids:
                    wvfm_factor = 2.
                if pos[0]==-1:
                    continue
                if (pos[2]==z) and (pos[1]==y):
                    this_zy_sum += wvfm_factor*light_wvfms[0][i,j].sum()
            for i in range(len(self.geometry.attrs['module_RO_bounds'])):
                for j in range(2):
                    z_offset = ((-1)**(j+1))*1.25
                    if this_zy_sum==0:
                        cmap_value = light_cmap_zero(0)
                    elif this_zy_sum > self.light_threshold:
                        cmap_value = light_cmap(light_norm(this_zy_sum))
                        if (abs(z-self.geometry.attrs['module_RO_bounds'][i][j][2]) < 1):
                            self.ax_bv.plot([self.geometry.attrs['module_RO_bounds'][i][j][2]+z_offset, self.geometry.attrs['module_RO_bounds'][i][j][2]+z_offset], \
                                     [y-2.25, y+2.25],color=cmap_value, alpha=1, linewidth=3.5, solid_capstyle='butt')

        # Plot light in XY projection               
        for x,y in itertools.product(self.sipm_unique_x,self.sipm_unique_y):
            if x==-1: continue
            if y==-1: continue
            this_xy_sum = 0
            for i,j in itertools.product(range(light_wvfms[0].shape[0]),range(light_wvfms[0].shape[1])):
                pos=self.sipm_abs_pos[(i,j)][0]
                #if i == 4 and j == 4:
                    #print("Position of SiPM where ADC == 4 and Channel == 4:",pos)
                det_id = self.light_det_id[(i,j)][0]
                wvfm_factor = 1.
                if det_id in acl_det_ids:
                    wvfm_factor = 2.
                if pos[0]==-1:
                    continue
                if (abs(pos[0]-x) < 0.5) and (pos[1]==y):
                    this_xy_sum += wvfm_factor*light_wvfms[0][i,j].sum()
            for i in range(len(self.geometry.attrs['module_RO_bounds'])):
                for j in range(2):
                    x_offset = ((-1)**(j+1))*1.25
                    if this_xy_sum==0:
                        cmap_value = light_cmap_zero(0)
                    elif this_xy_sum > self.light_threshold:
                        cmap_value = light_cmap(light_norm(this_xy_sum))
                        if (abs(x-self.geometry.attrs['module_RO_bounds'][i][j][0]) < 1):
                            self.ax_dv.plot([self.geometry.attrs['module_RO_bounds'][i][j][0]+x_offset, self.geometry.attrs['module_RO_bounds'][i][j][0]+x_offset], \
                                    [y-2.25, y+2.25],color=cmap_value, alpha=1, linewidth=4, solid_capstyle='butt')

        # Plot light for XYZ (beam, drift, vertical) 3D view: 
        for i,j in itertools.product(range(light_wvfms[0].shape[0]),range(light_wvfms[0].shape[1])):
            pos=self.sipm_abs_pos[(i,j)][0]
            det_id = self.light_det_id[(i,j)][0]
            wvfm_factor = 1.
            if det_id in acl_det_ids:
                wvfm_factor = 2.
            if pos[0]==-1:
                continue
            this_xyz_sum = wvfm_factor*light_wvfms[0][i,j].sum()
            if this_xyz_sum < self.light_threshold: continue
            if this_xyz_sum==0:
                cmap_value = light_cmap_zero(0)
                #print("Zero light sum at:", pos)
            else:
                cmap_value = light_cmap(light_norm(this_xyz_sum))
            z_diffs = [abs(pos[2]-self.geometry.attrs['module_RO_bounds'][k][l][2]) for k,l in itertools.product(range(len(self.geometry.attrs['module_RO_bounds'])),range(2))]
            z_diffs = np.reshape(z_diffs, np.shape(self.geometry.attrs['module_RO_bounds'][:,:,2]))
            min_z_diff = np.where(abs(z_diffs)<1)
            z_pos = self.geometry.attrs['module_RO_bounds'][:,:,2][min_z_diff][0]
            #print("Z pos:", z_pos)
            found_x = 0
            for k in range(len(self.geometry.attrs['module_RO_bounds'])):
                if (abs(pos[0]-self.geometry.attrs['module_RO_bounds'][k][0][0]) < 1):
                    x1 = self.geometry.attrs['module_RO_bounds'][k][0][0]
                    x2 = self.geometry.attrs['module_RO_bounds'][k][0][0]+self.geometry.attrs['max_drift_distance']
                    found_x = 1
                elif (abs(pos[0]-self.geometry.attrs['module_RO_bounds'][k][1][0]) < 1):
                    x1 = self.geometry.attrs['module_RO_bounds'][k][1][0]-self.geometry.attrs['max_drift_distance']
                    x2 = self.geometry.attrs['module_RO_bounds'][k][1][0]
                    found_x = 1
                if found_x ==1:
                    light_x, light_y, light_z = make_z_plane(x1,x2, pos[1]-2.25, pos[1]+2.25,z_pos)
                    self.ax_bdv.plot_surface(light_z,light_x,light_y,color=cmap_value, alpha=0.2, shade=False)
                    if self.show_event_mx2:
                        self.ax_mx2.plot_surface(light_z,light_x,light_y,color=cmap_value, alpha=0.1, shade=False)
                    #print("Light sum at:", pos, "is", this_xyz_sum)
                    break
                else: continue
#

# Helper functions outside of main class
def make_x_plane(y1, y2, z1, z2, x):
    y = np.linspace(y1, y2, 100)
    z = np.linspace(z1, z2, 100)
    Y, Z = np.meshgrid(y, z)
    X = np.full(Y.shape, x)
    return X, Y, Z

def make_z_plane(x1, x2, y1, y2, z):
    x = np.linspace(x1, x2, 100)
    y = np.linspace(y1, y2, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.full(X.shape, z)
    return X, Y, Z

