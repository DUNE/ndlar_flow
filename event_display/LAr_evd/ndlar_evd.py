# Contact: l.cremonesi@qmul.ac.uk @Linda Cremonesi on DUNE Slack
# Most important parts of the code were written by Elise Hinkle for the 2x2 Mx2 event displays
# All of the hacky bad code to get it to work for ND LAr and TMS were written by Linda
# The TMS geometry bounds and steel plates are hard-coded: this is bad practice, sorry!
# GOOD LUCK!

# Import packages to check for and install missing packages
import sys
import subprocess

# # Function to install missing packages
# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# # Ensure setuptools is installed to use pkg_resources
# try:
#     import pkg_resources
# except ImportError:
#     install('setuptools')
#     import pkg_resources

# # Ensure all non-standard packages are installed
# required_packages = [
#     'numpy', 'pandas', 'sqlalchemy', 'h5py', 'cmasher', 'IPython', 'PyMuPDF', 'matplotlib', 'pillow', 'uproot', 'h5flow', 'ipywidgets'
# ]

# installed_packages = {pkg.key for pkg in pkg_resources.working_set}
# missing_packages = [pkg for pkg in required_packages if pkg not in installed_packages]

# if missing_packages:
#     print("Missing packages:", missing_packages)
#     for package in missing_packages:
#         install(package)

# Import modules
import fitz
import numpy as np
import pandas as pd
from datetime import datetime
import ipywidgets as widgets
from io import BytesIO
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.proto_nd_flow.util.lut import LUT
#from h5flow.core import resources
import itertools
import math
import h5py
import cmasher as cmr
from IPython.display import display, clear_output
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.image as mpimg
from PIL import Image
from math import fabs
import uproot

import periodictable
from particle import Particle

class LArEventDisplay:

    ''' 
        Class to set up interactive ND LAr display for files run through proto_nd_flow. If also displaying TMS data, TMS file should be
        in ROOT format (vs. dst format).

        Inputs to this class are as follows:

            - filedir          (str): path to input file (minus filename)
            - filename         (str): name of flow file
            - nhits_min        (int): minimum number of hits (threshold) for events to be made available (default: 1)
            - nhits_max        (int): maximum number of hits allowed in event for events to be made available (default: 1e10)
            - ntrigs           (int): number of external triggers (threshold)  threshold for events to be made available (default: 0)
            - show_light       (bool): whether to show light information in display (default: True)
            - show_colorbars   (bool): whether to display color bars (default: True)
            - filepath_tms     (str): path to TMS file if using TMS data (default: None)
            - hist_projection  (bool): if True, hits are binned in a 2D histogram for the 2D charge hit projections. Bins with 0 charge are not displayed.
                                       If False, hits are plotted as scatter points for the 2D charge hit projections. (default: True)
            - charge_threshold (float): threshold for charge hits to be shown (default: None)
            - light_threshold  (float): threshold for light to be shown (default: 150000 ADC counts)
            - beam_only        (bool): whether to only show beam events (default: False)
            
        In order to run the display and interactively flip through events, set up a Jupyter Notebook, import everything in this file,
        and execute the run() method, e.g.:

        from lar_only_evd import *
        plt.ion()

        d = '/path/to/file/'
        f = 'filename'
        evd = LArEventDisplay(filedir=d, filename=f, nhits=1, ntrigs=1)
        evd.run()

        Alternatively, you can a display for a specific event by calling the display_event() method with the event ID as an argument, e.g.:
        evd.display_event(0). This can also be done within another python script. 

        Class methods:
            - save_to_pdf(ev_id, points_scaled_to_pixel_pitch=False): Save current display to PDF
            - run(): Run the display
            - clear_axes(): Clear axes for next event
            - get_event(ev_id): Get event information for a specific event
            - set_axes(cmap, mcharge, cmap_zero, mlight=None): Set up axes for display (e.g. limits, module locations)
            - display_event(ev_id): Display a specific event
            - plot_light(light_wvfms, light_norm, light_cmap, light_cmap_zero): Plot light information

    '''

    # Initialize class
    def __init__(self, filedir, filename, runsdb='sqlite:////global/cfs/cdirs/dune/www/data/2x2/DB/RunsDB/releases/tmsx2runs_v0.1_alpha3.sqlite', \
                 nhits_min=1, nhits_max=1e10, ntrigs=0, show_light=True, filepath_tms=None, \
                 show_colorbars=True, charge_threshold=None, light_threshold=150000, beam_only=False, \
                 hist_projection=False, single_neutrino=False):
        
        # Open files
        f = h5py.File(filedir+filename, 'r')
        if filepath_tms is not None:
            f_tms = uproot.open(filepath_tms)
            self.show_tms = True
        else:
            self.show_tms = False

        # Set general class-level variables from inputs
        self.show_event_tms = self.show_tms
        self.filedir = filedir
        self.filename = filename
        self.filepath_tms = filepath_tms
        try:
            self.runsdb = runsdb
            self.all_subruns_db = pd.read_sql_table('All_global_subruns', runsdb)
        except:
            self.runsdb = '0' #None
            self.all_subruns_db = None
        self.show_light = show_light
        self.show_event_light = show_light
        self.show_colorbars = show_colorbars
        self.charge_threshold = charge_threshold
        self.light_threshold = light_threshold
        self.beam_only = beam_only
        self.hist_projection = hist_projection
        self.single_neutrino = single_neutrino
        self.pick_nuint = True # LC: flag created to pick one neutrino interaction out of the spill

        # Set directory for saving files and finding logo image files
        self.lar_evd_dir = os.path.dirname(__file__)

        # Load DUNE and 2x2 logos
        dune_logo = os.path.join(self.lar_evd_dir, 'DUNElogo.pdf')
        self.dune_logo_pdf = fitz.open(dune_logo)#mpimg.imread(dune_logo)
        #subexp_logo=os.path.join(self.lar_evd_dir, '2x2logo.pdf')
        subexp_logo=os.path.join(self.lar_evd_dir, 'emptySquare.pdf')
        self.subexp_logo_pdf = fitz.open(subexp_logo)#mpimg.imread(subexp_logo)

        # Resize DUNE logo image to fit in display
        dune_logo_page = self.dune_logo_pdf.load_page(0)
        dune_logo_pixmap = dune_logo_page.get_pixmap(matrix=fitz.Matrix(5, 5), dpi=600)
        dune_logo_image = Image.frombytes("RGB", [dune_logo_pixmap.width, dune_logo_pixmap.height], dune_logo_pixmap.samples)
        dune_logo_buf = BytesIO()
        dune_logo_image.save(dune_logo_buf, format='png')
        dune_logo_buf.seek(0)
        self.dune_logo_png = mpimg.imread(dune_logo_buf, format='png')

        # Resize 2x2 logo image to fit in display
        subexp_logo_page = self.subexp_logo_pdf.load_page(0)
        subexp_logo_pixmap = subexp_logo_page.get_pixmap(matrix=fitz.Matrix(5, 5), dpi=600)
        subexp_logo_image = Image.frombytes("RGB", [int(subexp_logo_pixmap.width), int(subexp_logo_pixmap.height)], subexp_logo_pixmap.samples)
        #subexp_logo_height, subexp_logo_width = int(subexp_logo_image_full_size.height*.08), int(subexp_logo_image_full_size.width*.08)
        #subexp_logo_image = subexp_logo_image_full_size.resize((subexp_logo_width, subexp_logo_height), Image.LANCZOS)
        subexp_logo_buf = BytesIO()
        subexp_logo_image.save(subexp_logo_buf, format='png')
        subexp_logo_buf.seek(0)
        self.subexp_logo_png = mpimg.imread(subexp_logo_buf, format='png')

        # Load events dataset
        self.events = f['charge/events/data']

        # Load external triggers dataset
        self.exttrigs_full = f['charge/ext_trigs/data']
        self.exttrigs_ref = f['charge/events/ref/charge/ext_trigs/ref']
        self.exttrigs_region = f['charge/events/ref/charge/ext_trigs/ref_region']

        # Get beam trigger events
        self.exttrigs_beam = np.where(self.exttrigs_full['iogroup'] == 5)
        self.beam_events_ref = np.sort(self.exttrigs_ref[:,0][self.exttrigs_beam])
        self.beam_events = self.events[self.beam_events_ref]
        if self.beam_only:
            self.events = self.beam_events
        self.is_beam_event = beam_only

        # Filter events and beam events based on nhits and ntrigs
        self.events = self.events[self.events['nhit'] >= nhits_min]
        self.events = self.events[self.events['nhit'] <= nhits_max]
        self.events = self.events[self.events['n_ext_trigs'] >= ntrigs]
        self.beam_events = self.beam_events[self.beam_events['nhit'] >= nhits_min]
        self.beam_events = self.beam_events[self.beam_events['nhit'] <= nhits_max]
        self.beam_events = self.beam_events[self.beam_events['n_ext_trigs'] >= ntrigs]

        # Load geometry and other info
        self.geometry = f['geometry_info']
        self.info = {
            'vdrift': f['lar_info'].attrs['v_drift'],
            'clock_period': 0.1,
        }
        self.run_info = f['run_info']
        self.is_mc = self.run_info.attrs['is_mc']

        ## Setting for mc_truth
        self.truth_int = f['mc_truth/interactions/data']
        self.truth_stack = f['mc_truth/stack/data']
        
        # Load charge hits dataset
        self.hits_dset = 'calib_prompt_hits'
        self.hits_full = f['charge/'+self.hits_dset+'/data']
        self.hits_ref = f['charge/events/ref/charge/'+self.hits_dset+'/ref']
        self.hits_region = f['charge/events/ref/charge/'+self.hits_dset+'/ref_region']
        self.hits_per_event = nhits_min

        # Load light event and waveform datasets and light geometry info if using
        if self.show_light:
            self.light_events = f['light/events/data']
            self.charge_light_ref = f['charge/events/ref/light/events/ref']
            self.charge_light_region = f['charge/events/ref/light/events/ref_region']
            self.light_wvfms = f['light/wvfm/data']
            self.light_event_wvfm_ref = f['light/events/ref']['light/wvfm']['ref']
            self.light_event_wvfm_region = f['light/events/ref']['light/wvfm']['ref_region']

            self.sipm_abs_pos = LUT.from_array(f["geometry_info/sipm_abs_pos"].attrs["meta"],f["geometry_info/sipm_abs_pos/data"])
            self.sipm_rel_pos = LUT.from_array(f["geometry_info/sipm_rel_pos"].attrs["meta"],f["geometry_info/sipm_rel_pos/data"])
            self.light_det_id = LUT.from_array(f["geometry_info/det_id"].attrs["meta"],f["geometry_info/det_id/data"])

            self.all_sipm_pos = f["geometry_info/sipm_abs_pos/data"]["data"][1:]
            self.sipm_unique_x = np.unique([pos[0] for pos in self.all_sipm_pos])
            self.sipm_unique_z = np.unique([pos[2] for pos in self.all_sipm_pos])
            self.sipm_unique_y = np.unique([pos[1] for pos in self.all_sipm_pos])
        
        # Load tms data if using
        if self.show_tms:

#            self.tms_track_hit_pos = f_tms["Reco_Tree"]["TrackHitPos"].array(library="np")
            self.tms_track_hit_pos = f_tms["Reco_Tree"]["KalmanTruePos"].array(library="np") ## LINDA USING TRUE INFO FOR TMS FOR NOW

            self.tms_evt_no   = f_tms["Reco_Tree"]["EventNo"].array(library="np")
            self.tms_slice_no = f_tms["Reco_Tree"]["SliceNo"].array(library="np")
            self.tms_spill_no = f_tms["Reco_Tree"]["SpillNo"].array(library="np")
            self.tms_run_no   = f_tms["Reco_Tree"]["RunNo"].array(library="np")

#            self.tms_steel_regions_bounds = [ 1121, 1446, 1752, 1831.4 ] ## NEW STEEL REGION
            self.tms_steel_regions_bounds = [ 1136.2, 1350.0, 1831.4 ] ## OLD STEEL REGION used for MicroProd4.1

            self.tms_rough_bounds = [[-350., 350.], [-300, 50], [1136.2, 1831.4]]
            self.tms_fiducial_bounds = [[-330., 330.], [-285, 50], [1136.2, 1831.4]]
            
        # Set up figure and subplots
        # NOTE: This is very different if TMS is shown
        if self.show_tms: 
            # Setting figure WITH TMS
            self.fig = plt.figure(constrained_layout=False, figsize=(15, 15))
            # self.axes_mosaic = [["ax_bd", "ax_bd",  "ax_subexp_logo", "ax_subexp_logo", "ax_bdv", "ax_bdv", "ax_bdv", "ax_bdv"],\
            #                     ["ax_bd", "ax_bd",  "ax_subexp_logo", "ax_subexp_logo", "ax_bdv", "ax_bdv", "ax_bdv", "ax_bdv"],\
            #                     ["ax_bv", "ax_bv", "ax_dv", "ax_dv", "ax_bdv", "ax_bdv", "ax_bdv", "ax_bdv"],\
            #                     ["ax_bv", "ax_bv", "ax_dv", "ax_dv", "ax_bdv", "ax_bdv", "ax_bdv", "ax_bdv"],\
            #                     ["ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms"],\
            #                     ["ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms"],\
            #                     ["ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms"],\
            #                     ["ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms"],\
            #                     ["ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms"]]
            # self.axes_dict = self.fig.subplot_mosaic(self.axes_mosaic, \
            #                                         per_subplot_kw={"ax_bdv": {"projection": "3d"}, 
            #                                                        "ax_tms": {"projection": "3d"}})
            self.axes_mosaic = [["ax_bd", "ax_bd",  "ax_bd", "ax_bd", "ax_bd", "ax_bd", "ax_bd", "ax_bd"],\
                                ["ax_bd", "ax_bd",  "ax_bd", "ax_bd", "ax_bd", "ax_bd", "ax_bd", "ax_bd"],\
                                ["ax_bd", "ax_bd",  "ax_bd", "ax_bd", "ax_bd", "ax_bd", "ax_bd", "ax_bd"],\
                                ["ax_bd", "ax_bd",  "ax_bd", "ax_bd", "ax_bd", "ax_bd", "ax_bd", "ax_bd"],\
                                ["ax_bv", "ax_bv",  "ax_bv", "ax_bv", "ax_bv", "ax_bv", "ax_bv", "ax_bv"],\
                                ["ax_bv", "ax_bv",  "ax_bv", "ax_bv", "ax_bv", "ax_bv", "ax_bv", "ax_bv"],\
                                ["ax_bv", "ax_bv",  "ax_bv", "ax_bv", "ax_bv", "ax_bv", "ax_bv", "ax_bv"],\
                                ["ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms"],\
                                ["ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms"],\
                                ["ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms"],\
                                ["ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms"],\
                                ["ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms"],\
                                ["ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms"],\
                                ["ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms", "ax_tms"]]
            self.axes_dict = self.fig.subplot_mosaic(self.axes_mosaic, \
                                                    per_subplot_kw={ #"ax_bdv": {"projection": "3d"}, 
                                                                   "ax_tms": {"projection": "3d"}})
            
            # Setting colorbar axes (if showing) WITH tms
            if self.show_colorbars and not self.show_light:
                cbar_ax = self.fig.add_axes([0.15, 0.45, 0.675, 0.015])
            if self.show_colorbars and self.show_light:
                cbar_ax = self.fig.add_axes([0.08, 0.45, 0.38, 0.015])
                light_cbar_ax = self.fig.add_axes([0.51, 0.45, 0.38, 0.015])

            # Adjust subplot positioning WITH tms
            self.fig.subplots_adjust(top=0.93 ,bottom=0.001) 
            self.fig.subplots_adjust(wspace=0.02, hspace=0.02)

            # Set up tms axis
            self.ax_tms = self.axes_dict["ax_tms"]
            current_tms_pos = self.ax_tms.get_position()
            if self.show_colorbars:
                x0_shift = 0.13
                y0_shift = 0.04
            else: 
                x0_shift = 0.11
                y0_shift = 0.04
            new_tms_pos = [current_tms_pos.x0-x0_shift, current_tms_pos.y0-y0_shift, \
                           current_tms_pos.width*1.2, current_tms_pos.height*1]
            self.ax_tms.set_position(new_tms_pos)

            # Set up DUNE logo axis WITH tms
            ax_dune_logo = self.fig.add_axes([0.59, 0.942, 0.37, 0.057])

        else:
            # Setting figure WITHOUT tms
            self.fig = plt.figure(constrained_layout=False, figsize=(15, 8))
            self.axes_mosaic = [["ax_bd", "ax_subexp_logo", "ax_bdv", "ax_bdv"],["ax_bv", "ax_dv", "ax_bdv", "ax_bdv"],]
            self.axes_dict = self.fig.subplot_mosaic(self.axes_mosaic, \
                                                    per_subplot_kw={"ax_bdv": {"projection": "3d"}})
            # Setting colorbar axes (if showing) WITHOUT tms
            if self.show_colorbars and not self.show_light:
                cbar_ax = self.fig.add_axes([0.145, 0.001, 0.675, 0.025])
            if self.show_colorbars and self.show_light:
                cbar_ax = self.fig.add_axes([0.08, 0.001, 0.38, 0.025])
                light_cbar_ax = self.fig.add_axes([0.51, 0.001, 0.38, 0.025])

            # Adjust subplot positioning WITHOUT tms
            self.fig.subplots_adjust(bottom=0.1)
            self.fig.subplots_adjust(wspace=0.02, hspace=0.02)

            # Set up DUNE logo axis WITHOUT tms
            ax_dune_logo = self.fig.add_axes([0.56, 0.895, 0.43, 0.099])

        # Initialize axes for display (with or without tms)
        self.ax_dune_logo = ax_dune_logo
        self.ax_bd = self.axes_dict["ax_bd"]
        self.ax_bv = self.axes_dict["ax_bv"]
        if (self.show_tms is False):
            self.ax_dv = self.axes_dict["ax_dv"]
            self.ax_bdv = self.axes_dict["ax_bdv"]
            self.ax_subexp_logo = self.axes_dict["ax_subexp_logo"]
            self.dv_points = self.ax_dv.scatter([], [])
            self.bdv_points = self.ax_bdv.scatter([], [], [])

        if self.show_light and not self.single_neutrino:
            print("You chose an event display with ND LAr light using full spill events. This is is going to take a while. I suggest you go and make yourself a cup of tea while you wait!")
            
        if self.show_colorbars:
            self.cbar_ax = cbar_ax
            if self.show_light:
                self.light_cbar_ax = light_cbar_ax

        # Initialize point collections for plotting
        if self.show_tms:
            self.tms_lar_points = self.ax_tms.scatter([], [], [])
        self.bd_points = self.ax_bd.scatter([], [])
        self.bv_points = self.ax_bv.scatter([], [])

        # Set up 3D view angles and zooms for GIFs
        self.base_angle = list(range(-180,180,5))
        self.azimuths = [ba for ba in self.base_angle]
        self.zeniths = [fabs(ba*0.25) for ba in self.base_angle]
        self.offset = int(len(self.base_angle)/6) # Shift angles relative to improve perspectives
        self.azimuths = self.azimuths[self.offset:] + self.azimuths[:self.offset] # Shift angles relative to improve perspectives
        self.zeniths = self.zeniths[self.offset:] + self.zeniths[:self.offset] # Shift angles relative to improve perspectives
        self.zooms = [0.985,]*len(self.azimuths)  # Set view zoom

        # print('Base angle', self.base_angle)
        # print('Azimuths', self.azimuths)
        # print('Zeniths', self.zeniths)
        # print('Offset', self.offset)
        # print('Zooms', self.zooms)
    
        # Create sliders for elevation and azimuthal angles # TO DO: FIX/IMPLEMENT SLIDER WIDGET
        #self.elev_slider = widgets.FloatSlider(value=30, min=0, max=90, step=1, description='Elevation:')
        #self.azim_slider = widgets.FloatSlider(value=45, min=0, max=360, step=1, description='Azimuth:')
        

    ## Create the interactive widget # TO DO: FIX/IMPLEMENT SLIDER WIDGET
    #def update_plot(self, elev, azim):
    #    self.ax_bdv.view_init(elev=elev, azim=azim)
    #    display(plt.gcf(), self.elev_slider, self.azim_slider)


    # TO DO: Make hits, cmap, charge_norm class variables vs. separate objects carried around between class methods
    def save_to_pdf(self, ev_id, points_scaled_to_pixel_pitch=False, hits=None, cmap=None, charge_norm=None):
        
        # Adjust point size in plots if points are scaled to pixel pitch
        if points_scaled_to_pixel_pitch:
            pixel_pitch_sizes = np.full(self.hits_per_event, 0.3)       
            if self.hist_projection:
                self.bd_points.remove()
                self.bv_points.remove()
                self.dv_points.remove()
                z_bins = np.linspace(self.geometry.attrs['lar_detector_bounds'][0][2],self.geometry.attrs['lar_detector_bounds'][1][2],\
                                     int((self.geometry.attrs['lar_detector_bounds'][1][2]-self.geometry.attrs['lar_detector_bounds'][0][2])/0.4))
                y_bins = np.linspace(self.geometry.attrs['lar_detector_bounds'][0][1],self.geometry.attrs['lar_detector_bounds'][1][1],\
                                     int((self.geometry.attrs['lar_detector_bounds'][1][1]-self.geometry.attrs['lar_detector_bounds'][0][1])/0.4))
                x_bins = np.linspace(self.geometry.attrs['lar_detector_bounds'][0][0],self.geometry.attrs['lar_detector_bounds'][1][0],\
                                     int((self.geometry.attrs['lar_detector_bounds'][1][0]-self.geometry.attrs['lar_detector_bounds'][0][0])/0.4))

                bd_charge_hist, _, _ = np.histogram2d(hits['z'], hits['x'], bins=[z_bins,x_bins],weights=hits['Q'])
                bd_charge_hist_masked = np.where(bd_charge_hist==0, np.nan, bd_charge_hist) # TO DO: SHOULD CHARGE ==0 BE MASKED?
                ZX_Z, ZX_X = np.meshgrid(z_bins[:-1], x_bins[:-1])
                self.bd_points = self.ax_bd.pcolormesh(ZX_Z, ZX_X, bd_charge_hist_masked.T, cmap=cmap, norm=charge_norm, alpha=1)

                bv_charge_hist, _, _ = np.histogram2d(hits['z'], hits['y'], bins=[z_bins,y_bins],weights=hits['Q'])
                bv_charge_hist_masked = np.where(bv_charge_hist==0, np.nan, bv_charge_hist) # TO DO: SHOULD CHARGE ==0 BE MASKED?
                ZY_Z, ZY_Y = np.meshgrid(z_bins[:-1], y_bins[:-1])
                self.bv_points = self.ax_bv.pcolormesh(ZY_Z, ZY_Y, bv_charge_hist_masked.T, cmap=cmap, norm=charge_norm, alpha=1)

                dv_charge_hist, _, _ = np.histogram2d(hits['x'], hits['y'], bins=[x_bins,y_bins],weights=hits['Q'])
                dv_charge_hist_masked = np.where(dv_charge_hist==0, np.nan, dv_charge_hist) # TO DO: SHOULD CHARGE ==0 BE MASKED?
                XY_X, XY_Y = np.meshgrid(x_bins[:-1], y_bins[:-1])
                self.dv_points = self.ax_dv.pcolormesh(XY_X, XY_Y, dv_charge_hist_masked.T, cmap=cmap, norm=charge_norm, alpha=1)
            else:
                self.bd_points.set_sizes(pixel_pitch_sizes)
                self.bv_points.set_sizes(pixel_pitch_sizes)
                self.dv_points.set_sizes(pixel_pitch_sizes)

            if self.show_event_tms:
                self.tms_lar_points.set_sizes(pixel_pitch_sizes)
            self.bdv_points.set_sizes(pixel_pitch_sizes)

        save_dir = self.lar_evd_dir
        with_tms_str = ""
        with_light_str = ""
        if self.show_tms:
            with_tms_str += "_withTMS"
        if self.show_light:
            with_light_str += "_withLight"
        filename = self.filename.split('.')[0]+'_Run_'+str(self.event_run)+'_Event_'+str(ev_id)+with_light_str+with_tms_str+'.pdf'
        filename_jpg = self.filename.split('.')[0]+'_Run_'+str(self.event_run)+'_Event_'+str(ev_id)+with_light_str+with_tms_str+'.jpg'
        savepath = os.path.join(save_dir, filename)
        savepath_jpg = os.path.join(save_dir, filename_jpg)
        print("Saving to", savepath.split('.')[0]+'_Display.pdf')
        self.fig.savefig(savepath, bbox_inches='tight')
        # self.fig.savefig(savepath_jpg, bbox_inches='tight') ## JPG is not good quality ...

        # # Then, add metadata and add back vectorized DUNE logo to saved PDF
        # saved_pdf = fitz.open(savepath)
        # saved_pdf_metadata = saved_pdf.metadata
        # if self.filepath_tms is None:
        #     saved_pdf_metadata.update({'title' : "Event "+str(ev_id)+" from "+self.filedir+self.filename+" with NO tms file and using runs database "+self.runsdb})
        # else:
        #     saved_pdf_metadata.update({'title' : "Event "+str(ev_id)+" from "+self.filedir+self.filename+" with tms file "+self.filepath_tms+" and using runs database "+self.runsdb})
        # saved_pdf.set_metadata(saved_pdf_metadata)
        # saved_pdf_page = saved_pdf[0]
        # rect_max_x = saved_pdf_page.rect[2]
        # # include_dune_logo_rect = fitz.Rect(rect_max_x-300, 2, rect_max_x-5, 65)
        # # saved_pdf_page.show_pdf_page(include_dune_logo_rect, self.dune_logo_pdf, 0)
        # # if (self.show_tms is False):
        # #     include_subexp_logo_rect = fitz.Rect(rect_max_x-635, 78, rect_max_x-425, 288)
        # #     saved_pdf_page.show_pdf_page(include_subexp_logo_rect, self.subexp_logo_pdf, 0)
        # saved_pdf.save(savepath.split('.')[0]+'_Display.pdf')

        # # Remove initial PDF without DUNE logo
        # os.remove(savepath)
        

    def run(self):

        ## Link sliders to the update function
        #widgets.interactive(self.update_plot, elev=self.elev_slider, azim=self.azim_slider) # TO DO: FIX SLIDER WIDGET
  
        # Get event IDs and initialize event index
        event_ids = [ev['id'] for ev in self.events]
        ev_idx = 0
        ev_id = event_ids[ev_idx]

        # Display first event 
        hits, cmap, charge_norm = self.display_event(ev_id)

        # Displays event until user input determines next action
        # User can quit display (q), save current display to PDF (s), save current display to PDF
        # with points scaled to pixel pitch (p), skip to next event (enter),
        # make a GIF of an event (g), or skip to a specific event ID (type number)
        while True:

            display(plt.gcf()) #, self.elev_slider, self.azim_slider) # TO DO: FIX SLIDER WIDGET
            user_input = input(
                'Next event (q to exit/s to save to pdf/p to save to pdf with points scald to pixel pitch/g to create gif/enter for next/number to skip to event)?\n')
            if not user_input:
                clear_output(wait=True)
                ev_idx += 1
                ev_id = event_ids[ev_idx]
                hits, cmap, charge_norm = self.display_event(ev_id)
            elif user_input[0].lower() == 'q':
                sys.exit()
            elif user_input[0].lower() == 's':
                self.save_to_pdf(ev_id, hits=hits, cmap=cmap, charge_norm=charge_norm)
            elif user_input[0].lower() == 'p':
                self.save_to_pdf(ev_id, points_scaled_to_pixel_pitch=True, hits=hits, cmap=cmap, charge_norm=charge_norm)
            elif user_input[0].lower() == 'g':
                print("Creating GIF of Event Display")
                # Loop over 3D views
                gif_dir = self.lar_evd_dir
                frame_num = 0
                for (azi,zen,zoom) in zip(self.azimuths,self.zeniths,self.zooms):
                    print('Angles: ', zen, azi)
                    self.ax_bdv.view_init(zen, azi) 
                    # self.ax_bdv.dist = zoom    # LC: This does not work for ND-LAr
                    self.ax_bdv.set_box_aspect([1,1,1]) 
                    figname = gif_dir+'frame_%04d_%04d.png' % (ev_id, frame_num)
                    self.fig.savefig(figname)
                    frame_num += 1
                    print('saving', figname)
#                os.system("convert -delay 10 "+gif_dir+"frame*.png "+gif_dir+"animated_"+str(ev_id)+"no_axes.gif")
                os.system("magick -delay 10 "+gif_dir+"frame*.png "+gif_dir+"animated_"+str(ev_id)+"no_axes.gif")
            else:
                try:
                    clear_output(wait=True)
                    ev_id = int(user_input)
                    ev_idx = event_ids.index(ev_id)
                    hits, cmap, charge_norm = self.display_event(ev_id)
                except:
                    clear_output(wait=True)
                    print("Event number %s not valid" % user_input)
                    print("Proceeded to next available event instead")
                    ev_idx += 1
                    ev_id = event_ids[ev_idx]
                    hits, cmap, charge_norm = self.display_event(ev_id)                 
            if ev_id >= event_ids[-1]:
                print("End of file")
                sys.exit()
    

    def clear_axes(self):

        self.ax_bd.cla()
        self.ax_bv.cla()
        if self.show_colorbars:
            self.cbar_ax.cla()
            if self.show_light:
                self.light_cbar_ax.cla()
        if self.show_tms:
            self.ax_tms.cla()
        else:
            self.ax_bdv.cla()
            self.ax_dv.cla()
        self.fig.texts.clear()

    def get_event(self, ev_id):

        # To start, set show_event_tms to general show_tms value
        self.show_event_tms = self.show_tms
        
        # Get event ID information
        ev_idx = np.where(self.events['id'] == ev_id)[0][0]
        #print("Number of available events:", len(self.events))
        #print("For fast-forwarding purposes, here is every 10th event number in your sample:", [ev for ev in self.events['id'][9::10]])

        # Get event general information
        event = self.events[ev_idx]
        # Use event unix TS in seconds for hand-scanning campaign
        event_datetime = str(event['unix_ts']) #datetime.utcfromtimestamp(event['unix_ts']).strftime('%Y-%m-%d %H:%M:%S')
        if not self.is_mc:
            try:
                event_run_info = self.all_subruns_db[(self.all_subruns_db['start_time_unix'] <= event['unix_ts']) &
                                      (self.all_subruns_db['end_time_unix'] > event['unix_ts'])]
                event_subrun = event_run_info['global_subrun'].values[0]
                event_run = event_run_info['global_run'].values[0]
                self.event_run = event_run
            except:
                event_run = -1
                event_subrun = -1
                self.event_run = event_run
            data_sim_watermark = 'DATA'
            watermark_fs = 78
        elif self.is_mc:
        
            event_truth = self.truth_int[ev_idx]
            event_nu = event_truth['nu_pdg']
            event_enu = event_truth['Enu']/1000 # convert MeV to GeV 
            event_reaction = event_truth['reaction']
            event_target = event_truth['target']
            event_target_symbol = periodictable.elements[event_target].symbol
            event_isCC = event_truth['isCC']
            event_reaction_str = GENIE_code(event_reaction)
            event_lep_pdg = event_truth['lep_pdg']
            event_lep_E   = event_truth['Elep']/1000 # convert MeV to GeV 
            event_true_vtx_x = event_truth['x_vert']
            event_true_vtx_y = event_truth['y_vert']
            event_true_vtx_z = event_truth['z_vert']

            # print('Event true x, y, z',event_truth['x_vert'], event_truth['y_vert'], event_truth['z_vert'])
            
            event_run = int((event_truth['event_id']-ev_idx)/1000)
            event_subrun = 0
            self.event_run = event_run
            tmp_ev_idx = event_truth['event_id']- (event_run*1000)
            
            stack_diff = np.abs(self.truth_stack['event_id'] - event_truth['event_id'] )
            particles = np.argwhere(stack_diff < 1).reshape(1,-1)[0]

            outstring = ''


            if self.single_neutrino : 
                for particle in particles:
                    stack_info = self.truth_stack[particle]
                    part_pdg = pdgToString(stack_info['part_pdg'])
                    part_4mom = stack_info['part_4mom']
                    print(stack_info['event_id'], stack_info['part_pdg'])
                    if (stack_info['part_pdg'] < 1000000 and stack_info['part_pdg']!= event_nu and stack_info['part_pdg']!= event_lep_pdg):
                        outstring += ' + ' + part_pdg + " ({:.2f} GeV)".format(part_4mom[3]/1000.)
                if ( tmp_ev_idx != ev_idx ):
                    print ('DIFFERENT EVENT NUMBERS !!! ############################################################################################## ', tmp_ev_idx, ev_idx)
            
            
            data_sim_watermark = 'SIMULATION'
            watermark_fs = 26

  
        print("Number of external triggers in this event:", event['n_ext_trigs'])

        # Check if event is a beam event
        if not self.beam_only and ev_id in self.beam_events['id']:
            self.is_beam_event = True
        # # Check if event is a beam event for showing tms (TO DO: Is this necessary for tms matching?)
        # if not self.is_beam_event: 
        #     self.show_event_tms = False

        # Get event charge information
        hit_ref = self.hits_ref[self.hits_region[ev_id,'start']:self.hits_region[ev_id,'stop']]
        if self.pick_nuint:
            print (self.hits_region[ev_id,'start'],self.hits_region[ev_id,'stop'])
#            hit_ref = self.hits_ref[10000:50000]
        hit_ref = np.sort(hit_ref[hit_ref[:,0] == ev_id, 1])
        hits = self.hits_full[hit_ref]
        self.hits_per_event = len(hits)


            
        
        # Get event external trigger information
        exttrig_ref = self.exttrigs_ref[self.exttrigs_region[ev_id,'start']:self.exttrigs_region[ev_id,'stop']]
        exttrig_ref = np.sort(exttrig_ref[exttrig_ref[:,0] == ev_id, 1])
        exttrigs = self.exttrigs_full[exttrig_ref]
        #print("External trigger information:", exttrigs['iogroup'], exttrigs['ts'], exttrigs['ts_raw'])

        # Prepare color map for charge
        if len(hits) > 0:
            min_charge = min(hits['Q'])
            if max(hits['Q']) > min_charge:
                max_charge = max(hits['Q'])
            else: 
                max_charge = min_charge + 1
        else:
            min_charge = 0
            max_charge = 1
        # Set charge norm to hardcoded values for hand-scanning campaign
        charge_norm = mpl.colors.Normalize(vmin=-30,vmax=120) #mpl.colors.Normalize(vmin=min_charge,vmax=max_charge)
        cmap = cmr.get_sub_cmap('cmr.torch_r', 0.13,0.95)
        cmap_zero = cmr.get_sub_cmap('cmr.torch_r', 0.03, 0.95)
        mcharge = plt.cm.ScalarMappable(norm=charge_norm, cmap=cmap)

        if self.show_light:
            # Get event light matches
            self.show_event_light = True
            light_matches = self.charge_light_ref[self.charge_light_region[ev_id,'start']:self.charge_light_region[ev_id,'stop']]
            light_matches = np.sort(light_matches[light_matches[:,0] == ev_id, 1])
            light = self.light_events[light_matches]

            # If no light matches, set show_event_light to False and don't carry out other light display steps
            if len(light) == 0:
                print("No light information for event", ev_id)
                self.show_event_light = False

            # If light matches, continue to get light information
            if len(light) > 0:

                light_idx = light[0][0]

                # Get light waveforms
                light_wvfm_ref = self.light_event_wvfm_ref[self.light_event_wvfm_region[light_idx,'start']:self.light_event_wvfm_region[light_idx,'stop']]
                light_wvfm_ref = np.sort(light_wvfm_ref[light_wvfm_ref[:,0] == light_idx, 1])

                # Subtract pedestals for data: # TO DO: FIX PEDESTAL SUBTRACTION IF ADDITIONAL LIGHT CALIBRATIONS OCCUR
                light_wvfm_get_peds = np.mean(self.light_wvfms[light_wvfm_ref]["samples"][:, :, :, 0:50], axis=-1)
                light_wvfm_peds_exp = np.expand_dims(light_wvfm_get_peds, axis=-1)
                light_wvfm_peds = light_wvfm_peds_exp * np.ones((1, 1, 1, 1000))
                light_wvfms = self.light_wvfms[light_wvfm_ref]["samples"] - light_wvfm_peds

                # Prepare color map for light
                light_cmap=cmr.get_sub_cmap(cmr.voltage_r, 0.0, 0.55)
                light_cmap_zero=cmr.get_sub_cmap(cmr.voltage_r, 0.0, 0.55)
                min_light = self.light_threshold
                if light_wvfms[0].sum(axis=-1).max() > min_light:
                    max_light = light_wvfms[0].sum(axis=-1).max()*2
                else:
                    max_light = min_light*10
                light_norm = colors.LogNorm(min_light,max_light)
                mlight = plt.cm.ScalarMappable(norm=light_norm, cmap=light_cmap)


        # Get tms matched event info if using
        # TO DO: Make tms matching more robust
        # Debugging statements included to help with troubleshooting
        if self.show_event_tms:
            
            # print('TMS EVENT NUMBER:', self.tms_evt_no)
            # print('TMS SLICE NUMBER:', self.tms_slice_no)
            # print('TMS SPILL NUMBER:', self.tms_spill_no)
            # print('TMS RUN NUMBER:', self.tms_run_no)

            ### SPILL FOR ND LAr ev_idx = self.events['id'] 
            
            spill_diff = np.abs(self.tms_spill_no - ev_idx )
            trigger = np.argwhere(spill_diff < 1).reshape(1,-1)[0]
            print('WHHHHHHEEEERE ???? ', self.tms_spill_no[trigger[0]], ev_idx)
            print('Trigger', trigger)
            # charge_time = (event["unix_ts"] + event["ts_start"]/ 1e7)
            # #print("Charge Time:", charge_time)
            # #print("Hit t_drift max:", hits['t_drift'].max())
            # #print("Minerva time 1:", self.minerva_times[0])
            # #print("Minerva times:", len(self.minerva_times))
            # # find the index of the minerva_times that matches the charge_time
            # tms_charge_time_diffs = np.abs(self.minerva_times - np.full_like(self.minerva_times, charge_time))
            # trigger = np.argwhere(tms_charge_time_diffs < 0.5).reshape(1,-1)[0] # changed from 0.5 acceptance window
            # #print("Time Differences:", tms_charge_time_diffs[trigger])
            #print("Trigger:", trigger)
            xs = []
            ys = []
            zs = []
            qs = []
            for trig in trigger:
                track_hit_pos = self.tms_track_hit_pos[trig]
#                print(trig, track_hit_pos.shape)
                if (len(track_hit_pos)==0):
                    continue
                # print(self.tms_track_hit_pos[trig][1])
                # print(self.tms_track_hit_pos[trig][2])
                for idx in range(200):
                    if track_hit_pos[0][idx][0]==0: #<-1e8:
                        # print('out!')
                        break
                    else:
                        # print('Track found', track_hit_pos[0][idx][0], track_hit_pos[0][idx][1], track_hit_pos[0][idx][2])
                        xs.append(track_hit_pos[0][idx][0]/10.) # convert mm to cm        
                        ys.append(track_hit_pos[0][idx][1]/10.)        
                        zs.append(track_hit_pos[0][idx][2]/10.)        
                        qs.append(10.)      #### LINDA HACK !!! TO BE FIXED !!!
                        
            tms = {'tmsx': xs, 'tmsy':ys, 'tmsz':zs, 'tmsq':qs}
            if len(tms['tmsx']) == 0:
                print("No matched TMS events for ND LAr Event ", ev_id)
                self.show_event_tms = False
            else:
                # Prepare color map for tms (currently not used in plotting anywhere)
                tms_norm = mpl.colors.LogNorm(vmin=min(tms['tmsq']),vmax=max(tms['tmsq']))
                tms_cmap = cmr.get_sub_cmap('cmr.torch_r', 0.13,0.95) 
                mtms = plt.cm.ScalarMappable(norm=tms_norm, cmap=tms_cmap)

        # Set figure title (uses event information loaded in this method)
        if self.show_tms:
            title_y = 0.97
            subtitle_y = 0.95
            subtitle_truth_y = 0.935
            watermark_y = 0.91
            watermark_x = 0.87
        else:
            title_y = 0.945
            subtitle_y = 0.915
            subtitle_truth_y = 0.895
            watermark_y = 0.835
            watermark_x = 0.905

        interaction_str = " "
        if self.single_neutrino : 
            interaction_str += pdgToString(event_nu)
            
            interaction_str += " ({:.1f} GeV) + ".format(event_enu)
    
            interaction_str += event_target_symbol
            if (event_isCC == True):
                interaction_str += ' (CC '
            else:
                interaction_str += ' (NC '
    
            interaction_str += event_reaction_str + ') \u2192 ' + pdgToString(event_lep_pdg) + " ({:.1f} GeV)".format(event_lep_E) + outstring
        
        print(interaction_str)
        self.fig.text(s=" Run %i, Subrun %i" %
                          (event_run, event_subrun), x=0.05, y=title_y,\
                            size=24, weight='bold', ha='left', linespacing=1)
        self.fig.text(x=0.051, y=subtitle_y, s=" Event %i: %s UTC, nHits: %i" % (ev_id, event_datetime, self.hits_per_event),\
                            size=15, ha='left', style='italic', linespacing=1)
        if (self.is_mc):
            self.fig.text(x=0.051, y=subtitle_truth_y, s="%s" % (interaction_str),\
                            size=10, ha='left', style='italic', linespacing=1)
            
        self.fig.text(watermark_x, watermark_y, data_sim_watermark, fontsize=watermark_fs, color='black', alpha=0.15,
                      ha='right', va='center', weight='bold', style='italic', rotation=0)# zorder=-1)
        

        # Return event information including charge, light, and tms datasets for plotting and all color scale information
        if self.show_event_tms and self.show_event_light:
            return hits, tms, light_wvfms, mcharge, mtms, mlight, cmap, tms_cmap, light_cmap, charge_norm, tms_norm, light_norm, cmap_zero, light_cmap_zero
        elif self.show_event_tms and not self.show_event_light:
            return hits, tms, mcharge, mtms, cmap, tms_cmap, charge_norm, tms_norm, cmap_zero
        elif self.show_event_light and not self.show_event_tms:
            return hits, light_wvfms, mcharge, mlight, cmap, light_cmap, charge_norm, light_norm, cmap_zero, light_cmap_zero
        else:
            return hits, mcharge, cmap, charge_norm, cmap_zero
    

    def set_axes(self, cmap, mcharge, cmap_zero, mlight=None):

        '''
            cmap: charge color map
            mcharge: charge color map scale
            cmap_zero: charge color map scale with lower values (for backgrounds)
            mlight: light color map scale
        '''

        # Show DUNE logo
        self.ax_dune_logo.axis('off')
        self.ax_dune_logo.imshow(self.dune_logo_png)

        
        # Only show 3D view of ND LAr once if including TMS 
        if (self.show_tms is False):
            self.ax_subexp_logo.axis('off')
            self.ax_subexp_logo.imshow(self.subexp_logo_png)

            # Set axes for 3D canvas (Beam, Drift, Vertical)
            self.ax_bdv.set_xlabel('\nBeam Axis (z) [cm]', fontsize=14, weight='bold', linespacing=2) #z
            self.ax_bdv.set_ylabel('\nDrift Axis (x) [cm]', fontsize=14, weight='bold', linespacing=2) #x
            self.ax_bdv.set_zlabel('\nVertical Axis (y) [cm]', fontsize=14, weight='bold', linespacing=2) #y
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



        # NOTE: xlim and ylim for all 2D subplots are the same and based on the 
        #       maximum and minimum boundary values of the longest detector axis (beam, Z)
        # Set axes for Beam vs Drift (ZX) canvas
        self.ax_bd.set_ylabel('Drift Axis (x) [cm]', fontsize=14, weight='bold')
        self.ax_bd.set_ylim(self.geometry.attrs['lar_detector_bounds'][0][0], \
            self.geometry.attrs['lar_detector_bounds'][1][0])
        self.ax_bd.set_xlim(self.geometry.attrs['lar_detector_bounds'][0][2],\
            self.geometry.attrs['lar_detector_bounds'][1][2])
        self.ax_bd.tick_params(axis='y', which='major', labelsize=12.5)
        self.ax_bd.set_xticks([])
#        self.ax_bd.set_yticks(np.arange(-60,61,20))

        # Set axes for Beam vs Vertical (ZY) canvas
        self.ax_bv.set_xlabel('Beam Axis (z) [cm]', fontsize=14, weight='bold')
        self.ax_bv.set_ylabel('Vertical Axis [cm]', fontsize=14, weight='bold')
        self.ax_bv.set_xlim(self.geometry.attrs['lar_detector_bounds'][0][2],\
            self.geometry.attrs['lar_detector_bounds'][1][2])
        self.ax_bv.set_ylim(self.geometry.attrs['lar_detector_bounds'][0][1], \
            self.geometry.attrs['lar_detector_bounds'][1][1])
        self.ax_bv.tick_params(axis='both', which='major', labelsize=12.5)
#        self.ax_bv.set_xticks(np.arange(-60,61,20))
#        self.ax_bv.set_yticks(np.arange(-60,61,20))

        # Set axes for Drift vs Vertical (XY) canvas
        if (self.show_tms is False):
            self.ax_dv.set_xlabel('Drift Axis (x) [cm]', fontsize=14, weight='bold')
            #self.ax_dv.set_ylabel('Vertical Axis [cm]', fontsize=14) # Currently not showing y-axis label bc overlap with left subplot
            self.ax_dv.set_xlim(self.geometry.attrs['lar_detector_bounds'][0][0],\
                self.geometry.attrs['lar_detector_bounds'][1][0])
            self.ax_dv.set_ylim(self.geometry.attrs['lar_detector_bounds'][0][1], \
                self.geometry.attrs['lar_detector_bounds'][1][1])
            self.ax_dv.tick_params(axis='x', which='major', labelsize=12.5)
    #        self.ax_dv.set_xticks(np.arange(-60,61,20))
            self.ax_dv.set_yticks([])

        # Set tms axis if using
        if self.show_tms:
            zlim_tms = [self.geometry.attrs['lar_detector_bounds'][0][2]- 150, \
                self.geometry.attrs['lar_detector_bounds'][1][2] + 1100 ]
            xlim_tms = [self.geometry.attrs['lar_detector_bounds'][0][0] - 200, \
                self.geometry.attrs['lar_detector_bounds'][1][0] + 200 ]
            ylim_tms = [self.geometry.attrs['lar_detector_bounds'][0][1] - 200, \
                self.geometry.attrs['lar_detector_bounds'][1][1] + 50] 
            self.ax_tms.set_xlabel('\nBeam Axis [cm]', fontsize=14, weight='bold', linespacing=2) #z
            self.ax_tms.set_ylabel('\nDrift Axis [cm]', fontsize=14, weight='bold', linespacing=2) #x
            self.ax_tms.set_zlabel('\nVertical Axis [cm]', fontsize=14, weight='bold', linespacing=2) #y
            self.ax_tms.tick_params(axis='both', which='major', labelsize=12.5)
            self.ax_tms.set_xlim(zlim_tms[0], zlim_tms[1]) # beam
            self.ax_tms.set_ylim(xlim_tms[0], xlim_tms[1]) # drift
            self.ax_tms.set_zlim(ylim_tms[0], ylim_tms[1]) # vertical
            self.ax_tms.grid(False)
            self.ax_tms.xaxis.pane.fill = True
            self.ax_tms.yaxis.pane.fill = True
            self.ax_tms.zaxis.pane.fill = True
            self.ax_tms.xaxis.pane.set_facecolor('white')#cmap_zero(0))
            self.ax_tms.yaxis.pane.set_facecolor('white')#cmap_zero(0))
            self.ax_tms.zaxis.pane.set_facecolor('white')#cmap_zero(0))
            z_length = 10
            aspect_ratio_x = (zlim_tms[1] - zlim_tms[0])
            aspect_ratio_y = (xlim_tms[1] - xlim_tms[0])*z_length/aspect_ratio_x
            aspect_ratio_z = (ylim_tms[1] - ylim_tms[0])*z_length/aspect_ratio_x
            print(aspect_ratio_x, aspect_ratio_y, aspect_ratio_z)
            
#            self.ax_tms.set_box_aspect([7.75,3.5,3], zoom=1.65)
            self.ax_tms.set_box_aspect([z_length, aspect_ratio_y, aspect_ratio_z], zoom=1.65)
            self.ax_tms.view_init(azim=-75, elev=17) # default -75, 17
            #self.ax_tms.set_aspect('auto')

            # Update bounds of 2D views if SHOW TMS
            self.ax_bd.set_ylim(xlim_tms[0], xlim_tms[1])
            self.ax_bd.set_xlim(zlim_tms[0], zlim_tms[1])
            self.ax_bv.set_xlim(zlim_tms[0], zlim_tms[1])
            self.ax_bv.set_ylim(ylim_tms[0], ylim_tms[1])

            # Draw rough boundaries of TMS in 3D projection
            for j in range(2):    
                for k in range(2):
                    self.ax_tms.plot([self.tms_rough_bounds[2][j], self.tms_rough_bounds[2][j]], \
                                  [self.tms_rough_bounds[0][k], self.tms_rough_bounds[0][k]], \
                                  [self.tms_rough_bounds[1][0], self.tms_rough_bounds[1][1]], color='black', alpha=0.1, clip_on = False)
                    self.ax_tms.plot([self.tms_rough_bounds[2][j], self.tms_rough_bounds[2][j]], \
                                  [self.tms_rough_bounds[0][0], self.tms_rough_bounds[0][1]], \
                                  [self.tms_rough_bounds[1][k], self.tms_rough_bounds[1][k]], color='black', alpha=0.1, clip_on = False)
                    self.ax_tms.plot([self.tms_rough_bounds[2][0], self.tms_rough_bounds[2][1]], \
                                  [self.tms_rough_bounds[0][k], self.tms_rough_bounds[0][k]], \
                                  [self.tms_rough_bounds[1][j], self.tms_rough_bounds[1][j]], color='black', alpha=0.1, clip_on = False)


            # Draw rough boundaries of TMS in 2D projections
            for j in range(2):    
                self.ax_bd.plot([self.tms_rough_bounds[2][j], self.tms_rough_bounds[2][j]], \
                                  [self.tms_rough_bounds[0][0], self.tms_rough_bounds[0][1]], color='black', alpha=0.1, clip_on = False)
                self.ax_bd.plot([self.tms_rough_bounds[2][0], self.tms_rough_bounds[2][1]], \
                                  [self.tms_rough_bounds[0][j], self.tms_rough_bounds[0][j]], color='black', alpha=0.1, clip_on = False)
                self.ax_bv.plot([self.tms_rough_bounds[2][j], self.tms_rough_bounds[2][j]], \
                                  [self.tms_rough_bounds[1][0], self.tms_rough_bounds[1][1]], color='black', alpha=0.1, clip_on = False)
                self.ax_bv.plot([self.tms_rough_bounds[2][0], self.tms_rough_bounds[2][1]], \
                                  [self.tms_rough_bounds[1][j], self.tms_rough_bounds[1][j]], color='black', alpha=0.1, clip_on = False)

            x_steel = [ -350, -175, 0, 175 ] 
            steel_width = 175
            for j in range(4):
                self.ax_bd.fill([self.tms_steel_regions_bounds[0], self.tms_steel_regions_bounds[0], \
                                  self.tms_steel_regions_bounds[1], self.tms_steel_regions_bounds[1]], \
                                 [x_steel[j], x_steel[j]+steel_width, \
                                  x_steel[j]+steel_width, x_steel[j] ],\
                                 color='gray', alpha=0.05)
                self.ax_bd.fill([self.tms_steel_regions_bounds[1], self.tms_steel_regions_bounds[1], \
                                  self.tms_steel_regions_bounds[2], self.tms_steel_regions_bounds[2]], \
                                 [x_steel[j], x_steel[j]+steel_width, \
                                  x_steel[j]+steel_width, x_steel[j] ],\
                                 color='gray', alpha=0.15)
                # self.ax_bd.fill([self.tms_steel_regions_bounds[2], self.tms_steel_regions_bounds[2], \ # relevant for the new TMS geometry
                #                   self.tms_rough_bounds[2][1], self.tms_rough_bounds[2][1]], \
                #                  [x_steel[j], x_steel[j]+steel_width, \
                #                   x_steel[j]+steel_width, x_steel[j] ],\
                #                  color='gray', alpha=0.15)

            self.ax_bv.fill([self.tms_steel_regions_bounds[0], self.tms_steel_regions_bounds[0], \
                                  self.tms_steel_regions_bounds[1], self.tms_steel_regions_bounds[1]], \
                                 [self.tms_rough_bounds[1][0], self.tms_rough_bounds[1][1], \
                                  self.tms_rough_bounds[1][1], self.tms_rough_bounds[1][0] ],\
                                 color='gray', alpha=0.05)
            self.ax_bv.fill([self.tms_steel_regions_bounds[1], self.tms_steel_regions_bounds[1], \
                                  self.tms_steel_regions_bounds[2], self.tms_steel_regions_bounds[2]], \
                                 [self.tms_rough_bounds[1][0], self.tms_rough_bounds[1][1], \
                                  self.tms_rough_bounds[1][1], self.tms_rough_bounds[1][0] ],\
                                 color='gray', alpha=0.15)
            # self.ax_bv.fill([self.tms_steel_regions_bounds[2], self.tms_steel_regions_bounds[2], \  # relevant for the new TMS geometry
            #                       self.tms_rough_bounds[2][1], self.tms_rough_bounds[2][1]], \
            #                      [self.tms_rough_bounds[1][0], self.tms_rough_bounds[1][1], \
            #                       self.tms_rough_bounds[1][1], self.tms_rough_bounds[1][0] ],\
            #                      color='gray', alpha=0.15)



            
        
#         For top views the steel gaps are at x = 0 cm and x = ±186 cm (both are in the middle of the 2 cm air gap between the steel plates of size 1.85 m)
# In addition for side views (this can be considered optional of course) the thin steel part starts at z = 11210 mm, the thick steel part at z = 14460 mm and the double thick steel part at z = 17520 mm.
# The exact scintillator start position in z is then z = 11185 mm (or 11133 mm for double front) and the end position z = 18535 mm [both scintillator positions are the middle of the scintillator with a width of 17 mm]
                

        # Plot cathodes + module outlines for 3D view(s) and fill module volumes + plot cathodes for 2D LAr volume projections           
        for i in range(len(self.geometry.attrs['module_RO_bounds'])):

            # Plot cathodes for XYZ (beam, drift, vertical) 3D view (and 2x2+tms view if using):
            X_cathode, Y_cathode, Z_cathode = make_x_plane(self.geometry.attrs['module_RO_bounds'][i][0][1], self.geometry.attrs['module_RO_bounds'][i][1][1], \
                                                           self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2], 
                                                           self.geometry.attrs['module_RO_bounds'][i][0][0]+self.geometry.attrs['max_drift_distance']+self.geometry.attrs['cathode_thickness']/2)
            
            # if self.show_tms:
            #     self.ax_tms.plot_surface(Z_cathode,X_cathode,Y_cathode, color='gainsboro', alpha=0.05, clip_on = False)
            # else:
            #     self.ax_bdv.plot_surface(Z_cathode,X_cathode,Y_cathode, color='gainsboro', alpha=0.1) 

            for j in range(2):
                for k in range(2):
                    # Plot outlines of modules for XYZ (beam, drift, vertical) 3D view:
                    if self.show_tms:
                        self.ax_tms.plot([self.geometry.attrs['module_RO_bounds'][i][j][2], self.geometry.attrs['module_RO_bounds'][i][j][2]], \
                                [self.geometry.attrs['module_RO_bounds'][i][0][0], self.geometry.attrs['module_RO_bounds'][i][1][0]], \
                                [self.geometry.attrs['module_RO_bounds'][i][k][1], self.geometry.attrs['module_RO_bounds'][i][k][1]], color='black', alpha=0.05, clip_on = False)

                        self.ax_tms.plot([self.geometry.attrs['module_RO_bounds'][i][j][2], self.geometry.attrs['module_RO_bounds'][i][j][2]], \
                                [self.geometry.attrs['module_RO_bounds'][i][k][0], self.geometry.attrs['module_RO_bounds'][i][k][0]], \
                                [self.geometry.attrs['module_RO_bounds'][i][0][1], self.geometry.attrs['module_RO_bounds'][i][1][1]], color='black', alpha=0.05, clip_on = False)

                        self.ax_tms.plot([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                                [self.geometry.attrs['module_RO_bounds'][i][j][0], self.geometry.attrs['module_RO_bounds'][i][j][0]], \
                                [self.geometry.attrs['module_RO_bounds'][i][k][1], self.geometry.attrs['module_RO_bounds'][i][k][1]], color='black', alpha=0.05, clip_on = False)

                    # self.ax_bd.plot([self.geometry.attrs['module_RO_bounds'][i][j][2], self.geometry.attrs['module_RO_bounds'][i][j][2]], \
                        #         [self.geometry.attrs['module_RO_bounds'][i][k][0], self.geometry.attrs['module_RO_bounds'][i][k][0]], \
                        #         [self.geometry.attrs['module_RO_bounds'][i][0][1], self.geometry.attrs['module_RO_bounds'][i][1][1]], color='black', alpha=0.05, clip_on = False)
                    
                    
                    else:
                        self.ax_bdv.plot([self.geometry.attrs['module_RO_bounds'][i][j][2], self.geometry.attrs['module_RO_bounds'][i][j][2]], \
                            [self.geometry.attrs['module_RO_bounds'][i][0][0], self.geometry.attrs['module_RO_bounds'][i][1][0]], \
                            [self.geometry.attrs['module_RO_bounds'][i][k][1], self.geometry.attrs['module_RO_bounds'][i][k][1]], color='black', alpha=0.05, clip_on = False)

                        self.ax_bdv.plot([self.geometry.attrs['module_RO_bounds'][i][j][2], self.geometry.attrs['module_RO_bounds'][i][j][2]], \
                            [self.geometry.attrs['module_RO_bounds'][i][k][0], self.geometry.attrs['module_RO_bounds'][i][k][0]], \
                            [self.geometry.attrs['module_RO_bounds'][i][0][1], self.geometry.attrs['module_RO_bounds'][i][1][1]], color='black', alpha=0.05, clip_on = False)

                        self.ax_bdv.plot([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                            [self.geometry.attrs['module_RO_bounds'][i][j][0], self.geometry.attrs['module_RO_bounds'][i][j][0]], \
                            [self.geometry.attrs['module_RO_bounds'][i][k][1], self.geometry.attrs['module_RO_bounds'][i][k][1]], color='black', alpha=0.05, clip_on = False)

                    
                    # Plot module boundaries on beam-drift 2D projection
                    self.ax_bd.plot([self.geometry.attrs['module_RO_bounds'][i][j][2], self.geometry.attrs['module_RO_bounds'][i][j][2]], \
                                [self.geometry.attrs['module_RO_bounds'][i][0][0], self.geometry.attrs['module_RO_bounds'][i][1][0]], \
                                        color='black', alpha=0.05, clip_on = False)

                    self.ax_bd.plot([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                                [self.geometry.attrs['module_RO_bounds'][i][j][0], self.geometry.attrs['module_RO_bounds'][i][j][0]], \
                                        color='black', alpha=0.05, clip_on = False)
            

            # Fill modules for ZX (beam, drift) projections:
            # self.ax_bd.fill([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][0][2], \
            #           self.geometry.attrs['module_RO_bounds'][i][1][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
            #          [self.geometry.attrs['module_RO_bounds'][i][0][0], self.geometry.attrs['module_RO_bounds'][i][1][0], \
            #           self.geometry.attrs['module_RO_bounds'][i][1][0], self.geometry.attrs['module_RO_bounds'][i][0][0]],\
            #          color=cmap_zero(0), alpha=0.85)

            # Module boundaries on ZX (beam, drift) projections:
            if (i<5):
                self.ax_bv.plot([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][0][2]], \
                        [self.geometry.attrs['module_RO_bounds'][i][0][1], self.geometry.attrs['module_RO_bounds'][i][1][1]], \
                        color='black', alpha=0.1, clip_on = False)
                self.ax_bv.plot([self.geometry.attrs['module_RO_bounds'][i][1][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                        [self.geometry.attrs['module_RO_bounds'][i][0][1], self.geometry.attrs['module_RO_bounds'][i][1][1]], \
                        color='black', alpha=0.1, clip_on = False)
                self.ax_bv.plot([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                                [self.geometry.attrs['module_RO_bounds'][i][0][1], self.geometry.attrs['module_RO_bounds'][i][0][1]], \
                        color='black', alpha=0.1, clip_on = False)
                self.ax_bv.plot([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                                [self.geometry.attrs['module_RO_bounds'][i][1][1], self.geometry.attrs['module_RO_bounds'][i][1][1]], \
                        color='black', alpha=0.1, clip_on = False)


            
            # Only two modules represented in ZY and XY projections (i.e. projections after this line)
            if i >= 34: continue

            # # Fill modules for ZY (beam, vertical) projections: ##### LINDA TO UNDERSTAND THIS !
            # self.ax_bv.fill([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][0][2], \
            #           self.geometry.attrs['module_RO_bounds'][i][1][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
            #          [self.geometry.attrs['module_RO_bounds'][i][0][1], self.geometry.attrs['module_RO_bounds'][i][1][1], \
            #           self.geometry.attrs['module_RO_bounds'][i][1][1], self.geometry.attrs['module_RO_bounds'][i][0][1]],\
            #          color=cmap_zero(0), alpha=0.85)


            # Plot cathodes for XY (drift, vertical) projections:
            if (self.show_tms is False): 
                # # Fill modules for XY (drift, vertical) projections:
                # self.ax_dv.fill([self.geometry.attrs['module_RO_bounds'][i+1][0][0], self.geometry.attrs['module_RO_bounds'][i+1][0][0], \
                #       self.geometry.attrs['module_RO_bounds'][i+1][1][0], self.geometry.attrs['module_RO_bounds'][i+1][1][0]], \
                #      [self.geometry.attrs['module_RO_bounds'][i+1][0][1], self.geometry.attrs['module_RO_bounds'][i+1][1][1], \
                #       self.geometry.attrs['module_RO_bounds'][i+1][1][1], self.geometry.attrs['module_RO_bounds'][i+1][0][1]],\
                #      color=cmap_zero(0), alpha=0.85)

                self.ax_dv.plot([self.geometry.attrs['module_RO_bounds'][i+1][0][0] + self.geometry.attrs['max_drift_distance']+self.geometry.attrs['cathode_thickness']/2, \
                      self.geometry.attrs['module_RO_bounds'][i+1][0][0]+self.geometry.attrs['max_drift_distance']+self.geometry.attrs['cathode_thickness']/2],\
                     [self.geometry.attrs['module_RO_bounds'][i+1][0][1], self.geometry.attrs['module_RO_bounds'][i+1][1][1]], \
                      color='gainsboro', alpha=0.9, clip_on = False)

            
            # # If light is plotted in this event, the cathodes for ZX (beam, drift) projections are plotted after light in this view
            # if not self.show_event_light:
            #     # Plot cathodes for ZX (beam, drift) projections:
            #     for i in range(len(self.geometry.attrs['module_RO_bounds'])):
            #         self.ax_bd.plot([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
            #                  [self.geometry.attrs['module_RO_bounds'][i][0][0]+self.geometry.attrs['max_drift_distance']+self.geometry.attrs['cathode_thickness']/2, \
            #                   self.geometry.attrs['module_RO_bounds'][i][0][0]+self.geometry.attrs['max_drift_distance']+self.geometry.attrs['cathode_thickness']/2],\
            #                   color='gainsboro', alpha=0.9, linewidth=2,solid_capstyle='butt')
        
        # Set up colorbars            
        if self.show_colorbars:
            # Set charge colorbar
            cbar = self.fig.colorbar(mcharge, cax=self.cbar_ax, label=r'Charge [$10^3$ e]', orientation='horizontal')
            cbar.set_label(r'Charge [$\mathbf{10^3}$ e]', size=13, weight='bold')
            self.cbar_ax.tick_params(labelsize=11.5)

            if self.show_event_light:
                # Set light colorbar
                light_cbar = self.fig.colorbar(mlight, cax=self.light_cbar_ax, label=r'Light [ADC Counts]', orientation = 'horizontal')
                light_cbar.set_label(r'Light [ADC Counts]', size=13, weight='bold')
                self.light_cbar_ax.tick_params(labelsize=11.5)


    def display_event(self, ev_id):

        self.clear_axes()
        hits, *event_info = self.get_event(ev_id)
        if self.show_event_tms and self.show_event_light:
            tms, light_wvfms, mcharge, mtms, mlight, cmap, tms_cmap, light_cmap, charge_norm, tms_norm, light_norm, cmap_zero, light_cmap_zero = event_info
        elif self.show_event_tms and not self.show_event_light:
            tms, mcharge, mtms, cmap, tms_cmap, charge_norm, tms_norm, cmap_zero = event_info
        elif self.show_event_light and not self.show_event_tms:
            light_wvfms, mcharge, mlight, cmap, light_cmap, charge_norm, light_norm, cmap_zero, light_cmap_zero = event_info
        else:
            mcharge, cmap, charge_norm, cmap_zero = event_info

        # Check whether event is a beam trigger event
        if self.is_beam_event:
            print("Event " + str(ev_id) + " is a beam trigger event")
        else:
            print("Event " + str(ev_id) + " is NOT a beam trigger event")

        # Reset hits if charge threshold is set
        if self.charge_threshold is not None:
            hits = hits[hits['Q'] > self.charge_threshold]
            self.hits_per_event = len(hits)



        if self.show_tms:
            self.tms_lar_points = self.ax_tms.scatter(hits['z'], hits['x'], hits['y'], lw=0, ec='C0', \
                    c=cmap(charge_norm(hits['Q'])), s=2.75, alpha=1, marker="s", clip_on = False)

        if self.show_event_tms:
            # print(tms['tmsx'], tms['tmsy'], tms['tmsz'])
            #self.ax_tms.scatter(tms['mz'], tms['mx'], tms['my'], lw=0, ec='C0', \ # Currently not using tms energy info
            #                c=tms_cmap(tms_norm(tms['mq'])), s=15, alpha=1)       # Currently not using tms energy info
            self.ax_tms.scatter(tms['tmsz'], tms['tmsx'], tms['tmsy'], lw=0, ec='C0', \
                            c='red', s=4, alpha=1, clip_on = False) 
            self.bd_points = self.ax_bd.scatter(tms['tmsz'], tms['tmsx'], lw=0, ec='C0', c=cmap(
                    charge_norm(tms['tmsq'])), s=2.75, alpha=1, marker="s", clip_on = False)
            self.bv_points = self.ax_bv.scatter(tms['tmsz'], tms['tmsy'], lw=0, ec='C0', c=cmap(
                    charge_norm(tms['tmsq'])), s=2.75, alpha=1, marker="s", clip_on = False)
        if (self.show_tms is False):
            # Plot hits in 3D views first so that cathodes/anodes go over the hits
            self.bdv_points = self.ax_bdv.scatter(hits['z'], hits['x'], hits['y'], lw=0, ec='C0', \
                            c=cmap(charge_norm(hits['Q'])), s=0.75, alpha=1, marker="s", clip_on = False)

        
        # Set up axes for all views        
        if self.show_event_light:
            self.set_axes(cmap, mcharge, cmap_zero, mlight)
        else:
            self.set_axes(cmap, mcharge, cmap_zero)

        # Plot light information
        if self.show_event_light:
            self.plot_light(light_wvfms, light_cmap, light_norm, light_cmap_zero)

        # Plot 2D charge hits
        if self.hist_projection:

            ## LC: HIST PROJECTION NOT GREAT FOR NOW, SO USING SCATTER PLOT INSTEAD
            nz_bins = int((self.geometry.attrs['lar_detector_bounds'][1][2]-self.geometry.attrs['lar_detector_bounds'][0][2]) )*2 ## LC: one bin 0.5cm
            ny_bins = int((self.geometry.attrs['lar_detector_bounds'][1][1]-self.geometry.attrs['lar_detector_bounds'][0][1]) )*2
            nx_bins = int((self.geometry.attrs['lar_detector_bounds'][1][0]-self.geometry.attrs['lar_detector_bounds'][0][0]) )*2
            print ('N bins', nx_bins, ny_bins, nz_bins)
            z_bins = np.linspace(self.geometry.attrs['lar_detector_bounds'][0][2],self.geometry.attrs['lar_detector_bounds'][1][2],nz_bins) 
            y_bins = np.linspace(self.geometry.attrs['lar_detector_bounds'][0][1],self.geometry.attrs['lar_detector_bounds'][1][1],ny_bins)
            x_bins = np.linspace(self.geometry.attrs['lar_detector_bounds'][0][0],self.geometry.attrs['lar_detector_bounds'][1][0],nx_bins)

            
            bd_charge_hist, _, _ = np.histogram2d(hits['z'], hits['x'], bins=[z_bins,x_bins],weights=hits['Q']) ## LC: HACK! The /4 normalises the charge because of the binning above
            bd_charge_hist_masked = np.where(bd_charge_hist==0, np.nan, bd_charge_hist) # TO DO: SHOULD CHARGE ==0 BE MASKED?
            ZX_Z, ZX_X = np.meshgrid(z_bins[:-1], x_bins[:-1])
            self.bd_points = self.ax_bd.pcolormesh(ZX_Z, ZX_X, bd_charge_hist_masked.T, cmap=cmap, norm=charge_norm, alpha=1)

            bv_charge_hist, _, _ = np.histogram2d(hits['z'], hits['y'], bins=[z_bins,y_bins],weights=hits['Q'])
            bv_charge_hist_masked = np.where(bv_charge_hist==0, np.nan, bv_charge_hist) # TO DO: SHOULD CHARGE ==0 BE MASKED?
            ZY_Z, ZY_Y = np.meshgrid(z_bins[:-1], y_bins[:-1])
            self.bv_points = self.ax_bv.pcolormesh(ZY_Z, ZY_Y, bv_charge_hist_masked.T, cmap=cmap, norm=charge_norm, alpha=1)

            if (self.show_tms is False):
                dv_charge_hist, _, _ = np.histogram2d(hits['x'], hits['y'], bins=[x_bins,y_bins],weights=hits['Q'])
                dv_charge_hist_masked = np.where(dv_charge_hist==0, np.nan, dv_charge_hist) # TO DO: SHOULD CHARGE ==0 BE MASKED?
                XY_X, XY_Y = np.meshgrid(x_bins[:-1], y_bins[:-1])
                self.dv_points = self.ax_dv.pcolormesh(XY_X, XY_Y, dv_charge_hist_masked.T, cmap=cmap, norm=charge_norm, alpha=1)
        else:
            self.bd_points = self.ax_bd.scatter(hits['z'], hits['x'], lw=0, ec='C0', c=cmap(
                    charge_norm(hits['Q'])), s=0.75, alpha=1, marker="s", clip_on = False)
            self.bv_points = self.ax_bv.scatter(hits['z'], hits['y'], lw=0, ec='C0', c=cmap(
                    charge_norm(hits['Q'])), s=0.75, alpha=1, marker="s", clip_on = False)
            if (self.show_tms is False):
                self.dv_points = self.ax_dv.scatter(hits['x'], hits['y'], lw=0, ec='C0', c=cmap(
                    charge_norm(hits['Q'])), s=0.75, alpha=1, marker="s", clip_on = False)
            
        return hits, cmap, charge_norm


    def plot_light(self, light_wvfms, light_cmap, light_norm, light_cmap_zero):
        
        #acl_det_ids = [0,4,8,12] # Used in previous version of code to identify ACL detectors

        sipm_abs_pos = {}
        for i,j in itertools.product(range(light_wvfms[0].shape[0]),range(light_wvfms[0].shape[1])):
            sipm_abs_pos[(i,j)]=self.sipm_abs_pos[(i,j)][0]
        
        print("Sum light waveforms and plot light in ZX (beam, drift) projection")
        # Sum light waveforms and plot light in ZX (beam, drift) projection
        for x,z in itertools.product(self.sipm_unique_x,self.sipm_unique_z):
            if x==-1: continue
            if z==-1: continue
            this_xz_sum = 0
            # Get light sum for each SiPM in this z,x position
            for i,j in itertools.product(range(light_wvfms[0].shape[0]),range(light_wvfms[0].shape[1])):
                #det_id = self.light_det_id[(i,j)][0]
                pos=sipm_abs_pos[(i,j)]
                if pos[0]==-1:
                    continue
                if (abs(pos[0]-x) < 0.5) and (pos[2]==z):
                    this_xz_sum += light_wvfms[0][i,j].sum()

            # Plot light in ZX projection if SiPM sum is over threshold
            # This is a bit different from ZY because of previous issues with SiPM x positions
            for i in range(len(self.geometry.attrs['module_RO_bounds'])):
                for j in range(2):
                    z_offset = ((-1)**(j+1))*1.25
                    if (abs(z-self.geometry.attrs['module_RO_bounds'][i][j][2]) < 2): # LC changed from 1 to 2 to go through this statement
                        if this_xz_sum==0:
                            cmap_value = light_cmap_zero(0)
                        elif this_xz_sum > self.light_threshold:
                            cmap_value = light_cmap(light_norm(this_xz_sum))
                            if (abs(x-self.geometry.attrs['module_RO_bounds'][i][0][0]) < 2): # LC changed from 1 to 2 to go through this statement
                                self.ax_bd.plot([self.geometry.attrs['module_RO_bounds'][i][j][2]+z_offset, self.geometry.attrs['module_RO_bounds'][i][j][2]+z_offset], \
                                         [self.geometry.attrs['module_RO_bounds'][i][0][0], self.geometry.attrs['module_RO_bounds'][i][0][0]+self.geometry.attrs['max_drift_distance']],\
                                          color=cmap_value, alpha=1, linewidth=3.5, solid_capstyle='butt')
                            elif (abs(x-self.geometry.attrs['module_RO_bounds'][i][1][0]) < 2): # LC changed from 1 to 2 to go through this statement
                                self.ax_bd.plot([self.geometry.attrs['module_RO_bounds'][i][j][2]+z_offset, self.geometry.attrs['module_RO_bounds'][i][j][2]+z_offset], \
                                         [self.geometry.attrs['module_RO_bounds'][i][1][0]-self.geometry.attrs['max_drift_distance'], self.geometry.attrs['module_RO_bounds'][i][1][0]] ,\
                                          color=cmap_value, alpha=1, linewidth=3.5, solid_capstyle='butt')
        
        # print("Plot cathodes for ZX (beam, drift) projections (done for charge only case in set_axes method)")
        # # Plot cathodes for ZX (beam, drift) projections (done for charge only case in set_axes method):
        # for i in range(len(self.geometry.attrs['module_RO_bounds'])):
        #     self.ax_bd.plot([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
        #              [self.geometry.attrs['module_RO_bounds'][i][0][0]+self.geometry.attrs['max_drift_distance']+self.geometry.attrs['cathode_thickness']/2, \
        #               self.geometry.attrs['module_RO_bounds'][i][0][0]+self.geometry.attrs['max_drift_distance']+self.geometry.attrs['cathode_thickness']/2],\
        #               color='gainsboro', alpha=0.9, linewidth=2,solid_capstyle='butt')

        print("Plot light in ZY (beam, vertical) projection")
        # Plot light in ZY (beam, vertical) projection
        for z,y in itertools.product(self.sipm_unique_z,self.sipm_unique_y):
            if z==-1: continue
            if y==-1: continue
            this_zy_sum = 0
            # Get light sum for each SiPM in this z,y position
            for i,j in itertools.product(range(light_wvfms[0].shape[0]),range(light_wvfms[0].shape[1])):
                pos=sipm_abs_pos[(i,j)]
                #det_id = self.light_det_id[(i,j)][0]
                if pos[0]==-1:
                    continue
                if (pos[2]==z) and (pos[1]==y):
                    this_zy_sum += light_wvfms[0][i,j].sum()
            # Plot light in ZY projection if SiPM sum is over threshold
            for i in range(len(self.geometry.attrs['module_RO_bounds'])):
                for j in range(2):
                    z_offset = ((-1)**(j+1))*1.25
                    if this_zy_sum==0:
                        cmap_value = light_cmap_zero(0)
                    elif this_zy_sum > self.light_threshold:
                        cmap_value = light_cmap(light_norm(this_zy_sum))
                        if (abs(z-self.geometry.attrs['module_RO_bounds'][i][j][2]) < 2): # LC changed from 1 to 2 to go through this statement
                            self.ax_bv.plot([self.geometry.attrs['module_RO_bounds'][i][j][2]+z_offset, self.geometry.attrs['module_RO_bounds'][i][j][2]+z_offset], \
                                     [y-2.25, y+2.25],color=cmap_value, alpha=1, linewidth=3.5, solid_capstyle='butt')

                
        print("Plot light in XY (drift, vertical) projection")
        # Plot light in XY projection               
        for x,y in itertools.product(self.sipm_unique_x,self.sipm_unique_y):
            if x==-1: continue
            if y==-1: continue
            this_xy_sum = 0
            # Get light sum for each SiPM in this x,y position
            for i,j in itertools.product(range(light_wvfms[0].shape[0]),range(light_wvfms[0].shape[1])):
                pos=sipm_abs_pos[(i,j)]
                #det_id = self.light_det_id[(i,j)][0]
                if pos[0]==-1:
                    continue
                if (abs(pos[0]-x) < 0.5) and (pos[1]==y):
                    this_xy_sum += light_wvfms[0][i,j].sum()
            # Plot light in XY projection if SiPM sum is over threshold
            for i in range(len(self.geometry.attrs['module_RO_bounds'])):
                for j in range(2):
                    x_offset = ((-1)**(j+1))*1.25
                    if this_xy_sum==0:
                        cmap_value = light_cmap_zero(0)
                    elif this_xy_sum > self.light_threshold:
                        cmap_value = light_cmap(light_norm(this_xy_sum))
                        if (abs(x-self.geometry.attrs['module_RO_bounds'][i][j][0]) < 2): # LC changed from 1 to 2 to go through this statement
                            self.ax_dv.plot([self.geometry.attrs['module_RO_bounds'][i][j][0]+x_offset, self.geometry.attrs['module_RO_bounds'][i][j][0]+x_offset], \
                                    [y-2.25, y+2.25],color=cmap_value, alpha=1, linewidth=4, solid_capstyle='butt')


        print("Plot light in 3D projection")
        # Plot light for XYZ (beam, drift, vertical) 3D view (and ND LAr+tms view if using)
        for i,j in itertools.product(range(light_wvfms[0].shape[0]),range(light_wvfms[0].shape[1])):
            pos=sipm_abs_pos[(i,j)]
            #det_id = self.light_det_id[(i,j)][0]
            if pos[0]==-1:
                continue
            this_xyz_sum = light_wvfms[0][i,j].sum()
            if this_xyz_sum < self.light_threshold: continue
            if this_xyz_sum==0:
                cmap_value = light_cmap_zero(0)
            else:
                cmap_value = light_cmap(light_norm(this_xyz_sum))
            z_diffs = [abs(pos[2]-self.geometry.attrs['module_RO_bounds'][k][l][2]) for k,l in itertools.product(range(len(self.geometry.attrs['module_RO_bounds'])),range(2))]
            z_diffs = np.reshape(z_diffs, np.shape(self.geometry.attrs['module_RO_bounds'][:,:,2]))
            # print(z_diffs)
            min_z_diff = np.where(abs(z_diffs)<2) # LC changed this from 1 to 2 to get anything passing
            z_pos = self.geometry.attrs['module_RO_bounds'][:,:,2][min_z_diff][0]
            found_x = 0
            for k in range(len(self.geometry.attrs['module_RO_bounds'])):
                # print (abs(pos[0]-self.geometry.attrs['module_RO_bounds'][k][0][0]), abs(pos[0]-self.geometry.attrs['module_RO_bounds'][k][1][0]))
                if (abs(pos[0]-self.geometry.attrs['module_RO_bounds'][k][0][0]) < 2):  # LC changed this from 1 to 2 to get anything passing
                    x1 = self.geometry.attrs['module_RO_bounds'][k][0][0]
                    x2 = self.geometry.attrs['module_RO_bounds'][k][0][0]+self.geometry.attrs['max_drift_distance']
                    found_x = 1
                elif (abs(pos[0]-self.geometry.attrs['module_RO_bounds'][k][1][0]) < 2):  # LC changed this from 1 to 2 to get anything passing
                    x1 = self.geometry.attrs['module_RO_bounds'][k][1][0]-self.geometry.attrs['max_drift_distance']
                    x2 = self.geometry.attrs['module_RO_bounds'][k][1][0]
                    found_x = 1
                if found_x ==1:
                    light_x, light_y, light_z = make_z_plane(x1,x2, pos[1]-2.25, pos[1]+2.25,z_pos)
                    # print('Found', light_z )
                    if self.show_event_tms:
                        self.ax_tms.plot_surface(light_z,light_x,light_y,color=cmap_value, alpha=0.9, shade=False)
                    else:
                        self.ax_bdv.plot_surface(light_z,light_x,light_y,color=cmap_value, alpha=0.9, shade=False)
                    break
                else: continue



def GENIE_code(code):

    GENIE_dict = { 
        "QE"        : 1,
        "1Kaon"     : 2,
        "DIS"       : 3,
        "RES"       : 4,
        "COH"       : 5,
        "DFR"       : 6,
        "NuEEL"     : 7,
        "IMD"       : 8,
        "AMNuGamma" : 9,
        "MEC"       : 10,
        "CEvNS"     : 11,
        "IBD"       : 12,
        "GLR"       : 13,
        "IMDAnh"    : 14,
        "PhotonCOH" : 15,
        "PhotonRES" : 16,
        "1Pion"     : 17,
        "DMEL"      : 101,
        "DMDIS"     : 102,
        "DME"       : 103
        }    
    GENIE_dict_2 = dict((v,k) for k,v in GENIE_dict.items())
    return GENIE_dict_2[code]


def pdgToString(code):

    tmp = Particle.from_pdgid(code)
    output = '$' + tmp.latex_name + '$'
    return output

# Helper functions outside of main class
# Both allow for plotting planes in 3D space and are basically the same
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

def make_y_plane(x1, x2, z1, z2, y):
    x = np.linspace(x1, x2, 100)
    z = np.linspace(z1, z2, 100)
    X, Z = np.meshgrid(x, z)
    Y = np.full(X.shape, y)
    return X, Y, Z

