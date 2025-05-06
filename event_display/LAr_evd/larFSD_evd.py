# Based on Elise Hinkle 2x2 Event display
# Contact: nicolas.sallin@unibe.ch, @Nicolas Sallin on DUNE Slack
# INSTALLING AND IMPORTANT PYTHON MODULES
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
    'numpy', 'h5py', 'pandas', 'matplotlib'
]    
# , 'sqlalchemy', 'cmasher', 'IPython', 'PyMuPDF', 'pillow', 'uproot', 'h5flow', 'ipywidgets'
# ]

installed_packages = {pkg.key for pkg in pkg_resources.working_set}
missing_packages = [pkg for pkg in required_packages if pkg not in installed_packages]

if missing_packages:
    for package in missing_packages:
        install(package)

# Import modules
import pymupdf
import numpy as np
import pandas as pd
# from datetime import datetime
# import ipywidgets as widgets
from io import BytesIO
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.proto_nd_flow.util.lut import LUT
# from h5flow.core import resources
# import itertools
import math
import h5py
import cmasher as cmr
from IPython.display import display, clear_output
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from PIL import Image
# from math import fabs
from time import sleep
# import uproot


class LArEventDisplayFSD:
    ''' 
        Class to set up interactive FSD display for files run through proto_nd_flow

        Inputs to this class are as follows:

            - filedir          (str):   Path to input file (minus filename)
            - filename         (str):   Name of flow file
            - runsdb           (str):   Optional, run database (default = None)
            - nhits_min        (int):   Minimum number of hits (threshold) for events to be made available (default: 1)
            - nhits_max        (int):   Maximum number of hits allowed in event for events to be made available (default: 1e10)
            - ntrigs           (int):   Number of external triggers threshold for events to be made available (default: 0)
            - show_light       (bool):  Show light information in display (default: True)
            - light_event_only (bool):  Only the events containing light information are displayed (default: False)
            - show_colorbars   (bool):  Display the color bars (default: True)
            - hist_projection  (bool):  If True, hits are binned in a 2D histogram for the 2D charge hit projections. Bins with 0 charge are not displayed.
                                        If False, hits are plotted as scatter points for the 2D charge hit projections. (default: True)
            - charge_threshold (float): Threshold for charge hits to be shown (default: None)
            - light_threshold  (float): Threshold for light to be shown (default: 1000 ADC counts)
            - beam_only        (bool):  Only show beam events (default: False)
            - show_fig_wfms    (bool):  Show a second figure with the light waveforms and SiPM coordinates (default: False)
            - ouput_path       (str):   Path where to save the figures, if None: save in LAr_evd/FSD_eventDisplay/ (default: None)

        Class methods:

            - run()                 :   Interactive script to browse through the selected events
            - display_event(ev_id)  :   Display the initialized plots corresponding to the event 'ev_id'
            - save_plots(which_plot, events_id):    Save the plots in the output repository
            - events_id()           :   Return a list of events number that passed the selection

        In order to run the display and interactively flip through events, set up a Jupyter Notebook, import everything in this file,
        and execute the run() method, e.g.:

        from larFSD_evd import *
        plt.ion()

        d = '/path/to/file/'
        f = 'filename'
        evd = LArEventDisplay(filedir=d, filename=f, nhits=1, ntrigs=1)
        evd.run()

        Alternatively, you can display a specific event by calling the display_event() method with the event ID as an argument, e.g.:
        evd.display_event(123). 

            
    '''

    # Initialize class
    def __init__(self, filedir, filename, runsdb=None, 
                 nhits_min=1, nhits_max=1e10, ntrigs=0, show_light=True, 
                 show_colorbars=True, charge_threshold=None, light_threshold=1000, beam_only=False, 
                 hist_projection=True, light_event_only=False, show_fig_wfms=False, output_path=None):
        
        # Open files
        f = h5py.File(filedir+filename, 'r')

        # Set general class-level variables from inputs
        self.filedir = filedir
        self.filename = filename
        try:
            self.runsdb = runsdb
            self.all_subruns_db = pd.read_sql_table('All_global_subruns', runsdb)
        except:
            self.runsdb = None
            self.all_subruns_db = None
        
        # Set the 'show_*' and other selection variables
        self.show_light = show_light
        self.show_event_light = show_light
        self.light_event_only = light_event_only
        self.show_fig_wfms = show_fig_wfms

        # Check the consistency of the arguments
        if (self.show_fig_wfms is True):
            try:
                if (self.show_fig_wfms is True) and (self.show_light is False):
                    raise ValueError("The parameter show_light need to be set to True to display the waveform plot")
            except ValueError as e:
                print(e)
                # Asking the user for input to allow light display
                while True:
                    user_input = input("Do you want to set 'show_light' to True? [y/n]: ").strip().lower()
                    
                    if user_input in ["y", "yes"]:
                        print("'show_light' is set to True")
                        self.show_light = True
                        break
                    elif user_input in ["n", "no"]:
                        print("'show_light' is kept False, the waveform plot is not displayed")
                        self.show_fig_wfms = False
                        break
                    else:
                        print("Invalid input. Please enter 'y' or 'n'.")

        
        self.show_colorbars = show_colorbars
        self.beam_only = beam_only
        self.hist_projection = hist_projection

        # Set the thresholds
        self.charge_threshold = charge_threshold
        self.light_threshold = max(light_threshold, 10) # light_threshold should be bigger than 0 because of the log scale of the plot colorbar
        
        # Set directory for saving files and finding logo image files
        self.lar_evd_dir = os.path.dirname(__file__)
        
        # Set the output path
        if (output_path is None):
            self.output_path = os.path.join(os.path.dirname(__file__) ,'FSD_eventDisplay/')
        else:
            self.output_path = os.path.abspath(output_path)

        # Load DUNE
        dune_logo = os.path.join(self.lar_evd_dir, 'DUNElogo.pdf')
        self.dune_logo_pdf = pymupdf.open(dune_logo)
        # subexp_logo=os.path.join(self.lar_evd_dir, '2x2logo.pdf')
        # self.subexp_logo_pdf = fitz.open(subexp_logo)#mpimg.imread(subexp_logo)

        # Resize DUNE logo image to fit in display
        dune_logo_page = self.dune_logo_pdf.load_page(0)
        dune_logo_pixmap = dune_logo_page.get_pixmap(matrix=pymupdf.Matrix(5, 5), dpi=600)
        dune_logo_image = Image.frombytes("RGB", [dune_logo_pixmap.width, dune_logo_pixmap.height], dune_logo_pixmap.samples)
        dune_logo_buf = BytesIO()
        dune_logo_image.save(dune_logo_buf, format='png')
        dune_logo_buf.seek(0)
        self.dune_logo_png = mpimg.imread(dune_logo_buf, format='png')

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

        # Load charge hits dataset
        self.hits_dset = 'calib_prompt_hits'
        self.hits_full = f['charge/'+self.hits_dset+'/data']
        self.hits_ref = f['charge/events/ref/charge/'+self.hits_dset+'/ref']
        self.hits_region = f['charge/events/ref/charge/'+self.hits_dset+'/ref_region']
        self.hits_per_event = nhits_min

        # Load light events, waveform datasets and light geometry info if using
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

            # Filter event based on light info if requested
            if self.light_event_only:                
                self.charge_light_eventIDs = np.sort(self.charge_light_ref[:, 0])
                mask_charge_light_event = np.isin(self.events['id'], self.charge_light_eventIDs)                     
                self.events = self.events[mask_charge_light_event]

        
        
        # Get detector boundaries (For FSD the detector and module boundaries are the same so 10cm for light readout and 3 cm for visibility are added 
        #   for illustrative purpose of the light readout)
        light_RO_width = 10 #[cm]
        self.light_RO_shift = 3 #[cm]

        drift_bounds = self.geometry.attrs['lar_detector_bounds'][:,0] + np.array([-light_RO_width, light_RO_width]) + np.array([-self.light_RO_shift, self.light_RO_shift])
        beam_bounds = self.geometry.attrs['lar_detector_bounds'][:,2] + np.array([-light_RO_width, light_RO_width]) + np.array([-self.light_RO_shift, self.light_RO_shift])
        vert_bounds = self.geometry.attrs['lar_detector_bounds'][:,1]
        
        # Defined some module properties
        self.N_sipm_side = int(60)
        self.N_side_tpc = 2
        self.N_tpc = 2
        self.N_sipm_lightModule = 6
        self.N_LCM_lightModule = 3

        # Width ratio between 2D and 3D plots
        widthRatio2Dto3D = 0.9

        # Compute ratio for hight and width of 2D plots
        heightRatio_2D = [1., np.sum(abs(vert_bounds))/np.sum(abs(drift_bounds))]
        widthRatio_2D = [np.sum(abs(beam_bounds)), np.sum(abs(drift_bounds))]/(np.sum(abs(beam_bounds))+np.sum(abs(drift_bounds)))
        
        # Set up figure and subplots
        self.fig = plt.figure(layout='none', figsize=(15,9))
        self.axes_mosaic = [["ax_bd", "ax_subexp_logo", "ax_dvb"],["ax_bv", "ax_dv", "ax_dvb"],]
        self.axes_dict = self.fig.subplot_mosaic(self.axes_mosaic, \
                                                per_subplot_kw={"ax_dvb": {"projection": "3d"}}, \
                                                height_ratios= heightRatio_2D, width_ratios=[widthRatio2Dto3D*widthRatio_2D[0], widthRatio2Dto3D*widthRatio_2D[1], 1])
        
        # Adjust subplot positioning
        self.fig.subplots_adjust(bottom=0.1)
        self.fig.subplots_adjust(wspace=0.02, hspace=0.02)

        # Set up DUNE logo axis
        self.ax_dune_logo = self.fig.add_axes([0.56, 0.895, 0.43, 0.099])
        self.ax_dune_logo.axis('off')
        self.ax_dune_logo.imshow(self.dune_logo_png)

        # Initialize axes for display
        # 3D plot
        self.ax_dvb = self.axes_dict["ax_dvb"]
        # 2D plots
        self.ax_bd = self.axes_dict["ax_bd"]
        self.ax_bv = self.axes_dict["ax_bv"]
        self.ax_dv = self.axes_dict["ax_dv"]
        # Logo
        self.ax_subexp_logo = self.axes_dict["ax_subexp_logo"]
        # Colorbar
        if self.show_colorbars:
            if self.show_light:
                self.cbar_ax = self.fig.add_axes([0.08, 0.001, 0.38, 0.025])
                self.light_cbar_ax = self.fig.add_axes([0.51, 0.001, 0.38, 0.025])
            else:
                self.cbar_ax = self.fig.add_axes([0.145, 0.001, 0.675, 0.025])

        # Initialize point collections for plotting
        self.dvb_points = self.ax_dvb.scatter([], [], [])
        self.bd_points = self.ax_bd.scatter([], [])
        self.bv_points = self.ax_bv.scatter([], [])
        self.dv_points = self.ax_dv.scatter([], [])

        # Set axes for 3D canvas (Beam, Drift, Vertical)
        self.ax_dvb.view_init(azim=-60, elev=17, vertical_axis='y')
        self.ax_dvb.set_zlabel('Beam Axis (z) [cm]', fontsize=8, weight='bold')
        self.ax_dvb.set_xlabel('Drift Axis (x) [cm]', fontsize=8, weight='bold') 
        self.ax_dvb.set_ylabel('\nVertical Axis (y) [cm]', fontsize=10, weight='bold')

        self.ax_dvb.grid(False)
        self.ax_dvb.xaxis.pane.fill = True
        self.ax_dvb.yaxis.pane.fill = True
        self.ax_dvb.zaxis.pane.fill = True
        self.ax_dvb.xaxis.pane.set_facecolor('white')
        self.ax_dvb.yaxis.pane.set_facecolor('white')
        self.ax_dvb.zaxis.pane.set_facecolor('white')
        self.ax_dvb.tick_params(axis='both', which='major', labelsize=8)
        self.ax_dvb.set_xlim(drift_bounds)
        self.ax_dvb.set_ylim(vert_bounds[0], vert_bounds[1]+10)
        self.ax_dvb.set_zlim(beam_bounds)
        self.ax_dvb.set_aspect('equal', adjustable='box')
        
        

        # Set axes for Beam vs Drift (ZX) canvas (carful the matplotlib direction does not match the detector convention geometry)
        self.ax_bd.set_ylabel('Drift Axis (x) [cm]', fontsize=10, weight='bold')
        self.ax_bd.set_xlim(beam_bounds)
        self.ax_bd.set_ylim(drift_bounds)
        self.ax_bd.set_aspect('equal')
        self.ax_bd.set_xticklabels([])

        # Set axes for Beam vs Vertical (ZY) canvas
        self.ax_bv.set_xlabel('Beam Axis (z) [cm]', fontsize=10, weight='bold')
        self.ax_bv.set_ylabel('Vertical Axis [cm]', fontsize=10, weight='bold')
        self.ax_bv.set_xlim(beam_bounds)
        self.ax_bv.set_ylim(vert_bounds)
        self.ax_bv.set_aspect('equal')

        # Set axes for Drift vs Vertical (XY) canvas
        self.ax_dv.set_xlabel('Drift Axis (x) [cm]', fontsize=10, weight='bold')
        self.ax_dv.set_xlim(drift_bounds)
        self.ax_dv.set_ylim(vert_bounds)
        self.ax_dv.set_yticklabels([])
        self.ax_dv.set_aspect('equal')
        self.ax_dv.set_anchor((-0.5, 0.5))

        # Set axes for the logo
        self.ax_subexp_logo.axis('off')
        self.ax_subexp_logo.set_xlim(drift_bounds)
        self.ax_subexp_logo.set_ylim(drift_bounds)
        self.ax_subexp_logo.set_aspect('equal')

        # Set the module boundaries
        self.min_module_bounds = self.geometry.attrs['module_RO_bounds'][0][0]
        self.max_module_bounds = self.geometry.attrs['module_RO_bounds'][0][1]

        # Set the difference vertices and edges
        module_vertice = np.array([
            [self.min_module_bounds[0], self.min_module_bounds[1], self.min_module_bounds[2]], # bottom
            [self.max_module_bounds[0], self.min_module_bounds[1], self.min_module_bounds[2]], 
            [self.max_module_bounds[0], self.min_module_bounds[1], self.max_module_bounds[2]],
            [self.min_module_bounds[0], self.min_module_bounds[1], self.max_module_bounds[2]], 
            [self.min_module_bounds[0], self.max_module_bounds[1], self.min_module_bounds[2]], #top
            [self.max_module_bounds[0], self.max_module_bounds[1], self.min_module_bounds[2]],
            [self.max_module_bounds[0], self.max_module_bounds[1], self.max_module_bounds[2]],
            [self.min_module_bounds[0], self.max_module_bounds[1], self.max_module_bounds[2]]
            ])

        module_edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]

        bv_module_edges = [
            [0, 4], [4, 7], [7, 3], [0, 3] 
        ]

        dv_module_edges = [
            [3, 7], [7, 6], [6, 2], [2, 3] 
        ]

        bd_module_edges = [
            [4, 5], [5, 6], [6, 7], [7, 4]
        ]

        # Set up the module outlines
        for edge in module_edges:
            self.ax_dvb.plot([module_vertice[edge[0], 0], module_vertice[edge[1], 0]], 
                    [module_vertice[edge[0], 1], module_vertice[edge[1], 1]], 
                    [module_vertice[edge[0], 2], module_vertice[edge[1], 2]], color='black', zorder=5)

        for bv_edge in bv_module_edges:
            self.ax_bv.plot([module_vertice[bv_edge[0], 2], module_vertice[bv_edge[1], 2]], 
                    [module_vertice[bv_edge[0], 1], module_vertice[bv_edge[1], 1]], color='black')

        for dv_edge in dv_module_edges:
            self.ax_dv.plot([module_vertice[dv_edge[0], 0], module_vertice[dv_edge[1], 0]],  
                    [module_vertice[dv_edge[0], 1], module_vertice[dv_edge[1], 1]], color='black')
            
        for bd_edge in bd_module_edges:
            self.ax_bd.plot([module_vertice[bd_edge[0], 2], module_vertice[bd_edge[1], 2]],  
                    [module_vertice[bd_edge[0], 0], module_vertice[bd_edge[1], 0]], color='black')

        # Plot the cathode
        self.ax_dvb.plot_surface(np.array([[0,0], [0,0]]), 
                                 np.array([[self.min_module_bounds[1], self.max_module_bounds[1]], [self.min_module_bounds[1], self.max_module_bounds[1]]]), 
                                 np.array([[self.min_module_bounds[2], self.min_module_bounds[2]], [self.max_module_bounds[2], self.max_module_bounds[2]]]),
                                 color='lightgray', alpha=0.2)

        # Postion of the middle of the cathode
        self.midcathode_pos = 0

        # For illustrative purpose we want the cathode to have a minimal thickness on the 2D drawing
        width_cathode = float(max(2, self.geometry.attrs['cathode_thickness']))
                                 
        self.ax_dv.add_patch(
            Rectangle(xy=(self.midcathode_pos-0.5*width_cathode, self.min_module_bounds[1]), width=width_cathode,
                      height=(self.max_module_bounds[1]-self.min_module_bounds[1]), facecolor='lightgray', edgecolor='lightgray', alpha=0.9)
        )

        self.ax_bd.add_patch(
            Rectangle(xy=(self.min_module_bounds[2], self.midcathode_pos-0.5*width_cathode), width=(self.max_module_bounds[2]-self.min_module_bounds[2]),
                      height=width_cathode, facecolor='lightgray', edgecolor='lightgray', alpha=0.9)
        )

        # Set up the light readout axes
        self.v_binEdges_light = np.arange(-0.5, self.N_sipm_side+0.5, 1, dtype=float)
        self.projLeft_binEdges_light = np.array([-0.5, 0.5])
        self.projRight_binEdges_light = np.array([0.5, 1.5])
        self.d_binEdges_light = np.array([-0.5, 0.5, 1.5])

        light_zeroWeights = np.zeros((1,self.N_sipm_side)) 

        # Beam - Vertical plot
        self.ax_bv_light_side0 = self.ax_bv.inset_axes([beam_bounds[0], vert_bounds[0], light_RO_width, vert_bounds[1]-vert_bounds[0]], 
                                                      transform=self.ax_bv.transData)
        self.ax_bv_light_side1 = self.ax_bv.inset_axes([beam_bounds[1]-light_RO_width, vert_bounds[0], light_RO_width, vert_bounds[1]-vert_bounds[0]], 
                                                      transform=self.ax_bv.transData)
        
        self.h_bv_light_side0 = self.ax_bv_light_side0.hist2d([], [], bins=[self.projLeft_binEdges_light,self.v_binEdges_light], edgecolor='k', 
                                                            weights=light_zeroWeights, cmap='Greys', lw=0.5, alpha=0.6)[3]
        self.h_bv_light_side1 = self.ax_bv_light_side1.hist2d([], [], bins=[self.projRight_binEdges_light,self.v_binEdges_light], edgecolor='k', 
                                                              weights=light_zeroWeights, cmap='Greys', lw=0.5, alpha=0.6)[3]

        self.ax_bv_light_side0.axis('off')
        self.ax_bv_light_side1.axis('off')

        # Drift - Vert. plot
        self.ax_dv_light_tpc0 = self.ax_dv.inset_axes([drift_bounds[0], vert_bounds[0], light_RO_width, vert_bounds[1]-vert_bounds[0]], 
                                                      transform=self.ax_dv.transData)
        self.ax_dv_light_tpc1 = self.ax_dv.inset_axes([drift_bounds[1]-light_RO_width, vert_bounds[0], light_RO_width, vert_bounds[1]-vert_bounds[0]], 
                                                      transform=self.ax_dv.transData)
        
        self.h_dv_light_tpc0 = self.ax_dv_light_tpc0.hist2d([], [], bins=[self.projLeft_binEdges_light, self.v_binEdges_light], edgecolor='k', 
                                                            weights=light_zeroWeights, cmap='Greys', lw=0.5, alpha=0.6)[3]
        self.h_dv_light_tpc1 = self.ax_dv_light_tpc1.hist2d([], [], bins=[self.projRight_binEdges_light, self.v_binEdges_light], edgecolor='k', 
                                                              weights=light_zeroWeights, cmap='Greys', lw=0.5, alpha=0.6)[3]

        self.ax_dv_light_tpc0.axis('off')
        self.ax_dv_light_tpc1.axis('off')

        # Beam - Drift plot
        self.ax_bd_light_side0 = self.ax_bd.inset_axes([beam_bounds[0], self.min_module_bounds[0], light_RO_width, self.max_module_bounds[0]-self.min_module_bounds[0]], 
                                                      transform=self.ax_bd.transData)
        self.ax_bd_light_side1 = self.ax_bd.inset_axes([beam_bounds[1]-light_RO_width, self.min_module_bounds[0], light_RO_width, 
                                                        self.max_module_bounds[0]-self.min_module_bounds[0]], transform=self.ax_bd.transData)
        
        self.h_bd_light_side0 = self.ax_bd_light_side0.hist2d([], [], bins=[self.projLeft_binEdges_light, self.d_binEdges_light], edgecolor='k', weights=light_zeroWeights,
                                                            cmap='Greys', lw=0.8, alpha=0.8)[3]
        self.h_bd_light_side1 = self.ax_bd_light_side1.hist2d([], [], bins=[self.projRight_binEdges_light, self.d_binEdges_light], edgecolor='k', weights=light_zeroWeights,
                                                              cmap='Greys', lw=0.8, alpha=0.8)[3]

        self.ax_bd_light_side0.axis('off')
        self.ax_bd_light_side1.axis('off')
                            
        # Compute the SiPM contour to be used with a plot_surface function for the 3D plot
        self.d3D_binEdges_light = np.array([self.min_module_bounds[0], self.midcathode_pos, self.max_module_bounds[0]])
        self.v3D_binEdges_light = np.linspace(self.min_module_bounds[1], self.max_module_bounds[1], 61, endpoint=True)
        
        self.d3D_binEdges_light_grid, self.v3D_binEdges_light_grid = np.meshgrid(self.d3D_binEdges_light, self.v3D_binEdges_light)
        self.b3D_binEdges_light_grid_side0 = np.full_like(self.d3D_binEdges_light_grid, self.min_module_bounds[2]-self.light_RO_shift)
        self.b3D_binEdges_light_grid_side1 = np.full_like(self.d3D_binEdges_light_grid, self.max_module_bounds[2]+self.light_RO_shift)

        color_values = np.zeros((60,2))
        norm = Normalize()
        colors = plt.get_cmap('Greys')(norm(color_values))

        self.surf_dvb_light_side0 = self.ax_dvb.plot_surface(self.d3D_binEdges_light_grid, self.v3D_binEdges_light_grid, self.b3D_binEdges_light_grid_side0, 
                                                             rstride=1, cstride=1, facecolors=colors, shade=False, alpha=0.6, edgecolor='black', zorder=0)
        self.surf_dvb_light_side1 = self.ax_dvb.plot_surface(self.d3D_binEdges_light_grid, self.v3D_binEdges_light_grid, self.b3D_binEdges_light_grid_side1, 
                                                             rstride=1, cstride=1, facecolors=colors, shade=False, alpha=0.6, edgecolor='black', zorder=10)

        # Initialize a second figure to plot the light waveforms
        if (self.show_fig_wfms == True):

            # Init. the figure
            self.fig_wfms = plt.figure(layout='none', figsize=(15,32))
            self.axes_wfms_mosaic = [["ax_wfms_tpc0_side0", "ax_wfms_tpc0_side1", ".", "ax_wfms_tpc1_side0", "ax_wfms_tpc1_side1", ".",
                                      "ax_lightChan_tpc0_side0", "ax_lightChan_tpc0_side1", ".", "ax_lightChan_tpc1_side0", "ax_lightChan_tpc1_side1"]]
            self.axes_wfms_dict = self.fig_wfms.subplot_mosaic(self.axes_wfms_mosaic, width_ratios=[1, 1, 0.05, 1, 1, 0.25, 1, 1, 0.05, 1, 1])
            
            # List of axes
            self.axs_wfms = []
            self.axs_lightChan = []
            shift = 0

            # List of histograms and their parameters
            self.hs_wfms = []
            self.hs_lightChan = []
            self.txt_lightChan = []
            self.d_binEdges_light_modules = np.array([[drift_bounds[0], 0], [0, drift_bounds[1]]])
            self.v_binEdges_light_modules = np.linspace(vert_bounds[0], vert_bounds[1], num=self.N_sipm_side+1)
            light_module_zeroWeights = np.zeros((self.N_tpc*self.N_side_tpc,self.N_sipm_side))

            self.axs_p_wfms = []
            lw_light_module = 2


            for tpc in range(self.N_tpc):
                self.axs_wfms.append([])
                self.axs_lightChan.append([])
                self.axs_p_wfms.append([])

                self.hs_wfms.append([])
                self.hs_lightChan.append([])

                self.txt_lightChan.append([])

                self.fig_wfms.text(0.22+0.19*tpc, 0.89, f'TPC {tpc}', va='baseline', ha='center', weight='bold', fontsize=13, color='k')
                self.fig_wfms.text(0.62+0.19*tpc, 0.89, f'TPC {tpc}', va='baseline', ha='center', weight='bold', fontsize=13, color='k')

                for side in range(self.N_side_tpc):
                    i_wfms= tpc*2+side+shift
                    i_lightChan = tpc*2+side+self.N_tpc+self.N_side_tpc+2+shift

                    if (self.axes_wfms_mosaic[0][i_wfms] == "."):
                        shift += 1
                        i_wfms += 1
                        i_lightChan += 1

                    # Add the axes to the resp. list and set the parameters
                    self.axs_wfms[tpc].append(self.axes_wfms_dict[self.axes_wfms_mosaic[0][i_wfms]])

                    self.axs_wfms[tpc][side].grid(False)
                    self.axs_wfms[tpc][side].tick_params(axis='both', which='major', labelsize=8)
                    
                    self.axs_wfms[tpc][side].set_xlim(np.sort([drift_bounds[side],0]))
                    self.axs_wfms[tpc][side].set_ylim(vert_bounds)
                    self.axs_wfms[tpc][side].set_title(f'Side {side}', fontsize=13, color='k')

                    self.axs_lightChan[tpc].append(self.axes_wfms_dict[self.axes_wfms_mosaic[0][i_lightChan]])
                    self.axs_lightChan[tpc][side].grid(False)
                    self.axs_lightChan[tpc][side].tick_params(axis='both', which='major', labelsize=8)
                    self.axs_lightChan[tpc][side].set_title(f'Side {side}', fontsize=13, color='k')
                    
                    self.axs_lightChan[tpc][side].set_xlim(np.sort([drift_bounds[side],0]))
                    self.axs_lightChan[tpc][side].set_ylim(vert_bounds)

                    if (tpc+side != 0):
                        self.axs_wfms[tpc][side].set_yticklabels([])
                        self.axs_lightChan[tpc][side].set_yticklabels([])


                    # Init. the histograms
                    self.hs_wfms[tpc].append(self.axs_wfms[tpc][side].hist2d([], [], bins=[self.d_binEdges_light_modules[side], self.v_binEdges_light_modules], edgecolor='none', 
                                                                             weights=light_module_zeroWeights, cmap='Greys', alpha=0.8,  zorder=0)[3])
                    self.hs_lightChan[tpc].append(self.axs_lightChan[tpc][side].hist2d([], [], bins=[self.d_binEdges_light_modules[side], self.v_binEdges_light_modules],
                                                                                        edgecolor='none', weights=light_module_zeroWeights, cmap='Greys', 
                                                                                        alpha=0.8,  zorder=0)[3])
                    
                    N_sipm_LCM = self.N_sipm_lightModule/self.N_LCM_lightModule
                    self.txt_lightChan[tpc].append([])
                    self.axs_p_wfms[tpc].append([])

                    for n_sipm_side in range(self.N_sipm_side): 
                       
                        self.txt_lightChan[tpc][side].append(self.axs_lightChan[tpc][side].text(0, 0, ""))

                        d_width = self.d_binEdges_light_modules[:][1] - self.d_binEdges_light_modules[:][0]
                        v_width = self.v_binEdges_light_modules[1] - self.v_binEdges_light_modules[0]
                        self.axs_p_wfms[tpc][side].append(self.axs_wfms[tpc][side].inset_axes([self.d_binEdges_light_modules[side][0], self.v_binEdges_light_modules[n_sipm_side], d_width[side], v_width], 
                                                      transform=self.axs_wfms[tpc][side].transData))
                        self.axs_p_wfms[tpc][side][n_sipm_side].set_facecolor('none')
                        self.axs_p_wfms[tpc][side][n_sipm_side].set_xticklabels([])
                        self.axs_p_wfms[tpc][side][n_sipm_side].set_yticklabels([])
                        self.axs_p_wfms[tpc][side][n_sipm_side].tick_params(which='major', width=0, length=0)
                        self.axs_p_wfms[tpc][side][n_sipm_side].grid(visible=True, lw=0.5, color='gray', alpha=0.8, zorder=10)
                        
                        # Draw the modules outline
                        if (n_sipm_side%self.N_sipm_lightModule == 0):
                            n_light_module = int(n_sipm_side/self.N_sipm_lightModule)

                            if (n_light_module%2 == 0): # LCM module
                                for n_LCM in range(self.N_LCM_lightModule):
                                    i_sipm = int(n_sipm_side+n_LCM*N_sipm_LCM)
                                    moduleOutline_wfms = Rectangle(
                                        (self.d_binEdges_light_modules[side][0], self.v_binEdges_light_modules[i_sipm]), 
                                        self.d_binEdges_light_modules[side][1] - self.d_binEdges_light_modules[side][0],  # width
                                        self.v_binEdges_light_modules[int(i_sipm+N_sipm_LCM)] - self.v_binEdges_light_modules[i_sipm],  # height
                                        linewidth=lw_light_module, edgecolor='k', facecolor='none', zorder=10)
                                    self.axs_wfms[tpc][side].add_patch(moduleOutline_wfms)
                                    moduleOutline_lightChan = Rectangle(
                                        (self.d_binEdges_light_modules[side][0], self.v_binEdges_light_modules[i_sipm]), 
                                        self.d_binEdges_light_modules[side][1] - self.d_binEdges_light_modules[side][0],  # width
                                        self.v_binEdges_light_modules[int(i_sipm+N_sipm_LCM)] - self.v_binEdges_light_modules[i_sipm],  # height
                                        linewidth=lw_light_module, edgecolor='k', facecolor='none', zorder=10)
                                    self.axs_lightChan[tpc][side].add_patch(moduleOutline_lightChan)

                            else : # ArCLight module
                                i_sipm = n_sipm_side
                                moduleOutline_wfms = Rectangle(
                                    (self.d_binEdges_light_modules[side][0], self.v_binEdges_light_modules[i_sipm]), 
                                    self.d_binEdges_light_modules[side][1] - self.d_binEdges_light_modules[side][0],  # width
                                    self.v_binEdges_light_modules[i_sipm+self.N_sipm_lightModule] - self.v_binEdges_light_modules[i_sipm],  # height
                                    linewidth=lw_light_module, edgecolor='k', facecolor='none', zorder=10)
                                self.axs_wfms[tpc][side].add_patch(moduleOutline_wfms)

                                moduleOutline_lightChan = Rectangle(
                                    (self.d_binEdges_light_modules[side][0], self.v_binEdges_light_modules[i_sipm]), 
                                    self.d_binEdges_light_modules[side][1] - self.d_binEdges_light_modules[side][0],  # width
                                    self.v_binEdges_light_modules[i_sipm+self.N_sipm_lightModule] - self.v_binEdges_light_modules[i_sipm],  # height
                                    linewidth=lw_light_module, edgecolor='k', facecolor='none', zorder=10)
                                self.axs_lightChan[tpc][side].add_patch(moduleOutline_lightChan)

            # Set up colorbar axes
            if self.show_colorbars: 
                self.light_cbar_ax_wfms = self.fig_wfms.add_axes([0.15, 0.065, 0.7, 0.01])

            # Add the label for the axes
            self.fig_wfms.text(0.5, 0.085, 'Drift Axes (x) [cm]', ha='center', va='baseline', fontsize=13, color='k', weight='bold')
            self.axs_wfms[0][0].set_ylabel('Vertical Axis (y) [cm]', fontsize=13, weight='bold')

            # Adjust subplot positioning
            self.fig_wfms.subplots_adjust(bottom=0.1)
            self.fig_wfms.subplots_adjust(wspace=0.0, hspace=0.01)

        # Information about the selection:
        print(f'Processing file {filedir+filename}')
        print(f'The output path is set to {self.output_path}')
        print(f"Number of events in the selection: {len(self.events)}")
        
 
    def run(self):
        '''
        Interactive script to browse through the selected events

        Args:
            None

        Return:
            None
        '''
    
        # Get event IDs and initialize event index
        event_ids = [ev['id'] for ev in self.events]
        ev_idx = 0

        # Display first event 
        self.display_event(event_ids[ev_idx])

        # Displays event until user input determines next action
        # User can:
        # - skip to next event (enter)
        # - skip to a specific event ID (type number)
        # - list the event ID in the selection (l)
        # - save current display to png (s)
        # - quit display (q)
        while True:

            sleep(0.5) # needed to run on nersc, otherwise sometime the output is cleared after the input box is displayed
            user_input = input(
                "Next event (Enter: go to next event/'ev_id' + Enter: skip to event 'ev_id'/ l + Enter: list the events ID in the selection/ s + Enter: save the current plot as png/ q + Enter: exit/)?\n")
            if not user_input:
                clear_output(wait=True)
                ev_idx += 1
                self.display_event(event_ids[ev_idx])
            elif user_input[0].lower() == 'q':
                sys.exit()
            elif user_input[0].lower() == 's':
                clear_output(wait=True)
                self.save_plots(events_id = event_ids[ev_idx])

            elif user_input[0].lower() == 'l':
                clear_output(wait=True)
                print(self.events_id())
            else:
                clear_output(wait=True)
                try:
                    ev_id = int(user_input)
                except ValueError as e:
                    print(f'{e}\nPlease enter a valid input')
                    continue

                self.display_event(ev_id)

                # Set the idx to the input event
                for idx in range(len(event_ids)):
                    if self.current_ev_id == event_ids[idx]:
                        ev_idx = idx
                        break
                
            if ev_idx >= len(event_ids):
                print("End of file")
                sys.exit()
    

    def _clear_axes(self):
        '''
        Clear the initialized axis 

        Args: 
            None

        Return:
            None
        '''
        # 3D plot
        if self.dvb_points in self.ax_dvb.get_children():
            self.dvb_points.remove()

        if self.surf_dvb_light_side0 in self.ax_dvb.get_children():
            self.surf_dvb_light_side0.remove()

        if self.surf_dvb_light_side1 in self.ax_dvb.get_children():
            self.surf_dvb_light_side1.remove()


        if self.dvb_points in self.ax_dvb.get_children():
            self.dvb_points.remove()

        # 2D plots

            # Charge
        if self.bv_points in self.ax_bv.get_children():
            self.bv_points.remove()

        if self.dv_points in self.ax_dv.get_children():
            self.dv_points.remove()

        if self.bd_points in self.ax_bd.get_children():
            self.bd_points.remove()

            # Light
        if self.h_dv_light_tpc0 in self.ax_dv_light_tpc0.get_children():
            self.h_dv_light_tpc0.remove()

        if self.h_dv_light_tpc1 in self.ax_dv_light_tpc1.get_children():
            self.h_dv_light_tpc1.remove()

        if self.h_bv_light_side0 in self.ax_bv_light_side0.get_children():
            self.h_bv_light_side0.remove()

        if self.h_bv_light_side1 in self.ax_bv_light_side1.get_children():
            self.h_bv_light_side1.remove()

        if self.h_bd_light_side0 in self.ax_bd_light_side0.get_children():
            self.h_bd_light_side0.remove()

        if self.h_bd_light_side1 in self.ax_bd_light_side1.get_children():
            self.h_bd_light_side1.remove()
        
        if self.show_colorbars:
            self.cbar_ax.cla()
            if self.show_light:
                self.light_cbar_ax.cla()
        self.fig.texts.clear()

        if self.show_fig_wfms:
            for tpc in range(self.N_tpc):
                for side in range(self.N_side_tpc):
                    if self.hs_wfms[tpc][side] in self.axs_wfms[tpc][side].get_children():
                        self.hs_wfms[tpc][side].remove()
                    if self.hs_lightChan[tpc][side] in self.axs_lightChan[tpc][side].get_children():
                        self.hs_lightChan[tpc][side].remove()
                    for n_sipm in range(self.N_sipm_side):    
                        if self.txt_lightChan[tpc][side][n_sipm] in self.axs_lightChan[tpc][side].get_children():
                            self.txt_lightChan[tpc][side][n_sipm].remove()
                        self.axs_p_wfms[tpc][side][n_sipm].cla()
                        self.axs_p_wfms[tpc][side][n_sipm].set_facecolor('none')
                        self.axs_p_wfms[tpc][side][n_sipm].set_xticklabels([])
                        self.axs_p_wfms[tpc][side][n_sipm].set_yticklabels([])
                        self.axs_p_wfms[tpc][side][n_sipm].tick_params(which='major', width=0, length=0)
                        self.axs_p_wfms[tpc][side][n_sipm].grid(visible=True, lw=0.5)

        return None             

    def _get_event(self, ev_id):
        '''
        Retrieve the event 'ev_id' information.

        Args:
            ev_id: ID of the event to plot

        Return: 
            hits: charge information
            light_wvfms: light waveforms
            mchareg: the ScalarMappable of the charge colorbar
            mlight: the ScalarMappable of the light colorbar
            cmap: charge colormap
            light_cmap: light colormap
            charge_norm: normalization to use for the charge colorbar
            light_norm: normalization to use for the light colorbar
            event_subrun: subrun number of event 'ev_id'
        '''

        # Get event ID information
        ev_idx = np.where(self.events['id'] == ev_id)[0][0]

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
            except:
                event_run = -1
                event_subrun = -1
            data_sim_watermark = 'DATA'
            watermark_fs = 78
        elif self.is_mc:
            event_run = 2
            event_subrun = 2
            data_sim_watermark = 'SIMULATION'
            watermark_fs = 68

        # print(f"Number of external triggers in this event: {event['n_ext_trigs']}")

        # Check if event is a beam event
        if not self.beam_only and ev_id in self.beam_events['id']:
            self.is_beam_event = True

        # Get event charge information
        hit_ref = self.hits_ref[self.hits_region[ev_id,'start']:self.hits_region[ev_id,'stop']]
        hit_ref = np.sort(hit_ref[hit_ref[:,0] == ev_id, 1])
        hits = self.hits_full[hit_ref]
        self.hits_per_event = len(hits)

        # Get event external trigger information
        exttrig_ref = self.exttrigs_ref[self.exttrigs_region[ev_id,'start']:self.exttrigs_region[ev_id,'stop']]
        exttrig_ref = np.sort(exttrig_ref[exttrig_ref[:,0] == ev_id, 1])
        exttrigs = self.exttrigs_full[exttrig_ref]

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

        charge_norm = mpl.colors.Normalize(vmin=min_charge,vmax=max_charge)
        cmap = cmr.get_sub_cmap('cmr.torch_r', 0.13,0.95)
        mcharge = plt.cm.ScalarMappable(norm=charge_norm, cmap=cmap)

        if self.show_light:
            # Get event light matches
            self.show_event_light = True
            light_matches = self.charge_light_ref[self.charge_light_region[ev_id,'start']:self.charge_light_region[ev_id,'stop']]
            light_matches = np.sort(light_matches[light_matches[:,0] == ev_id, 1])
            light = self.light_events[light_matches]

            # If no light matches, set show_event_light to False and don't carry out other light display steps
            if len(light) == 0:
                print(f"No light information for event {ev_id}")
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
                light_wvfm_peds = light_wvfm_peds_exp * np.ones((1, 1, 1, 600))
                light_wvfms = self.light_wvfms[light_wvfm_ref]["samples"] - light_wvfm_peds

                # Prepare color map for light
                light_cmap=cmr.get_sub_cmap(cmr.voltage_r, 0.0, 0.55)
                min_light = self.light_threshold
                if light_wvfms[0].sum(axis=-1).max() > min_light:
                    max_light = light_wvfms[0].sum(axis=-1).max()*2
                else:
                    max_light = min_light*10

                light_norm = colors.LogNorm(min_light,max_light)
                mlight = plt.cm.ScalarMappable(norm=light_norm, cmap=light_cmap)

        # Set figure title (uses event information loaded in this method)
        title_y = 0.945
        subtitle_y = 0.905
        watermark_y = 0.835
        watermark_x = 0.905
        self.fig.text(s=" Run %i, Subrun %i" %
                          (event_run, event_subrun), x=0.05, y=title_y,\
                            size=26, weight='bold', ha='left', linespacing=1)
        self.fig.text(x=0.051, y=subtitle_y, s=" Event %i: %s UTC, nHits: %i" % (ev_id, event_datetime, self.hits_per_event),\
                            size=22, ha='left', style='italic', linespacing=1)
        self.fig.text(watermark_x, watermark_y, data_sim_watermark, fontsize=watermark_fs, color='black', alpha=0.15,
                      ha='right', va='center', weight='bold', style='italic', rotation=0)
        

        # Return event information including charge and light for plotting and all color scale information
        if self.show_event_light:
            return hits, light_wvfms, mcharge, mlight, cmap, light_cmap, charge_norm, light_norm, event_subrun
        else:
            return hits, mcharge, cmap, charge_norm, event_subrun
    

    def _plot_colorbars(self, mcharge, mlight=None):
        '''
        Plot the colorbars associated to the 2D and 3D views.

        Args:
            mchareg: the ScalarMappable of the charge colorbar
            mlight: the ScalarMappable of the light colorbar
        '''
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


    def _plot_light(self, light_I, light_cmap, light_norm):
        '''
        Plot the integrated light data on the 2D and 3D views.

        Args:
            light_I: array containing the integral of the waveforms
            light_cmap: colormap
            light_norm: normalization to use for the colorbar

        Return:
            None
        '''

        # Initialzed the plot coordinate
        projLeft_x_data = np.zeros(60)
        projRight_x_data = projLeft_x_data+1
        y_data = np.arange(0,60,1)

        # Project the light integral w.r.t. the view
        bv_light_I = np.sum(light_I, axis=0, where=(light_I != -1))
        dv_light_I = np.sum(light_I, axis=1, where=(light_I != -1))
        bd_light_I = np.sum(light_I, axis=2, where=(light_I != -1))
       

        # Plot the 2D views
        self.h_bv_light_side0 = self.ax_bv_light_side0.hist2d(projLeft_x_data, y_data, bins=[self.projLeft_binEdges_light,self.v_binEdges_light], 
                                                            weights=bv_light_I[0], edgecolor='k', cmap=light_cmap, norm=light_norm, lw=0.5, alpha=1)[3]
        self.h_bv_light_side1 = self.ax_bv_light_side1.hist2d(projRight_x_data, y_data, bins=[self.projRight_binEdges_light,self.v_binEdges_light], 
                                                              weights=bv_light_I[1], edgecolor='k', cmap=light_cmap, norm=light_norm, lw=0.5, alpha=1)[3]

        self.h_dv_light_tpc0 = self.ax_dv_light_tpc0.hist2d(projLeft_x_data, y_data, bins=[self.projLeft_binEdges_light, self.v_binEdges_light], 
                                                            weights=dv_light_I[0], edgecolor='k', cmap=light_cmap, norm=light_norm, lw=0.5, alpha=1)[3]
        self.h_dv_light_tpc1 = self.ax_dv_light_tpc1.hist2d(projRight_x_data, y_data, bins=[self.projRight_binEdges_light, self.v_binEdges_light], 
                                                              weights=dv_light_I[1], edgecolor='k', cmap=light_cmap, norm=light_norm, lw=0.5, alpha=1)[3]

        self.h_bd_light_side0 = self.ax_bd_light_side0.hist2d([0,0], [0,1], bins=[self.projLeft_binEdges_light, self.d_binEdges_light], 
                                                            weights= bd_light_I[:,0], edgecolor='k', cmap=light_cmap, norm=light_norm, lw=1, alpha=1)[3]
        self.h_bd_light_side1 = self.ax_bd_light_side1.hist2d([1,1], [0,1], bins=[self.projRight_binEdges_light, self.d_binEdges_light], 
                                                              weights= bd_light_I[:,1], edgecolor='k', cmap=light_cmap, norm=light_norm, lw=1, alpha=1)[3]

        # Compute the color value associated to the light integral for the 3D plot
        original_shape = light_I.shape
        normed_light_I = light_norm(light_I.flatten()).reshape(original_shape)
        light_I_colors = plt.get_cmap(light_cmap)(normed_light_I)
        light_I_colors = np.transpose(light_I_colors, (2,1,0,3))
        
        # Plot the 3D view
        self.surf_dvb_light_side0 = self.ax_dvb.plot_surface(self.d3D_binEdges_light_grid, self.v3D_binEdges_light_grid, self.b3D_binEdges_light_grid_side0,
                                                            rstride=1, cstride=1, facecolors=light_I_colors[:,0, :, :], shade=False, alpha=0.3, edgecolor='black', zorder=0)
        self.surf_dvb_light_side1 = self.ax_dvb.plot_surface(self.d3D_binEdges_light_grid, self.v3D_binEdges_light_grid, self.b3D_binEdges_light_grid_side1,
                                                             rstride=1, cstride=1, facecolors=light_I_colors[:,1, :, :], shade=False, alpha=0.3, edgecolor='black', zorder=11)
        
        return None

    def _plot_wvfms(self, light_wvfms, light_I, light_cmap, light_norm, mlight):
        '''
        Plot the light waveforms.

        Args:
            light_wvfms: array containing the single waveforms
            light_I: array containing the integral of the waveforms
            light_cmap: colormap
            light_norm: normalization to use for the colorbar
            mlight: the ScalarMappable of the colorbar

        Return:
            None
        '''
        # Initialize the data, range and the plot parameter
        x_data = [np.full((self.N_sipm_side), -30), np.full((self.N_sipm_side), 30)]
        vert_bounds = self.geometry.attrs['lar_detector_bounds'][:,1]
        y_data, size_sipm = np.linspace(vert_bounds[0],vert_bounds[1], num=self.N_sipm_side, endpoint=False, retstep=True)
        y_data_center = y_data+size_sipm/2

        x_data_wvfm = np.arange(0, light_wvfms.shape[-1], 1)

        light_wvfms_range = np.array([np.min(light_wvfms), np.max(light_wvfms)])
        shift_light_wvfms_range = (light_wvfms_range[1]-light_wvfms_range[0])*0.1 # Plot parameter 
        light_wvfms_range = light_wvfms_range + np.array([-shift_light_wvfms_range, shift_light_wvfms_range])

        for tpc in range(self.N_tpc):
            for side in range(self.N_side_tpc):
                # Set the background histos
                self.hs_wfms[tpc][side]= self.axs_wfms[tpc][side].hist2d(x_data[side], y_data_center, bins=[self.d_binEdges_light_modules[side], self.v_binEdges_light_modules], 
                                                                         edgecolor='none', weights=light_I[tpc][side], cmap=light_cmap, norm=light_norm, lw=0.6,
                                                                         alpha=0.6,  zorder=0)[3]
                self.hs_lightChan[tpc][side] = self.axs_lightChan[tpc][side].hist2d(x_data[side], y_data_center, bins=[self.d_binEdges_light_modules[side], 
                                                                                    self.v_binEdges_light_modules], edgecolor='none', weights=light_I[tpc][side],
                                                                                    cmap=light_cmap, norm=light_norm, lw=0.6, alpha=0.6,  zorder=0)[3]
                
                for sipm in range(self.N_sipm_side):

                    adc, chan = self.sipm_rel_pos.get_keys_from_val([tpc,side,sipm])

                    # Plot the waveforms
                    self.axs_p_wfms[tpc][side][sipm].set_xlim(0, light_wvfms.shape[-1])
                    self.axs_p_wfms[tpc][side][sipm].set_ylim(light_wvfms_range[0], light_wvfms_range[1])
                    self.axs_p_wfms[tpc][side][sipm].plot(x_data_wvfm, light_wvfms[0][adc][chan],'red', lw=0.75, zorder=10)
                    self.axs_p_wfms[tpc][side][sipm].set_yticks(np.linspace(light_wvfms_range[0], light_wvfms_range[1], num=4))

                    # Set the sipm informations
                    self.txt_lightChan[tpc][side][sipm] = self.axs_lightChan[tpc][side].text(x_data[side][sipm], y_data_center[sipm], 
                                                                                             f"TPC: {tpc}, Side: {side}, SiPM: {sipm}\n ADC: {adc}, Channel: {chan}",
                               ha="center", va="center", color="k", fontsize = 6, zorder=10)

        # Plot the colorbar
        light_wvfms_cbar = self.fig_wfms.colorbar(mlight, cax=self.light_cbar_ax_wfms, label=r'Light [ADC Counts]', orientation = 'horizontal')      
        light_wvfms_cbar.set_label(r'Light [ADC Counts]', size=13, weight='bold')
        self.light_cbar_ax_wfms.tick_params(labelsize=11.5)

        return None
        
        
    def _plot_event(self, ev_id):
        '''
        Plot the event 'ev_id'.

        Args:
            ev_id: ID of the event to plot

        Return:
            None
        '''
        
        self._clear_axes()

        # Check that the event ID is in the selection
        if ev_id in self.events['id']:
            self.current_ev_id = ev_id
        else:
            print("The specified event number is not part of the current selection.")
            while True: 
                response = input("Would you like to proceed to the next available event? (y/n):").strip().lower()
                
                if response in ['y', 'yes']:
                    greater_ev_id = self.events['id'][self.events['id'] >= ev_id]
                    if greater_ev_id.size > 0:
                        self.current_ev_id = greater_ev_id[0]
                        break
                    else:
                        raise ValueError("No subsequent event found in the selection.")
                elif response in ['n', 'no']:
                    sys.exit(1)
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
        
        # Get the event informations
        hits, *event_info = self._get_event(self.current_ev_id)

        if self.show_event_light:
            light_wvfms, mcharge, mlight, cmap, light_cmap, charge_norm, light_norm, event_subrun = event_info
        else:
            mcharge, cmap, charge_norm, event_subrun = event_info

        # Set the current event subrun number (Note that it include the run nunber)
        self.current_ev_subrun = event_subrun

        # Reset hits if charge threshold is set
        if self.charge_threshold is not None:
            hits = hits[hits['Q'] > self.charge_threshold]
            self.hits_per_event = len(hits)

        # Plot 3D charge hits
        self.dvb_points = self.ax_dvb.scatter(hits['x'], hits['y'], hits['z'], lw=0, ec='C0', \
                            c=cmap(charge_norm(hits['Q'])), s=0.75, alpha=1, marker="s", zorder=10)
        

        # Plot 2D charge hits
        if self.hist_projection:
            z_bins = np.linspace(self.geometry.attrs['lar_detector_bounds'][0][2],self.geometry.attrs['lar_detector_bounds'][1][2],\
                                 int((self.geometry.attrs['lar_detector_bounds'][1][2]-self.geometry.attrs['lar_detector_bounds'][0][2])/0.5))
            y_bins = np.linspace(self.geometry.attrs['lar_detector_bounds'][0][1],self.geometry.attrs['lar_detector_bounds'][1][1],\
                                 int((self.geometry.attrs['lar_detector_bounds'][1][1]-self.geometry.attrs['lar_detector_bounds'][0][1])/0.5))
            x_bins = np.linspace(self.geometry.attrs['lar_detector_bounds'][0][0],self.geometry.attrs['lar_detector_bounds'][1][0],\
                                 int((self.geometry.attrs['lar_detector_bounds'][1][0]-self.geometry.attrs['lar_detector_bounds'][0][0])/0.5))

            bd_charge_hist, _, _ = np.histogram2d(hits['z'], hits['x'], bins=[z_bins,x_bins],weights=hits['Q'])
            bd_charge_hist_masked = np.where(bd_charge_hist==0, np.nan, bd_charge_hist) 
            ZX_Z, ZX_X = np.meshgrid(z_bins[:-1], x_bins[:-1])
            self.bd_points = self.ax_bd.pcolormesh(ZX_Z, ZX_X, bd_charge_hist_masked.T, cmap=cmap, norm=charge_norm, alpha=1, zorder=10)

            bv_charge_hist, _, _ = np.histogram2d(hits['z'], hits['y'], bins=[z_bins,y_bins],weights=hits['Q'])
            bv_charge_hist_masked = np.where(bv_charge_hist==0, np.nan, bv_charge_hist)
            ZY_Z, ZY_Y = np.meshgrid(z_bins[:-1], y_bins[:-1]) 
            self.bv_points = self.ax_bv.pcolormesh(ZY_Z, ZY_Y, bv_charge_hist_masked.T, cmap=cmap, norm=charge_norm, alpha=1, zorder=10)

            dv_charge_hist, _, _ = np.histogram2d(hits['x'], hits['y'], bins=[x_bins,y_bins],weights=hits['Q'])
            dv_charge_hist_masked = np.where(dv_charge_hist==0, np.nan, dv_charge_hist) 
            XY_X, XY_Y = np.meshgrid(x_bins[:-1], y_bins[:-1])
            self.dv_points = self.ax_dv.pcolormesh(XY_X, XY_Y, dv_charge_hist_masked.T, cmap=cmap, norm=charge_norm, alpha=1, zorder=10)
        else:
            self.bd_points = self.ax_bd.scatter(hits['z'], hits['x'], lw=0, ec='C0', c=cmap(
                    charge_norm(hits['Q'])), s=0.75, alpha=1, marker="s", zorder=10)
            self.bv_points = self.ax_bv.scatter(hits['z'], hits['y'], lw=0, ec='C0', c=cmap(
                    charge_norm(hits['Q'])), s=0.75, alpha=1, marker="s", zorder=10)
            self.dv_points = self.ax_dv.scatter(hits['x'], hits['y'], lw=0, ec='C0', c=cmap(
                    charge_norm(hits['Q'])), s=0.75, alpha=1, marker="s", zorder=10)
            
        # Plot light information
        if self.show_event_light==True:

            # Integrate the waveforms
            light_wvfms_I = np.sum(light_wvfms, axis=-1)


            # Reshape the light integrals array and set the negative values to -1:
            # light_I[tpc][side][vertical position] with
            #   tpc: module index
            #   side: chamber index w.r.t the drift axis
            #   vertical position: index starting from bottom (0-59)
            light_I = np.full((self.N_tpc, self.N_side_tpc, self.N_sipm_side), -1) 
            module = 0
            for adc in range(np.size(light_wvfms_I, 1)):
                for chan in range(np.size(light_wvfms_I, 2)):
                    sipm_rel_pos = np.array(self.sipm_rel_pos[(adc, chan)][0])
                    if -1 not in sipm_rel_pos:
                        if light_wvfms_I[module][adc][chan] >= 0:
                            light_I[sipm_rel_pos[0], sipm_rel_pos[1], sipm_rel_pos[2]] = light_wvfms_I[module][adc][chan]
                        else: 
                            light_I[sipm_rel_pos[0], sipm_rel_pos[1], sipm_rel_pos[2]] = -1

            
            self._plot_light(light_I, light_cmap, light_norm)

            if self.show_fig_wfms:
                self._plot_wvfms(light_wvfms, light_I, light_cmap, light_norm, mlight)

            # Set up colorbars 
        if self.show_event_light:

            self._plot_colorbars(mcharge, mlight)
        else:
            self._plot_colorbars(mcharge)

        return None
            
    def _save_plot(self, which_plot = None):    
        '''
        Save the currently ploted event

        Args: 
            which_plot: See save_plots() for the details (default: None)

        Return:
            None
        '''
        # Save the plots
        file_name= f'FSDDisplay_Run{self.current_ev_subrun}_Ev{self.current_ev_id}'
        ext = '.png'
        if which_plot == None:
            self.fig.savefig(os.path.join(self.output_path, file_name+ext), bbox_inches='tight')
            if self.show_fig_wfms:
                self.fig_wfms.savefig(os.path.join(self.output_path, file_name+'_wvfms'+ext), bbox_inches='tight')
        
        elif which_plot == 'display':
            self.fig.savefig(os.path.join(self.output_path, file_name+ext), bbox_inches='tight')

        elif which_plot == 'wvfms':
            if self.show_fig_wfms:
                self.fig_wfms.savefig(os.path.join(self.output_path, file_name+'_wvfms'+ext), bbox_inches='tight')
            else:
                raise TypeError("The waveforms plot was not initialzed. Set 'show_fig_wfms=True' to enable it.")

        else :
            raise ValueError("The argument passed to 'which_plot' is not supported. The expected values are None, 'display' or 'wvfms'")
        
        return None
    
    def display_event(self, ev_id):
        '''
        Display the initialized plots corresponding to the event 'ev_id'.

        Args:
            ev_id: ID of the event to display

        Return: 
            None
        '''

        self._plot_event(ev_id)

        

        display(self.fig)
        
        if (self.show_fig_wfms == True):
            display(self.fig_wfms)

        return None
    
    def save_plots(self, which_plot = None, events_id = None):
        '''
        Save the plots in the output repository.

        Args:
            which_plot (str): Select the subset of plot to be saved: 
                    - None      : default value, display all the initialized figures
                    - 'display' : only save the display figure
                    - 'wvfms'   : only save the waveforms and channel figure

            events_id (int or list[int]): event id(s) to save, if None given, save the last displayed event, int (default: None)

        Return:
            None
        '''
        # Create the output directories if necessary
        os.makedirs(self.output_path, exist_ok=True)

        # Check the event ID argument
        if events_id is None:
            if (self.current_ev_id != -1):
                self._save_plot(which_plot)
                print(f'Event {self.current_ev_id} was saved under {self.output_path}')
                return None
            else:
                raise TypeError("Missing required 'event_id'. Please run 'display_event()' first or provide 'event_id' explicitly.")
        
        if isinstance(events_id, int) or isinstance(events_id, np.uint64):
            events_id = [events_id]

        for ev_id in events_id:
            self._plot_event(ev_id)
            self._save_plot(which_plot) 

        if (len(events_id)==1):
            print(f'The event {events_id[0]} was saved under {self.output_path}')
        else:
            print(f'The {len(events_id)} events  were saved under {self.output_path}')

        return None
    
    def events_id(self):
        '''
        Return a list of events number that passed the selection.
        
        Args:
            None
        Return:
            events_id (numpy.ndarray([numpy.uint64])): list of events number that passed the selection
        '''
        return self.events['id']