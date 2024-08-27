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
    'numpy', 'h5py', 'cmasher', 'IPython', 'PyMuPDF', 'matplotlib', 'pillow', 'uproot', 'h5flow', 'ipywidgets'
]

installed_packages = {pkg.key for pkg in pkg_resources.working_set}
missing_packages = [pkg for pkg in required_packages if pkg not in installed_packages]

if missing_packages:
    for package in missing_packages:
        install(package)

# Import modules
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
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.image as mpimg
from PIL import Image
from math import fabs
import uproot


class LArEventDisplay:

    ''' 
        Class to set up interactive 2x2 display for files run through proto_nd_flow. If also displaying Mx2 data, Mx2 file should be
        in ROOT format (vs. dst format).

        Inputs to this class are as follows:

            - filedir          (str): path to input file (minus filename)
            - filename         (str): name of flow file
            - nhits            (int): number of hits (threshold) for events to be made available (default: 1)
            - ntrigs           (int): number of external triggers (threshold)  threshold for events to be made available (default: 0)
            - show_light       (bool): whether to show light information in display (default: True)
            - show_colorbars   (bool): whether to display color bars (default: True)
            - filepath_mx2     (str): path to Mx2 file if using Mx2 data (default: None)
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
    def __init__(self, filedir, filename, nhits=1, ntrigs=0, show_light=True, filepath_mx2=None, \
                 show_colorbars=True, charge_threshold=None, light_threshold=150000, beam_only=False, \
                 hist_projection=True):
        
        # Open files
        f = h5py.File(filedir+filename, 'r')
        if filepath_mx2 is not None:
            f_mx2 = uproot.open(filepath_mx2)
            self.show_mx2 = True
        else:
            self.show_mx2 = False

        # Set general class-level variables from inputs
        self.show_event_mx2 = self.show_mx2
        self.filename = filename
        self.filepath_mx2 = filepath_mx2
        self.show_light = show_light
        self.show_event_light = show_light
        self.show_colorbars = show_colorbars
        self.charge_threshold = charge_threshold
        self.light_threshold = light_threshold
        self.beam_only = beam_only
        self.hist_projection = hist_projection

        # Set directory for saving files and finding logo image files
        self.lar_evd_dir = os.path.dirname(__file__)

        # Load DUNE and 2x2 logos
        dune_logo = os.path.join(self.lar_evd_dir, 'DUNElogo.pdf')
        self.dune_logo_pdf = fitz.open(dune_logo)#mpimg.imread(dune_logo)
        subexp_logo=os.path.join(self.lar_evd_dir, '2x2logo.pdf')
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
        self.events = self.events[self.events['nhit'] > nhits]
        self.events = self.events[self.events['n_ext_trigs'] >= ntrigs]
        self.beam_events = self.beam_events[self.beam_events['nhit'] > nhits]
        self.beam_events = self.beam_events[self.beam_events['n_ext_trigs'] >= ntrigs]

        # Load geometry and other info
        self.geometry = f['geometry_info']
        self.info = {
            'vdrift': f['lar_info'].attrs['v_drift'],
            'clock_period': 0.1,
        }

        # Load charge hits dataset
        self.hits_dset = 'calib_prompt_hits'
        self.hits_full = f['charge/'+self.hits_dset+'/data']
        self.hits_ref = f['charge/events/ref/charge/'+self.hits_dset+'/ref']
        self.hits_region = f['charge/events/ref/charge/'+self.hits_dset+'/ref_region']
        self.hits_per_event = nhits

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

        # Set up figure and subplots
        # NOTE: This is very different if Mx2 is shown
        if self.show_mx2: 
            
            # Setting figure WITH Mx2
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
            
            # Setting colorbar axes (if showing) WITH Mx2
            if self.show_colorbars and not self.show_light:
                cbar_ax = self.fig.add_axes([0.15, 0.45, 0.675, 0.015])
            if self.show_colorbars and self.show_light:
                cbar_ax = self.fig.add_axes([0.08, 0.45, 0.38, 0.015])
                light_cbar_ax = self.fig.add_axes([0.51, 0.45, 0.38, 0.015])

            # Adjust subplot positioning WITH Mx2
            self.fig.subplots_adjust(top=0.94,bottom=0.001)
            self.fig.subplots_adjust(wspace=0.02, hspace=0.02)

            # Set up Mx2 axis
            self.ax_mx2 = self.axes_dict["ax_mx2"]
            current_mx2_pos = self.ax_mx2.get_position()
            if self.show_colorbars:
                x0_shift = 0.13
                y0_shift = 0.04
            else: 
                x0_shift = 0.11
                y0_shift = 0.04
            new_mx2_pos = [current_mx2_pos.x0-x0_shift, current_mx2_pos.y0-y0_shift, \
                           current_mx2_pos.width*1.2, current_mx2_pos.height*1]
            self.ax_mx2.set_position(new_mx2_pos)

            # Set up DUNE logo axis WITH Mx2
            ax_dune_logo = self.fig.add_axes([0.59, 0.942, 0.37, 0.057])

        else:

            # Setting figure WITHOUT Mx2
            self.fig = plt.figure(constrained_layout=False, figsize=(15, 8))
            self.axes_mosaic = [["ax_bd", "ax_subexp_logo", "ax_bdv", "ax_bdv"],["ax_bv", "ax_dv", "ax_bdv", "ax_bdv"],]
            self.axes_dict = self.fig.subplot_mosaic(self.axes_mosaic, \
                                                    per_subplot_kw={"ax_bdv": {"projection": "3d"}})
            
            # Setting colorbar axes (if showing) WITHOUT Mx2
            if self.show_colorbars and not self.show_light:
                cbar_ax = self.fig.add_axes([0.145, 0.001, 0.675, 0.025])
            if self.show_colorbars and self.show_light:
                cbar_ax = self.fig.add_axes([0.08, 0.001, 0.38, 0.025])
                light_cbar_ax = self.fig.add_axes([0.51, 0.001, 0.38, 0.025])

            # Adjust subplot positioning WITHOUT Mx2
            self.fig.subplots_adjust(bottom=0.1)
            self.fig.subplots_adjust(wspace=0.02, hspace=0.02)

            # Set up DUNE logo axis WITHOUT Mx2
            ax_dune_logo = self.fig.add_axes([0.56, 0.895, 0.43, 0.099])

        # Initialize axes for display (with or without Mx2)
        self.ax_dune_logo = ax_dune_logo
        self.ax_bdv = self.axes_dict["ax_bdv"]
        self.ax_bd = self.axes_dict["ax_bd"]
        self.ax_bv = self.axes_dict["ax_bv"]
        self.ax_dv = self.axes_dict["ax_dv"]
        self.ax_subexp_logo = self.axes_dict["ax_subexp_logo"]
        if self.show_colorbars:
            self.cbar_ax = cbar_ax
            if self.show_light:
                self.light_cbar_ax = light_cbar_ax

        # Initialize point collections for plotting
        if self.show_mx2:
            self.mx2_lar_points = self.ax_mx2.scatter([], [], [])
        self.bdv_points = self.ax_bdv.scatter([], [], [])
        self.bd_points = self.ax_bd.scatter([], [])
        self.bv_points = self.ax_bv.scatter([], [])
        self.dv_points = self.ax_dv.scatter([], [])

        # Set up 3D view angles and zooms for GIFs
        self.base_angle = list(range(-180,180,2))
        self.azimuths = [ba for ba in self.base_angle]
        self.zeniths = [fabs(ba*0.25) for ba in self.base_angle]
        self.offset = int(len(self.base_angle)/6) # Shift angles relative to improve perspectives
        self.azimuths = self.azimuths[self.offset:] + self.azimuths[:self.offset] # Shift angles relative to improve perspectives
        self.zeniths = self.zeniths[self.offset:] + self.zeniths[:self.offset] # Shift angles relative to improve perspectives
        self.zooms = [0.985,]*len(self.azimuths)  # Set view zoom

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

            if self.show_event_mx2:
                self.mx2_lar_points.set_sizes(pixel_pitch_sizes)
            self.bdv_points.set_sizes(pixel_pitch_sizes)

        # First, save figure to PDF (clearing DUNE and 2x2 logo axes)      
        save_dir = self.lar_evd_dir
        filename = self.filename.split('.')[0]+'_Event_'+str(ev_id)+'.pdf'
        savepath = os.path.join(save_dir, filename)
        print("Saving to", savepath.split('.')[0]+'_Display.pdf')
        self.ax_dune_logo.clear()
        self.ax_dune_logo.axis('off')
        self.ax_subexp_logo.clear()
        self.ax_subexp_logo.axis('off')
        self.fig.savefig(savepath, bbox_inches='tight')

        # Then, add back vectorized DUNE logo to saved PDF
        saved_pdf = fitz.open(savepath)
        saved_pdf_page = saved_pdf[0]
        rect_max_x = saved_pdf_page.rect[2]
        rect_max_y = saved_pdf_page.rect[3]
        include_dune_logo_rect = fitz.Rect(rect_max_x-300, 2, rect_max_x-5, 65)
        saved_pdf_page.show_pdf_page(include_dune_logo_rect, self.dune_logo_pdf, 0)
        include_subexp_logo_rect = fitz.Rect(rect_max_x-635, 78, rect_max_x-425, 288)
        saved_pdf_page.show_pdf_page(include_subexp_logo_rect, self.subexp_logo_pdf, 0)
        saved_pdf.save(savepath.split('.')[0]+'_Display.pdf')

        # Remove initial PDF without DUNE logo
        os.remove(savepath)
        

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
                    self.ax_bdv.view_init(zen, azi)  
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

        self.ax_bdv.cla()
        self.ax_bd.cla()
        self.ax_bv.cla()
        self.ax_dv.cla()
        if self.show_colorbars:
            self.cbar_ax.cla()
            if self.show_light:
                self.light_cbar_ax.cla()
        if self.show_mx2:
            self.ax_mx2.cla()


    def get_event(self, ev_id):

        # To start, set show_event_mx2 to general show_mx2 value
        self.show_event_mx2 = self.show_mx2

        # Get event ID information
        ev_idx = np.where(self.events['id'] == ev_id)[0][0]
        print("Number of available events:", len(self.events))
        print("For fast-forwarding purposes, here is every 10th event number in your sample:", [ev for ev in self.events['id'][9::10]])

        # Get event general information
        event = self.events[ev_idx]
        event_datetime = datetime.utcfromtimestamp(
                event['unix_ts']).strftime('%Y-%m-%d %H:%M:%S')
  
        print("Number of external triggers in this event:", event['n_ext_trigs'])

        # Check if event is a beam event
        if not self.beam_only and ev_id in self.beam_events['id']:
            self.is_beam_event = True
        # Check if event is a beam event for showing Mx2 (TO DO: Is this necessary for Mx2 matching?)
        if not self.is_beam_event: 
            self.show_event_mx2 = False

        # Get event charge information
        hit_ref = self.hits_ref[self.hits_region[ev_id,'start']:self.hits_region[ev_id,'stop']]
        hit_ref = np.sort(hit_ref[hit_ref[:,0] == ev_id, 1])
        hits = self.hits_full[hit_ref]
        self.hits_per_event = len(hits)

        # Get event external trigger information
        exttrig_ref = self.exttrigs_ref[self.exttrigs_region[ev_id,'start']:self.exttrigs_region[ev_id,'stop']]
        exttrig_ref = np.sort(exttrig_ref[exttrig_ref[:,0] == ev_id, 1])
        exttrigs = self.exttrigs_full[exttrig_ref]
        print("External trigger information:", exttrigs['iogroup'], exttrigs['ts'], exttrigs['ts_raw'])

        # Prepare color map for charge
        min_charge = min(hits['Q'])
        if max(hits['Q']) > min_charge:
            max_charge = max(hits['Q'])
        else: 
            max_charge = min_charge + 1
        charge_norm = mpl.colors.Normalize(vmin=min_charge,vmax=max_charge)
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

        # Get Mx2 matched event info if using
        # TO DO: Make Mx2 matching more robust
        # Debugging statements included to help with troubleshooting
        if self.show_event_mx2:
            
            charge_time = (event["unix_ts"] + event["ts_start"]/ 1e7)
            print("Charge Time:", charge_time)
            print("Hit t_drift max:", hits['t_drift'].max())
            #print("Minerva time 1:", self.minerva_times[0])
            #print("Minerva times:", len(self.minerva_times))
            # find the index of the minerva_times that matches the charge_time
            mx2_charge_time_diffs = np.abs(self.minerva_times - np.full_like(self.minerva_times, charge_time))
            trigger = np.argwhere(mx2_charge_time_diffs < 0.5).reshape(1,-1)[0] # changed from 0.5 acceptance window
            print("Time Differences:", mx2_charge_time_diffs[trigger])
            #print("Trigger:", trigger)
            xs = []
            ys = []
            zs = []
            qs = []
            #print("All Minerva tracks:", self.minerva_trk_index[trigger][0])
            for trig in trigger:
                print("Minerva time:", self.minerva_times[trig])
                print("Minerva time -1:", self.minerva_times[trig-1])
                print("Minerva time +1:", self.minerva_times[trig+1])
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
                # Prepare color map for Mx2 (currently not used in plotting anywhere)
                mx2_norm = mpl.colors.LogNorm(vmin=min(mx2['mq']),vmax=max(mx2['mq']))
                mx2_cmap = cmr.get_sub_cmap('cmr.torch_r', 0.13,0.95) 
                mmx2 = plt.cm.ScalarMappable(norm=mx2_norm, cmap=mx2_cmap)

        # Set figure title (uses event information loaded in this method)
        if self.show_mx2:
            top_adjust=''
            bottom_adjust="\n\n\n\n\n\n\n\n\n\n\n\n"
        else:
            top_adjust="\n"
            bottom_adjust=""
        self.fig.suptitle(top_adjust+" Event %i: %s UTC" %
                          (ev_id, event_datetime)+bottom_adjust, x=0.05, size=28, weight='bold', ha='left', linespacing=0.15)
        

        # Return event information including charge, light, and Mx2 datasets for plotting and all color scale information
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
            cmap_zero: charge color map scale with lower values (for backgrounds)
            mlight: light color map scale
        '''

        # Show 2x2 and DUNE logos
        self.ax_subexp_logo.axis('off')
        self.ax_subexp_logo.imshow(self.subexp_logo_png)
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

        # NOTE: xlim and ylim for all 2D subplots are the same and based on the 
        #       maximum and minimum boundary values of the longest detector axis (beam, Z)
        # Set axes for Beam vs Drift (ZX) canvas
        #self.ax_bd.set_xlabel('Beam Axis [cm]', fontsize=14) # Currently not showing x-axis label bc overlap with lower subplot
        self.ax_bd.set_ylabel('Drift Axis [cm]', fontsize=14, weight='bold')
        self.ax_bd.set_ylim(self.geometry.attrs['lar_detector_bounds'][0][2]-3, \
            self.geometry.attrs['lar_detector_bounds'][1][2]+3)
        self.ax_bd.set_xlim(self.geometry.attrs['lar_detector_bounds'][0][2]-3,\
            self.geometry.attrs['lar_detector_bounds'][1][2]+3)
        self.ax_bd.tick_params(axis='y', which='major', labelsize=12.5)
        self.ax_bd.set_xticks([])
        self.ax_bd.set_yticks(np.arange(-60,61,20))

        # Set axes for Beam vs Vertical (ZY) canvas
        self.ax_bv.set_xlabel('Beam Axis [cm]', fontsize=14, weight='bold')
        self.ax_bv.set_ylabel('Vertical Axis [cm]', fontsize=14, weight='bold')
        self.ax_bv.set_xlim(self.geometry.attrs['lar_detector_bounds'][0][2]-3,\
            self.geometry.attrs['lar_detector_bounds'][1][2]+3)
        self.ax_bv.set_ylim(self.geometry.attrs['lar_detector_bounds'][0][2]-3, \
            self.geometry.attrs['lar_detector_bounds'][1][2]+3)
        self.ax_bv.tick_params(axis='both', which='major', labelsize=12.5)
        self.ax_bv.set_xticks(np.arange(-60,61,20))
        self.ax_bv.set_yticks(np.arange(-60,61,20))

        # Set axes for Drift vs Vertical (XY) canvas
        self.ax_dv.set_xlabel('Drift Axis [cm]', fontsize=14, weight='bold')
        #self.ax_dv.set_ylabel('Vertical Axis [cm]', fontsize=14) # Currently not showing y-axis label bc overlap with left subplot
        self.ax_dv.set_xlim(self.geometry.attrs['lar_detector_bounds'][0][2]-3,\
            self.geometry.attrs['lar_detector_bounds'][1][2]+3)
        self.ax_dv.set_ylim(self.geometry.attrs['lar_detector_bounds'][0][2]-3, \
            self.geometry.attrs['lar_detector_bounds'][1][2]+3)
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
            self.ax_mx2.set_box_aspect([7.75,3.5,3], zoom=1.35)
            self.ax_mx2.view_init(azim=-75, elev=17) # default -75, 17
            #self.ax_mx2.set_aspect('auto')
    
            # Plot Mx2 outlines
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
                                    [y_base[i], y_base[(i + 1) % len(x_base)]], color="grey", alpha=0.85)

                    self.ax_mx2.plot([z_base[j][1], z_base[j][1]], 
                                    [x_base[i], x_base[(i + 1) % len(x_base)]], 
                                    [y_base[i], y_base[(i + 1) % len(x_base)]], color="grey", alpha=0.85)

                    self.ax_mx2.plot([z_base[j][0], z_base[j][1]], 
                                    [x_base[i], x_base[i]], 
                                    [y_base[i], y_base[i]], color="blue", alpha=0.85)

                    self.ax_mx2.plot([z_base[j][0], z_base[j][1]], 
                                    [x_base[i], x_base[i]], 
                                    [y_base[i], y_base[i]], color="grey", alpha=0.85)

        # Plot cathodes + module outlines for 3D view(s) and fill module volumes + plot cathodes for 2D LAr volume projections           
        for i in range(len(self.geometry.attrs['module_RO_bounds'])):

            # Plot cathodes for XYZ (beam, drift, vertical) 3D view (and 2x2+Mx2 view if using):
            X_cathode, Y_cathode, Z_cathode = make_x_plane(self.geometry.attrs['module_RO_bounds'][i][0][1], self.geometry.attrs['module_RO_bounds'][i][1][1], \
                                                           self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2], 
                                                           self.geometry.attrs['module_RO_bounds'][i][0][0]+self.geometry.attrs['max_drift_distance']+self.geometry.attrs['cathode_thickness']/2)
            self.ax_bdv.plot_surface(Z_cathode,X_cathode,Y_cathode, color='gainsboro', alpha=0.1)
            if self.show_mx2:
                self.ax_mx2.plot_surface(Z_cathode,X_cathode,Y_cathode, color='gainsboro', alpha=0.05)
            
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
                                [self.geometry.attrs['module_RO_bounds'][i][k][1], self.geometry.attrs['module_RO_bounds'][i][k][1]], color='black', alpha=0.2)

                        self.ax_mx2.plot([self.geometry.attrs['module_RO_bounds'][i][j][2], self.geometry.attrs['module_RO_bounds'][i][j][2]], \
                                [self.geometry.attrs['module_RO_bounds'][i][k][0], self.geometry.attrs['module_RO_bounds'][i][k][0]], \
                                [self.geometry.attrs['module_RO_bounds'][i][0][1], self.geometry.attrs['module_RO_bounds'][i][1][1]], color='black', alpha=0.2)

                        self.ax_mx2.plot([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                                [self.geometry.attrs['module_RO_bounds'][i][j][0], self.geometry.attrs['module_RO_bounds'][i][j][0]], \
                                [self.geometry.attrs['module_RO_bounds'][i][k][1], self.geometry.attrs['module_RO_bounds'][i][k][1]], color='black', alpha=0.2)

            # Fill modules for ZX (beam, drift) projections:
            self.ax_bd.fill([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][0][2], \
                      self.geometry.attrs['module_RO_bounds'][i][1][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                     [self.geometry.attrs['module_RO_bounds'][i][0][0], self.geometry.attrs['module_RO_bounds'][i][1][0], \
                      self.geometry.attrs['module_RO_bounds'][i][1][0], self.geometry.attrs['module_RO_bounds'][i][0][0]],\
                     color=cmap_zero(0), alpha=0.85)

            # Only two modules represented in ZY and XY projections (i.e. projections after this line)
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

            # If light is plotted in this event, the cathodes for ZX (beam, drift) projections are plotted after light in this view
            if not self.show_event_light:

                # Plot cathodes for ZX (beam, drift) projections:
                for i in range(len(self.geometry.attrs['module_RO_bounds'])):
                    self.ax_bd.plot([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                             [self.geometry.attrs['module_RO_bounds'][i][0][0]+self.geometry.attrs['max_drift_distance']+self.geometry.attrs['cathode_thickness']/2, \
                              self.geometry.attrs['module_RO_bounds'][i][0][0]+self.geometry.attrs['max_drift_distance']+self.geometry.attrs['cathode_thickness']/2],\
                              color='gainsboro', alpha=0.9, linewidth=2,solid_capstyle='butt')
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
        if self.show_event_mx2 and self.show_event_light:
            mx2, light_wvfms, mcharge, mmx2, mlight, cmap, mx2_cmap, light_cmap, charge_norm, mx2_norm, light_norm, cmap_zero, light_cmap_zero = event_info
        elif self.show_event_mx2 and not self.show_event_light:
            mx2, mcharge, mmx2, cmap, mx2_cmap, charge_norm, mx2_norm, cmap_zero = event_info
        elif self.show_event_light and not self.show_event_mx2:
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

        # Plot hits in 3D views first so that cathodes/anodes go over the hits
        self.bdv_points = self.ax_bdv.scatter(hits['z'], hits['x'], hits['y'], lw=0, ec='C0', \
                            c=cmap(charge_norm(hits['Q'])), s=0.75, alpha=1, marker="s")

        if self.show_event_mx2:
            self.mx2_lar_points = self.ax_mx2.scatter(hits['z'], hits['x'], hits['y'], lw=0, ec='C0', \
                    c=cmap(charge_norm(hits['Q'])), s=0.75, alpha=1, marker="s")
            #self.ax_mx2.scatter(mx2['mz'], mx2['mx'], mx2['my'], lw=0, ec='C0', \ # Currently not using Mx2 energy info
            #                c=mx2_cmap(mx2_norm(mx2['mq'])), s=15, alpha=1)       # Currently not using Mx2 energy info
            self.ax_mx2.scatter(mx2['mz'], mx2['mx'], mx2['my'], lw=0, ec='C0', \
                            c='red', s=4, alpha=1)    

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
            z_bins = np.linspace(self.geometry.attrs['lar_detector_bounds'][0][2],self.geometry.attrs['lar_detector_bounds'][1][2],\
                                 int((self.geometry.attrs['lar_detector_bounds'][1][2]-self.geometry.attrs['lar_detector_bounds'][0][2])/0.5))
            y_bins = np.linspace(self.geometry.attrs['lar_detector_bounds'][0][1],self.geometry.attrs['lar_detector_bounds'][1][1],\
                                 int((self.geometry.attrs['lar_detector_bounds'][1][1]-self.geometry.attrs['lar_detector_bounds'][0][1])/0.5))
            x_bins = np.linspace(self.geometry.attrs['lar_detector_bounds'][0][0],self.geometry.attrs['lar_detector_bounds'][1][0],\
                                 int((self.geometry.attrs['lar_detector_bounds'][1][0]-self.geometry.attrs['lar_detector_bounds'][0][0])/0.5))

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
            self.bd_points = self.ax_bd.scatter(hits['z'], hits['x'], lw=0, ec='C0', c=cmap(
                    charge_norm(hits['Q'])), s=0.75, alpha=1, marker="s")
            self.bv_points = self.ax_bv.scatter(hits['z'], hits['y'], lw=0, ec='C0', c=cmap(
                    charge_norm(hits['Q'])), s=0.75, alpha=1, marker="s")
            self.dv_points = self.ax_dv.scatter(hits['x'], hits['y'], lw=0, ec='C0', c=cmap(
                    charge_norm(hits['Q'])), s=0.75, alpha=1, marker="s")
            
        return hits, cmap, charge_norm


    def plot_light(self, light_wvfms, light_cmap, light_norm, light_cmap_zero):
        
        #acl_det_ids = [0,4,8,12] # Used in previous version of code to identify ACL detectors

        # Sum light waveforms and plot light in ZX (beam, drift) projection
        for x,z in itertools.product(self.sipm_unique_x,self.sipm_unique_z):
            if x==-1: continue
            if z==-1: continue
            this_xz_sum = 0
            # Get light sum for each SiPM in this z,x position
            for i,j in itertools.product(range(light_wvfms[0].shape[0]),range(light_wvfms[0].shape[1])):
                #det_id = self.light_det_id[(i,j)][0]
                pos=self.sipm_abs_pos[(i,j)][0]
                if pos[0]==-1:
                    continue
                if (abs(pos[0]-x) < 0.5) and (pos[2]==z):
                    this_xz_sum += light_wvfms[0][i,j].sum()

            # Plot light in ZX projection if SiPM sum is over threshold
            # This is a bit different from ZY because of previous issues with SiPM x positions
            for i in range(len(self.geometry.attrs['module_RO_bounds'])):
                for j in range(2):
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

        # # Sum light waveforms and plot light in ZY (beam, vertical) projection
        for z,y in itertools.product(self.sipm_unique_z,self.sipm_unique_y):
            if z==-1: continue
            if y==-1: continue
            this_zy_sum = 0
            # Get light sum for each SiPM in this z,y position
            for i,j in itertools.product(range(light_wvfms[0].shape[0]),range(light_wvfms[0].shape[1])):
                pos=self.sipm_abs_pos[(i,j)][0]
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
                        if (abs(z-self.geometry.attrs['module_RO_bounds'][i][j][2]) < 1):
                            self.ax_bv.plot([self.geometry.attrs['module_RO_bounds'][i][j][2]+z_offset, self.geometry.attrs['module_RO_bounds'][i][j][2]+z_offset], \
                                     [y-2.25, y+2.25],color=cmap_value, alpha=1, linewidth=3.5, solid_capstyle='butt')

        # Plot light in XY projection               
        for x,y in itertools.product(self.sipm_unique_x,self.sipm_unique_y):
            if x==-1: continue
            if y==-1: continue
            this_xy_sum = 0
            # Get light sum for each SiPM in this x,y position
            for i,j in itertools.product(range(light_wvfms[0].shape[0]),range(light_wvfms[0].shape[1])):
                pos=self.sipm_abs_pos[(i,j)][0]
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
                        if (abs(x-self.geometry.attrs['module_RO_bounds'][i][j][0]) < 1):
                            self.ax_dv.plot([self.geometry.attrs['module_RO_bounds'][i][j][0]+x_offset, self.geometry.attrs['module_RO_bounds'][i][j][0]+x_offset], \
                                    [y-2.25, y+2.25],color=cmap_value, alpha=1, linewidth=4, solid_capstyle='butt')

        # Plot light for XYZ (beam, drift, vertical) 3D view (and 2x2+Mx2 view if using)
        for i,j in itertools.product(range(light_wvfms[0].shape[0]),range(light_wvfms[0].shape[1])):
            pos=self.sipm_abs_pos[(i,j)][0]
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
            min_z_diff = np.where(abs(z_diffs)<1)
            z_pos = self.geometry.attrs['module_RO_bounds'][:,:,2][min_z_diff][0]
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
                    self.ax_bdv.plot_surface(light_z,light_x,light_y,color=cmap_value, alpha=0.1, shade=False)
                    if self.show_event_mx2:
                        self.ax_mx2.plot_surface(light_z,light_x,light_y,color=cmap_value, alpha=0.1, shade=False)
                    break
                else: continue


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

