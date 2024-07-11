import sys
import warnings
import numpy as np
from datetime import datetime
from proto_nd_flow.util.lut import LUT
from h5flow.core import resources
import itertools
import os
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



class LArEventDisplay:

    ''' 
        Class to set up interactive 2x2 display for files run through proto_nd_flow.

        Inputs are as follows:

            - filedir         (str): path to input file
            - flow_file       (str): path to input file
            - dune_logo       (str): path to DUNE logo image
            - subexp_logo     (str): path to subexperiment logo image
            - nhits           (int): hit threshold for events to be made available in interactive display
            - show_light      (bool): whether to show light information in display (default: True)
            - public          (bool): whether to display for public i.e. include extra charge/light thresholds (default: False)
            
        In order to run the display, set up a Jupyter Notebook, import everything in this file,
        and execute the run() method, e.g.:

        from lar_only_evd import *
        plt.ion()

        f = '/path/to/file'
        evd = ProtoNDFlowEventDisplay(flow_file=f, nhits=1, show_light=True)
        evd.run()
    '''

    def __init__(self, filedir,filename, dune_logo, subexp_logo, nhits=1, ntrigs=0, show_light=True, public=False):
        
        f = h5py.File(filedir+filename, 'r')
        # ARTIFICIALLY ADDING LIGHT INFO:
        lf = h5py.File('/global/cfs/cdirs/dune/users/calivers/elise_files/mpd_run_hvramp_rctl_091_p39.FLOW.hdf5', 'r')
        self.filename = filename
        self.show_light = show_light
        self.show_event_light = show_light
        self.public = public
        self.dune_logo = mpimg.imread(dune_logo)
        self.subexp_logo = mpimg.imread(subexp_logo)

        # Resize DUNE logo image to fit in display
        self.original_dune_logo_shape = self.dune_logo.shape
        self.dune_logo = Image.fromarray((self.dune_logo * 255).astype(np.uint8))
        shrink_factor=(1/3)
        new_size = (int(self.dune_logo.size[0] * shrink_factor), int(self.dune_logo.size[1] * shrink_factor))
        self.dune_logo = self.dune_logo.resize(new_size, Image.LANCZOS)
        self.dune_logo = np.array(self.dune_logo) / 255
        print("Resized DUNE logo shape:", self.dune_logo.shape)
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
        self.light_threshold = 3e5
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
        self.fig = plt.figure(constrained_layout=False, figsize=(26, 13))
        self.axes_mosaic = [["ax_bd", "ax_logo", "ax_bdv", "ax_bdv"],["ax_bv", "ax_dv", "ax_bdv", "ax_bdv"],]
        self.axes_dict = self.fig.subplot_mosaic(self.axes_mosaic, \
                                                 per_subplot_kw={"ax_bdv": {"projection": "3d"}})

        cbar_ax = self.fig.add_axes([0.95, 0.12, 0.015, 0.75])
        self.fig.subplots_adjust(right=0.92)
        if self.show_light:
            light_cbar_ax = self.fig.add_axes([0.15, 0.005, 0.75, 0.03])
        self.fig.subplots_adjust(bottom=0.1)
        self.fig.subplots_adjust(wspace=0.02, hspace=0.02)

        self.ax_bdv = self.axes_dict["ax_bdv"]
        self.ax_bd = self.axes_dict["ax_bd"]
        self.ax_bv = self.axes_dict["ax_bv"]
        self.ax_dv = self.axes_dict["ax_dv"]
        self.ax_logo = self.axes_dict["ax_logo"]
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


    def run(self):
  
        print("Number of available events:", len(self.events))
        ev_id = 0

        # Displays event until user input determines next action
        # User can quit display (q), save current display to PDF (s), skip to next event (enter),
        # or skip to a specific event number out of the total number of events to display (type number)
        while True:

            clear_output(wait=True)
            self.display_event(ev_id)
            display(plt.gcf())
            user_input = input(
                'Next event (q to exit/s to save to pdf/enter for next/number to skip to position)?\n')
            if not user_input:
                clear_output(wait=True)
                ev_id += 1
                print(ev_id)
                self.display_event(ev_id)
            elif user_input[0].lower() == 'q':
                sys.exit()
            elif user_input[0].lower() == 's':
                plt.savefig(self.filename+'_Event_'+str(ev_id)+'.pdf')
            elif user_input[0].lower() == 'g':
                print("Creating GIF of Event Display")
                # Loop over 3D views
                gif_dir = '/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/2x2_sim/run-ndlar-flow/ndlar_flow/event_display/LAr_evd/'
                frame_num = 0
                for (azi,zen,zoom) in zip(self.azimuths,self.zeniths,self.zooms):
                    self.ax_bdv.view_init(zen, azi)   # see mpl_toolkits.mplot3d.axes3d.Axes3D.view_init
                    self.ax_bdv.dist = zoom
                    figname = gif_dir+'frame_%04d_%04d.png' % (ev_id, frame_num)
                    self.fig.savefig(figname)
                    frame_num += 1
                os.system("convert -delay 10 "+gif_dir+"frame*.png animated_"+str(ev_id)+".gif")
            else:
                try:
                    clear_output(wait=True)
                    ev_id = int(user_input)
                    print(ev_id)
                    self.display_event(ev_id)
                except ValueError:
                    print("Event number %s not valid" % user_input)
            if ev_id >= len(self.events):
                print("End of file")
                sys.exit()
    
    def clear_axes(self):

        self.ax_bdv.cla()
        self.ax_bd.cla()
        self.ax_bv.cla()
        self.ax_dv.cla()
        self.cbar_ax.cla()
        if self.show_light:
            self.light_cbar_ax.cla()


    def get_event(self, ev_id):

        # Get event charge information
        event = self.events[ev_id]
        event_datetime = datetime.utcfromtimestamp(
                event['unix_ts']).strftime('%Y-%m-%d %H:%M:%S')
        # DEBUGGING TIMESTAMPS
        print("Charge Unix TS:", event['unix_ts'])
        print("Charge TS Start:", event['ts_start'])
        print("Charge TS End:", event['ts_end'])
        print("Module RO Bounds:", self.geometry.attrs['module_RO_bounds'])
        ev_id = event['id']
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
        # Prepare color map for charge
        #print("Min charge:", min(hits['Q']), "Max charge:", max(hits['Q']))
        if self.public:
            min_charge = self.charge_threshold
        else:
            min_charge = min(hits['Q'])
        charge_norm = mpl.colors.Normalize(vmin=min_charge,vmax=max(max(hits['Q']), 1.))
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
            light_norm = colors.LogNorm(min_light,light_wvfms[0].sum(axis=-1).max()*2)
            light_norm_single = colors.LogNorm(1,light_wvfms[0].sum(axis=-1).max())
            c = light_norm(light_wvfms[0].sum(axis=-1))
            mlight = plt.cm.ScalarMappable(norm=light_norm, cmap=light_cmap)

        # Set figure title
        self.fig.suptitle("\nEvent %i - %s UTC" %
                          (ev_id, event_datetime), x=0.38, size=48, weight='bold', linespacing=0.3)

        if self.show_event_light:
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
        self.ax_logo.axis('off')
        self.ax_logo.imshow(self.subexp_logo)
        if self.show_light:
            self.fig.figimage(self.dune_logo, xo=1632, \
                                yo=1215, origin='upper')
        else:
            self.fig.figimage(self.dune_logo, xo=1632, \
                                yo=1085, origin='upper')
        print("Number of available events:", len(self.events))

        # Set axes for 3D canvas (Beam, Drift, Vertical)
        self.ax_bdv.set_xlabel('\nBeam Direction [cm]', fontsize=22, weight='bold', linespacing=2) #z
        self.ax_bdv.set_ylabel('\nDrift Direction [cm]', fontsize=22, weight='bold', linespacing=2) #x
        self.ax_bdv.set_zlabel('\nVertical Direction [cm]', fontsize=22, weight='bold', linespacing=2) #y
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
        self.ax_bdv.xaxis.pane.set_facecolor(cmap_zero(0))
        self.ax_bdv.yaxis.pane.set_facecolor(cmap_zero(0))
        self.ax_bdv.zaxis.pane.set_facecolor(cmap_zero(0))
        self.ax_bdv.tick_params(axis='both', which='major', labelsize=20)

        # Set axes for Beam vs Drift (ZX) canvas
        #self.ax_bd.set_xlabel('Beam Direction [cm]', fontsize=20)
        self.ax_bd.set_ylabel('Drift Direction [cm]', fontsize=20, weight='bold')
        self.ax_bd.set_ylim(self.geometry.attrs['lar_detector_bounds'][0][0]-0.5, \
            self.geometry.attrs['lar_detector_bounds'][1][0]+0.5)
        self.ax_bd.set_xlim(self.geometry.attrs['lar_detector_bounds'][0][2]-3.15,\
            self.geometry.attrs['lar_detector_bounds'][1][2]+3)
        self.ax_bd.tick_params(axis='y', which='major', labelsize=18)
        self.ax_bd.set_xticks([])

        # Set axes for Beam vs Vertical (ZY) canvas
        self.ax_bv.set_xlabel('Beam Direction [cm]', fontsize=20, weight='bold')
        self.ax_bv.set_ylabel('Vertical Direction [cm]', fontsize=20, weight='bold')
        self.ax_bv.set_xlim(self.geometry.attrs['lar_detector_bounds'][0][2]-3.05,\
            self.geometry.attrs['lar_detector_bounds'][1][2]+2.9)
        self.ax_bv.set_ylim(self.geometry.attrs['lar_detector_bounds'][0][1]-0.5, \
            self.geometry.attrs['lar_detector_bounds'][1][1]+0.5)
        self.ax_bv.tick_params(axis='both', which='major', labelsize=18)

        # Set axes for Drift vs Vertical (XY) canvas
        self.ax_dv.set_xlabel('Drift Direction [cm]', fontsize=20, weight='bold')
        #self.ax_dv.set_ylabel('Vertical Direction [cm]', fontsize=20)
        self.ax_dv.set_xlim(self.geometry.attrs['lar_detector_bounds'][0][0]-2.95,\
            self.geometry.attrs['lar_detector_bounds'][1][0]+2.8)
        self.ax_dv.set_ylim(self.geometry.attrs['lar_detector_bounds'][0][1]-0.4, \
            self.geometry.attrs['lar_detector_bounds'][1][1]+0.5)
        self.ax_dv.tick_params(axis='x', which='major', labelsize=18)
        self.ax_dv.set_yticks([])

        for i in range(len(self.geometry.attrs['module_RO_bounds'])):

            # Plot cathodes for XYZ (beam, drift, vertical) 3D view:
            X_cathode, Y_cathode, Z_cathode = make_x_plane(self.geometry.attrs['module_RO_bounds'][i][0][1], self.geometry.attrs['module_RO_bounds'][i][1][1], \
                                                           self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2], 
                                                           self.geometry.attrs['module_RO_bounds'][i][0][0]+self.geometry.attrs['max_drift_distance']+self.geometry.attrs['cathode_thickness']/2)
            self.ax_bdv.plot_surface(Z_cathode,X_cathode,Y_cathode, color='gainsboro', alpha=0.1)

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

            if not self.show_light:

                # Plot cathodes for ZX (beam, drift) projections:
                for i in range(len(self.geometry.attrs['module_RO_bounds'])):
                    for j in range(2):
                        self.ax_bd.plot([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                                 [self.geometry.attrs['module_RO_bounds'][i][0][0]+self.geometry.attrs['max_drift_distance']+self.geometry.attrs['cathode_thickness']/2, \
                                  self.geometry.attrs['module_RO_bounds'][i][0][0]+self.geometry.attrs['max_drift_distance']+self.geometry.attrs['cathode_thickness']/2],\
                                  color='gainsboro', alpha=0.9, linewidth=2,solid_capstyle='butt')

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
        if self.show_event_light:
            light_wvfms, mcharge, mlight, cmap, light_cmap, charge_norm, light_norm, cmap_zero, light_cmap_zero = event_info
        else:
            mcharge, cmap, charge_norm, cmap_zero = event_info

        # Adjust drift velocity
        # USED DURING RAMP, MOSTLY UNNECESSARY NOW
        #drift_dir = np.full_like(hits['x'], 1)
        #io_group_mask = hits['io_group'] % 2 == 0
        #drift_dir[io_group_mask] = -1
#
        #orig_drift_v=0.16 #mm/clock cycles 
        #new_drift_v=0.158#mm/clock cycles, for example. not sure what it is now
        #orig_drift = hits['x'] # cm
        #drift_time = hits['t_drift']*0.1 # ticks (0.1 us)
        ##print("Drift time:", drift_time[:10])
        ##print("Original drift:", orig_drift[:10])
        #corrected_drift = ( orig_drift - (drift_dir)*drift_time * np.full_like(drift_time, orig_drift_v)) \
        #    + (drift_dir)*drift_time* np.full_like(drift_time, new_drift_v)
        ##print("Corrected drift:", corrected_drift[:10])
        #corrected_drift = hits['x']

        if self.public:
            hits = hits[hits['Q'] > self.charge_threshold]
        # Plot hits in 3D view first so that cathodes/anodes go over the hits
        self.ax_bdv.scatter(hits['z'], hits['x'], hits['y'], lw=0, ec='C0', \
                            c=cmap(charge_norm(hits['Q'])), s=5, alpha=1)
        if self.show_event_light:
            self.set_axes(cmap, mcharge, cmap_zero, mlight)
        else:
            self.set_axes(cmap, mcharge, cmap_zero)

        if self.show_event_light:
            self.plot_light(light_wvfms, light_cmap, light_norm, light_cmap_zero)

        # Plot 2D charge hits
        self.ax_bd.scatter(hits['z'], hits['x'], lw=0, ec='C0', c=cmap(
                charge_norm(hits['Q'])), s=5, alpha=1)
        self.ax_bv.scatter(hits['z'], hits['y'], lw=0, ec='C0', c=cmap(
                charge_norm(hits['Q'])), s=5, alpha=1)
        self.ax_dv.scatter(hits['x'], hits['y'], lw=0, ec='C0', c=cmap(
                charge_norm(hits['Q'])), s=5, alpha=1)
        # Lines below are for geometry debugging purposes
        #self.ax_bd.scatter(hits[hits['io_group'] == 8]['z'], hits[hits['io_group'] == 8]['x'], lw=0, ec='C0', c=cmap(
        #        charge_norm(hits[hits['io_group'] == 8]['Q'])), s=5, alpha=1)
        

    # Temporary fix for bug in light geometry LUTs
    def convert_light_x(self, light_lut_idx1, light_lut_idx2, light_x):
        
            rel_pos = self.sipm_rel_pos[(light_lut_idx1, light_lut_idx2)][0]
            if rel_pos[0]%2 == 0:
                light_x_true = light_x #- (4*15.215) # ARTIFICIALLY ADDING LIGHT INFO:
            elif rel_pos[0]%2 == 1:
                light_x_true = light_x #+ (4*15.215) # ARTIFICIALLY ADDING LIGHT INFO:

            return light_x_true


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
                true_x = self.convert_light_x(i,j,pos[0])
                #print("True x:", true_x, "Pos[0]:", pos[0])
                if (abs(true_x-x) < 0.5) and (pos[2]==z):
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
                                          color=cmap_value, alpha=1, linewidth=5, solid_capstyle='butt')
                            elif (abs(x-self.geometry.attrs['module_RO_bounds'][i][1][0]) < 1):
                                self.ax_bd.plot([self.geometry.attrs['module_RO_bounds'][i][j][2]+z_offset, self.geometry.attrs['module_RO_bounds'][i][j][2]+z_offset], \
                                         [self.geometry.attrs['module_RO_bounds'][i][1][0]-self.geometry.attrs['max_drift_distance'], self.geometry.attrs['module_RO_bounds'][i][1][0]] ,\
                                          color=cmap_value, alpha=1, linewidth=5, solid_capstyle='butt')

        # Plot cathodes for ZX (beam, drift) projections (done for charge only case in set_axes method):
        for i in range(len(self.geometry.attrs['module_RO_bounds'])):
            for j in range(2):
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
                                     [y-2, y+2],color=cmap_value, alpha=1, linewidth=5, solid_capstyle='butt')

        # Plot light in XY projection               
        for x,y in itertools.product(self.sipm_unique_x,self.sipm_unique_y):
            if x==-1: continue
            if y==-1: continue
            this_xy_sum = 0
            for i,j in itertools.product(range(light_wvfms[0].shape[0]),range(light_wvfms[0].shape[1])):
                pos=self.sipm_abs_pos[(i,j)][0]
                det_id = self.light_det_id[(i,j)][0]
                wvfm_factor = 1.
                if det_id in acl_det_ids:
                    wvfm_factor = 2.
                if pos[0]==-1:
                    continue
                true_x = self.convert_light_x(i,j,pos[0])
                if (abs(true_x-x) < 0.5) and (pos[1]==y):
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
                                    [y-2, y+2],color=cmap_value, alpha=1, linewidth=5, solid_capstyle='butt')

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
            true_x = self.convert_light_x(i,j,pos[0])
            found_x = 0
            for k in range(len(self.geometry.attrs['module_RO_bounds'])):
                if (abs(true_x-self.geometry.attrs['module_RO_bounds'][k][0][0]) < 1):
                    x1 = self.geometry.attrs['module_RO_bounds'][k][0][0]
                    x2 = self.geometry.attrs['module_RO_bounds'][k][0][0]+self.geometry.attrs['max_drift_distance']
                    found_x = 1
                elif (abs(true_x-self.geometry.attrs['module_RO_bounds'][k][1][0]) < 1):
                    x1 = self.geometry.attrs['module_RO_bounds'][k][1][0]-self.geometry.attrs['max_drift_distance']
                    x2 = self.geometry.attrs['module_RO_bounds'][k][1][0]
                    found_x = 1
                if found_x ==1:
                    light_x, light_y, light_z = make_z_plane(x1,x2, pos[1]-2, pos[1]+2,z_pos)
                    self.ax_bdv.plot_surface(light_z,light_x,light_y,color=cmap_value, alpha=0.1, shade=False)
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

