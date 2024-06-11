import sys
import warnings
import numpy as np
from datetime import datetime
from proto_nd_flow.util.lut import LUT
import itertools
import os
import math
import h5py
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

class LArEventDisplay:

    ''' 
        Class to set up interactive 2x2 display for files run through proto_nd_flow.

        Inputs are as follows:

            - filedir         (str): path to input file
            - flow_file       (str): path to input file
            - nhits           (int): hit threshold for events to be made available in interactive display
            - show_light      (bool): whether to show light information in display (default: True)
            
        In order to run the display, set up a Jupyter Notebook, import everything in this file,
        and execute the run() method, e.g.:

        from lar_only_evd import *
        plt.ion()

        f = '/path/to/file'
        evd = ProtoNDFlowEventDisplay(flow_file=f, nhits=1, show_light=True)
        evd.run()
    '''

    def __init__(self, filedir,filename, nhits=1, show_light=True):
        
        f = h5py.File(filedir+filename, 'r')
        self.filename = filename
        self.show_light = show_light

        # Load events dataset
        events = f['charge/events/data']
        self.events = events[events['nhit'] > nhits]

        # Load charge hits dataset
        self.hits_dset = 'calib_prompt_hits'
        self.hits_full = f['charge/'+self.hits_dset+'/data']
        self.hits_ref = f['charge/events/ref/charge/'+self.hits_dset+'/ref']
        self.hits_region = f['charge/events/ref/charge/'+self.hits_dset+'/ref_region']
        
        # Load light event and waveform datasets
        self.light_events = f['light/events/data']
        self.charge_light_ref = f['charge/events/ref/light/events/ref']
        self.charge_light_region = f['charge/events/ref/light/events/ref_region']
        self.light_wvfms = f['light/wvfm/data']
        self.light_event_wvfm_ref = f['light/events/ref']['light/wvfm']['ref']
        self.light_event_wvfm_region = f['light/events/ref']['light/wvfm']['ref_region']

        # Load geometry and other info
        self.geometry = f['geometry_info']
        self.info = {
            'vdrift': f['lar_info'].attrs['v_drift'],
            'clock_period': 0.1,
        }

        self.sipm_abs_pos = LUT.from_array(f["geometry_info/sipm_abs_pos"].attrs["meta"],f["geometry_info/sipm_abs_pos/data"])
        self.sipm_rel_pos = LUT.from_array(f["geometry_info/sipm_rel_pos"].attrs["meta"],f["geometry_info/sipm_rel_pos/data"])

        self.all_sipm_pos = f["geometry_info/sipm_abs_pos/data"]["data"][1:]
        self.sipm_unique_x = np.unique([pos[0] for pos in self.all_sipm_pos])
        self.sipm_unique_z = np.unique([pos[2] for pos in self.all_sipm_pos])
        self.sipm_unique_y = np.unique([pos[1] for pos in self.all_sipm_pos])

        # Set up figure and subplots
        self.fig = plt.figure(constrained_layout=False, figsize=(13, 13))
        #gs_zyxy = self.fig.add_gridspec(nrows=1, ncols=3, top=0.93, width_ratios=[1, 1, 0.05],
        #                                left=0.15, right=0.5, bottom=0.58,
        #                                hspace=0, wspace=0)
        ax_bdv  = self.fig.add_subplot(222, projection='3d')
        ax_bd = self.fig.add_subplot(221)
        ax_bv = self.fig.add_subplot(223)
        ax_dv = self.fig.add_subplot(224)
        cbar_ax = self.fig.add_axes([0.95, 0.12, 0.03, 0.75])
        light_cbar_ax = self.fig.add_axes([0.15, 0.005, 0.75, 0.03])
        self.fig.subplots_adjust(right=0.9)
        self.fig.subplots_adjust(bottom=0.1)
        #ax_xy = self.fig.add_subplot(gs_zyxy[0])
        #ax_zy = self.fig.add_subplot(gs_zyxy[1], sharey=ax_xy)

        self.ax_bdv = ax_bdv
        self.ax_bd = ax_bd
        self.ax_bv = ax_bv
        self.ax_dv = ax_dv
        self.cbar_ax = cbar_ax
        self.light_cbar_ax = light_cbar_ax


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
        self.light_cbar_ax.cla()


    def get_event(self, ev_id):

        # Get event charge information
        event = self.events[ev_id]
        event_datetime = datetime.utcfromtimestamp(
                event['unix_ts']).strftime('%Y-%m-%d %H:%M:%S')
        ev_id = event['id']
        hit_ref = self.hits_ref[self.hits_region[ev_id,'start']:self.hits_region[ev_id,'stop']]
        hit_ref = np.sort(hit_ref[hit_ref[:,0] == ev_id, 1])
        hits = self.hits_full[hit_ref]

        # Get event light information
        light_matches = self.charge_light_ref[self.charge_light_region[ev_id,'start']:self.charge_light_region[ev_id,'stop']]
        light_matches = np.sort(light_matches[light_matches[:,0] == ev_id, 1])
        
        light = self.light_events[light_matches]
        light_idx = light[0][0]

        light_wvfm_ref = self.light_event_wvfm_ref[self.light_event_wvfm_region[light_idx,'start']:self.light_event_wvfm_region[light_idx,'stop']]
        light_wvfm_ref = np.sort(light_wvfm_ref[light_wvfm_ref[:,0] == light_idx, 1])
        light_wvfms = self.light_wvfms[light_wvfm_ref]["samples"]

        # Prepare color map for charge
        charge_norm = mpl.colors.LogNorm(vmin=min(hits['Q']),vmax=max(hits['Q']))
        cmap = plt.cm.get_cmap('jet')
        mcharge = plt.cm.ScalarMappable(norm=charge_norm, cmap=cmap)

        #Prepare color map for light
        light_cmap=cm.afmhot
        light_norm = colors.LogNorm(1,light_wvfms[0].sum(axis=-1).max()*100)
        light_norm_single = colors.LogNorm(1,light_wvfms[0].sum(axis=-1).max())
        c = light_norm(light_wvfms[0].sum(axis=-1))
        mlight = plt.cm.ScalarMappable(norm=light_norm, cmap=light_cmap)

        # Set figure title
        self.fig.suptitle("Event %i, ID %i - %s UTC" %
                          (ev_id, event['id'], event_datetime), size=24)

        return hits, light_wvfms, mcharge, mlight, cmap, light_cmap, charge_norm, light_norm
    

    def set_axes(self, cmap, mcharge, mlight):

        '''
            cmap: charge color map
            mcharge: charge color map scale
            mlight: light color map scale
        '''

        # Set axes for 3D canvas (Beam, Drift, Vertical)
        self.ax_bdv.set_xlabel('Beam Direction [cm]', fontsize=15) #z
        self.ax_bdv.set_ylabel('Drift Direction [cm]', fontsize=15) #x
        self.ax_bdv.set_zlabel('Vertical Direction [cm]', fontsize=15) #y
        self.ax_bdv.set_xlim(self.geometry.attrs['lar_detector_bounds'][0][2], \
            self.geometry.attrs['lar_detector_bounds'][1][2])
        self.ax_bdv.set_ylim(self.geometry.attrs['lar_detector_bounds'][0][0], \
            self.geometry.attrs['lar_detector_bounds'][1][0])
        self.ax_bdv.set_zlim(self.geometry.attrs['lar_detector_bounds'][0][1], \
            self.geometry.attrs['lar_detector_bounds'][1][1])
        self.ax_bdv.grid(False)
        self.ax_bdv.tick_params(axis='both', which='major', labelsize=14)

        # Set axes for Beam vs Drift (ZX) canvas
        self.ax_bd.set_xlabel('Beam Direction [cm]', fontsize=15)
        self.ax_bd.set_ylabel('Drift Direction [cm]', fontsize=15)
        self.ax_bd.set_ylim(self.geometry.attrs['lar_detector_bounds'][0][0]-0.5, \
            self.geometry.attrs['lar_detector_bounds'][1][0]+0.5)
        self.ax_bd.set_xlim(self.geometry.attrs['lar_detector_bounds'][0][2]-3.15,\
            self.geometry.attrs['lar_detector_bounds'][1][2]+3)
        self.ax_bd.tick_params(axis='both', which='major', labelsize=14)

        # Set axes for Beam vs Vertical (ZY) canvas
        self.ax_bv.set_xlabel('Beam Direction [cm]', fontsize=15)
        self.ax_bv.set_ylabel('Vertical Direction [cm]', fontsize=15)
        self.ax_bv.set_xlim(self.geometry.attrs['lar_detector_bounds'][0][2]-3.05,\
            self.geometry.attrs['lar_detector_bounds'][1][2]+2.9)
        self.ax_bv.set_ylim(self.geometry.attrs['lar_detector_bounds'][0][1]-0.5, \
            self.geometry.attrs['lar_detector_bounds'][1][1]+0.5)
        self.ax_bv.tick_params(axis='both', which='major', labelsize=14)

        # Set axes for Drift vs Vertical (XY) canvas
        self.ax_dv.set_xlabel('Drift Direction [cm]', fontsize=15)
        self.ax_dv.set_ylabel('Vertical Direction [cm]', fontsize=15)
        self.ax_dv.set_xlim(self.geometry.attrs['lar_detector_bounds'][0][0]-2.95,\
            self.geometry.attrs['lar_detector_bounds'][1][0]+2.8)
        self.ax_dv.set_ylim(self.geometry.attrs['lar_detector_bounds'][0][1]-0.4, \
            self.geometry.attrs['lar_detector_bounds'][1][1]+0.5)
        self.ax_dv.tick_params(axis='both', which='major', labelsize=14)

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
                            [self.geometry.attrs['module_RO_bounds'][i][k][1], self.geometry.attrs['module_RO_bounds'][i][k][1]], color=cmap(0), alpha=0.35)

                    self.ax_bdv.plot([self.geometry.attrs['module_RO_bounds'][i][j][2], self.geometry.attrs['module_RO_bounds'][i][j][2]], \
                            [self.geometry.attrs['module_RO_bounds'][i][k][0], self.geometry.attrs['module_RO_bounds'][i][k][0]], \
                            [self.geometry.attrs['module_RO_bounds'][i][0][1], self.geometry.attrs['module_RO_bounds'][i][1][1]], color=cmap(0), alpha=0.35)

                    self.ax_bdv.plot([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                            [self.geometry.attrs['module_RO_bounds'][i][j][0], self.geometry.attrs['module_RO_bounds'][i][j][0]], \
                            [self.geometry.attrs['module_RO_bounds'][i][k][1], self.geometry.attrs['module_RO_bounds'][i][k][1]], color=cmap(0), alpha=0.35)

                # Plot outlines of modules for ZX (beam, drift) projections:
                self.ax_bd.plot([self.geometry.attrs['module_RO_bounds'][i][j][2], self.geometry.attrs['module_RO_bounds'][i][j][2]], \
                         [self.geometry.attrs['module_RO_bounds'][i][0][0], self.geometry.attrs['module_RO_bounds'][i][1][0]],\
                         color=cmap(0), alpha=1)
                self.ax_bd.plot([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                         [self.geometry.attrs['module_RO_bounds'][i][j][0], self.geometry.attrs['module_RO_bounds'][i][j][0]],\
                         color=cmap(0), alpha=1)

                # Only two modules represented in ZY and XY projections
                if i >= 2: continue

                # Plot outlines of modules for ZY (beam, vertical) projections:
                self.ax_bv.plot([self.geometry.attrs['module_RO_bounds'][i][j][2], self.geometry.attrs['module_RO_bounds'][i][j][2]], \
                         [self.geometry.attrs['module_RO_bounds'][i][0][1], self.geometry.attrs['module_RO_bounds'][i][1][1]],\
                         color=cmap(0), alpha=1)
                self.ax_bv.plot([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                         [self.geometry.attrs['module_RO_bounds'][i][j][1], self.geometry.attrs['module_RO_bounds'][i][j][1]],\
                         color=cmap(0), alpha=1)

                # Plot outlines of modules for XY (drift, vertical) projections:
                # X footprints are the same for i=0 and i=1, but different for i=1 and i=2
                self.ax_dv.plot([self.geometry.attrs['module_RO_bounds'][i+1][j][0], self.geometry.attrs['module_RO_bounds'][i+1][j][0]], \
                         [self.geometry.attrs['module_RO_bounds'][i+1][0][1], self.geometry.attrs['module_RO_bounds'][i+1][1][1]],\
                         color=cmap(0), alpha=1)
                self.ax_dv.plot([self.geometry.attrs['module_RO_bounds'][i+1][0][0], self.geometry.attrs['module_RO_bounds'][i+1][1][0]], \
                         [self.geometry.attrs['module_RO_bounds'][i+1][j][1], self.geometry.attrs['module_RO_bounds'][i+1][j][1]],\
                         color=cmap(0), alpha=1)


            # Fill modules for ZX (beam, drift) projections:
            self.ax_bd.fill([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][0][2], \
                      self.geometry.attrs['module_RO_bounds'][i][1][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                     [self.geometry.attrs['module_RO_bounds'][i][0][0], self.geometry.attrs['module_RO_bounds'][i][1][0], \
                      self.geometry.attrs['module_RO_bounds'][i][1][0], self.geometry.attrs['module_RO_bounds'][i][0][0]],\
                     color=cmap(0), alpha=0.85)

            # Only two modules represented in ZY and XY projections
            if i >= 2: continue

            # Fill modules for ZY (beam, vertical) projections:
            self.ax_bv.fill([self.geometry.attrs['module_RO_bounds'][i][0][2], self.geometry.attrs['module_RO_bounds'][i][0][2], \
                      self.geometry.attrs['module_RO_bounds'][i][1][2], self.geometry.attrs['module_RO_bounds'][i][1][2]], \
                     [self.geometry.attrs['module_RO_bounds'][i][0][1], self.geometry.attrs['module_RO_bounds'][i][1][1], \
                      self.geometry.attrs['module_RO_bounds'][i][1][1], self.geometry.attrs['module_RO_bounds'][i][0][1]],\
                     color=cmap(0), alpha=0.85)

            # Fill modules for XY (drift, vertical) projections:
            self.ax_dv.fill([self.geometry.attrs['module_RO_bounds'][i+1][0][0], self.geometry.attrs['module_RO_bounds'][i+1][0][0], \
                      self.geometry.attrs['module_RO_bounds'][i+1][1][0], self.geometry.attrs['module_RO_bounds'][i+1][1][0]], \
                     [self.geometry.attrs['module_RO_bounds'][i+1][0][1], self.geometry.attrs['module_RO_bounds'][i+1][1][1], \
                      self.geometry.attrs['module_RO_bounds'][i+1][1][1], self.geometry.attrs['module_RO_bounds'][i+1][0][1]],\
                     color=cmap(0), alpha=0.85)

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
        cbar.set_label(r'Charge [$10^3$ e]', size=15)
        self.cbar_ax.tick_params(labelsize=14)

        # Set light colorbar
        light_cbar = self.fig.colorbar(mlight, cax=self.light_cbar_ax, label=r'Light [ADC Counts]', orientation = 'horizontal')
        light_cbar.set_label(r'Light [ADC Counts]', size=15)
        self.light_cbar_ax.tick_params(labelsize=14)


    def display_event(self, ev_id):

        self.clear_axes()
        hits, light_wvfms, mcharge, mlight, cmap, light_cmap, charge_norm, light_norm = self.get_event(ev_id)

        # Plot hits in 3D view first so that cathodes/anodes go over the hits
        self.ax_bdv.scatter(hits['z'], hits['x'], hits['y'], lw=0, ec='C0', \
                            c=cmap(charge_norm(hits['Q'])), s=5, alpha=1)

        self.set_axes(cmap, mcharge, mlight)

        if self.show_light:
            self.plot_light(light_wvfms, light_cmap, light_norm)

        # Plot 2D charge hits
        self.ax_bd.scatter(hits['z'], hits['x'], lw=0, ec='C0', c=cmap(
                charge_norm(hits['Q'])), s=5, alpha=1)
        self.ax_bv.scatter(hits['z'], hits['y'], lw=0, ec='C0', c=cmap(
                charge_norm(hits['Q'])), s=5, alpha=1)
        self.ax_dv.scatter(hits['x'], hits['y'], lw=0, ec='C0', c=cmap(
                charge_norm(hits['Q'])), s=5, alpha=1)
        

    # temporary fix for bug in light geometry LUTs
    def convert_light_x(self, light_lut_idx1, light_lut_idx2, light_x):
        
            rel_pos = self.sipm_rel_pos[(light_lut_idx1, light_lut_idx2)][0]
            if rel_pos[0]%2 == 0:
                light_x_true = light_x - (4*15.215)
            elif rel_pos[0]%2 == 1:
                light_x_true = light_x + (4*15.215)

            return light_x_true


    def plot_light(self, light_wvfms, light_cmap, light_norm):
        
        # Plot light in XZ (drift, beam) projection
        for x,z in itertools.product(self.sipm_unique_x,self.sipm_unique_z):
            if x==-1: continue
            if z==-1: continue
            this_xz_sum = 0
            for i,j in itertools.product(range(light_wvfms[0].shape[0]),range(light_wvfms[0].shape[1])):
                #if i!=0: continue
                pos=self.sipm_abs_pos[(i,j)][0]
                rel_pos = self.sipm_rel_pos[(i,j)][0]
                if pos[0]==-1:
                    continue
                true_x = self.convert_light_x(i,j,pos[0])
                #print("True x:", true_x, "Pos[0]:", pos[0])
                if (abs(true_x-x) < 0.5) and (pos[2]==z):
                    this_xz_sum += light_wvfms[0][i,j].sum()
                    #print("SIPM:",i,j,"Abs pos:",pos,"Rel pos:",rel_pos,"Light sum:",light_wvfms[0][i,j].sum())
            for i in range(len(self.geometry.attrs['module_RO_bounds'])):
                for j in range(2):
                    #if count != chosen_count: continue
                    z_offset = ((-1)**(j+1))*1.25
                    if (abs(z-self.geometry.attrs['module_RO_bounds'][i][j][2]) < 1):
                        if this_xz_sum==0:
                            cmap_value = light_cmap(0)
                        else:
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
                if pos[0]==-1:
                    continue
                if (pos[2]==z) and (pos[1]==y):
                    this_zy_sum += light_wvfms[0][i,j].sum()
            for i in range(len(self.geometry.attrs['module_RO_bounds'])):
                for j in range(2):
                    z_offset = ((-1)**(j+1))*1.25
                    if this_zy_sum==0:
                        cmap_value = light_cmap(0)
                    else:
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
                if pos[0]==-1:
                    continue
                true_x = self.convert_light_x(i,j,pos[0])
                if (abs(true_x-x) < 0.5) and (pos[1]==y):
                    this_xy_sum += light_wvfms[0][i,j].sum()
            for i in range(len(self.geometry.attrs['module_RO_bounds'])):
                for j in range(2):
                    x_offset = ((-1)**(j+1))*1.25
                    if this_xy_sum==0:
                        cmap_value = light_cmap(0)
                    else:
                        cmap_value = light_cmap(light_norm(this_xy_sum))
                    if (abs(x-self.geometry.attrs['module_RO_bounds'][i][j][0]) < 1):
                        self.ax_dv.plot([self.geometry.attrs['module_RO_bounds'][i][j][0]+x_offset, self.geometry.attrs['module_RO_bounds'][i][j][0]+x_offset], \
                                 [y-2, y+2],color=cmap_value, alpha=1, linewidth=5, solid_capstyle='butt')

        # Plot light for XYZ (beam, drift, vertical) 3D view: 
        for i,j in itertools.product(range(light_wvfms[0].shape[0]),range(light_wvfms[0].shape[1])):
            pos=self.sipm_abs_pos[(i,j)][0]
            if pos[0]==-1:
                continue
            this_xyz_sum = light_wvfms[0][i,j].sum()
            if this_xyz_sum==0:
                cmap_value = light_cmap(0)
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
                elif (abs(true_x-geometry.attrs['module_RO_bounds'][k][1][0]) < 1):
                    x1 = self.geometry.attrs['module_RO_bounds'][k][1][0]-self.geometry.attrs['max_drift_distance']
                    x2 = self.geometry.attrs['module_RO_bounds'][k][1][0]
                    found_x = 1
                if found_x ==1:
                    light_x, light_y, light_z = make_z_plane(x1,x2, pos[1]-2, pos[1]+2,z_pos)
                    self.ax_bdv.plot_surface(light_z,light_x,light_y,color=cmap_value, alpha=0.15, shade=False)
                    #print("Light sum at:", pos, "is", this_xyz_sum)
                    break
                else: continue


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

