import sys
import warnings
from datetime import datetime
import numpy as np
import h5py
from IPython.display import display, clear_output
import yaml
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

class ProtoNDFlowEventDisplay:

    '''
        Class to set up interactive single module displays for files run through proto_nd_flow.
        
        Inputs are as follows:

            - filedir       (str): path to input file
            - filename      (str): name of file; must be flow file run through module0_flow
            - geometry_file (str): full path and name of geometry file describing module to be displayed
            - nhits         (int): hit threshold for events to be made available in interactive display
            - hits_dset     (str): dataset of hits within the file that you want to display
                                   options are 'raw_hits', 'calib_prompt_hits', and 'calib_final_hits'
        
        In order to run the display, set up a Jupyter Notebook, import everything in this file,
        and execute the run() method, e.g.:
        
        from protond0flow_evd import *
        plt.ion()
        
        d = '/path/to/file'
        f = 'name_of_file'
        g = '/path/to/geometry/file/name_of_geometry_file'
        hd = 'hits_dataset_you_want_to_display'

        evd = ProtoNDFlowEventDisplay(filedir=d, filename=f, geometry_file=g,nhits=1, hits_dset=hd)
        test_evd.run()
    '''
    def __init__(self, filedir, filename, geometry_file=None, nhits=1, hits_dset='calib_final_hits'):
        f = h5py.File(filedir+filename, 'r')
        self.filename = filename

        # Set name of hits dataset to be used
        self.hits_dset = hits_dset

        # Load datasets
        events = f['charge/events/data']
        self.events = events[events['nhit'] > nhits]
        try:
            self.tracks = f['combined/tracklets/data']
            self.tracks_ref = f['charge/events/ref/combined/tracklets/ref']
            self.tracks_region = f['charge/events/ref/combined/tracklets/ref_region']
            self.hits_trk_ref = f['combined/tracklets/ref/charge/hits/ref']
            self.hits_trk_region = f['combined/tracklets/ref/charge/hits/ref_region']
            self.hits_drift = f['combined/hit_drift/data']
        except KeyError:
            print("No tracklets found")
        self.hits = f['charge/'+self.hits_dset+'/data']
        self.hits_ref = f['charge/events/ref/charge/'+self.hits_dset+'/ref']
        self.hits_region = f['charge/events/ref/charge/'+self.hits_dset+'/ref_region']
        self.ext_trigs = f['charge/ext_trigs/data']
        self.ext_trigs_ref = f['charge/events/ref/charge/ext_trigs/ref']
        self.ext_trigs_region = f['charge/events/ref/charge/ext_trigs/ref_region']
        self.info = {
            'vdrift': f['lar_info'].attrs['v_drift'],
            'clock_period': 0.1,
        }

        # Set dataset-dependent field names and settings
        # charge/raw_hits dataset needs charge/packets information for effective plotting
        if self.hits_dset == 'raw_hits':
            
            self.charge = 'ADC'
            self.x_vals = 'x_pix' 
            self.y_vals = 'y_pix'
            self.z_vals = 'z_pix' 
            self.convert_to_mm = 10 # xyz starts in cm
            self.y_offset = 0. # mm
            self.packets = f['charge/packets/data']
            self.packets_hits_ref = f['charge/'+self.hits_dset+'/ref/charge/packets/ref']
            self.packets_hits_region = f['charge/'+self.hits_dset+'/ref/charge/packets/ref_region']
            self.ped_mv = 580.
            self.vcm_mv = 288.
            self.vref_mv = 1300.
        
        else: # e.g. for calib_final_hits and calib_prompt_hits
            
            self.charge = 'Q'
            self.x_vals = 'x'
            self.y_vals = 'y'
            self.z_vals = 'z'
            self.convert_to_mm = 10 # xyz starts in cm
            self.y_offset = 0. # mm

        # Set up figure and subplots
        self.fig = plt.figure(constrained_layout=False, figsize=(8.5, 6.))
        gs_zyxy = self.fig.add_gridspec(nrows=1, ncols=3, top=0.93, width_ratios=[1, 1, 0.05],
                                        left=0.15, right=0.5, bottom=0.58,
                                        hspace=0, wspace=0)

        ax_xy = self.fig.add_subplot(gs_zyxy[0])
        ax_zy = self.fig.add_subplot(gs_zyxy[1], sharey=ax_xy)

        cax = self.fig.add_subplot(gs_zyxy[2])
        ip = InsetPosition(ax_zy, [1.1, 0, 0.1, 1])
        cax.set_axes_locator(ip)
        self.cax = cax
        self.fig.text(
            0.04, 0.25, 'charge [10$^3$ e]', ha='center', va='center', rotation='vertical')

        gs_time = self.fig.add_gridspec(nrows=2, ncols=1,
                                        left=0.1, right=0.57, bottom=0.08, top=0.47,
                                        hspace=0.09)
        ax_time_1 = self.fig.add_subplot(gs_time[0])
        ax_time_2 = self.fig.add_subplot(gs_time[1], sharex=ax_time_1)

        gs_zyx = self.fig.add_gridspec(nrows=1, ncols=1,
                                       left=0.52, right=0.99, bottom=0.1, top=0.95,
                                       wspace=0)
        ax_zyx = self.fig.add_subplot(gs_zyx[0], projection='3d')
        ax_zyx.set_facecolor('none')

        # Use geometry file to set up Module display
        if not geometry_file:
            geometry_file = self.info['geometry_file']

        with open(geometry_file, 'r') as gf:
            tile_layout = yaml.load(gf, Loader=yaml.FullLoader)

        mm2cm = 0.1
        pixel_pitch = tile_layout['pixel_pitch'] * mm2cm
        chip_channel_to_position = tile_layout['chip_channel_to_position']
        tile_chip_to_io = tile_layout['tile_chip_to_io']
        self.io_group_io_channel_to_tile = {}
        for tile in tile_chip_to_io:
            for chip in tile_chip_to_io[tile]:
                io_group_io_channel = tile_chip_to_io[tile][chip]
                io_group = io_group_io_channel//1000
                io_channel = io_group_io_channel % 1000
                self.io_group_io_channel_to_tile[(io_group, io_channel)] = tile

        cm2mm = 10

        xs = np.array(list(chip_channel_to_position.values()))[
            :, 0] * pixel_pitch * cm2mm
        ys = np.array(list(chip_channel_to_position.values()))[
            :, 1] * pixel_pitch * cm2mm
        tile_borders = np.zeros((2, 2))
        tpc_borders = np.zeros((0, 3, 2))
        tpc_centers = np.array(list(tile_layout['tpc_centers'].values()))
        tile_borders[0] = [-(max(xs)+pixel_pitch)/2, (max(xs)+pixel_pitch)/2]
        tile_borders[1] = [-(max(ys)+pixel_pitch)/2, (max(ys)+pixel_pitch)/2]

        tile_positions = np.array(list(tile_layout['tile_positions'].values()))
        tile_orientations = np.array(
            list(tile_layout['tile_orientations'].values()))
        self.tile_positions = tile_positions
        self.tile_orientations = tile_orientations
        tpcs = np.unique(tile_positions[:, 0])
        tpc_borders = np.zeros((len(tpcs), 3, 2))

        self.drift_length = abs(tile_positions[0][0])
        self.drift_time = self.drift_length / \
            self.info['vdrift']/self.info['clock_period'] if self.info \
            else self.drift_length / 1.648 / 0.1

        for itpc, tpc_id in enumerate(tpcs):
            this_tpc_tile = tile_positions[tile_positions[:, 0] == tpc_id]
            this_orientation = tile_orientations[tile_positions[:, 0] == tpc_id]

            x_border = min(this_tpc_tile[:, 2])+tile_borders[0][0]+tpc_centers[itpc][0], \
                max(this_tpc_tile[:, 2]) + \
                tile_borders[0][1]+tpc_centers[itpc][0]
            y_border = min(this_tpc_tile[:, 1])+tile_borders[1][0]+tpc_centers[itpc][1], \
                max(this_tpc_tile[:, 1]) + \
                tile_borders[1][1]+tpc_centers[itpc][1]
            z_border = min(this_tpc_tile[:, 0])+tpc_centers[itpc][2], \
                max(this_tpc_tile[:, 0])+self.drift_length * \
                this_orientation[:, 0][0]+tpc_centers[itpc][2]

            tpc_borders[itpc] = (x_border, y_border, z_border)

        self.tpc_borders = tpc_borders
        self.ax_zyx = ax_zyx
        self.ax_zy = ax_zy
        self.ax_time_1 = ax_time_1
        self.ax_time_2 = ax_time_2
        self.ax_xy = ax_xy

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
                plt.savefig(self.filename+'_Event_'+str(ev_id)+'_using_'+self.hits_dset+'.pdf')
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

    # Short method to return tile_id from io_group and io_channel
    def _get_tile_id(self, io_group, io_channel):
        if (io_group, io_channel) in self.io_group_io_channel_to_tile:
            tile_id = self.io_group_io_channel_to_tile[io_group, io_channel]
        else:
            warnings.warn("IO group %i, IO channel %i not found" %
                          (io_group, io_channel))
            return 0

        return tile_id

    
    # Short method to return z coordinate from io_group, io_channel, and time
    def _get_z_coordinate(self, io_group, io_channel, time):
        tile_id = self._get_tile_id(io_group, io_channel)

        z_anode = self.tile_positions[tile_id-1][0]
        drift_direction = self.tile_orientations[tile_id-1][0]
        return z_anode + time*self.info['vdrift']*self.info['clock_period']*drift_direction
    
    # Short method to convert ADC to ke- (only necessary for charge/raw_hits dataset)
    def charge_from_ADC(self, dw, vref, vcm, ped):
        return (dw / 256. * (vref - vcm) + vcm - ped) / 4.

    def get_event_start_time(self, event):
        """Estimate the event start time"""
        if event['n_ext_trigs']:
            # First Choice: Use earliest light system trigger in event
            ev_id = event['id']
            ext_trig_ref = self.ext_trigs_ref[self.ext_trigs_region[ev_id,'start']:self.ext_trigs_region[ev_id,'stop']]
            ext_trig_ref = np.sort(ext_trig_ref[ext_trig_ref[:,0] == ev_id, 1])

            return np.min(self.ext_trigs[ext_trig_ref]['ts'])
        # Second Choice:
        #  Try to determine the start time from a 'bump' in charge.
        #  This is only valid if some part of the event hits one of the anodes.
        ticks_per_qsum = 10  # clock ticks per time bin
        t0_charge_threshold = 200.0 / 4. # Rough qsum threshold
        ev_id = event['id']
        hit_ref = self.hits_ref[self.hits_region[ev_id,'start']:self.hits_region[ev_id,'stop']]
        hit_ref = np.sort(hit_ref[hit_ref[:,0] == ev_id, 1])
        hits = self.hits[hit_ref]
        # determine charge vs time in enlarged window
        min_ts = np.amin(hits['ts_pps'])
        max_ts = np.amax(hits['ts_pps'])
        # If event long enough, calculate qsum vs time
        if (max_ts - min_ts) > ticks_per_qsum:
            time_bins = np.arange(min_ts-ticks_per_qsum,
                                  max_ts+ticks_per_qsum)
            # integrate q in sliding window to produce qsum profile
            #  histogram charge
            if self.hits_dset == 'raw_hits':
                q_vs_t = np.histogram(hits['ts_pps'],
                                  bins=time_bins,
                                  weights=self.charge_from_ADC(hits[self.charge], self.vref_mv, self.vcm_mv, self.ped_mv))[0]
            else:
                q_vs_t = np.histogram(hits['ts_pps'],
                                  bins=time_bins,
                                  weights=hits[self.charge])[0]
            #  calculate rolling qsum
            qsum_vs_t = np.convolve(q_vs_t,
                                    np.ones(ticks_per_qsum, dtype=int),
                                    'valid')
            t0_bin_index = np.argmax(qsum_vs_t > t0_charge_threshold)
            t0_bin_index += ticks_per_qsum
            start_time = time_bins[t0_bin_index]
            # Check if qsum exceed threshold
            if start_time < max_ts:
                return start_time
        # Fallback is to use the first hit
        return event['ts_start']

    # Set up axes
    def set_axes(self):

        self.ax_time_2.set_xlabel(r"timestamp [0.1 $\mathrm{\mu}$s]")
        self.ax_time_1.set_title("TPC 1", fontsize=10, x=0.5, y=0.75)
        self.ax_time_2.set_title("TPC 2", fontsize=10, x=0.5, y=0.75)

        self.ax_zy.set_xlim(np.min(self.tpc_borders[:, 2, :]), np.max(
            self.tpc_borders[:, 2, :]))
        self.ax_zy.set_ylim(np.min(self.tpc_borders[:, 1, :]), np.max(
            self.tpc_borders[:, 1, :]))
        self.ax_zy.set_aspect('equal')
        self.ax_zy.set_xlabel("Z [mm]")
        for tk in self.ax_zy.get_yticklabels():
            tk.set_visible(False)
        # self.ax_zy.set_yticklabels([])

        self.ax_xy.set_xlim(np.min(self.tpc_borders[:, 0, :]), np.max(
            self.tpc_borders[:, 0, :]))
        self.ax_xy.set_ylim(np.min(self.tpc_borders[:, 1, :]), np.max(
            self.tpc_borders[:, 1, :]))
        self.ax_xy.set_aspect('equal')
        self.ax_xy.set_xlabel("X [mm]")
        self.ax_xy.set_ylabel("Y [mm]")

        self.ax_xy.axvline(0, c='gray')

        anode1 = plt.Rectangle((self.tpc_borders[0][0][0], self.tpc_borders[0][1][0]),
                               self.tpc_borders[0][0][1] -
                               self.tpc_borders[0][0][0],
                               self.tpc_borders[0][1][1] -
                               self.tpc_borders[0][1][0],
                               linewidth=1, fc='none',
                               edgecolor='gray')
        self.ax_zyx.add_patch(anode1)
        art3d.pathpatch_2d_to_3d(anode1, z=self.tpc_borders[0][2][0], zdir="y")

        anode2 = plt.Rectangle((self.tpc_borders[0][0][0], self.tpc_borders[0][1][0]),
                               self.tpc_borders[0][0][1] -
                               self.tpc_borders[0][0][0],
                               self.tpc_borders[0][1][1] -
                               self.tpc_borders[0][1][0],
                               linewidth=1, fc='none',
                               edgecolor='gray')
        self.ax_zyx.add_patch(anode2)
        art3d.pathpatch_2d_to_3d(anode2, z=self.tpc_borders[1][2][0], zdir="y")

        cathode = plt.Rectangle((self.tpc_borders[0][0][0], self.tpc_borders[0][1][0]),
                                self.tpc_borders[0][0][1] -
                                self.tpc_borders[0][0][0],
                                self.tpc_borders[0][1][1] -
                                self.tpc_borders[0][1][0],
                                linewidth=1, fc='gray', alpha=0.25,
                                edgecolor='gray')
        self.ax_zyx.add_patch(cathode)
        art3d.pathpatch_2d_to_3d(cathode, z=0, zdir="y")

        self.ax_zyx.plot((self.tpc_borders[0][0][0], self.tpc_borders[0][0][0]),
                         (self.tpc_borders[0][2][0],
                          self.tpc_borders[1][2][0]),
                         (self.tpc_borders[0][1][0], self.tpc_borders[0][1][0]), lw=1, color='gray')

        self.ax_zyx.plot((self.tpc_borders[0][0][0], self.tpc_borders[0][0][0]),
                         (self.tpc_borders[0][2][0],
                          self.tpc_borders[1][2][0]),
                         (self.tpc_borders[0][1][1], self.tpc_borders[0][1][1]), lw=1, color='gray')

        self.ax_zyx.plot((self.tpc_borders[0][0][1], self.tpc_borders[0][0][1]),
                         (self.tpc_borders[0][2][0],
                          self.tpc_borders[1][2][0]),
                         (self.tpc_borders[0][1][0], self.tpc_borders[0][1][0]), lw=1, color='gray')

        self.ax_zyx.plot((self.tpc_borders[0][0][1], self.tpc_borders[0][0][1]),
                         (self.tpc_borders[0][2][0],
                          self.tpc_borders[1][2][0]),
                         (self.tpc_borders[0][1][1], self.tpc_borders[0][1][1]), lw=1, color='gray')

        self.ax_zyx.set_ylim(
            np.min(self.tpc_borders[:, 2, :]), np.max(self.tpc_borders[:, 2, :]))
        self.ax_zyx.set_xlim(
            np.min(self.tpc_borders[:, 0, :]), np.max(self.tpc_borders[:, 0, :]))
        self.ax_zyx.set_zlim(
            np.min(self.tpc_borders[:, 1, :]), np.max(self.tpc_borders[:, 1, :]))
        self.ax_zyx.grid(False)
        self.ax_zyx.set_xlabel("Z [mm]")
        self.ax_zyx.set_ylabel("X [mm]")
        self.ax_zyx.set_zlabel("Y [mm]")
        self.ax_zyx.set_box_aspect((2, 2, 4))
        self.ax_zyx.xaxis.set_major_locator(plt.MaxNLocator(3))
        self.ax_zyx.yaxis.set_major_locator(plt.MaxNLocator(3))
        self.ax_zyx.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        self.ax_zyx.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        self.ax_zyx.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        self.ax_zyx.zaxis.labelpad = 20

    def clear_axes(self):
        self.ax_time_1.cla()
        self.ax_time_2.cla()
        self.ax_zyx.cla()
        self.ax_xy.cla()
        self.ax_zy.cla()
        self.cax.cla()
    
    # Main method to control plotting of information
    def display_event(self, ev_id):
        self.clear_axes()
        self.set_axes()

        event = self.events[ev_id]
        event_datetime = datetime.utcfromtimestamp(
            event['unix_ts']).strftime('%Y-%m-%d %H:%M:%S')
        self.fig.suptitle("Event %i, ID %i - %s UTC" %
                          (ev_id, event['id'], event_datetime))
        ev_id = event['id']
        hit_ref = self.hits_ref[self.hits_region[ev_id,'start']:self.hits_region[ev_id,'stop']]
        hit_ref = np.sort(hit_ref[hit_ref[:,0] == ev_id, 1])
        ext_trig_ref = self.ext_trigs_ref[self.ext_trigs_region[ev_id,'start']:self.ext_trigs_region[ev_id,'stop']]
        ext_trig_ref = np.sort(ext_trig_ref[ext_trig_ref[:,0] == ev_id, 1])

        event_start_time = self.get_event_start_time(event)

        hits = self.hits[hit_ref]

        if self.hits_dset == 'raw_hits':
            packets_hits_ref_mask = np.isin(self.packets_hits_ref[:,0], hit_ref)
            packets_hits_ref_masked = self.packets_hits_ref[packets_hits_ref_mask]

            io_groups = np.zeros(hit_ref.shape)
            for i, hit_element in enumerate(hit_ref):
                argwhere = np.argwhere(packets_hits_ref_masked[:,0] == hit_element)[0,0]
                packet_index = packets_hits_ref_masked[argwhere][1]
                io_groups[i] = self.packets[packet_index]['io_group']
                
            packets = self.packets[hit_ref]
        else:
            io_groups = hits['io_group']

        cmap = plt.cm.get_cmap('plasma')

        # Need to convert charge if using charge/raw_hits dataset
        if self.hits_dset == 'raw_hits':
            norm = matplotlib.colors.Normalize(
                vmin=min(self.charge_from_ADC(self.hits[hit_ref][self.charge], self.vref_mv, self.vcm_mv, self.ped_mv)),
                vmax=max(self.charge_from_ADC(self.hits[hit_ref][self.charge], self.vref_mv, self.vcm_mv, self.ped_mv)))
        else:
            norm = matplotlib.colors.Normalize(
                vmin=min(self.hits[hit_ref][self.charge]),
                vmax=max(self.hits[hit_ref][self.charge]))           
        mcharge = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        hits_anode1 = hits[io_groups == 1]
        hits_anode2 = hits[io_groups == 2]

        #hits_anode1 = hits[packets['io_group']*self.convert_to_mm <= 0]
        #hits_anode2 = hits[hits[self.x_vals]*self.convert_to_mm > 0]
        if self.hits_dset == 'raw_hits':
            q_anode1 = self.charge_from_ADC(hits_anode1[self.charge], self.vref_mv, self.vcm_mv, self.ped_mv) 
            q_anode2 = self.charge_from_ADC(hits_anode2[self.charge], self.vref_mv, self.vcm_mv, self.ped_mv)
        else:
            q_anode1 = hits_anode1[self.charge] 
            q_anode2 = hits_anode2[self.charge]
            
        if self.hits_dset == 'raw_hits':
            t_anode1 = hits_anode1['ts_pps']-event_start_time
            t_anode2 = hits_anode2['ts_pps']-event_start_time
        else:
            t_anode1 = hits_anode1['t_drift']
            t_anode2 = hits_anode2['t_drift']
        
        self.ax_time_1.hist(t_anode1, weights=q_anode1,
                            bins=200,  # np.linspace(0,self.drift_time,200),
                            histtype='step', label='binned')
        self.ax_time_2.hist(t_anode2, weights=q_anode2,
                            bins=200,  # np.linspace(0,self.drift_time,200),
                            histtype='step', label='binned')

        if q_anode1.any():
            self.ax_time_1.scatter(
                t_anode1, q_anode1, c='r', s=5, label='hits', zorder=999)
            self.ax_time_1.legend()

        if q_anode2.any():
            self.ax_time_2.scatter(
                t_anode2, q_anode2, c='r', s=5, label='hits', zorder=999)
            if not q_anode1.any():
                self.ax_time_2.legend()

        self.fig.colorbar(mcharge, cax=self.cax, label=r'Charge [$10^3$] e')

        if event['n_ext_trigs']:
            trig_delta = self.ext_trigs[ext_trig_ref]['ts']-event_start_time
            for trig in trig_delta:
                self.ax_time_1.axvline(x=trig, c='g')
                self.ax_time_2.axvline(x=trig, c='g')

        unassoc_hit_mask = np.ones(event['nhit']).astype(bool)

        if 'ntracks' in event.dtype.name and event['ntracks']:
            track_ref = event['track_ref']
            tracks = self.tracks[track_ref]
            track_start = tracks['start']
            track_end = tracks['end']
            for i, track in enumerate(tracks):

                hit_trk_ref = track['hit_ref']
                hits_trk = self.hits[hit_trk_ref]

                # Difference between the z coordinate using the event ts_start (used in the track fitter)
                # and the start time found by get_event_start_time
                z_correction = (self._get_z_coordinate(hits_trk['iogroup'][0], hits_trk['iochannel'][0], event_start_time)
                                - self._get_z_coordinate(hits_trk['iogroup'][0], hits_trk['iochannel'][0], event['ts_start']))

                self.ax_zy.plot((track_start[i][0], track_end[i][0]),
                                (track_start[i][1], track_end[i][1]),
                                c='C{}'.format(i+1), alpha=0.75, lw=1)

                self.ax_xy.plot((track_start[i][2], track_end[i][2]),
                                (track_start[i][1], track_end[i][1]),
                                c='C{}'.format(i+1), alpha=0.75, lw=1)

                hits_anode1 = hits_trk[hits_trk[self.x_vals]*self.convert_to_mm <= 0]
                hits_anode2 = hits_trk[hits_trk[self.x_vals]*self.convert_to_mm >0]

                if self.hits_dset == 'raw_hits':
                    self.ax_zy.scatter(hits_trk['px'], hits_trk['py'], lw=0.2, ec='C{}'.format(
                        i+1), c=cmap(norm(self.charge_from_ADC(hits_trk[self.charge]), self.vref_mv, self.vcm_mv, self.ped_mv)), s=5, alpha=0.75)

                    hit_xvals = [self._get_z_coordinate(io_group, io_channel, time) for io_group, io_channel, time in zip(
                        hits_trk['iogroup'], hits_trk['iochannel'], hits_trk['ts']-track['t0'])]

                    self.ax_xy.scatter(hit_xvals, hits_trk['py'], lw=0.2, ec='C{}'.format(
                        i+1), c=cmap(norm(self.charge_from_ADC(hits_trk[self.charge], self.vref_mv, self.vcm_mv, self.ped_mv))), s=5, alpha=0.75)
                    self.ax_zyx.scatter(hits_trk['px'], hit_xvals, hits_trk['py'], lw=0.2, ec='C{}'.format(
                        i+1), c=cmap(norm(self.charge_from_ADC(hits_trk[self.charge], self.vref_mv, self.vcm_mv, self.ped_mv))), s=5, alpha=0.75)
                else:
                    self.ax_zy.scatter(hits_trk['px'], hits_trk['py'], lw=0.2, ec='C{}'.format(
                        i+1), c=cmap(norm(hits_trk[self.charge])), s=5, alpha=0.75)

                    hit_xvals = [self._get_z_coordinate(io_group, io_channel, time) for io_group, io_channel, time in zip(
                        hits_trk['iogroup'], hits_trk['iochannel'], hits_trk['ts']-track['t0'])]

                    self.ax_xy.scatter(hit_xvals, hits_trk['py'], lw=0.2, ec='C{}'.format(
                        i+1), c=cmap(norm(hits_trk[self.charge])), s=5, alpha=0.75)
                    self.ax_zyx.scatter(hits_trk['px'], hit_xvals, hits_trk['py'], lw=0.2, ec='C{}'.format(
                        i+1), c=cmap(norm(hits_trk[self.charge])), s=5, alpha=0.75)

                self.ax_zyx.plot((track_start[i][0], track_end[i][0]),
                                 (track_start[i][2], track_end[i][2]),
                                 (track_start[i][1], track_end[i][1]),
                                 c='C{}'.format(i+1), alpha=0.5, lw=4)

                unassoc_hit_mask[np.in1d(hits['hid'], hits_trk['hid'])] = 0
                

        ev_id = event['id']
        
        ''' For now, all tracklet plotting is just commented out'''
        '''
        track_ref = self.tracks_ref[self.tracks_region[ev_id,'start']:self.tracks_region[ev_id,'stop']]
        track_ref = np.sort(track_ref[track_ref[:,0] == ev_id, 1])
        tracks = self.tracks[track_ref]
        track_start = tracks['start']
        track_end = tracks['end']
        for itrk, (ts, te) in enumerate(zip(track_start, track_end)):
            hit_ref = self.hits_trk_ref[self.hits_trk_region[tracks[itrk]['id'],'start']:self.hits_trk_region[tracks[itrk]['id'],'stop']]
            hit_ref = np.sort(hit_ref[hit_ref[:,0] == tracks[itrk]['id'], 1])
            hits_trk = self.hits[hit_ref]
            hits_drift_trk = self.hits_drift[hit_ref]
            self.ax_zyx.scatter(hits_trk['px'], hits_drift_trk[self.z_vals]*self.convert_to_mm, hits_trk['py'], lw=0.2, ec='C{}'.format(
                itrk+1), c=cmap(norm(hits_trk[self.charge])), s=5, alpha=0.75)
            self.ax_xy.scatter(hits_drift_trk[self.z_vals]*self.convert_to_mm, hits_trk['py'], lw=0.2, ec='C{}'.format(
                itrk+1), c=cmap(norm(hits_trk[self.charge])), s=5, alpha=0.75)
            self.ax_zy.scatter(hits_trk['px'], hits_trk['py'], lw=0.2, ec='C{}'.format(
                itrk+1), c=cmap(norm(hits_trk[self.charge])), s=5, alpha=0.75)
            self.ax_zy.plot((ts[0], te[0]),
                            (ts[1], te[1]),
                            c='C{}'.format(itrk+1), alpha=0.75, lw=1)
            self.ax_xy.plot((ts[2], te[2]),
                            (ts[1], te[1]),
                            c='C{}'.format(itrk+1), alpha=0.75, lw=1)
            self.ax_zyx.plot((ts[0], te[0]),
                             (ts[2], te[2]),
                             (ts[1], te[1]),
                             c='C{}'.format(itrk+1), alpha=0.5, lw=4)
            unassoc_hit_mask[np.in1d(hits['id'], hits_trk['id'])] = 0
        if np.any(unassoc_hit_mask):
        '''

        unassoc_hits = hits#[unassoc_hit_mask]
        BG = np.asarray([1., 1., 1., ])
        my_cmap = cmap(np.arange(cmap.N))
        alphas = np.linspace(0, 1, cmap.N)
        # Mix the colors with the background
        for i in range(cmap.N):
            my_cmap[i, :-1] = my_cmap[i, :-1] * \
                alphas[i] + BG * (1.-alphas[i])
        my_cmap = ListedColormap(my_cmap)
        
        if self.hits_dset == 'raw_hits':
            packets_hits_ref_mask = np.isin(self.packets_hits_ref[:,0], hit_ref)
            packets_hits = self.packets[:][self.packets_hits_ref[packets_hits_ref_mask][:,1]]
            #packets_hits = self.packets[:][self.packets_hits_ref[:,1]][hit_ref] # also works, but is slower
            hit_xvals = [self._get_z_coordinate(io_group, io_channel, time) for io_group, io_channel, time in zip(
                packets_hits['io_group'], packets_hits['io_channel'], unassoc_hits['ts_pps']-event_start_time)]
            self.ax_zyx.scatter(unassoc_hits[self.z_vals]*self.convert_to_mm, hit_xvals, unassoc_hits[self.y_vals]*self.convert_to_mm+self.y_offset, lw=0, ec='C0', c=cmap(
                norm(self.charge_from_ADC(unassoc_hits[self.charge], self.vref_mv, self.vcm_mv, self.ped_mv))), s=5, alpha=1)
            self.ax_zy.scatter(unassoc_hits[self.z_vals]*self.convert_to_mm, unassoc_hits[self.y_vals]*self.convert_to_mm+self.y_offset, lw=0, ec='C0', c=cmap(
                norm(self.charge_from_ADC(unassoc_hits[self.charge], self.vref_mv, self.vcm_mv, self.ped_mv))), s=5, alpha=1)
            self.ax_xy.scatter(hit_xvals, unassoc_hits[self.y_vals]*self.convert_to_mm+self.y_offset, lw=0, ec='C0', c=cmap(
                norm(self.charge_from_ADC(unassoc_hits[self.charge], self.vref_mv, self.vcm_mv, self.ped_mv))), s=5, alpha=1)
        else:
            hit_xvals = unassoc_hits[self.x_vals]*self.convert_to_mm
            self.ax_zyx.scatter(unassoc_hits[self.z_vals]*self.convert_to_mm, hit_xvals, unassoc_hits[self.y_vals]*self.convert_to_mm+self.y_offset, lw=0, ec='C0', c=cmap(
                norm(unassoc_hits[self.charge])), s=5, alpha=1)
            self.ax_zy.scatter(unassoc_hits[self.z_vals]*self.convert_to_mm, unassoc_hits[self.y_vals]*self.convert_to_mm+self.y_offset, lw=0, ec='C0', c=cmap(
                norm(unassoc_hits[self.charge])), s=5, alpha=1)
            self.ax_xy.scatter(hit_xvals, unassoc_hits[self.y_vals]*self.convert_to_mm+self.y_offset, lw=0, ec='C0', c=cmap(
                norm(unassoc_hits[self.charge])), s=5, alpha=1)
