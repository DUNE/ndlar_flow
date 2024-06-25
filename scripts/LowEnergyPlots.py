import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.colors as colors
from matplotlib.colors import LogNorm

class LowEnergyPlots:
    def __init__(self, filelist):
        with h5py.File(filelist[0], 'r') as f:
            self.drift = np.array(f['combined/hit_drift/data']['drift_coordinate'])
            self.y = np.array(f['charge/raw_hits/data']['y_pix'])
            self.z = np.array(f['charge/raw_hits/data']['z_pix'])
            self.io = np.array(f['charge/raw_hits/data']['iogroup'])
            self.x_anode_max = np.max(f['charge/raw_hits/data']['x_pix'])
            self.x_anode_min = np.min(f['charge/raw_hits/data']['x_pix'])
        
        for file in filelist[1:]:
            with h5py.File(file, 'r') as f:
                self.drift = np.concatenate((self.drift, f['combined/hit_drift/data']['drift_coordinate']))
                self.y = np.concatenate((self.y, f['charge/raw_hits/data']['y_pix']))
                self.z = np.concatenate((self.z, f['charge/raw_hits/data']['z_pix']))
                self.io = np.concatenate((self.io, f['charge/raw_hits/data']['iogroup']))
    
    def ZY_Hist2D(self, figTitle=None, vmin=1e0, vmax=1e3, imageName=None, bins=None, hist_range=None, isModule2=False):
        ### plot hits across pixel planes
        if bins is not None:
            z_bins = bins[0]
            y_bins = bins[1]
        elif bins is None and isModule2:
            z_bins=163
            y_bins=2*z_bins
        else:
            z_bins=140
            y_bins=2*z_bins
        if hist_range is not None:
            z_min_max = hist_range[0]
            y_min_max = hist_range[1]
        else:
            z_min_max = [-31,31]
            y_min_max = [-62,62]

        fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(8,6))
        cmap = plt.cm.jet

        TPC1_mask = (self.io == 1) & (self.drift > self.x_anode_min) & (self.drift < 0)
        TPC2_mask = (self.io == 2) & (self.drift < self.x_anode_max) & (self.drift > 0)

        H1 = axes[0].hist2d(self.z[TPC1_mask], self.y[TPC1_mask], range=[z_min_max, y_min_max],bins = [z_bins,y_bins], weights=np.ones_like(self.z[TPC1_mask]),norm = colors.LogNorm(vmin=vmin,vmax=vmax))
        fig.colorbar(H1[3], ax=axes[0])
        H2 = axes[1].hist2d(self.z[TPC2_mask], self.y[TPC2_mask], range=[z_min_max, y_min_max], bins = [z_bins,y_bins], weights=np.ones_like(self.z[TPC2_mask]),norm = colors.LogNorm(vmin=vmin,vmax=vmax))

        fig.colorbar(H2[3], ax=axes[1])
        axes[0].set_title(f'TPC 1')
        axes[1].set_title(f'TPC 2')
        fig.suptitle(figTitle, fontsize=10)
        axes[0].set_xlabel('Z [cm]')
        axes[1].set_xlabel('Z [cm]')
        axes[0].set_ylabel('Y [cm]')
        #axes[0].set_ylim(y_min_max[0], y_min_max[1])
        #axes[0].set_xlim(x_min_max[0], x_min_max[1])
        #axes[1].set_ylim(y_min_max[0], y_min_max[1])
        #axes[1].set_xlim(x_min_max[0], x_min_max[1])
        if imageName is not None:
            plt.savefig(imageName)
        plt.show()

    def XZ_Hist2D(self, figTitle=None, vmin=1e0, vmax=1e3, imageName=None, bins=None, hist_range=None, isModule2=False):
        ### plot hists across drift coordinate and pixel z coordinate
        if hist_range is None:
            x_min_max = [-31,31]
            z_min_max = [-31,31]
        else:
            x_min_max = hist_range[0]
            z_min_max = hist_range[1]
        if bins is not None:
            x_bins = bins[0]
            z_bins = bins[1]
        elif bins is None and isModule2:
            x_bins = 163
            z_bins = 163
        else:
            x_bins = 140
            z_bins = x_bins
            
        fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(9,6))
        cmap = plt.cm.jet

        TPC1_mask = (self.io == 1) & (self.drift > self.x_anode_min) & (self.drift < 0)
        TPC2_mask = (self.io == 2) & (self.drift < self.x_anode_max) & (self.drift > 0)
        mask = TPC1_mask | TPC2_mask
        H1 = axes.hist2d(self.z[mask], self.drift[mask], \
                         range=[z_min_max, x_min_max],bins = [z_bins,x_bins], \
                         norm = colors.LogNorm(vmin=vmin,vmax=vmax))
        fig.colorbar(H1[3], ax=axes)
        axes.set_xlabel('Z [cm]')
        axes.set_ylabel('Drift Coordinate X [cm]')
        #axes.set_xlim(z_min_max[0], z_min_max[1])
        #axes.set_ylim(x_min_max[0], x_min_max[1])
        fig.suptitle(figTitle)
        if imageName is not None:
            plt.savefig(imageFileName)
        plt.show()