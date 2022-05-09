import numpy as np
import numpy.ma as ma
import os

from h5flow import H5FLOW_MPI
from h5flow.core import H5FlowStage, resources

from module0_flow.util import units


class LightIntensityMapGenerator(H5FlowStage):
    '''
    Generates a 3D histogram of tracks not crossing the light detectors along with a 3+1D histogram of the light signal.
    '''
    x_bins = np.linspace(-310.38,310.38,15)
    y_bins = np.linspace(-620.76,620.76,27)
    z_bins = np.linspace(-304.31,304.31,17)
    d_bins = np.arange(33)

    edge_cut = 20

    sum_files = True

    q_file = 'light_intensity_q.sim.lut.v2.npz'
    s_file = 'light_intensity_s.sim.lut.v2.npz'

    def __init__(self, **params):
        super().__init__(**params)
        if self.rank == 0:
            print('x_bins', self.x_bins.shape)
            print('y_bins', self.y_bins.shape)
            print('z_bins', self.z_bins.shape)
            print('d_bins', self.d_bins.shape)        
        
        self.q_bins = self.x_bins, self.y_bins, self.z_bins
        self.q_hist = self.fill_hist(None, np.zeros(1),np.zeros(1),np.zeros(1), weights=np.zeros(1), bins=self.q_bins)
        
        self.s_bins = self.x_bins, self.y_bins, self.z_bins, self.d_bins
        self.s_hist = self.fill_hist(None, np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1), weights=np.zeros(1), bins=self.s_bins)

    @staticmethod
    def fill_hist(hist, x, y, z, d=None, weights=None, bins=None):
        '''
        Update the given histogram with new values

        :param hist: ND array, ``shape: (nbins_x, nbins_y, nbins_z[, nbins_d])``

        :param x: 1D array of x values

        :param y: 1D array of y values

        :param z: 1D array of z values

        :param d: 1D array of detector index (optional)

        :param weights: 1D array of entry weights

        :param bins: ``tuple`` of x,y,z[,d] bin edges
        '''
        if weights is None:
            weights = np.ones_like(x)

        mask = np.ones_like(weights, dtype=bool)
        for a in [x,y,z,d if d is not None else weights]:
            if hasattr(a, 'mask'):
                mask = mask & ~a.mask

        x,y,z,weights,d_broadcast = np.broadcast_arrays(
            x,y,z,weights,d if d is not None else weights)

        sample = np.column_stack([x[mask], y[mask], z[mask]] + ([d_broadcast[mask]] if d is not None else []))
        sample = sample.reshape(-1, 3 + int(d is not None))
        h,edges = np.histogramdd(sample, weights=weights[mask], bins=bins)

        return hist + h if hist is not None else h

    def finish(self, source_name):
        # fetch histograms from other threads
        if H5FLOW_MPI:
            q_hist = np.ascontiguousarray(np.zeros_like(self.q_hist))
            self.comm.Reduce(np.ascontiguousarray(self.q_hist), q_hist, root=0)
            s_hist = np.ascontiguousarray(np.zeros_like(self.s_hist))
            self.comm.Reduce(np.ascontiguousarray(self.s_hist), s_hist, root=0)
        else:
            q_hist = self.q_hist
            s_hist = self.s_hist
            
        # save to file
        if self.rank == 0:
            print(q_hist.sum())
            if os.path.exists(self.q_file) and self.sum_files == True:
                print('Loading existing data from file...')
                q_hist += np.load(self.q_file)['hist']
            if os.path.exists(self.s_file) and self.sum_files == True:
                s_hist += np.load(self.s_file)['hist']
            print(q_hist.sum())
            np.savez_compressed(self.q_file, hist=q_hist, bins=self.q_bins)
            np.savez_compressed(self.s_file, hist=s_hist, bins=self.s_bins)

    def run(self, source_name, source_slice, cache):
        super().run(source_name, source_slice, cache)
        
        n_ev = cache[source_name].shape[0]
        
        # get charge hits for event
        hits = cache['charge/hits']
        hit_drift = cache['combined/hit_drift'].reshape(hits.shape)

        x = hits['px'].compressed()
        y = hits['py'].compressed()
        z = hit_drift['z'].compressed()
        xyz = np.c_[x,y,z]
        q = hits['q'].compressed() * 221 # mV -> e

        # avoid activity near field cage walls (except top and bottom)
        if np.any((~resources['Geometry'].in_fid(xyz, field_cage_fid=self.edge_cut)) & ((y < 595) & (y > -595))):
            return
        if q.sum() < 1e6 or q.sum() > 2e7: # ~10cm -> 200cm MIP energy
            return
        
        # get light detector signal
        if np.all(cache['light/swvfm'].mask['samples']):
            return
        light_signal = cache['light/swvfm'].reshape((n_ev, -1,))['samples'].filled(0.).sum(axis=-1).sum(axis=1)
        tpc_id, det_id = np.indices(light_signal.shape)[-2:]
        
        # calculate solid angle weight
        omega = resources['Geometry'].solid_angle(
            np.c_[hits['px'].compressed(), hits['py'].compressed(), hit_drift['z'].compressed()],
            tpc_id, det_id)
        omega = omega.reshape(-1, np.prod(light_signal.shape[-2:]))
        light_signal = light_signal.reshape(-1, omega.shape[-1])
        det_id = det_id.reshape(light_signal.shape)
        tpc_id = tpc_id.reshape(light_signal.shape)        

        norm = (omega * q[:, np.newaxis]).sum(axis=0, keepdims=True)
        q_weight = ((q[:,np.newaxis] * omega) / np.clip(norm, 1e-15, None))
        q_weight[:,(norm < 1e-15).ravel()] = 0
        
        # update histograms
        self.q_hist = self.fill_hist(self.q_hist, x, y, z, weights=q, bins=self.q_bins)
        self.s_hist = self.fill_hist(self.s_hist, x[:,np.newaxis], y[:,np.newaxis], z[:,np.newaxis], det_id + tpc_id * 16, weights=light_signal * q_weight, bins=self.s_bins)
        
