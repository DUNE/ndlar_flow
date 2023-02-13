import numpy as np
import numpy.ma as ma
import os

from h5flow import H5FLOW_MPI
from h5flow.core import H5FlowStage, resources

from module0_flow.util import units


class LightIntensityMapGenerator(H5FlowStage):
    '''
    Generates a 3D histogram of tracks not crossing the light detectors along with a 3+1D histogram of the light signal.

    Uses the LArData resource
    '''
    x_bins = np.linspace(-310.38,310.38,15)
    y_bins = np.linspace(-620.76,620.76,27)
    z_bins = np.linspace(-304.31,304.31,17)
    d_bins = np.arange(33)

    edge_cut = 20 # avoid tracks that come within edge_cut mm of light detectors

    sum_files = True
    charge_weighting = False
    use_adc_channel_id = False

    wvfm_dset_name = 'light/swvfm' # must have a samples field
    tracks_dset_name = 'combined/tracks'

    q_file = 'light_intensity_q.sim.lut.v2.npz'
    s_file = 'light_intensity_s.sim.lut.v2.npz'

    def __init__(self, **params):
        super().__init__(**params)

        self.q_file = params.get('q_file', self.q_file)
        self.s_file = params.get('s_file', self.s_file)

        self.sum_files = params.get('sum_files', self.sum_files)
        self.charge_weighting = params.get('charge_weighting', self.charge_weighting)
        self.use_adc_channel_id = params.get('use_adc_channel_id', self.use_adc_channel_id)

        self.wvfm_dset_name = params.get('wvfm_dset_name', self.wvfm_dset_name)
        self.charge_dset_name = params.get('charge_dset_name', self.charge_dset_name)
        self.tracks_dset_name = params.get('tracks_dset_name', self.tracks_dset_name)
        
        self.q_bins = []
        self.q_hist = None
        
        self.s_bins = []
        self.s_hist = None

        self.nevents = 0


    def init(self, source_name):
        ntpc, ndet, _ = self.data_manager.get_dset(self.wvfm_dset_name).dtype['samples'].shape
        self.d_bins = np.arange(ntpc * ndet + 1)

        self.q_bins = self.x_bins, self.y_bins, self.z_bins
        self.s_bins = self.x_bins, self.y_bins, self.z_bins, self.d_bins

        if self.rank == 0:
            print('x_bins', self.x_bins.shape)
            print('y_bins', self.y_bins.shape)
            print('z_bins', self.z_bins.shape)
            print('d_bins', self.d_bins.shape)        

        self.q_hist = self.fill_hist(None, np.zeros(1),np.zeros(1),np.zeros(1), weights=np.zeros(1), bins=self.q_bins)
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
            self.nevents = self.comm.reduce(self.nevents, root=0)
        else:
            q_hist = self.q_hist
            s_hist = self.s_hist
            
        # save to file
        if self.rank == 0:
            print(f'current events: {self.nevents}')
            if os.path.exists(self.q_file) and self.sum_files == True:
                print('Loading existing data from file...')
                q_hist += np.load(self.q_file)['hist']
            if os.path.exists(self.s_file) and self.sum_files == True:
                s_hist += np.load(self.s_file)['hist']
                self.nevents += np.load(self.s_file)['nevents']

            print(f'total events: {self.nevents}')
            np.savez_compressed(self.q_file, hist=q_hist, bins=self.q_bins, nevents=self.nevents)
            np.savez_compressed(self.s_file, hist=s_hist, bins=self.s_bins, nevents=self.nevents)

    def run(self, source_name, source_slice, cache):
        super().run(source_name, source_slice, cache)
        
        n_ev = cache[source_name].shape[0]
        
        # get charge hits for event
        events = cache['charge/events']
        hits = cache['charge/hits']
        hit_charge = cache[self.charge_dset_name].reshape(hits.shape)
        hit_drift = cache['combined/hit_drift'].reshape(hits.shape)
        t0 = cache['combined/t0']

        # ensure good T0
        if np.any(t0['type'] != 1):
            return
        if np.any(events['n_ext_trigs'] != 2):
            return

        x = hits['px'].compressed()
        y = hits['py'].compressed()
        z = hit_drift['z'].compressed()
        xyz = np.c_[x,y,z]

        # apply gain calibration
        q = hit_charge['q'].compressed() * gain # mV -> e
        q = np.where(resources['Geometry'].in_fid(xyz) & (hit_drift['t_drift'].compressed() < 1900) & (hit_drift['t_drift'].compressed() > 0), q, 0)

        # get tracks for events
        tracks = cache[self.tracks_dset_name]

        # require at least 1 decent track length in the event
        if not np.any(tracks['length'] > 100):
            return
        # check the dQ/dx
        avg_dqdx = np.mean(gain * tracks['dq'] / np.sqrt(np.sum(tracks['dx']**2, axis=-1)).clip(4, None), axis=-1)
        if np.any((avg_dqdx > 8e3) | (avg_dqdx < 4e3)):
            return
        # check for bad track angle
        if np.all(np.abs(np.diff(tracks['trajectory'], axis=-1)) < 20):
            return
        # avoid activity near field cage walls (except top and bottom)
        if np.any((~resources['Geometry'].in_fid(xyz, field_cage_fid=self.edge_cut)) & ((y < 595) & (y > -595)) & resources['Geometry'].in_fid(xyz)):
            return
        # ~10cm -> 200cm MIP energy
        if q.sum() < 1e6 or q.sum() > 2e7:
            return
        
        # get light detector signal
        if cache[self.wvfm_dset_name] is None or np.all(cache[self.wvfm_dset_name].mask['samples']):
            return

        light_signal = cache[self.wvfm_dset_name].reshape((n_ev, -1,))['samples'].filled(0.).astype(float).sum(axis=-1).sum(axis=1)
        ntpc = light_signal.shape[-2]
        ndet = light_signal.shape[-1]
        tpc_id, det_id = np.indices(light_signal.shape)[-2:]

        light_signal = light_signal.reshape(-1, ntpc*ndet)
        tpc_id = tpc_id.reshape(-1, ntpc*ndet)
        det_id = det_id.reshape(-1, ntpc*ndet)

        # set output index in resulting histogram
        s_hist_index = det_id + tpc_id * ndet

        # calculate solid angle weight
        if self.use_adc_channel_id == True:
            # reinterprets waveform shape as (adc index, detector index)
            adc_id = tpc_id.copy()
            channel_id = det_id.copy()

            tpc_id = resources['Geometry'].tpc_id[(adc_id.ravel(), channel_id.ravel())]
            det_id = resources['Geometry'].det_id[(adc_id.ravel(), channel_id.ravel())]

            tpc_id = tpc_id.reshape(adc_id.shape)
            det_id = det_id.reshape(channel_id.shape)

        omega = resources['Geometry'].solid_angle(xyz, tpc_id, det_id)
        omega = omega.reshape(xyz.shape[0:1] + light_signal.shape[1:])
        # enforce cathode non-visibility
        omega[(xyz[:,2][:,np.newaxis] < 0) & (tpc_id != 0)] = 0
        omega[(xyz[:,2][:,np.newaxis] > 0) & (tpc_id != 1)] = 0
        # insure that any invalid hits are not counted in sum
        omega[~resources['Geometry'].in_fid(xyz)] = 0
        omega[((xyz[:,0] == xyz[:,1]) & (xyz[:,1] == 0)).ravel()] = 0
        q[((xyz[:,0] == xyz[:,1]) & (xyz[:,1] == 0)).ravel()] = 0

        if self.charge_weighting == False:
            norm = omega.sum(axis=0, keepdims=True)
            q_weight = omega / np.clip(norm, 1e-15, None)
        else:
            norm = (omega * q[:, np.newaxis]).sum(axis=0, keepdims=True)
            q_weight = ((q[:,np.newaxis] * omega) / np.clip(norm, 1e-15, None))
        q_weight[:,(norm < 1e-15).ravel()] = 0
        
        # update histograms
        self.q_hist = self.fill_hist(self.q_hist, x, y, z, weights=q, bins=self.q_bins)
        self.s_hist = self.fill_hist(self.s_hist, x[:,np.newaxis], y[:,np.newaxis], z[:,np.newaxis], s_hist_index, weights=light_signal * q_weight, bins=self.s_bins)

        self.nevents += 1

