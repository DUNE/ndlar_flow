import numpy as np
import numpy.ma as ma
import logging
from scipy.interpolate import interp1d
import scipy.stats as stats
import scipy.ndimage as ndimage
import scipy.optimize as optimize
from copy import deepcopy

from h5flow.core import H5FlowStage, resources
from h5flow.data import dereference_chain

from module0_flow.util.func import mode, condense_array


class StoppingMuonSelection(H5FlowStage):
    '''
        Perform a selection for stopping muons. A stopping event
        is defined by no more than one merged track segment that enters the
        detector fiducial volume and does not leave it. Creates a boolean array
        of 1:1 with events indicating stopping events, and creates a boolean
        array 1:1 with merged track segments if they individually meet the
        stopping criteria.

        A dQ/dx profile is generated per event and used to discriminate
        stopping protons and muons, as well as through-going muons.

        If the file is a MC file, also generates boolean arrays with the true
        value.


    '''
    class_version = '0.0.0'

    default_fid_cut = 20  # mm
    default_cathode_fid_cut = 20  # mm
    default_length_cut = 100  # mm
    default_profile_dx = 20  # mm
    default_profile_max_range = 1600  # mm
    default_larpix_gain = 250  # e/mV
    default_larpix_noise = 0  # e
    default_lifetime = 2.2e3 # us
    default_profile_search_dx = 50 # mm
    default_curvature_rr_correction = 22.6647 / 22
    default_density_dx_correction_params = [0.78497819,  -3.41826874, 198.93022888]
    default_hits_dset_name = 'charge/hits'
    default_merged_dset_name = 'combined/tracklets/merged'
    default_t0_dset_name = 'combined/t0'
    default_mc_trajectory_path = ['combined/tracklets/merged',
                                  'charge/hits', 'charge/packets',
                                  'mc_truth/tracks',
                                  'mc_truth/trajectories']
    default_path = 'analysis/stopping_muons'

    event_sel_dset_name = 'event_sel_reco'
    event_profile_dset_name = 'event_profile'
    event_sel_truth_dset_name = 'event_sel_truth'

    event_sel_dtype = np.dtype([('sel', 'u1'),
                                ('stop', 'u1'),
                                ('muon_loglikelihood_mean', 'f8'),
                                ('proton_loglikelihood_mean', 'f8'),
                                ('mip_loglikelihood_mean', 'f8'),
                                ('stop_pt', 'f8', (3,))])

    @staticmethod
    def event_profile_dtype(dx, max_range):
        profile_bins = int(max_range / dx)
        return np.dtype([
            ('seed_pt', 'f8', (3,)),
            ('profile_rr', 'f8', (profile_bins,)),
            ('profile_dqdx', 'f8', (profile_bins,)),
            ('profile_n', 'i8', (profile_bins,)),
            ('profile_dx', 'f8', (profile_bins,)),
            ('profile_pos', 'f8', (profile_bins, 3)),
            ('muon_likelihood', 'f8', (profile_bins,2)),
            ('proton_likelihood', 'f8', (profile_bins,2)),
            ('mip_likelihood', 'f8', (profile_bins,2)),
        ])

    def __init__(self, **params):
        super(StoppingMuonSelection, self).__init__(**params)

        self.path = params.get('path', self.default_path)
        self.fid_cut = params.get('fid_cut', self.default_fid_cut)
        self.cathode_fid_cut = params.get('cathode_fid_cut', self.default_cathode_fid_cut)
        self.length_cut = params.get('length_cut', self.default_length_cut)
        self.larpix_gain = params.get('larpix_gain', self.default_larpix_gain)
        self.larpix_noise = params.get('larpix_noise', self.default_larpix_noise)
        self.lifetime = params.get('lifetime', self.default_lifetime)
        self.profile_dx = params.get('profile_dx', self.default_profile_dx)
        self.profile_search_dx = params.get('profile_search_dx', self.default_profile_search_dx)
        self.profile_max_range = params.get('profile_max_range',
                                            self.default_profile_max_range)
        self.curvature_rr_correction = params.get('curvature_rr_correction',
                                            self.default_curvature_rr_correction)
        self.density_dx_correction_params = params.get('density_dx_correction_params',
                                            self.default_density_dx_correction_params)
        self.hits_dset_name = params.get('hits_dset_name',
                                         self.default_hits_dset_name)
        self.t0_dset_name = params.get('t0_dset_name',
                                       self.default_t0_dset_name)
        self.hits_dset_name = params.get('hits_dset_name',
                                         self.default_hits_dset_name)
        self.merged_dset_name = params.get('merged_dset_name',
                                           self.default_merged_dset_name)
        self.mc_trajectory_path = params.get('mc_trajectory_path',
                                             self.default_mc_trajectory_path)

        self.event_profile_dtype = self.event_profile_dtype(self.profile_dx,
                                                            self.profile_max_range)

        self.regions = []

    def init(self, source_name):
        self.is_mc = resources['RunData'].is_mc

        self.data_manager.set_attrs(self.path,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    fid_cut=self.fid_cut,
                                    length_cut=self.length_cut,
                                    profile_dx=self.profile_dx,
                                    profile_search_dx=self.profile_search_dx,
                                    profile_max_range=self.profile_max_range,
                                    curvature_rr_correction=self.curvature_rr_correction,
                                    density_dx_correction_params=self.density_dx_correction_params,
                                    hits_dset_name=self.hits_dset_name,
                                    t0_dset_name=self.t0_dset_name,
                                    merged_dset_name=self.merged_dset_name
                                    )
        self.data_manager.create_dset(f'{self.path}/{self.event_sel_dset_name}',
                                      self.event_sel_dtype)
        self.data_manager.create_dset(f'{self.path}/{self.event_profile_dset_name}',
                                      self.event_profile_dtype)
        if self.is_mc:
            self.data_manager.create_dset(f'{self.path}/{self.event_sel_truth_dset_name}',
                                          self.event_sel_dtype)

        self.create_regions()
        self.data_manager.set_attrs(self.path, regions=np.array(self.regions))

        self.create_dqdx_profile_templates()
        self.data_manager.set_attrs(self.path,
                                    proton_dqdx=self.proton_range_table['dqdx'],
                                    muon_dqdx=self.muon_range_table['dqdx'],
                                    proton_dqdx_width=self.proton_range_table['dqdx_width'],
                                    muon_dqdx_width=self.muon_range_table['dqdx_width'],
                                    proton_dedx=self.proton_range_table['dedx_mpv'],
                                    muon_dedx=self.muon_range_table['dedx_mpv'],
                                    proton_range=self.proton_range_table['range'],
                                    muon_range=self.muon_range_table['range'],
                                    proton_recom=self.proton_range_table['recomb'],
                                    muon_recom=self.muon_range_table['recomb'])
        
    def finish(self, source_name):
        sel_dset_name = f'{self.path}/{self.event_sel_dset_name}'
        
        total = len(self.data_manager.get_dset(sel_dset_name))
        nstopping = np.sum(self.data_manager.get_dset(sel_dset_name)['stop'])
        nselected = np.sum(self.data_manager.get_dset(sel_dset_name)['sel'])
        print(f'Stopping: {nstopping} / {total} ({nstopping/total:0.03f})')
        print(f'Selected: {nselected} / {total} ({nselected/total:0.03f})')
        
        if self.is_mc:
            sel_truth_dset_name = f'{self.path}/{self.event_sel_truth_dset_name}'
            true_stopping = np.sum(self.data_manager.get_dset(sel_truth_dset_name)['stop'])
            true_stopping_muon = np.sum(self.data_manager.get_dset(sel_truth_dset_name)['sel'])
            print(f'True stopping: {true_stopping} / {total} ({true_stopping/total:0.03f})')
            print(f'True stopping muons: {true_stopping_muon} / {total} ({true_stopping_muon/total:0.03f})')
            
            correct = np.sum(self.data_manager.get_dset(sel_truth_dset_name)['sel'] &
                             self.data_manager.get_dset(sel_dset_name)['sel'])
            
            print(f'Purity: {correct} / {nselected} ({correct/nselected:0.03f})')
            print(f'Efficiency: {correct} / {true_stopping_muon} ({correct/true_stopping_muon:0.03f})')

    def create_dqdx_profile_templates(self):
        # create range tables used for dQ/dx profile discrimination
        self.muon_range_table = dict()
        self.proton_range_table = dict()
        
        # only consider reasonable range values
        muon_mask = resources['ParticleData'].muon_range_table['range'] > 0.1
        for key,val in deepcopy(resources['ParticleData'].muon_range_table).items():
            self.muon_range_table[key] = val[muon_mask]
        proton_mask = resources['ParticleData'].proton_range_table['range'] > 0.1
        for key,val in deepcopy(resources['ParticleData'].proton_range_table).items():
            self.proton_range_table[key] = val[proton_mask]

        # convert mean dE/dx entries to MPV dE/dx
        self.muon_range_table['dedx_mpv'] = resources['ParticleData'].landau_peak(
            self.muon_range_table['t'], resources['ParticleData'].mu_mass,
            self.profile_dx) / self.profile_dx
        self.proton_range_table['dedx_mpv'] = resources['ParticleData'].landau_peak(
            self.proton_range_table['t'], resources['ParticleData'].p_mass,
            self.profile_dx) / self.profile_dx

        # calculate recombination correction
        muon_r = resources['LArData'].ionization_recombination(
            self.muon_range_table['dedx_mpv'])
        proton_r = resources['LArData'].ionization_recombination(
            self.proton_range_table['dedx_mpv'])
        w = resources['LArData'].ionization_w
        self.muon_range_table['recomb'] = muon_r
        self.proton_range_table['recomb'] = proton_r

        self.muon_range_table['dqdx'] = (muon_r * self.muon_range_table['dedx_mpv'] / w)
        self.proton_range_table['dqdx'] = (proton_r * self.proton_range_table['dedx_mpv'] / w)
        self.muon_range_table['dqdx_width'] = (
            muon_r / w * resources['ParticleData'].landau_width(self.muon_range_table['t'],
                                                                resources['ParticleData'].mu_mass,
                                                                self.profile_dx) / self.profile_dx)
        self.proton_range_table['dqdx_width'] = (
            proton_r / w * resources['ParticleData'].landau_width(self.proton_range_table['t'],
                                                                  resources['ParticleData'].p_mass,
                                                                  self.profile_dx) / self.profile_dx)
        noise = (self.larpix_noise * np.sqrt(self.profile_dx / resources['Geometry'].pixel_pitch)
                 / resources['Geometry'].pixel_pitch)
        
        self.muon_range_table['mcs_angle'] = resources['ParticleData'].mcs_angle(self.muon_range_table['t'],
                                                                  resources['ParticleData'].mu_mass,
                                                                  self.profile_dx)
        self.proton_range_table['mcs_angle'] = resources['ParticleData'].mcs_angle(self.proton_range_table['t'],
                                                                  resources['ParticleData'].p_mass,
                                                                  self.profile_dx)

        # self.apply_position_resolution(self.muon_range_table, noise=noise)
        # self.apply_position_resolution(self.proton_range_table, noise=noise)

    def apply_position_resolution(self, range_table, noise=0):
        ''' Update the range table ``dqdx`` and ``dqdx_width`` by smearing the range values by a gaussian ``profile_dx`` '''
        # interpolate dQ/dx MPV and width to apply a gaussian smear
        interpolation_pts, dx = np.linspace(-500, 2000, 10 * int(2500 / self.profile_dx),
                                            retstep=True)

        # interpolate central value
        rr = np.r_[-5000, 0, range_table['range']]
        dqdx = np.r_[0, 0, range_table['dqdx']]
        dqdx_width = np.r_[0, 0, range_table['dqdx_width']]
        interp_rr = interp1d(rr, dqdx)
        dqdx = interp_rr(interpolation_pts)
        # apply a position resolution smearing
        dqdx_smear = ndimage.uniform_filter(dqdx, int(self.profile_dx / dx), mode='nearest')
#         dqdx_smear = ndimage.uniform_filter(dqdx, 1, mode='nearest')

        # interpolate width
        interp_rr_width = interp1d(rr, dqdx_width)
        dqdx_width = interp_rr_width(interpolation_pts)
        # combine position resolution, intrinsic width, and noise contributions
        dqdx_width = np.sqrt(
        #     ndimage.uniform_filter(np.abs(ndimage.convolve(dqdx * dx, [-1, 1], mode='nearest')), int(self.profile_dx / dx), mode='nearest')**2
            ndimage.uniform_filter(np.abs(ndimage.convolve(dqdx * dx, [0], mode='nearest')), 1, mode='nearest')**2
            + dqdx_width**2
            + noise**2)

        # re-align to max
        high_val_align = interpolation_pts[np.argmax(dqdx_smear + dqdx_width)]
        high_val_interp = interp1d(interpolation_pts - high_val_align,
                                   dqdx_smear + dqdx_width)
        low_val_align = interpolation_pts[np.argmax(dqdx_smear - dqdx_width)]
        low_val_interp = interp1d(interpolation_pts - low_val_align,
                                  dqdx_smear - dqdx_width)
        high_val_interp, low_val_interp = (np.maximum(high_val_interp, low_val_interp), np.minimum(high_val_interp, low_val_interp))

        # set values
        _min, _max = (max(np.min(interpolation_pts - dx * low_val_align), np.min(interpolation_pts - dx * high_val_align)),
                      min(np.max(interpolation_pts - dx * low_val_align), np.max(interpolation_pts - dx * high_val_align)))
        range_table['dqdx'] = 0.5 * (high_val_interp(np.clip(rr[2:], _min, _max)) + low_val_interp(np.clip(rr[2:], _min, _max)))
        range_table['dqdx_width'] = 0.5 * (high_val_interp(np.clip(rr[2:], _min, _max)) - low_val_interp(np.clip(rr[2:], _min, _max)))

    def create_regions(self):
        x = resources['Geometry'].pixel_xy.compress((0,))
        y = resources['Geometry'].pixel_xy.compress((1,))
        z = resources['Geometry'].anode_z.compress()

        min_x, max_x = np.min(x) + self.fid_cut, np.max(x) - self.fid_cut
        min_y, max_y = np.min(y) + self.fid_cut, np.max(y) - self.fid_cut
        min_z, max_z = np.min(z) + self.fid_cut, np.max(z) - self.fid_cut

        self.regions.append((np.array([min_x, min_y, min_z]),
                             np.array([max_x, max_y, -self.cathode_fid_cut])))
        self.regions.append((np.array([min_x, min_y, self.cathode_fid_cut]),
                             np.array([max_x, max_y, max_z])))

    def in_fid(self, xyz):
        '''
            :param xyz: array ``shape: (N,3)``

            :returns: array ``shape: (N,)``

        '''
        coord_in_fid = ma.concatenate([np.expand_dims((xyz < np.expand_dims(boundary[1], 0))
                                                      & (xyz > np.expand_dims(boundary[0], 0)), axis=-1)
                                       for boundary in self.regions], axis=-1)
        in_fid = ma.all(coord_in_fid, axis=1)
        in_any_fid = ma.any(in_fid, axis=-1)
        return in_any_fid

    def contained(self, start_xyz, end_xyz):
        '''
            :param start_xyz: array ``shape: (N,3)``

            :param end_xyz: array ``shape: (N,3)``

            :returns: array ``shape: (N,)``

        '''
        start_in_fid = self.in_fid(start_xyz)
        end_in_fid = self.in_fid(end_xyz)
        track_contained = (~start_in_fid & end_in_fid) | (start_in_fid & ~end_in_fid)
        return track_contained

    def through_going(self, start_xyz, end_xyz):
        start_in_fid = self.in_fid(start_xyz)
        end_in_fid = self.in_fid(end_xyz)
        track_through_going = ~start_in_fid & ~end_in_fid
        return track_through_going

    def downward(self, start_xyz, end_xyz):
        end_in_fid = self.in_fid(end_xyz)
        dy = end_xyz[:, 1] - start_xyz[:, 1]
        return ((end_in_fid & (dy < 0)) | (~end_in_fid & (dy > 0)))
    
    @staticmethod
    def density_dx_correction(rr, *params):
        rr = np.clip(rr, 0, None)
        rv = params[0] * np.exp(-rr / params[2]) + params[1]
        return rv

    @staticmethod
    def profile_likelihood(profile_rr, profile_dqdx, profile_pos, range_table, type=''):
        '''
            Calculates the likelihood of a given dqdx v. residual range profile
            using a Moyal-distribution approximation.

            Likelihood data is passed via the ``range_table`` parameter which is
            a ``dict`` with the following arrays:

                - ``range``: residual range values used in interpolation ``shape: (n_interp_pts,)``
                - ``dqdx``: dQ/dx values used in interpolation ``shape: (n_interp_pts,)``
                - ``dqdx_width``: dQ/dx sigma values ``shape: (n_interp_pts,)``

            :param profile_rr: residual range ``shape: (..., n)``

            :param profile_dqdx: dqdx ``shape: (..., n)``
            
            :param profile_pos: bin position ``shape: (..., n, 3)``

            :param range_table: ``dict``, see above.
            
            :param type: likelihood pdf name, one of ``'abs_exp'``, ``'moyal'``, ``'gaus'``

            :returns: likelihood ``shape: (..., n)``

        '''
        interp = interp1d(range_table['range'], range_table['dqdx'])
        interp_width = interp1d(range_table['range'], range_table['dqdx_width'])
        interp_angle_width = interp1d(range_table['range'], range_table['mcs_angle'])
        min_range = np.min(range_table['range'])
        max_range = np.max(range_table['range'])

        interp_dqdx = interp(np.clip(profile_rr, min_range, max_range))
        interp_dqdx[profile_rr < min_range] = 0
        interp_dqdx_width = interp_width(np.clip(profile_rr, min_range, max_range))
        interp_angle_width = interp_angle_width(np.clip(profile_rr, min_range, max_range))
        
        # calculate dQ/dx log-likelihood
        if type == 'abs_exp':
            dqdx_term = -np.abs(profile_dqdx - interp_dqdx) / interp_dqdx_width * np.log(interp_dqdx_width / 2)
        elif type == 'moyal':
            dqdx_term = stats.moyal.logpdf(profile_dqdx, loc=interp_dqdx, scale=interp_dqdx_width)
        elif type == 'gaus':
            dqdx_term = stats.norm.logpdf(profile_dqdx, loc=interp_dqdx, scale=interp_dqdx_width)
        else:
            dqdx_term = np.zeros_like(profile_dqdx)
            
        # calculate MCS log-likelihood
        d = profile_pos[...,1:,:] - profile_pos[...,:-1,:]
        d = d * ((np.linalg.norm(profile_pos[...,1:,:], axis=-1, keepdims=True) > 0) *
                 (np.linalg.norm(profile_pos[...,:-1,:], axis=-1, keepdims=True) > 0))
        angle = np.zeros_like(profile_dqdx)
        norm = np.linalg.norm(d[...,1:,:], axis=-1) * np.linalg.norm(d[...,:-1,:], axis=-1)
        angle[...,1:-1] = np.maximum(np.sum(d[...,1:,:] * d[...,:-1,:], axis=-1), 1e-15) / np.maximum(norm, 1e-15)
        angle[...,1:-1] = np.arccos(angle[...,1:-1])
        
        angle_term = stats.norm.logpdf(angle, loc=0, scale=interp_angle_width) + np.log(2)
        angle_term[...,0] = 0 # skip first and last bins
        angle_term[...,-1] = 0
        angle_term[(np.array(profile_dqdx) <= 0)] = 0 # remove empty bins
        np.put_along_axis(angle_term, np.argmin(np.abs(profile_rr), axis=-1)[...,np.newaxis], -np.log(2*np.pi), axis=-1)
        
        return dqdx_term, angle_term
    
    @staticmethod
    def pixel_intersection(pt, n, pixel, pixel_pitch, mask=None):
        '''
            Calculate the line intersection defined by 3D position ``pt`` and
            direction unit vector ``n`` with a pixel at 2D position ``pixel``
            and a pixel width of ``pixel_pitch``
            
            :param pt: 3D position ``shape: (..., 3)``
            
            :param n: 3D vector ``shape: (..., 3)``
            
            :param pixel: 2D pixel location ``shape: (..., 2)``
            
            :param pixel_pitch: pixel width ``float``
        '''
        n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-15)
        
        # calculate pixel edges
        pixel_offset = [np.empty(pixel.shape) for _ in range(4)]
        pixel_n = [np.empty(pixel.shape) for _ in range(4)]
        for i,d in enumerate([(0,1),(1,0),(0,-1),(-1,0)]):
            pixel_offset[i][...,:] = np.array(d) * pixel_pitch/2
        for i,d in enumerate([(1,0),(0,1),(1,0),(0,1)]):
            pixel_n[i][...,:] = d
            
        # calculate intersections
        sol = [np.empty(np.maximum(n.shape[:-1], pixel.shape[:-1]).tolist() + [2,1]) for _ in range(4)]
        for i in range(4):
            mat = np.empty(sol[i].shape[:-1] + (2,))
            mat[...,0,0] = -n[...,0]
            mat[...,0,1] = pixel_n[i][...,0]
            mat[...,1,0] = -n[...,1]
            mat[...,1,1] = pixel_n[i][...,1]
            inv = np.linalg.pinv(mat)
            d = np.expand_dims((pt[...,0:2] - (pixel + pixel_offset[i])), axis=-1)
            sol[i][...,:] = np.sum(inv * d, axis=-1, keepdims=True)
        
        s_mask = [(np.abs(sol[i][...,1,:]) <= pixel_pitch/2) 
                  & np.all(n != 0, axis=-1, keepdims=True) 
                  for i in range(4)]
        if mask is not None:
            s_mask = [s_mask[i] & mask for i in range(4)]
        s_mask = np.concatenate(s_mask, axis=-1)
        s = np.concatenate(sol, axis=-1)[...,0,:]
        masked_s = ma.array(s, mask=~s_mask)
        s_min = ma.min(masked_s, axis=-1)
        s_max = ma.max(masked_s, axis=-1)
        # handle case for segment parallel to pixel (but no intersect)
        new_mask = (s_min == s_max) | s_min.mask | s_max.mask
        return ma.array(s_min, mask=new_mask), ma.array(s_max, mask=new_mask)

    @staticmethod
    def profiled_dqdx(tracks, seed_pt, hit_xyz, hit_q, dx, max_range, search_dx, pixel_pitch, mask=None):
        '''
            Generate dQ/dx profile. Algorithm is the following:

             1. Find track nearest to seed point

             2. Orient track in direction away from seed point

             3. Collect hits that lie within ``dx`` mm along the track trajectory

             4. Assign hits an overall position along event according to the projection onto the track trajectory

             5. Fill dQ/dx bins with hit charge according to hit position

             6. Set next iteration seed point as current track end point

             7. Repeat, removing previously used tracks from each iteration

            The output array contains ``int(max_range / dx)`` bins with the
            first and last bins being overflow bins.

            This is an expensive operation so a ``mask`` can be provided to skip
            the analysis of certain events. The returned array will not contain
            meaningful values for these entries.

            :param tracks: masked array, shape: (..., M)

            :param seed_pt: masked array, shape: (..., 3)

            :param hit_xyz: masked array, shape: (..., n, 3)

            :param hit_q: masked array, shape: (..., n)
            
            :param dx: float, bin size
            
            :param search_dx: float, distance to look for next track
            
            :param max_range: float, maximum bin
            
            :param pixel_pitch: float, pixel pitch for dx correction

            :returns: ``tuple`` of masked arrays, shape: (..., m). ``dq``, ``dn``, ``start_pt``, ``end_pt``, ``pos``
        '''
        orig_len = len(tracks)
        if mask is not None:
            tracks = tracks[mask]
            seed_pt = seed_pt[mask]
            hit_xyz = hit_xyz[mask]
            hit_q = hit_q[mask]

        seed_d = np.minimum(np.linalg.norm(tracks['trajectory'][..., 0, :] - seed_pt, axis=-1),
                            np.linalg.norm(tracks['trajectory'][..., -1, :] - seed_pt, axis=-1))
        seed_track = np.expand_dims(np.argmin(seed_d, axis=-1), axis=-1)

        start_pt = seed_pt.copy()

        dq = np.zeros((len(tracks), int(max_range / dx)))
        dn = np.zeros((len(tracks), int(max_range / dx)), dtype=int)
        pos = np.zeros((len(tracks), int(max_range / dx), 3))
        s_min = np.full((len(tracks), int(max_range / dx)), max_range+1.)
        s_max = np.full((len(tracks), int(max_range / dx)), -max_range-1.)

        track_mask = ~tracks['id'].mask.copy()  # True == include
        iter_mask = np.zeros_like(~tracks['id'].mask) # True == include
        hit_mask = ~(hit_q.mask | np.any(hit_xyz.mask, axis=-1))  # True == include
        
        np.put_along_axis(iter_mask, seed_track, True, axis=-1)

        s = 0
        bins = np.linspace(0, max_range, dq.shape[-1])
        for _ in range(tracks.shape[-1]):

            # find seed trajectory
            traj = np.take_along_axis(tracks, seed_track, axis=-1)['trajectory'].copy()

            # orient trajectory according to previous seed point
            traj_start_d = np.linalg.norm(traj[..., 0, :] - seed_pt, axis=-1, keepdims=True)
            traj_end_d = np.linalg.norm(traj[..., -1, :] - seed_pt, axis=-1, keepdims=True)
            is_reversed_traj = (traj_end_d < traj_start_d)
            is_reversed_traj = np.expand_dims(is_reversed_traj, axis=-1)
            traj = np.where(is_reversed_traj, traj[..., ::-1, :], traj)
            # (N, 1, npt, 3)

            # get trajectory displacement vectors
            traj_start = traj[..., :-1, :]
            traj_end = traj[..., 1:, :]
            traj_dx = traj_end - traj_start
            traj_length = np.linalg.norm(traj_dx, axis=-1, keepdims=True)
            traj_length = np.clip(traj_length, 1e-15, None)
            traj_n = traj_dx / traj_length
            # (N, 1, npt-1, 3)

            # collect hits within dx of trajectory
            hit_dx = np.expand_dims(hit_xyz, axis=-2) - traj_start
            hit_td = np.linalg.norm(hit_dx - np.sum(hit_dx * traj_n, axis=-1, keepdims=True) * traj_n, axis=-1)
            hit_alpha = np.sum(hit_dx * traj_n, axis=-1) / traj_length[..., 0]
            hit_on_traj = (hit_alpha < 1) & (hit_alpha > 0)
            hit_near_traj_pt = ((np.linalg.norm(np.expand_dims(hit_xyz, axis=-2) - traj_start, axis=-1) < dx)
                                | (np.linalg.norm(np.expand_dims(hit_xyz, axis=-2) - traj_end, axis=-1) < dx))
            itraj_min_td = np.expand_dims(np.argmin(
                ma.array(hit_td, mask=~(hit_on_traj | hit_near_traj_pt)),
                axis=-1), axis=-1)
            hit_min_td = np.take_along_axis(hit_td, itraj_min_td, axis=-1)
            traj_hit_mask = ((hit_min_td < dx)[..., 0] & hit_mask
                             & (np.any(hit_on_traj, axis=-1) | np.any(hit_near_traj_pt, axis=-1)))
            hit_mask = hit_mask & ~traj_hit_mask  # remove from next iteration
            # (N, nhit)

            # project hits onto track
            traj_s = np.cumsum(traj_length[..., 0], axis=-1) + s - traj_length[..., 0]  # (N, 1, npts-1)
            hit_s = hit_alpha * traj_length[..., 0] + traj_s  # (N, nhit, npts-1) * (N, 1, npt-1, 1)
            hit_s = np.take_along_axis(hit_s, itraj_min_td, axis=-1)[..., 0]
            # (N, nhit)

            # and create a histogram
            i_bin = np.expand_dims(np.clip(np.digitize(hit_s, bins=bins) - 1, 0, len(bins) - 1), axis=-1)
            # (N, nhit, nbin)
            q_mask = ((i_bin == np.expand_dims(np.indices(dq.shape)[-1], axis=-2))
                      & np.expand_dims(traj_hit_mask, axis=-1)
                      & np.expand_dims(np.take_along_axis(track_mask, seed_track, axis=-1), axis=-1)
                      & np.expand_dims(np.take_along_axis(iter_mask, seed_track, axis=-1), axis=-1)
                     )
            masked_q = ma.array(np.broadcast_to(hit_q[..., np.newaxis], q_mask.shape), mask=~q_mask)
            xyz_mask = np.broadcast_to(q_mask[..., np.newaxis], q_mask.shape + (3,))
            masked_xyz = ma.array(np.broadcast_to(hit_xyz[..., np.newaxis, :], xyz_mask.shape), mask=~xyz_mask)
            update_elem = np.any(~q_mask, axis=-2)
            dq[update_elem] = dq[update_elem] + ma.sum(masked_q, axis=-2).filled(0)[update_elem]
            dn[update_elem] = dn[update_elem] + ma.sum(~masked_q.mask, axis=-2).filled(0)[update_elem]
            pos[update_elem] = pos[update_elem] + ma.sum(masked_xyz, axis=-3).filled(0)[update_elem]
            
            masked_s = ma.array(np.broadcast_to(hit_s[..., np.newaxis], q_mask.shape), mask=~q_mask)
            s_min[update_elem] = ma.minimum(
                s_min[update_elem], ma.min(masked_s, axis=-2).filled(max_range+1)[update_elem])
            s_max[update_elem] = ma.maximum(
                s_max[update_elem], ma.max(masked_s, axis=-2).filled(-max_range-1)[update_elem])

            # termination conditions
            np.put_along_axis(track_mask, seed_track, False, axis=-1)
            if (np.all([s > max_range]) or not np.any(track_mask) or not np.any(hit_mask)):
                break

            # update variables
            s = s + np.sum(traj_length[..., 0], axis=-1)[..., np.newaxis]
            seed_pt = traj[..., -1, :].copy()
            traj_distance = np.sqrt(ma.sum((tracks['trajectory'] - np.expand_dims(seed_pt, axis=-2))**2, axis=-1))
            traj_mask = np.any((traj_distance < search_dx) & track_mask[...,np.newaxis], axis=-1)
            seed_track = np.expand_dims(np.argmax(traj_mask, axis=-1), axis=-1)
            
            iter_mask[:] = False
            np.put_along_axis(iter_mask, seed_track, np.take_along_axis(traj_mask, seed_track, axis=-1), axis=-1)

            if not np.any(traj_mask):
                break

        pos /= np.maximum(dn[..., np.newaxis], 1e-15)
        end_pt = np.take_along_axis(pos, np.argmax(dq/np.maximum(s_max-s_min, 1e-15) * (dn > 0), axis=-1)[..., np.newaxis, np.newaxis], axis=-2)

        r_dq = np.zeros((orig_len,) + dq.shape[1:])
        r_dn = np.zeros((orig_len,) + dn.shape[1:])
        r_start_pt = np.zeros((orig_len,) + start_pt.shape[1:])
        r_end_pt = np.zeros((orig_len,) + end_pt.shape[1:])
        r_pos = np.zeros((orig_len,) + pos.shape[1:])
        r_ds = np.zeros((orig_len,) + s_min.shape[1:])

        np.place(r_dq, np.broadcast_to(mask[..., np.newaxis], r_dq.shape), dq)
        np.place(r_dn, np.broadcast_to(mask[..., np.newaxis], r_dn.shape), dn)
        np.place(r_ds, np.broadcast_to(mask[..., np.newaxis], r_ds.shape), (s_max-s_min))
        np.place(r_start_pt, np.broadcast_to(mask[..., np.newaxis, np.newaxis], r_start_pt.shape), start_pt)
        np.place(r_end_pt, np.broadcast_to(mask[..., np.newaxis, np.newaxis], r_end_pt.shape), end_pt)
        np.place(r_pos, np.broadcast_to(mask[..., np.newaxis, np.newaxis], r_pos.shape), pos)

        return r_dq, r_dn, r_ds, r_start_pt, r_end_pt, r_pos

    @staticmethod
    def mean_neg_loglikelihood(r0, range_table, profile_n, profile_dqdx, profile_rr, profile_pos):
        profile_rr = profile_rr - r0
        profile_rr, profile_n, profile_dqdx = np.broadcast_arrays(profile_rr, profile_n, profile_dqdx)
        pt_likelihood_dqdx, pt_likelihood_mcs = StoppingMuonSelection.profile_likelihood(
            profile_rr, profile_dqdx, profile_pos, range_table, type='gaus')
        pt_likelihood_mcs = ma.masked_where((profile_n <= 0), pt_likelihood_mcs)
        pt_likelihood_dqdx = ma.masked_where((profile_n <= 0) & (profile_rr <= 0), pt_likelihood_dqdx)
        mean_likelihood = -pt_likelihood_dqdx.mean(axis=-1) - pt_likelihood_mcs.mean(axis=-1)
        return mean_likelihood

    def load_track_trajectories(self, source_name, source_slice, path):
        chain = list(zip([source_name] + path[:-1], path))

        data = self.data_manager.get_dset(path[-1])
        ref, ref_dir = list(zip(*[self.data_manager.get_ref(p, c) for p, c in chain]))
        regions = [self.data_manager.get_ref_region(p, c) for p, c in chain]

        return dereference_chain(source_slice, ref, data=data, regions=regions, ref_directions=ref_dir)

    def run(self, source_name, source_slice, cache):
        hits = cache[self.hits_dset_name]
        tracks = cache[self.merged_dset_name]
        t0 = cache[self.t0_dset_name].reshape(cache[source_name].shape)

        # calculate hit positions
        v_drift = resources['LArData'].v_drift
        crs_ticks = resources['RunData'].crs_ticks
        t_drift = (hits['ts'] - np.expand_dims(t0['ts'], axis=-1)) * crs_ticks
        z = resources['Geometry'].get_z_coordinate(hits['iogroup'], hits['iochannel'],
                                                    t_drift * v_drift)
        hit_q = self.larpix_gain * hits['q'] / np.exp(-t_drift/self.lifetime)  # convert mV -> ke
        hit_xyz = np.concatenate([
            hits['px'][..., np.newaxis], hits['py'][..., np.newaxis],
            z[..., np.newaxis]], axis=-1)

        # find all tracks that end in the fiducial volume
        track_start = tracks.ravel()['trajectory'][..., 0, :]
        track_stop = tracks.ravel()['trajectory'][..., -1, :]
        track_length = tracks.ravel()['length']
        is_stopping = (self.contained(track_start, track_stop)
                       & (track_length > self.length_cut))
        is_throughgoing = self.through_going(track_start, track_stop)
        is_downward = self.downward(track_start, track_stop)
        is_stopping = is_stopping.reshape(tracks.shape)
        is_throughgoing = is_throughgoing.reshape(tracks.shape)
        is_downward = is_downward.reshape(tracks.shape)

        if self.is_mc:
            # lookup the track's true trajectory
            track_traj = self.load_track_trajectories(source_name, source_slice,
                                                      self.mc_trajectory_path)
            track_traj = track_traj.reshape(tracks.shape + (-1,))
            track_traj = condense_array(track_traj, track_traj['trackID'].mask)

            track_true_id = mode(track_traj['trackID'])
            track_true_traj = np.take_along_axis(track_traj,
                                                 np.expand_dims(np.argmax(track_traj['trackID'] == track_true_id, axis=-1), axis=-1),
                                                 axis=-1)
            track_true_traj = track_true_traj.reshape(tracks.shape)

            # find if trajectory ends in the fiducial volume
            true_xyz_start = track_true_traj.ravel()['xyz_start'].copy()
            true_xyz_end = track_true_traj.ravel()['xyz_end'].copy()
            # FIXME: coordinates are weird between sim and data
            true_xyz_start[:, 1] += 218.236
            true_xyz_end[:, 1] += 218.236
            new_x_start = true_xyz_start[:, 2].copy()
            new_x_end = true_xyz_end[:, 2].copy()
            new_z_start = true_xyz_start[:, 0].copy()
            new_z_end = true_xyz_end[:, 0].copy()
            true_xyz_start[:, 0] = new_x_start
            true_xyz_start[:, 2] = new_z_start
            true_xyz_end[:, 0] = new_x_end
            true_xyz_end[:, 2] = new_z_end
            is_muon = ma.abs(track_true_traj['pdgId']) == 13
            is_proton = track_true_traj['pdgId'] == 2212
            is_true_stopping = self.contained(true_xyz_start, true_xyz_end)
            is_true_stopping = is_true_stopping.reshape(tracks.shape)

        # define a stopping event as one with exclusively 1 track that ends in fid.
        event_is_stopping = ((ma.sum(is_stopping, axis=-1) == 1)
                             & (t0['type'] != 0)
                             & (ma.sum(is_throughgoing, axis=-1) == 0)
                             & (ma.sum(is_stopping & is_downward, axis=-1) == 1))
        seed_pt = np.where(np.expand_dims(self.in_fid(track_start), axis=-1),
                           track_stop, track_start).reshape(tracks.shape + (3,))
        seed_pt = np.take_along_axis(seed_pt, np.argmax(is_stopping, axis=-1)[..., np.newaxis, np.newaxis],
                                     axis=-2)

        # now check the likelihood of a stopping muon
        dq, dn, ds, start_pt, end_pt, pos = self.profiled_dqdx(tracks, seed_pt, hit_xyz,
                                                           hit_q,
                                                           mask=event_is_stopping,
                                                           dx=self.profile_dx,
                                                           search_dx=self.profile_search_dx,
                                                           max_range=self.profile_max_range,
                                                           pixel_pitch=resources['Geometry'].pixel_pitch)

        profile_dqdx = dq / ma.maximum(ds, 1e-15) * (dn > 0)
        profile_dqdx[dn <= 0] = 0
        profile_dqdx = ma.masked_where((dn <= 0), profile_dqdx)

        profile_rr = ((np.expand_dims(np.argmax(profile_dqdx, axis=-1), axis=-1)
                       - np.indices(profile_dqdx.shape)[-1]) * self.profile_dx)
        profile_rr = ma.masked_where(dn <= 0, profile_rr)
        # apply a curvature correction
        profile_rr = profile_rr * self.curvature_rr_correction

        profile_n = ma.masked_where(dn <= 0, dn)
        
        # apply a hit density correction
        profile_dqdx = profile_dqdx * ds / ma.maximum(ds - self.density_dx_correction(profile_rr, *self.density_dx_correction_params), 1e-15) * (dn > 0)

        muon_r0 = np.zeros(profile_dqdx.shape[:-1])
        proton_r0 = np.zeros(profile_dqdx.shape[:-1])    
        for i in range(muon_r0.shape[0]):
            if np.any(~profile_n.mask[i] & (profile_n > 0) & ~profile_rr.mask[i] & ~profile_dqdx.mask[i]):
                rr_range = -np.max(profile_rr[i]) - self.profile_search_dx, -np.min(profile_rr[i]) + self.profile_search_dx
                rr_offset = np.expand_dims(np.linspace(rr_range[0], rr_range[1], 2*int(np.diff(rr_range)/self.profile_dx)), axis=-1)

                muon_likelihood = self.mean_neg_loglikelihood(
                    rr_offset, self.muon_range_table, profile_n[i], profile_dqdx[i], profile_rr[i], pos[i][np.newaxis,...])
                muon_r0[i] = rr_offset[np.argmin(muon_likelihood, axis=0)]
                
                proton_likelihood = self.mean_neg_loglikelihood(
                    rr_offset, self.proton_range_table, profile_n[i], profile_dqdx[i], profile_rr[i], pos[i][np.newaxis,...])
                proton_r0[i] = rr_offset[np.argmin(proton_likelihood, axis=0)]

        
        # calculate likelihood scores
        muon_likelihood_dqdx, muon_likelihood_mcs = self.profile_likelihood(
            (profile_rr - muon_r0[...,np.newaxis]).filled(-9999), profile_dqdx, pos,
                                                  self.muon_range_table, type='gaus')
        proton_likelihood_dqdx, proton_likelihood_mcs = self.profile_likelihood(
            (profile_rr - proton_r0[...,np.newaxis]).filled(-9999), profile_dqdx, pos,
                                                    self.proton_range_table, type='gaus')
        mip_likelihood_dqdx, mip_likelihood_mcs = self.profile_likelihood(
            np.clip(profile_rr, 1500, 1500).filled(1500),
                                                 profile_dqdx, pos,
                                                 self.muon_range_table, type='gaus')

        muon_likelihood_mcs = ma.masked_where((dn == 0), muon_likelihood_mcs)
        proton_likelihood_mcs = ma.masked_where((dn == 0), proton_likelihood_mcs)
        mip_likelihood_mcs = ma.masked_where((dn == 0), mip_likelihood_mcs)
        muon_likelihood_dqdx = ma.masked_where((dn == 0) & (profile_rr - muon_r0[...,np.newaxis] > 0), muon_likelihood_dqdx)
        proton_likelihood_dqdx = ma.masked_where((dn == 0) & (profile_rr - proton_r0[...,np.newaxis] > 0), proton_likelihood_dqdx)
        mip_likelihood_dqdx = ma.masked_where((dn == 0), mip_likelihood_dqdx)
        
        profile_rr = profile_rr - muon_r0[...,np.newaxis]
        i_stop = np.argmin(np.abs(profile_rr), axis=-1)[...,np.newaxis,np.newaxis]
#         stop_dir = np.take_along_axis(pos, i_stop-1, axis=-2) - np.take_along_axis(pos, i_stop-2, axis=-2)
#         stop_dir /= np.maximum(np.linalg.norm(stop_dir, axis=-1, keepdims=True), 1e-15)
        end_pt = np.take_along_axis(pos, i_stop, axis=-2) 
#                   - stop_dir * muon_r0[...,np.newaxis,np.newaxis])
        

        # select stopping muons
        event_is_stopping_muon = (event_is_stopping
                                  & (np.mean(muon_likelihood_dqdx, axis=-1) 
                                     + np.mean(muon_likelihood_mcs, axis=-1) 
                                     - np.mean(proton_likelihood_dqdx, axis=-1)
                                     - np.mean(proton_likelihood_mcs, axis=-1) > 0)
                                  & (np.mean(muon_likelihood_dqdx, axis=-1) 
                                     + np.mean(muon_likelihood_mcs, axis=-1) 
                                     - np.mean(mip_likelihood_dqdx, axis=-1)
                                     - np.mean(mip_likelihood_mcs, axis=-1) > 0))

        if self.is_mc:
            # define true stopping events as events with at least 1 muon that ends in fid.
            event_is_true_stopping = ma.sum(is_muon & is_true_stopping, axis=-1) >= 1
            true_stop_pt = np.where(np.expand_dims(self.in_fid(true_xyz_start), axis=-1),
                                    true_xyz_start, true_xyz_end).reshape(tracks.shape + (3,))
            true_stop_pt = np.take_along_axis(true_stop_pt,
                                              np.argmax(is_true_stopping, axis=-1)[..., np.newaxis, np.newaxis],
                                              axis=-2)

        # prep arrays to write to file
        event_sel = np.zeros(len(tracks), dtype=self.event_sel_dtype)
        event_sel['sel'] = event_is_stopping_muon
        event_sel['stop'] = event_is_stopping
        event_sel['muon_loglikelihood_mean'] = np.mean(muon_likelihood_mcs, axis=-1) + np.mean(muon_likelihood_dqdx, axis=-1)
        event_sel['proton_loglikelihood_mean'] = np.mean(proton_likelihood_mcs, axis=-1) + np.mean(proton_likelihood_dqdx, axis=-1)
        event_sel['mip_loglikelihood_mean'] = np.mean(mip_likelihood_mcs, axis=-1) + np.mean(mip_likelihood_dqdx, axis=-1)
        event_sel['stop_pt'] = end_pt.reshape(event_sel['stop_pt'].shape)

        event_profile = np.zeros(len(tracks), dtype=self.event_profile_dtype)
        event_profile['seed_pt'] = start_pt.reshape(event_profile['seed_pt'].shape)
        event_profile['profile_rr'] = profile_rr
        event_profile['profile_dqdx'] = profile_dqdx
        event_profile['profile_n'] = dn
        event_profile['profile_dx'] = ds
        event_profile['profile_pos'] = pos
        event_profile['muon_likelihood'][...,0] = muon_likelihood_dqdx
        event_profile['proton_likelihood'][...,0] = proton_likelihood_dqdx
        event_profile['mip_likelihood'][...,0] = mip_likelihood_dqdx
        event_profile['muon_likelihood'][...,1] = muon_likelihood_mcs
        event_profile['proton_likelihood'][...,1] = proton_likelihood_mcs
        event_profile['mip_likelihood'][...,1] = mip_likelihood_mcs

        if self.is_mc:
            event_true_sel = np.zeros(len(tracks), dtype=self.event_sel_dtype)
            event_true_sel['sel'] = event_is_true_stopping
            event_true_sel['stop'] = ma.sum(is_true_stopping, axis=-1) >= 1
            event_true_sel['muon_loglikelihood_mean'] = ma.sum(is_muon & is_true_stopping, axis=-1) >= 1
            event_true_sel['proton_loglikelihood_mean'] = ma.sum(is_proton & is_true_stopping, axis=-1) >= 1
            event_true_sel['mip_loglikelihood_mean'] = ma.sum(is_muon & ~is_true_stopping, axis=-1) >= 1
            event_true_sel['stop_pt'] = true_stop_pt.reshape(event_true_sel['stop_pt'].shape)

        # reserve data space
        event_sel_slice = self.data_manager.reserve_data(
            f'{self.path}/{self.event_sel_dset_name}', len(event_sel))
        event_profile_slice = self.data_manager.reserve_data(
            f'{self.path}/{self.event_profile_dset_name}', len(event_profile))
        if self.is_mc:
            event_sel_truth_slice = self.data_manager.reserve_data(
                f'{self.path}/{self.event_sel_truth_dset_name}',
                len(event_true_sel))

        # write
        self.data_manager.write_data(f'{self.path}/{self.event_sel_dset_name}',
                                     event_sel_slice, event_sel)
        self.data_manager.write_data(f'{self.path}/{self.event_profile_dset_name}',
                                     event_profile_slice, event_profile)
        if self.is_mc:
            self.data_manager.write_data(
                f'{self.path}/{self.event_sel_truth_dset_name}',
                event_sel_truth_slice, event_true_sel)
