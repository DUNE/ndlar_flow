import numpy as np
import numpy.ma as ma
import logging
from scipy.interpolate import interp1d, pchip_interpolate
import scipy.integrate as integrate
import scipy.stats as stats
import scipy.ndimage as ndimage
import scipy.optimize as optimize
from copy import deepcopy

# Need to have both h5flow and ndlar-flow installed
from h5flow.core import H5FlowStage, resources
from h5flow.data import dereference_chain

from module0_flow.util.func import mode, condense_array
import proto_nd_flow.util.units as units

class HIPSelection(H5FlowStage):
    '''
        Perform a selection for highly ionizing particles. A HIP event is defined 
        using the following criteria:
         - 
         - 
         -
        Creates a boolean array of 1:1 with events indicating HIP events, and 
        creates a boolean array 1:1 with "tracklets if they individually meet the
        HIP criteria.

        NOT CURRENTLY IMPLEMENTED:
        A dQ/dx profile is generated per event and used to discriminate
        stopping protons and muons, as well as through-going muons.

        If the file is a MC file, also generates boolean arrays with the true
        value.

    '''
    class_version = '2.0.0' # change for getting around assertion error in geometry files

    default_params = dict(
        fid_cut=5.0, # cm
        cathode_fid_cut=0.0, # cm
        anode_fid_cut=5.0, # cm
        profile_dx=2.2, # cm
        profile_max_range=200.0, # cm
        larpix_gain=250, # e/mV
        larpix_noise=5000,  # e/cm
        proton_classifier_cut=-1.0,
        muon_classifier_cut=-1.0,
        dqdx_peak_cut=5e4, # e/cm
        profile_search_dx=2.2, # cm
        remaining_e_cut=85e9, # keV

        curvature_rr_correction=22.6647 / 22,
        density_dx_correction_params=[0.78497819, -3.41826874, 198.93022888],

        hits_dset_name='charge/calib_prompt_hits', # '/data' directory may not be necessary ... unclear
        hit_drift_dset_name='charge/calib_prompt_hits', # TO DO: Calibrate for electron lifetime
        tracklet_dset_name='combined/tracklets', #/merged', # no merged part?
        t0_dset_name='combined/t0', # 
        ext_trigs_dset_name='charge/ext_trigs',
        truth_trajectories_dset_name='mc_truth/trajectories',
        charge_dset_name = 'charge/calib_prompt_hits',
        path='high_purity_sel/hips') # path within hdf5 file vs. file path

    sel_dset_name = 'sel_reco'
    sel_truth_dset_name = 'sel_truth'
    #event_tracks_dset_name = 'event_tracks_reco'
    #event_hits_dset_name = 'event_hits_reco'

    sel_dtype = np.dtype([('sel', 'u1'), 
                          ('hip', 'f8'),
                          ('pdg_id', 'f8',(1000,)),
                          ('nhits_over_thresh', 'f8'),
                          ('event_id', 'f8'),
                          ('ntracks', 'f8')])
                          #('max_dqdx', 'f4'),                                
                          #('muon_loglikelihood_mean', 'f8'),
                          #('proton_loglikelihood_mean', 'f8'),
                          #('mip_loglikelihood_mean', 'f8'),
                          #('pid_muon_proton', 'f8'),
                          #('pid_mip_proton', 'f8')])


    def __init__(self, **params):
        super(HIPSelection, self).__init__(**params)
        
        for key,val in self.default_params.items():
            setattr(self, key, params.get(key, val))

        self.curvature_rr_correction = params.get('curvature_rr_correction', dict())
        self.density_dx_correction_params = params.get('density_dx_correction_params', dict())
        self.larpix_gain = params.get('larpix_gain', dict())


    def init(self, source_name):
        super(HIPSelection, self).init(source_name)
        
        self.is_mc = resources['RunData'].is_mc
        correction_key = ('medm')
        correction_key = ('mc' if self.is_mc
                          else 'medm')
        correction_key = ('high' if (not self.is_mc
                                    and resources['RunData'].charge_thresholds == 'high')
                          else correction_key)
        self.curvature_rr_correction = self.curvature_rr_correction.get(correction_key, self.default_params['curvature_rr_correction'])
        self.density_dx_correction_params = self.density_dx_correction_params.get(correction_key, self.default_params['density_dx_correction_params'])
        self.larpix_gain = self.larpix_gain.get(correction_key, self.default_params['larpix_gain'])

        attrs = dict()
        for key in self.default_params:
            attrs[key] = getattr(self, key)
        #print(attrs)
        self.data_manager.set_attrs(self.path,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    **attrs)
        self.data_manager.create_dset(f'{self.path}/{self.sel_dset_name}',
                                      self.sel_dtype)
        #self.data_manager.create_dset(f'{self.path}/{self.event_tracks_dset_name}',
        #                              self.event_tracks_dtype)
        #self.data_manager.create_dset(f'{self.path}/{self.hit_profile_dset_name}',
        #                              self.hit_profile_dtype)
        #self.data_manager.create_ref(f'{self.path}/{self.hit_profile_dset_name}', self.hits_dset_name)
        if self.is_mc:
            self.data_manager.create_dset(f'{self.path}/{self.sel_truth_dset_name}',
                                          self.sel_dtype)

        #self.create_dqdx_profile_templates()
        #self.data_manager.set_attrs(self.path,
        #                           proton_dqdx=self.proton_range_table['dqdx'],
        #                           muon_dqdx=self.muon_range_table['dqdx'],
        #                           proton_dqdx_width=self.proton_range_table['dqdx_width'],
        #                           muon_dqdx_width=self.muon_range_table['dqdx_width'],
        #                           proton_dedx=self.proton_range_table['dedx_mpv'],
        #                           muon_dedx=self.muon_range_table['dedx_mpv'],
        #                           proton_range=self.proton_range_table['range'],
        #                           muon_range=self.muon_range_table['range'],
        #                           proton_recom=self.proton_range_table['recomb'],
        #                           muon_recom=self.muon_range_table['recomb'])

    def finish(self, source_name):
        super(HIPSelection, self).finish(source_name)
        sel_dset_name = f'{self.path}/{self.sel_dset_name}'

        if self.rank == 0:
            total = len(self.data_manager.get_dset(sel_dset_name))
            #min_tracks = np.min(self.data_manager.get_dset(sel_dset_name)['ntracks'])
            #print("Minimum tracks in an event:", min_tracks)
            #max_tracks = np.max(self.data_manager.get_dset(sel_dset_name)['ntracks'])
            #print("Maximum tracks in an event:", max_tracks)
#
            #min_hits = np.min(self.data_manager.get_dset(sel_dset_name)['nhits'])
            #print("Minimum hits in an event:", min_hits)
            #max_hits = np.max(self.data_manager.get_dset(sel_dset_name)['nhits'])
            #print("Maximum hits in an event:", max_hits)
#
            #min_charge = np.min(self.data_manager.get_dset(sel_dset_name)['event_charge'])
            #print("Minimum charge in an event:", min_charge)
            #max_charge = np.max(self.data_manager.get_dset(sel_dset_name)['event_charge'])
            #print("Maximum charge in an event:", max_charge)
#
            #min_ext_trigs = np.min(self.data_manager.get_dset(sel_dset_name)['next_trigs'])
            #print("Minimum ext_trigs in an event:", min_ext_trigs)
            #max_ext_trigs = np.max(self.data_manager.get_dset(sel_dset_name)['next_trigs'])
            #print("Maximum ext_trigs in an event:", max_ext_trigs)

            #nstopping = np.sum(self.data_manager.get_dset(sel_dset_name)['stop'])
            nselected = np.sum(self.data_manager.get_dset(sel_dset_name)['sel'])
            #print(f'Stopping: {nstopping} / {total} ({nstopping/total:0.03f})')
            print(f'Selected: {nselected} / {total} ({nselected/total:0.03f})')
            sel_events_mask = self.data_manager.get_dset(sel_dset_name)['sel'] == 1
            sel_events = self.data_manager.get_dset(sel_dset_name)[sel_events_mask]['event_id']
            print("Sample events:", sel_events)

            if self.is_mc:
                sel_truth_dset_name = f'{self.path}/{self.sel_truth_dset_name}'
                true_proton = np.sum(self.data_manager.get_dset(sel_truth_dset_name)['hip'])
                true_contained_proton = np.sum(self.data_manager.get_dset(sel_truth_dset_name)['sel'])
                print(f'True protons: {true_proton} / {total} ({true_proton/total:0.03f})')
                print(f'True contained proton events: {true_contained_proton} / {total} ({true_contained_proton/total:0.03f})')
                correct = np.sum(self.data_manager.get_dset(sel_truth_dset_name)['sel'] &
                                 self.data_manager.get_dset(sel_dset_name)['sel'])
                print(f'Purity: {correct} / {nselected} ({correct/nselected:0.03f})')
                print(f'Efficiency: {correct} / {true_contained_proton} ({correct/true_contained_proton:0.03f})')
############ START OF CODE PORTED FROM STOPPING MUON CODE FOR PID




    def create_dqdx_profile_templates(self):
        # create range tables used for dQ/dx profile discrimination
        self.muon_range_table = dict()
        self.proton_range_table = dict()

        # only consider reasonable range values
        muon_mask = resources['ParticleData'].muon_range_table['range'] > 0.1
        for key, val in deepcopy(resources['ParticleData'].muon_range_table).items():
            self.muon_range_table[key] = val[muon_mask]
        proton_mask = resources['ParticleData'].proton_range_table['range'] > 0.1
        for key, val in deepcopy(resources['ParticleData'].proton_range_table).items():
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
        noise = (self.larpix_noise * np.sqrt(self.profile_dx / resources['Geometry'].pixel_pitch*units.cm) # converting cm -> mm
                 / resources['Geometry'].pixel_pitch*units.cm) # converting cm -> mm
        post_dedx = resources['ParticleData'].landau_peak(50 * units.MeV,
                                                          resources['ParticleData'].e_mass,
                                                          self.profile_dx) / self.profile_dx
        post_dedx_width = resources['ParticleData'].landau_width(50 * units.MeV,
                                                                 resources['ParticleData'].e_mass,
                                                                 self.profile_dx) / self.profile_dx
        self.muon_range_table['post_dqdx'] = post_dedx * resources['LArData'].ionization_recombination(post_dedx) / w
        self.proton_range_table['post_dqdx'] = 1
        self.muon_range_table['post_dqdx_width'] = post_dedx_width * resources['LArData'].ionization_recombination(post_dedx) / w
        self.proton_range_table['post_dqdx_width'] = 1

        self.muon_range_table['mcs_angle'] = resources['ParticleData'].mcs_angle(self.muon_range_table['t'],
                                                                                 resources['ParticleData'].mu_mass,
                                                                                 self.profile_dx)
        self.proton_range_table['mcs_angle'] = resources['ParticleData'].mcs_angle(self.proton_range_table['t'],
                                                                                   resources['ParticleData'].p_mass,
                                                                                   self.profile_dx)
        self.muon_range_table['post_mcs_angle'] = resources['ParticleData'].mcs_angle(50 * units.MeV,
                                                                                      resources['ParticleData'].e_mass,
                                                                                      self.profile_dx)
        self.proton_range_table['post_mcs_angle'] = 1e-9

        self.muon_range_table['dqdx_gaus_width'] = self.larpix_noise
        self.proton_range_table['dqdx_gaus_width'] = self.larpix_noise

        #self.apply_position_resolution(self.muon_range_table, noise=noise)
        #self.apply_position_resolution(self.proton_range_table, noise=noise)

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
            #     ndimage.uniform_filtein_fid(self, xyz, cathode_fid=0.0, field_cage_fid=0.0)inr(np.abs(ndimage.convolve(dqdx * dx, [-1, 1], mode='nearest')), int(self.profile_dx / dx), mode='nearest')**2
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

    @staticmethod
    def density_dx_correction(rr, *params):
        rr = np.clip(rr, 0, None)
        rv = params[0] * np.exp(-rr / params[2]) + params[1]
        return rv

    @staticmethod
    def dx_estimate(profile_pos, hit_xyz, hit_idx, pixel_pitch, nsamples=10, tol=0.1):
        '''
            Calculate the track dx to be associated with each profile point.

            First finds the furthest point along the line that falls on a hit pixel.
            Then samples the track length between those points, checking to see if the sample point falls onto a
            disabled channel. The track length is calculated as the length between the furthest points, minus the
            approximate length on disabled channels

            :param profile_pos: xyz position of each profile point ``shape: (..., nprof, 3)``

            :param hit_xyz: xyz position of each hit ``shape: (..., nhit, 3)``

            :param hit_idx: index into ``profile_pos`` of each hit ``shape: (..., nhit)``

            :param nsamples: number of sample points to estimate disabled fraction of track

            :returns: dx to be associated with each profile point ``shape: (..., nprof)``

        '''
        dx = np.zeros(profile_pos.shape[:-1])
        for iprof in range(profile_pos.shape[-2]):
            if ~np.any(hit_idx >= iprof):
                break
            hit_mask = hit_idx == iprof
            if ~np.any(hit_mask):
                continue

            xyz = ma.array(hit_xyz, mask=np.broadcast_to(~hit_mask[...,np.newaxis], hit_xyz.shape)) # (nev, nhit, 3)
            valid = np.any(~xyz.mask[...,0], axis=-1) # (nev,)

            # get profile centroid
            pos = profile_pos[...,iprof,:] # (nev, 3)

            # get profile trajectory segment directions
            dirs = [profile_pos[...,iprof+1,:] - pos if iprof < profile_pos.shape[-2]-1 else profile_pos[...,-2,:] - pos,
                    profile_pos[...,iprof-1,:] - pos if iprof > 0 else profile_pos[...,1,:] - pos]
            dirs = np.concatenate([dr[...,np.newaxis,np.newaxis,np.newaxis,:] for dr in dirs], axis=1) # (nev, ndirection, 1, 1, 3)
            
            for idr in range(dirs.shape[1]):
                invalid_dir = np.all(dirs[:,idr] == 0., axis=-1)
                dirs[:,idr][invalid_dir] = -dirs[:,(idr+1)%2][invalid_dir]
            dirs = dirs / np.clip(np.linalg.norm(dirs, axis=-1, keepdims=True), 1e-15, None)

            # get active volume
            min_xyz,max_xyz = np.min(xyz, axis=-2) - pixel_pitch/2, np.max(xyz, axis=-2) + pixel_pitch/2
            min_xyz = min_xyz.reshape(-1,1,1,1,3)
            max_xyz = max_xyz.reshape(-1,1,1,1,3)
            c = np.concatenate([min_xyz, max_xyz], axis=2) # (nev, 1, ncorner, 1, 3)
            n = np.array([(1,0,0), (0,1,0), (0,0,1)]).reshape(1,1,1,3,3) # (1, 1, 1, naxes, 3)

            # find intersections with active volume planes
            pos = pos.reshape(-1, 1, 1, 1, 3)
            intersection = HIPSelection.intersection(pos, dirs, c, n)
            alpha = np.sum(dirs * (intersection - pos), axis=-1)

            # only use intersections that are within active volume (and in the correct direction relative to the trajectory segment)
            within_active_region = ((intersection[...,0] - max_xyz[...,0] <= tol)
                                    & (intersection[...,0] - min_xyz[...,0] >= -tol)
                                    & (intersection[...,1] - max_xyz[...,1] <= tol)
                                    & (intersection[...,1] - min_xyz[...,1] >= -tol)
                                    & (intersection[...,2] - max_xyz[...,2] <= tol)
                                    & (intersection[...,2] - min_xyz[...,2] >= -tol)
                                    & (alpha > 0) & valid.reshape(-1,1,1,1)) # (nev, ndirection, ncorner, naxes)

            intersection = np.take_along_axis(intersection, np.argmax(within_active_region[...,np.newaxis], axis=-2)[...,np.newaxis], axis=-2) # (nev, ndirection, ncorner, 1, 3)
            within_active_region = np.take_along_axis(within_active_region[...,np.newaxis], np.argmax(within_active_region[...,np.newaxis], axis=-2)[...,np.newaxis], axis=-2)
            intersection = np.take_along_axis(intersection, np.argmax(within_active_region, axis=-3)[...,np.newaxis], axis=-3) # (nev, ndirection, 1, 1, 3)
            within_active_region = np.take_along_axis(within_active_region, np.argmax(within_active_region, axis=-3)[...,np.newaxis], axis=-3)

            # calculate track length in active volume
            prof_dx = np.linalg.norm(intersection - pos, axis=-1) # (nev, ndirection, 1, 1)

            # correct for disabled channels
            disabled_fraction = np.zeros_like(prof_dx)
            if 'DisabledChannels' in resources:
                sample_pts = np.linspace(pos, intersection, nsamples, axis=0)
                sample_pt_disabled = ~resources['DisabledChannels'].is_active(sample_pts).reshape(sample_pts.shape[:-1])
                disabled_fraction = np.sum(sample_pt_disabled, axis=0) / nsamples

            prof_dx *= (1 - disabled_fraction)

            # collect result
            dx[...,iprof] = (prof_dx * within_active_region[...,0]).sum(axis=(1,2,3)) # (nev,)

        return dx
        

    @staticmethod
    def profile_likelihood(profile_rr, profile_dqdx, profile_pos, range_table, type='', mcs_weight=0.0625):
        '''
            Calculates the log-likelihood score of a given dqdx v. residual range profile
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

            :param type: likelihood pdf name, one of ``'abs_exp'``, ``'moyal'``, ``'moyal_gaus'``, ``'gaus'``

            :returns: likelihood ``shape: (..., n)``

        '''
        profile_rr, profile_dqdx = np.broadcast_arrays(profile_rr, profile_dqdx)
        profile_pos = np.broadcast_to(profile_pos, profile_rr.shape + (3,))

        interp = interp1d(np.r_[0, range_table['range']], np.r_[range_table['post_dqdx'], range_table['dqdx']])
        interp_width = interp1d(np.r_[0, range_table['range']], np.r_[range_table['post_dqdx_width'], range_table['dqdx_width']])
        interp_angle_width = interp1d(np.r_[0, range_table['range']], np.r_[range_table['post_mcs_angle'], range_table['mcs_angle']])
        min_range = np.min(range_table['range'])
        max_range = np.max(range_table['range'])

        # calculate dQ/dx log-likelihood
        interp_dqdx = interp(np.clip(profile_rr, min_range, max_range))
        interp_dqdx_width = interp_width(np.clip(profile_rr, min_range, max_range))

        if type == 'abs_exp':
            dqdx_term = stats.expon.logpdf(np.abs(profile_dqdx - interp_dqdx), scale=interp_dqdx_width) + np.log(2)
            #dqdx_term = -np.abs(profile_dqdx - interp_dqdx) / interp_dqdx_width - np.log(interp_dqdx_width / 2)
        elif type == 'moyal':
            dqdx_term = stats.moyal.logpdf(profile_dqdx, loc=interp_dqdx, scale=interp_dqdx_width)
        elif type == 'moyal_gaus':
            #             interp_gaus_width = range_table['dqdx_gaus_width']
            interp_gaus_width = interp1d(range_table['range'], range_table['dqdx_gaus_width'])(np.clip(profile_rr, min_range, max_range))
            smear_values = np.linspace(-5 * interp_gaus_width, +5 * interp_gaus_width, 25).reshape(profile_rr.shape + (25,))
            smeared_profile_dqdx = profile_dqdx[..., np.newaxis] + smear_values
            dqdx_term = np.log(np.sum(ma.maximum(stats.moyal.pdf(
                smeared_profile_dqdx, loc=interp_dqdx[..., np.newaxis], scale=interp_dqdx_width[..., np.newaxis]), 1e-300)
                * ma.maximum(stats.norm.pdf(smear_values, scale=interp_gaus_width[..., np.newaxis]), 1e-300), axis=-1))
        elif type == 'gaus':
            dqdx_term = stats.norm.logpdf(profile_dqdx, loc=interp_dqdx, scale=interp_dqdx_width)
        else:
            dqdx_term = -np.abs(profile_dqdx - interp_dqdx) / np.abs(interp_dqdx)

        # calculate MCS log-likelihood
        # pack profile pts
        valid_mask = (profile_dqdx > 0)
        any_valid = np.any(valid_mask)
        npts = np.sum(valid_mask, axis=-1, keepdims=True)
        if any_valid:
            max_npts = npts.max()
        else:
            max_npts = 0

        packed_pos = np.zeros(valid_mask.shape[:-1] + (max_npts, 3))
        packed_dqdx = np.zeros(valid_mask.shape[:-1] + (max_npts,))
        packed_rr = np.zeros(valid_mask.shape[:-1] + (max_npts,))
        place_mask = np.indices(packed_pos.shape)[-2] < npts[..., np.newaxis]
        np.place(packed_pos, place_mask, profile_pos[valid_mask])
        place_mask = np.indices(packed_dqdx.shape)[-1] < npts
        np.place(packed_dqdx, place_mask, profile_dqdx[valid_mask])
        np.place(packed_rr, place_mask, profile_rr[valid_mask])

        #interp_angle_width = interp_angle_width(np.clip(packed_rr, min_range, max_range))
        interp_angle_width = interp_angle_width(np.clip(profile_rr, min_range, max_range))

        d = packed_pos[..., 1:, :] - packed_pos[..., :-1, :]
        d = d * place_mask[...,1:,np.newaxis] * place_mask[...,:-1,np.newaxis]
        angle = np.zeros_like(packed_dqdx)
        norm = np.linalg.norm(d[..., 1:, :], axis=-1) * np.linalg.norm(d[..., :-1, :], axis=-1)
        if any_valid and angle.shape[-1] > 1:
            angle[..., 2:] = np.sum(d[..., 1:, :] * d[..., :-1, :], axis=-1) / np.maximum(norm, 1e-15)
            angle = np.arccos(angle)
            angle[..., 0] = 0
            angle[..., 1] = 0
            #angle[..., 0] = angle[..., 1]
            #angle[..., -1] = angle[..., -2]

        # and now unpack profile pts
        rv_angle_term = np.zeros(valid_mask.shape)
        np.place(rv_angle_term, valid_mask, angle[place_mask])

        #angle_term = stats.norm.logpdf(angle, loc=0, scale=interp_angle_width) + np.log(2)
        rv_angle_term = stats.expon.logpdf(rv_angle_term, scale=interp_angle_width) + np.log(2)
#         angle_term = -np.abs(angle) / np.pi
        #if any_valid:
        #    np.put_along_axis(angle_term, np.argmin(np.abs(packed_rr), axis=-1)[..., np.newaxis], -np.log(2), axis=-1)
        if any_valid:
            # don't count the last profile point towards score
            np.put_along_axis(rv_angle_term, np.argmin(np.abs(profile_rr), axis=-1)[..., np.newaxis], -np.log(2), axis=-1)

        return dqdx_term, rv_angle_term * mcs_weight

    @staticmethod
    def intersection(xyz, dxyz, pxyz, pnorm):
        '''
            calculate the intersection of lines with planes

            :param xyz: (..., 3) array representing line origins
            :param dxyz: (..., 3) array representing line directions (unit norm)
            :param pxyz: (..., 3) array representing a point on the plane
            :param pnorm: (..., 3) array representing plane normal (unit norm)

            :returns: (..., 3) array representing the intersection point
        '''
        with np.errstate(divide='ignore', invalid='ignore'):
            d = np.sum((pxyz - xyz) * pnorm, axis=-1) / np.sum(dxyz * pnorm, axis=-1)
            return xyz + dxyz * d[..., np.newaxis]


    @staticmethod
    def profiled_dqdx_kalman(tracks, seed_pt, hit_xyz, hit_q, dx, max_range, search_dx, pixel_pitch, mask=None):
        orig_len = len(tracks)
        if mask is not None:
            tracks = tracks[mask]
            seed_pt = seed_pt[mask]
            hit_xyz = hit_xyz[mask]
            hit_q = hit_q[mask]
        
        n = len(tracks)
        sample_points = int(max_range / dx)

        dq = np.zeros((n, sample_points))
        dn = np.zeros((n, sample_points), dtype=int)
        ds = np.zeros((n, sample_points))
        pos = np.zeros((n, sample_points, 3))
        hit_prof_idx = np.full(hit_q.shape, -1, dtype=int)
        hit_prof_s = np.full(hit_q.shape, 0, dtype=float)

        hit_mask = ~hit_q.mask

        # find initial point and direction
        traj = np.zeros((n, sample_points, 3))
        start_pt = seed_pt[...,0:1,:]
        end_pt = start_pt.copy()
        traj[...,0:1,:] = seed_pt.copy()
        local_mask = np.linalg.norm(hit_xyz - seed_pt, axis=-1, keepdims=True) < search_dx
        local_mask = np.broadcast_to(hit_mask[...,np.newaxis] & local_mask, hit_xyz.shape)
        curr_direction = ma.array(hit_xyz - seed_pt, mask=~local_mask).mean(axis=-2)
        curr_direction /= np.clip(np.linalg.norm(curr_direction, axis=-1, keepdims=True),1e-15,None)
        hit_mask = hit_mask & ~local_mask[...,0]

        disabled_channels = resources.get('DisabledChannels', None)
        
        i = 0
        while (i < sample_points-1) and np.any(hit_mask):
            i += 1
            
            # collect hits in local region
            dr = (hit_xyz - traj[...,i-1,np.newaxis,:])
            dl = np.sum(dr * curr_direction[...,np.newaxis,:], axis=-1, keepdims=True)
            forward = dl > 0
            dt = np.linalg.norm(dr - dl * curr_direction[...,np.newaxis,:], axis=-1, keepdims=True)
            local_mask = (dl < dx) & (dt < dx/2) & hit_mask[...,np.newaxis] & forward

            # if none found, expand search
            if np.any(~((local_mask[...,0]).any(axis=-1)) & hit_mask.any(axis=-1)):
                r = np.linalg.norm(dr, axis=-1, keepdims=True)

                # if disabled channels list present and next step is a disabled region, search in a longer line first
                if disabled_channels is not None:
                    proposed_step = traj[...,i-1,:] + curr_direction * dx
                    step_is_disabled = ~disabled_channels.is_active(proposed_step)
                    local_mask = local_mask | (
                        (dl < 2*dx) & (dt < 3*dx/4) & hit_mask[...,np.newaxis] & forward
                        & step_is_disabled[...,np.newaxis,np.newaxis]
                        & ~(local_mask).any(axis=-2, keepdims=True))

                # then search in a sphere in ever expanding circles
                search_factor = 1
                while np.any(~(local_mask[...,0]).any(axis=-1) & hit_mask.any(axis=-1)):
                    local_mask = (local_mask | (
                        (r < search_factor * search_dx) & hit_mask[...,np.newaxis]
                        & ~(local_mask).any(axis=-2, keepdims=True)))
                    search_factor += 1
                    if search_factor > 5:
                        break

            # if no more hits found, continue
            if not np.any(local_mask):
                break

            # calculate new sample point (charge weighted average position)
            traj[...,i,:] = ma.average(ma.array(hit_xyz, mask=~np.broadcast_to(local_mask, hit_xyz.shape)),
                                       weights=np.broadcast_to(hit_q[...,np.newaxis], hit_xyz.shape), axis=-2)
            end_pt = traj[...,i:i+1,:]

            # calculate new direction
            curr_direction = traj[...,i,:] - traj[...,i-1,:]
            curr_direction /= np.clip(np.linalg.norm(curr_direction, axis=-1, keepdims=True), 1e-15, None)

            # mask off used hits
            hit_mask = hit_mask & ~local_mask[...,0]

        # project hits onto trajectory segments
        dr = (hit_xyz[...,np.newaxis,:] - traj[...,np.newaxis,:-1,:]) # (ev, hit, traj-1, 3)
        traj_dr = traj[...,np.newaxis,1:,:] - traj[...,np.newaxis,:-1,:] # (ev, 1, traj-1, 3)
        traj_l = np.clip(np.linalg.norm(traj_dr, axis=-1, keepdims=True), 1e-15, None) # (ev, 1, traj-1, 1)
        traj_dr /= traj_l
        alpha = np.sum(dr * traj_dr, axis=-1) / traj_l[...,0] # (ev, hit, traj-1)

        # find closest segment
        d = np.linalg.norm(dr - traj_dr * np.clip(alpha[...,np.newaxis], 0, 1) * traj_l, axis=-1) # (ev, hit, traj-1)
        d = ma.array(d, mask=(hit_q.mask[...,np.newaxis] | (d > dx/2)))
        d.mask[...,i-1:] = True # remove invalid segments
        iseg_min = np.argmin(d, axis=-1) # (ev, hit)
        iseg_min[np.take_along_axis(d.mask, iseg_min[...,np.newaxis], axis=-1).reshape(iseg_min.shape)] = -1

        # calculate segment range
        s = np.concatenate([np.zeros(traj_l.shape[:-2] + (1,1)), np.cumsum(traj_l, axis=-2)], axis=-2) # (ev, 1, traj-1, 1)
        hit_s = np.take_along_axis(s, iseg_min[...,np.newaxis,np.newaxis], axis=-2)
        hit_s = hit_s + np.take_along_axis(traj_l * alpha[...,np.newaxis], iseg_min[...,np.newaxis,np.newaxis], axis=-2) # (ev, hit, 1, 1)
        hit_s = hit_s[...,0,0]
        
        # fill bins
        bins = np.linspace(0, max_range, sample_points)
        hit_prof_idx = np.clip(np.digitize(hit_s, bins=bins) - 1, 0, sample_points-1)
        hit_prof_idx[hit_q.mask] = -1

        sample_point_s = np.zeros_like(ds)
        prev_pos = traj[...,0,:]
        for i in range(sample_points):
            #if not np.any(hit_prof_idx >= i):
            #    break
            
            # grab hits from current trajectory point
            hit_mask = (hit_prof_idx == i) & (~hit_q.mask)
            any_hit_mask = hit_mask.any(axis=-1)
            #if not np.any(any_hit_mask):
            #    continue

            # re-estimate position and only use "local" hits
            traj_hit_s = ma.array(hit_s, mask=~hit_mask)
            local_pos = (ma.average(ma.array(hit_xyz, mask=~np.broadcast_to(hit_mask[...,np.newaxis], hit_xyz.shape)),
                                        weights=np.broadcast_to(hit_q[...,np.newaxis], hit_xyz.shape), axis=-2)
                             * any_hit_mask[...,np.newaxis])
            local_pos[~any_hit_mask,:] = prev_pos[~any_hit_mask,:]
            prev_pos = local_pos
            
            hit_mask = hit_mask & (np.linalg.norm(hit_xyz - local_pos[...,np.newaxis,:], axis=-1) < dx)
            any_hit_mask = hit_mask.any(axis=-1)

            #if not np.any(any_hit_mask):
            #    continue

            # fill output arrays
            pos[...,i,:] = local_pos
            dq[...,i] = (np.sum(ma.array(hit_q, mask=~hit_mask), axis=-1)) * any_hit_mask
            dn[...,i] = (np.sum(hit_mask, axis=-1)) * any_hit_mask            
            local_dir = pos[...,i,:] - pos[...,i-1,:] if i > 0 else traj[...,1,:] - traj[...,0,:]
            if i > 0:
                sample_point_s[...,i:] += np.linalg.norm(local_dir, axis=-1)[...,np.newaxis]
            local_dir /= np.clip(np.linalg.norm(local_dir, axis=-1, keepdims=True), 1e-15, None)
            local_s = ma.array(np.sum((hit_xyz - pos[...,i:i+1,:]) * local_dir[...,np.newaxis,:], axis=-1), mask=~hit_mask)
            hit_prof_s[hit_mask] = (local_s + sample_point_s[...,i:i+1])[hit_mask]
            ds[...,i] = (np.max(local_s, axis=-1) - np.min(local_s, axis=-1)) * any_hit_mask

        r_dq = np.zeros((orig_len,) + dq.shape[1:])
        r_dn = np.zeros((orig_len,) + dn.shape[1:], dtype=int)
        r_start_pt = np.zeros((orig_len,) + start_pt.shape[1:])
        r_end_pt = np.zeros((orig_len,) + end_pt.shape[1:])
        r_pos = np.zeros((orig_len,) + pos.shape[1:])
        r_ds = np.zeros((orig_len,) + ds.shape[1:])
        r_hit_prof_idx = np.zeros((orig_len,) + hit_prof_idx.shape[1:], dtype=int) - 1
        r_hit_prof_s = np.zeros((orig_len,) + hit_prof_s.shape[1:], dtype=float)    

        np.place(r_dq, np.broadcast_to(mask[..., np.newaxis], r_dq.shape), dq)
        np.place(r_dn, np.broadcast_to(mask[..., np.newaxis], r_dn.shape), dn)
        np.place(r_ds, np.broadcast_to(mask[..., np.newaxis], r_ds.shape), ds)
        np.place(r_start_pt, np.broadcast_to(mask[..., np.newaxis, np.newaxis], r_start_pt.shape), start_pt)
        np.place(r_end_pt, np.broadcast_to(mask[..., np.newaxis, np.newaxis], r_end_pt.shape), end_pt)
        np.place(r_pos, np.broadcast_to(mask[..., np.newaxis, np.newaxis], r_pos.shape), pos)
        np.place(r_hit_prof_idx, np.broadcast_to(mask[..., np.newaxis], r_hit_prof_idx.shape), hit_prof_idx)
        np.place(r_hit_prof_s, np.broadcast_to(mask[..., np.newaxis], r_hit_prof_s.shape), hit_prof_s)        

        return r_dq, r_dn, r_ds, r_start_pt, r_end_pt, r_pos, r_hit_prof_idx, r_hit_prof_s

    @staticmethod
    def mean_neg_loglikelihood(r0, range_table, profile_n, profile_dqdx, profile_rr, profile_pos):
        profile_rr = profile_rr - r0
        pt_likelihood_dqdx, pt_likelihood_mcs = HIPSelection.profile_likelihood(
            profile_rr, profile_dqdx, profile_pos, range_table)
        profile_n, profile_dqdx, profile_rr = np.broadcast_arrays(profile_n, profile_dqdx, profile_rr)
        pt_likelihood_mcs = ma.masked_where((profile_n <= 0) | (profile_rr <= 0), pt_likelihood_mcs)
        #pt_likelihood_dqdx = ma.masked_where((profile_rr <= 0), pt_likelihood_dqdx)
        pt_likelihood_dqdx = ma.masked_where((profile_n <= 0) | (profile_rr <= 0), pt_likelihood_dqdx)

        mean_likelihood = -pt_likelihood_dqdx.mean(axis=-1) - pt_likelihood_mcs.mean(axis=-1)
        return mean_likelihood





    def run(self, source_name, source_slice, cache):
        super(HIPSelection, self).run(source_name, source_slice, cache)
        
        # load arrays of event-level, hit-level, and track-level info
        events = cache[source_name]
        t0 = cache[self.t0_dset_name].reshape(cache[source_name].shape)
        hits = ma.array(cache[self.hits_dset_name], shrink=False)
        q = ma.array(cache[self.charge_dset_name], shrink=False)
        q = q.reshape(hits.shape)
        tracks = ma.array(cache[self.tracklet_dset_name], shrink=False)
        #hit_drift = ma.array(cache[self.hit_drift_dset_name].reshape(hits.shape), shrink=False)
        #track_hits = ma.array(cache[self.track_hits_dset_name], shrink=False)
        #track_hit_drift = ma.array(cache[self.track_hit_drift_dset_name], shrink=False)
        #print("Track shape:", tracks.shape)
        #print("Track hits shape:", track_hits.shape)
        #print("Track hit drift shape:", track_hit_drift.shape)

        if events.shape[0]:

            ## EVENT-LEVEL CALCULATIONS

            # calculate hit positions and charge
            hit_q = q['Q'] # convert mV -> ke
            # filter out bad channel ids            
            hit_mask = (hits['y'] != 0.0) & (hits['z'] != 0.0) & ~hit_q.mask & ~hits['t_drift'].mask            
            hit_q.mask = hit_q.mask | ~hit_mask
            hit_xyz = ma.array(np.concatenate([
                hits['x'][..., np.newaxis], hits['y'][..., np.newaxis],
                hits['z'][..., np.newaxis]], axis=-1), shrink=False, mask=np.zeros(hits['y'].shape + (3,), dtype=bool) | hit_q.mask[...,np.newaxis] | ~hit_mask[...,np.newaxis])
            hit_in_fid = resources['Geometry'].in_fid(hit_xyz.reshape(-1, 3), cathode_fid=self.cathode_fid_cut, \
                                                      field_cage_fid=self.fid_cut, anode_fid=self.anode_fid_cut).reshape(hit_xyz.shape[:-1])
            hit_q.mask = hit_q.mask | ~hit_in_fid

            # find value for the most charge in one hit in each event
            #print("HIt q", hit_q[~hit_q.mask]/self.larpix_gain)
            #print("HIt q", hit_q.shape)
            max_hit_charge = ma.array([int(ma.filled(hit_q[i,:].astype(float) > 37.5, False).sum()) for i in range(len(hits))]) # 75 ke -> 300 mV with conversion in calib hits
            #print("test max hit charge:", ma.filled(hit_q[0,:].astype(float)/self.larpix_gain > 300., False)== True)
            #print("NEW MAX HIT CHARGE SHAPE:", max_hit_charge.shape)
            #print("Max Hit Charge:", max_hit_charge)
            ## TRACK-LEVEL CALCULATIONS

            # find all tracks that end in the fiducial volume
            track_start = tracks.ravel()['trajectory'][..., 0, :]
            track_stop = tracks.ravel()['trajectory'][..., -1, :]
            #print("Track start:", track_start)
            #print("Track stop:", track_stop)
            #track_dqdx = tracks.ravel()['dq']/np.sqrt(np.sum(tracks.ravel()['dx']**2, axis=-1))
            #print("Tracks dq/dx:", track_dqdx[:5])
            #print("Tracks dn:", tracks.ravel()['dn'][:5])
            #track_dqdx_start = track_dqdx[track_dqdx != 0][..., 0]
            #track_dqdx_stop = track_dqdx[track_dqdx != 0][..., -1]
            #track_dn_start = tracks.ravel()['dn'][..., 0]
            #track_dn_stop = tracks.ravel()['dn'][..., -1]
            #print("Tracks start shape:", track_start.shape)
            #print("Tracks dq shape:", track_dqdx_start.shape)


            start_in_fid = resources['Geometry'].in_fid(
                track_start, field_cage_fid=self.fid_cut, anode_fid=self.anode_fid_cut)
            start_in_fid = start_in_fid.reshape(tracks.shape)
            stop_in_fid = resources['Geometry'].in_fid(
                track_stop, field_cage_fid=self.fid_cut, anode_fid=self.anode_fid_cut)
            stop_in_fid = stop_in_fid.reshape(tracks.shape)
            contained_in_fid = start_in_fid & stop_in_fid
            #print("Track start:", track_start[:5,:])
            #print("Track stop:", track_stop[:5,:])
            #print("Track dq start:", track_dqdx_start[:5])
            #print("Track dq stop:", track_dqdx_stop[:5])
            #print("Track dnhits start:", track_dn_start[:5])
            #print("Track dnhits stop:", track_dn_stop[:5])
            #print("Shape of start_in_fid:", start_in_fid.shape)
            #print("Start in fid:", start_in_fid)
            #print("Stop in fid:", stop_in_fid)
            #print("Contained in fid:", contained_in_fid)
            event_ntracks_in_fid = np.zeros(len(tracks), dtype=int)
            #print("Start in FID masked tracks:", np.array([int(start_in_fid[i].sum()) for i in range(len(tracks))]))

            # prep arrays to write to file
            event_ids = events['id']
            event_next_trigs = events['n_ext_trigs']
            #print("Shape of one event's tracks:", tracks['id'][0].shape)
            #print("One event's tracks's mask:", tracks['id'][0].mask)
            #print("One event's tracks's ids:", tracks['id'][0])
            #print("Number of valid events for one event:", int((~tracks['id'][0].mask).sum()))
            event_ntracks = np.array([int((~tracks['id'][i].mask).sum()) for i in range(len(tracks))])
            event_nhits = events['nhit']
            #event_charge = events['q']
            for i in range(len(tracks)):
                if event_ntracks[i] > 0: 
                    event_ntracks_in_fid[i] = int(contained_in_fid[i].sum())
                else:
                    event_ntracks_in_fid[i] = 0

            nhits_cut = (event_nhits > 50) & (event_nhits < 5000) 
            #print("Number of hits:", nhits_cut)
            hit_charge_threshold_cut = (max_hit_charge > 1) # cut on number of hits over threshold, which is currently 300 mV
            external_trigger_cut = (event_next_trigs > 0)
            ntracks_in_fid_cut = (event_ntracks_in_fid == event_ntracks) & (event_ntracks == 1)# & (event_ntracks <= 3)
            event_level_cuts = nhits_cut & hit_charge_threshold_cut & external_trigger_cut & ntracks_in_fid_cut
            print("Event level cuts:", event_level_cuts)

            
            print("Passing Tracks:", tracks['trajectory'][event_level_cuts])
            print("Event id:", events['id'][event_level_cuts])

            max_tracks = contained_in_fid.shape[1]
            #print("Max tracks shape:",max_tracks)
            #print("Event level cuts shape:", event_level_cuts.shape)
            # make the array of all initial cuts the same length as the tracks array
            #event_level_cuts_ext = np.array([np.full(max_tracks, event_level_cuts[i]) for i in range(len(event_level_cuts))])
            #print("Event level cuts extended shape:", event_level_cuts_ext.shape)

            #all_initial_cuts_ext = contained_in_fid & event_level_cuts_ext
            #contained_in_fid_red = np.logical_and.reduce(contained_in_fid, -1, dtype=bool)
            #print("Contained in FID Reduced:", contained_in_fid_red)
            
            #print("All initial cuts shape:", all_initial_cuts.shape)

            # Look into unique channels:
            #hits_with_channels = ma.array([hits['iogroup'], hits['iochannel'], hits['chipid'], hits['channelid']])
            #print("IO Group:", hits['iogroup'])
            #print("IO Channel:", hits['iochannel'])
            #print("Chip ID:", hits['chipid'])
            #print("Channel ID:", hits['channelid'])
            #print("Shape of hits with channels:", hits_with_channels.shape)


            '''if self.is_mc:
                # lookup the track's true trajectory
                track_traj = cache[self.truth_trajectories_dset_name]
                #print("True Trajectory PID situation:", track_traj['pdgId'])

                if track_traj.shape[0]:
                    #print("track ids pre-reshaping:", track_traj['trackID'])
                    track_traj = track_traj.reshape(tracks.shape[0:1] + (-1,))
                    track_traj = condense_array(track_traj, track_traj['trackID'].mask)
                    track_pdg = condense_array(track_traj, track_traj['pdgId'].mask)
                    
                    #print("track ids post-reshaping:", track_traj['trackID'])
                    #print("pdg ids post-reshaping:", track_pdg['pdgId'])
                    proton_mask_true = track_pdg['pdgId'] == 2212
                    proton_mask = np.tile(proton_mask_true[..., np.newaxis], (1,1,3))


                    #print("Proton mask:", proton_mask)
                    true_xyz_start =  ma.masked_where(~proton_mask, track_traj['xyz_start'])
                    true_xyz_end =  ma.masked_where(~proton_mask, track_traj['xyz_end'])

                    #n_protons = len(track_pdg[proton_mask_true])
                    #true_xyz_start = true_xyz_start[~true_xyz_start.mask].reshape((n_protons,3))
                    #true_xyz_end = true_xyz_end[~true_xyz_end.mask].reshape((n_protons,3))
                    #print("True xyz start:", true_xyz_start)
                    #print("Proton trajectories:", proton_trajectories)
                    # Look at all possible proton trajectories
                    #i_primary_traj = proton_trajectories
                    #print("Track trajectory shape:", track_traj.shape)
                    #print("Proton trajectories axis shape:", i_primary_traj.shape)
                    #track_true_traj = d
                    #print("Track true trajectories only protons:", track_true_traj)
                    #track_true_traj = track_true_traj.reshape(-1)
                    #true_xyz_start = proton_trajectories['xyz_start'] #track_true_traj['xyz_start'] 
                    #true_xyz_end = proton_trajectories['xyz_end']#track_true_traj['xyz_end']

                    # find if trajectory ends in the fiducial volume
                    #is_muon = ma.abs(track_true_traj['pdgId']) == 1
                    #print("PDG ID shape:", track_traj['pdgId'].shape)
                    #is_proton = ma.array([int(((track_traj['pdgId'][i] == 2212).astype(float)).sum(axis=-1))>=1 for i in range(len(track_traj))])

                    is_proton = np.empty(tracks.shape[0], dtype=bool)
                    for i in range(len(tracks)):
                        all_pdg = track_traj['pdgId'][i].ravel() == 2212
                        is_proton[i] = np.sum(all_pdg.astype(int))
                    #print("What is is_proton?:", is_proton)

                    #if len(is_proton):
                        #print("True start:", true_xyz_start[is_proton])
                        #print("True start:", true_xyz_end[is_proton])
                else:
                    track_true_traj = np.empty(tracks.shape[0], dtype=track_traj.dtype)
                    is_muon = np.zeros(track_true_traj.shape, dtype=bool)
                    is_proton = np.zeros(track_true_traj.shape, dtype=bool)
                    true_xyz_start = track_true_traj['xyz_start']
                    true_xyz_end = track_true_traj['xyz_end']
            # define seed point based on 
            start_in_fid = resources['Geometry'].in_fid(
                track_start, field_cage_fid=self.fid_cut, anode_fid=self.anode_fid_cut)
            seed_pt = track_start.reshape(tracks.shape + (3,))
            seed_track_mask = all_initial_cuts
            max_seed_pts = int(max(np.sum((ma.filled(seed_track_mask.astype(float), 0.0)), axis=-1).max(), 1))
            seed_pt_idx = ma.argsort(ma.array(seed_track_mask, mask=~seed_track_mask | seed_track_mask.mask), axis=-1, fill_value=0)[..., ::-1, np.newaxis]
            seed_pt = np.take_along_axis(seed_pt, seed_pt_idx, axis=1)[...,:max_seed_pts,:]
            seed_pt = ma.array(seed_pt, mask=np.indices(seed_pt.shape)[1] >= np.sum(seed_track_mask, axis=-1, keepdims=True)[...,np.newaxis])
            #print("Seed point:", seed_pt[~seed_pt.mask])
            #print("Seed point shape:", seed_pt.shape)

            event_passes_initial_cuts = ((t0['type'] != 0)
                                 #& (veto_q < self.veto_charge_cut)
                                 #& (active_proj_length > self.projected_length_cut)
                                 #& (ma.sum(is_throughgoing, axis=-1) == 0)
                                 & (ma.sum(seed_track_mask, axis=-1) >= 1))
            
            # now check the likelihood of a stopping muon

            # broadcast into appropriate shape for kalman fit
            tracks_km = np.broadcast_to(tracks[:,np.newaxis], (tracks.shape[0], max_seed_pts, tracks.shape[1]), subok=True).reshape(-1, tracks.shape[1])
            tracks_km.mask = np.broadcast_to(tracks.mask[:,np.newaxis], (tracks.shape[0], max_seed_pts, tracks.shape[1]), subok=True).reshape(-1, tracks.shape[1])
            hit_xyz_km = np.broadcast_to(hit_xyz[:,np.newaxis], (hit_xyz.shape[0], max_seed_pts) + hit_xyz.shape[1:], subok=True).reshape(-1, *hit_xyz.shape[1:])
            hit_xyz_km.mask = np.broadcast_to(hit_xyz.mask[:,np.newaxis], (hit_xyz.shape[0], max_seed_pts) + hit_xyz.shape[1:], subok=True).reshape(-1, *hit_xyz.shape[1:])
            hit_q_km = np.broadcast_to(hit_q[:,np.newaxis], (hit_q.shape[0], max_seed_pts) + hit_q.shape[1:], subok=True).reshape(-1, *hit_q.shape[1:])
            hit_q_km.mask = np.broadcast_to(hit_q.mask[:,np.newaxis], (hit_q.shape[0], max_seed_pts) + hit_q.shape[1:], subok=True).reshape(-1, *hit_q.shape[1:])
            kalman_mask = (event_passes_initial_cuts[...,np.newaxis] & ~seed_pt.mask[...,0]).ravel()
        
            # first generate the dQ/dx profile
            dq, dn, ds, start_pt, end_pt, pos, hit_prof_idx, hit_prof_s = self.profiled_dqdx_kalman(
                tracks_km, seed_pt.reshape(-1, 1, 3), hit_xyz_km, hit_q_km,
                mask=kalman_mask,
                dx=self.profile_dx, search_dx=self.profile_search_dx,
                max_range=self.profile_max_range, pixel_pitch=resources['Geometry'].pixel_pitch)
            #ds += resources['Geometry'].pixel_pitch # correct for pixel edges
            ds = self.dx_estimate(pos, hit_xyz_km, hit_prof_idx, resources['Geometry'].pixel_pitch)
            profile_n = dn
            profile_dqdx = dq / ma.maximum(ds, resources['Geometry'].pixel_pitch) * (dn > 0)
            profile_dqdx[dn <= 0] = 0

            # make an initial guess for the stopping point (maximum 2 dQ/dx bins)
            profile_rr = np.linalg.norm(pos[...,1:,:] - pos[...,:-1,:], axis=-1)
            profile_rr = np.concatenate((np.zeros(profile_rr.shape[:-1]+(1,)), profile_rr), axis=-1)
            profile_rr = np.cumsum(profile_rr, axis=-1)
        
            i_max = np.argsort(profile_dqdx, axis=-1)[...,-2:]
            profile_offset0 = np.take_along_axis(profile_rr, i_max[...,0:1], axis=-1)
            profile_offset1 = np.take_along_axis(profile_rr, i_max[...,1:2], axis=-1)
        
            # refine guess by using the hit with the largest charge
            hit_near_stop0 = (hit_prof_idx == i_max[...,0:1])
            hit_near_stop1 = (hit_prof_idx == i_max[...,1:2])
            profile_offset0[hit_near_stop0.any(axis=-1)] = np.take_along_axis(
                hit_prof_s, np.argmax(ma.array(hit_q_km, mask=~hit_near_stop0), axis=-1)[...,np.newaxis], axis=-1)[hit_near_stop0.any(axis=-1)]
            profile_offset1[hit_near_stop1.any(axis=-1)] = np.take_along_axis(
                hit_prof_s, np.argmax(ma.array(hit_q_km, mask=~hit_near_stop1), axis=-1)[...,np.newaxis], axis=-1)[hit_near_stop1.any(axis=-1)]

            profile_rr0 = profile_offset0 - profile_rr
            profile_rr1 = profile_offset1 - profile_rr

            # perform a fit for the stopping point assuming a muon or a proton
            proton_score = np.full(profile_dqdx.shape[:-1], 1e+303)
            muon_r0 = np.zeros(profile_dqdx.shape[:-1])
            proton_r0 = np.zeros(profile_dqdx.shape[:-1])
            max_range = 0 #self.profile_dx  # within +/- 1 profile bins
            sample_factor = 1 #20  # resolution is profile bin/10

            for i in range(proton_r0.shape[0]):
                if np.any((profile_n[i] > 0)):
                    valid_mask = profile_n[i] > 0

                    muon_offset = []
                    proton_offset = []
                    muon_likelihood = []
                    proton_likelihood = []         
                
                    for j,rr in enumerate([profile_rr0[i], profile_rr1[i]]):
                        rr_range = (np.maximum(-max_range, rr[valid_mask].min()),
                                    np.minimum(+max_range, rr[valid_mask].max()))
                        rr_offset = np.expand_dims(
                            np.linspace(rr_range[0], rr_range[1],
                                        np.clip(sample_factor * int(np.diff(rr_range) / self.profile_dx),1,None)),
                            axis=-1)
                        close_dqdx = np.take_along_axis(profile_dqdx[i:i + 1], np.argmin(np.abs(rr[np.newaxis,...] - rr_offset), axis=-1)[..., np.newaxis], axis=-1)
                        mask = np.ones_like((close_dqdx > self.dqdx_peak_cut)) # ignore dQ/dx mask
                        #if not np.any(mask):
                        #    continue

                        muon_likelihood.append(self.mean_neg_loglikelihood(
                            rr_offset + muon_r0[i], self.muon_range_table, profile_n[i:i + 1], profile_dqdx[i:i + 1], rr[np.newaxis,...], pos[i:i + 1]))
                        #muon_r0[i] = rr_offset[ma.argmin(ma.array(muon_likelihood, mask=~mask), axis=0)] + muon_r0[i]
                        muon_offset.append(rr_offset[ma.argmin(ma.array(muon_likelihood[j], mask=~mask), axis=0)])

                        proton_likelihood.append(self.mean_neg_loglikelihood(
                            rr_offset + proton_r0[i], self.proton_range_table, profile_n[i:i + 1], profile_dqdx[i:i + 1], rr[np.newaxis,...], pos[i:i + 1]))
                        #proton_r0[i] = rr_offset[ma.argmin(ma.array(proton_likelihood, mask=~mask), axis=0)] + proton_r0[i]
                        proton_offset.append(rr_offset[ma.argmin(ma.array(proton_likelihood[j], mask=~mask), axis=0)])

                    muon_j_min = np.argmin([np.min(ll) if ll is not np.nan else 1e+303 for ll in muon_likelihood])
                    proton_j_min = np.argmin([np.min(ll) if ll is not np.nan else 1e+303 for ll in proton_likelihood])
                    proton_score[i] = ma.filled(proton_likelihood[proton_j_min].astype(float), 1e+303)
                    muon_r0[i] = muon_offset[muon_j_min]
                    proton_r0[i] = proton_offset[proton_j_min]
                    profile_rr[i] = [profile_rr0[i], profile_rr1[i]][muon_j_min]

            # use only the dQ/dx profile from the most proton-like seed point
            ibest_seed = ma.argmin(ma.array(proton_score, mask=np.all(profile_n == 0, axis=-1)).reshape(-1, max_seed_pts), axis=-1)[...,np.newaxis]
            profile_dqdx = np.take_along_axis(profile_dqdx.reshape(ibest_seed.shape[0], max_seed_pts, -1), ibest_seed[...,np.newaxis], axis=1)[:,0]
            profile_n = np.take_along_axis(profile_n.reshape(ibest_seed.shape[0], max_seed_pts, -1), ibest_seed[...,np.newaxis], axis=1)[:,0]
            pos = np.take_along_axis(pos.reshape(ibest_seed.shape[0], max_seed_pts, -1, 3), ibest_seed[...,np.newaxis,np.newaxis], axis=1)[:,0]
            profile_rr = np.take_along_axis(profile_rr.reshape(ibest_seed.shape[0], max_seed_pts, -1), ibest_seed[...,np.newaxis], axis=1)[:,0]
            muon_r0 = np.take_along_axis(muon_r0.reshape(ibest_seed.shape[0], max_seed_pts), ibest_seed, axis=1)[:,0]
            proton_r0 = np.take_along_axis(proton_r0.reshape(ibest_seed.shape[0], max_seed_pts), ibest_seed, axis=1)[:,0]
            dq = np.take_along_axis(dq.reshape(ibest_seed.shape[0], max_seed_pts, -1), ibest_seed[...,np.newaxis], axis=1)[:,0]
            dn = np.take_along_axis(dn.reshape(ibest_seed.shape[0], max_seed_pts, -1), ibest_seed[...,np.newaxis], axis=1)[:,0]
            ds = np.take_along_axis(ds.reshape(ibest_seed.shape[0], max_seed_pts, -1), ibest_seed[...,np.newaxis], axis=1)[:,0]
            start_pt = np.take_along_axis(start_pt.reshape(ibest_seed.shape[0], max_seed_pts, -1, 3), ibest_seed[...,np.newaxis,np.newaxis], axis=1)[:,0]
            end_pt = np.take_along_axis(end_pt.reshape(ibest_seed.shape[0], max_seed_pts, -1, 3), ibest_seed[...,np.newaxis,np.newaxis], axis=1)[:,0]
            hit_prof_idx = np.take_along_axis(hit_prof_idx.reshape(ibest_seed.shape[0], max_seed_pts, -1), ibest_seed[...,np.newaxis], axis=1)[:,0]
            hit_prof_s = np.take_along_axis(hit_prof_s.reshape(ibest_seed.shape[0], max_seed_pts, -1), ibest_seed[...,np.newaxis], axis=1)[:,0]

            # calculate likelihood scores for refined dQ/dx profile
            muon_likelihood_dqdx, muon_likelihood_mcs = self.profile_likelihood(
                (profile_rr - muon_r0[..., np.newaxis]), profile_dqdx, pos,
                self.muon_range_table)
            proton_likelihood_dqdx, proton_likelihood_mcs = self.profile_likelihood(
                (profile_rr - proton_r0[..., np.newaxis]), profile_dqdx, pos,
                self.proton_range_table)
            mip_likelihood_dqdx, mip_likelihood_mcs = self.profile_likelihood(
                np.clip(profile_rr, 1500, 1500), profile_dqdx, pos,
                self.muon_range_table)

            muon_likelihood_mcs = ma.masked_where(
                (dn == 0) | (profile_rr - muon_r0[..., np.newaxis] <= 0),
                muon_likelihood_mcs)
            proton_likelihood_mcs = ma.masked_where(
                (dn == 0) | (profile_rr - proton_r0[..., np.newaxis] <= 0),
                proton_likelihood_mcs)
            mip_likelihood_mcs = ma.masked_where(
                (dn == 0) | (profile_rr - muon_r0[..., np.newaxis] <= 0),
                mip_likelihood_mcs)
            muon_likelihood_dqdx = ma.masked_where(
                (dn == 0) | (profile_rr - muon_r0[..., np.newaxis] <= 0),
                muon_likelihood_dqdx)
            proton_likelihood_dqdx = ma.masked_where(
                (dn == 0) | (profile_rr - proton_r0[..., np.newaxis] <= 0),
                proton_likelihood_dqdx)
            mip_likelihood_dqdx = ma.masked_where(
                (dn == 0) | (profile_rr - muon_r0[..., np.newaxis] <= 0),
                mip_likelihood_dqdx)

            # get end point (for stopping muon assumption)
            profile_rr = ma.array(profile_rr - muon_r0[..., np.newaxis], mask=(profile_n <= 0))
            i_stop = np.argmin(np.abs(profile_rr), axis=-1)[..., np.newaxis, np.newaxis]
            end_pt = np.take_along_axis(pos, i_stop, axis=-2)

            # correct for rounding error
            stop_rr = np.take_along_axis(profile_rr, i_stop[...,0], axis=-1)[...,np.newaxis]
            n = end_pt - np.take_along_axis(pos, np.clip(i_stop-1,0,None), axis=-2)
            n /= np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-15, None)
            end_pt_corr = stop_rr * n

            # check if endpoint in fiducial volume
            end_pt_in_fid = resources['Geometry'].in_fid(
                end_pt.reshape(-1, 3), cathode_fid=self.cathode_fid_cut, field_cage_fid=self.fid_cut, anode_fid=self.anode_fid_cut)
            end_pt_in_fid = end_pt_in_fid.reshape(tracks.shape[0])

            # estimate residual range for each hit
            hit_prof_rr = profile_rr.max(axis=-1, keepdims=True) - hit_prof_s     
            
            # calculate "additional" energy (all energy not associated to the parent muon) assuming nominal michel dE/dx
            q_sum = hit_q.sum(axis=-1) - ma.array(dq, mask=(np.around(profile_rr/self.profile_dx) * self.profile_dx < 0) | (profile_n <= 0)).sum(axis=-1)
            michel_dedx = resources['ParticleData'].landau_peak(50 * units.MeV, resources['ParticleData'].e_mass, resources['Geometry'].pixel_pitch)
            e = q_sum * resources['LArData'].ionization_w / resources['LArData'].ionization_recombination(michel_dedx)

            # calculate active distance to exit detector
            #active_proj_length = self.extrapolated_intersection(pos[...,0,:], end_pt.reshape(-1,3))

            # apply a hit density correction
            #profile_dqdx = profile_dqdx * ds / ma.maximum(ds - self.density_dx_correction(profile_rr, *self.density_dx_correction_params), resources['Geometry'].pixel_pitch) * (dn > 0)
            # apply a curvature correction
            profile_rr = profile_rr * self.curvature_rr_correction

            # find max dqdx
            max_dqdx = profile_dqdx.max(axis=-1)

            # select stopping muons
            #event_is_contained_muon = (event_is_contained & end_pt_in_fid  # stops in fiducial volume
            #                          & (e < self.remaining_e_cut)  # has additional energy consistent with a Michel or less
            #                          & (max_dqdx > self.dqdx_peak_cut)  # has a prominent dQ/dx peak
            #                          #& (ma.sum(is_stopping & ~is_near_edge, axis=-1) == 1)  # only one track stopping in fiducial volume
            #                          & (np.mean(muon_likelihood_dqdx, axis=-1)
            #                             + np.mean(muon_likelihood_mcs, axis=-1) * 0
            #                             - np.mean(proton_likelihood_dqdx, axis=-1)
            #                             - np.mean(proton_likelihood_mcs, axis=-1) * 0 < self.proton_classifier_cut)  # dQ/dx profile more consistent with stopping muon than proton
            #                          & (np.mean(muon_likelihood_dqdx, axis=-1)
            #                             + np.mean(muon_likelihood_mcs, axis=-1) * 0
            #                             - np.mean(mip_likelihood_dqdx, axis=-1)
            #                             - np.mean(mip_likelihood_mcs, axis=-1) * 0 > self.muon_classifier_cut))  # dQ/dx profile more consistent with stopping muon than MIP

            #PID score muon/proton  = (2/pi)arctan((loglikelihood muon - loglikelihood proton) / 100) close to 1 = muon, close to -1 = proton
            #print("MUON likelihood dqdx shape:", muon_likelihood_dqdx.shape)
            #print("MUON likelihood dqdx:", muon_likelihood_dqdx)
            #print("Mean MUON likelihood dqdx -1:", np.mean(muon_likelihood_dqdx, axis=-1))
            #print("Mean Muon likelihood dqdx general:", np.mean(muon_likelihood_dqdx))
            pid_muon_proton = (np.mean(muon_likelihood_dqdx, axis=-1) - (np.mean(proton_likelihood_dqdx, axis=-1)))

            #PID score mip/proton  = (2/pi)arctan((loglikelihood mip - loglikelihood proton) / 100) close to 1 = mip, close to -1 = proton
            pid_mip_proton = (np.mean(mip_likelihood_dqdx, axis=-1) - (np.mean(proton_likelihood_dqdx, axis=-1)))

            '''
            event_sel = (event_level_cuts)
                         #& (pid_muon_proton > -1.)
                         #& (pid_mip_proton > -1.))   # dQ/dx profile more consistent with stopping muon than MIP
                        #& (-np.mean(muon_likelihood_dqdx, axis=-1)
                        #    + np.mean(proton_likelihood_dqdx, axis=-1)> self.proton_classifier_cut))

            #track_nhits = tracks.ravel()['nhit'][~tracks['nhit'].mask]
            #track_length = tracks.ravel()['length'][~tracks['length'].mask]
            #track_theta = tracks.ravel()['theta'][~tracks['theta'].mask]
            #track_phi = tracks.ravel()['phi'][~tracks['phi'].mask]
            #track_q = tracks.ravel()['q'][~tracks['q'].mask]

            #passing_events = len(event_ids[event_sel])
            
            #print("Max Track length:", max_track_length)

        '''if self.is_mc and len(is_proton):
            # define true proton events contained in fid as any event with 
            # at least one proton contained in fid
            event_is_true_proton = is_proton
            #print("Shape of is_proton:", is_proton.shape)
            true_contained = np.full_like(is_proton, False)
            for i in range(len(true_contained)):
                true_xyz_start_in_fid = resources['Geometry'].in_fid(
                    true_xyz_start[i], cathode_fid=self.cathode_fid_cut, field_cage_fid=self.fid_cut, anode_fid=self.anode_fid_cut)
                true_xyz_end_in_fid = resources['Geometry'].in_fid(
                    true_xyz_end[i], cathode_fid=self.cathode_fid_cut, field_cage_fid=self.fid_cut, anode_fid=self.anode_fid_cut)
                contained = true_xyz_start_in_fid & true_xyz_end_in_fid
                true_contained[i] = bool(np.sum(contained))
                #print("Contained:", contained)
                #print("Sum contained", bool(np.sum(contained)))
            #true_contained.reshape(len(tracks))
            #print("True contained shape:", true_contained.shape)'''


        sel = np.zeros(len(tracks), dtype=self.sel_dtype)
       #print("Event selection:", event_sel)

        if len(sel):
            #print("Selection identified:", str(len(sel))+"/32")
            sel['sel'] = event_sel
            #print("Selected:", sel['sel'])
            sel['event_id'] = event_ids
            sel['hip'] = event_sel #((pid_muon_proton > -1.)& (pid_mip_proton > -1.))
            sel['nhits_over_thresh'] = max_hit_charge
            sel['pdg_id'] = np.zeros(1000)
            #sel['muon_loglikelihood_mean'] = np.mean(muon_likelihood_mcs, axis=-1) * 0 + np.mean(muon_likelihood_dqdx, axis=-1)
            #sel['proton_loglikelihood_mean'] = np.mean(proton_likelihood_mcs, axis=-1) * 0 + np.mean(proton_likelihood_dqdx, axis=-1)
            #sel['mip_loglikelihood_mean'] = np.mean(mip_likelihood_mcs, axis=-1) * 0 + np.mean(mip_likelihood_dqdx, axis=-1)
            #sel['max_dqdx'] = max_dqdx
            sel['ntracks'] = event_ntracks_in_fid
            #sel['pid_muon_proton'] = pid_muon_proton
            #sel['pid_mip_proton'] = pid_mip_proton
            #sel['next_trigs'] = event_next_trigs[event_sel]
            #sel['ntracks'] = event_ntracks[event_sel]
            #sel['nhits'] = event_nhits[event_sel]
            #sel['event_charge'] = event_charge[event_sel]

        '''if self.is_mc:
            event_true_sel = np.zeros(len(tracks), dtype=self.sel_dtype)
            if len(event_true_sel):
                event_true_sel['sel'] = event_is_true_proton & true_contained
                event_true_sel['hip'] = event_is_true_proton
                event_true_sel['event_id'] = event_ids
                event_true_sel['pdg_id'] = np.concatenate((track_traj['pdgId'],np.zeros((len(tracks), 1000-len(track_traj['pdgId'][0])))), axis=-1)
                #event_true_sel['muon_loglikelihood_mean'] = ma.sum(is_muon, axis=-1) >= 1
                #event_true_sel['proton_loglikelihood_mean'] = ma.sum(is_proton & is_true_stopping, axis=-1) >= 1
                #event_true_sel['mip_loglikelihood_mean'] = ma.sum(is_muon & ~is_true_stopping, axis=-1) >= 1'''
#
        #event_tracks = np.zeros(len(track_length), dtype=self.event_tracks_dtype)

        #if len(event_tracks):
        #    event_tracks['nhits'] = track_nhits
        #    event_tracks['length'] = track_length
        #    event_tracks['theta'] = track_theta
        #    event_tracks['phi'] = track_phi
        #    event_tracks['track_q'] = track_q
        #    
        #hit_profile = np.zeros(hits.shape, dtype=self.hit_profile_dtype)
        #if len(hit_profile):
        #    hit_profile['idx'] -= 1
        #    hit_profile['idx'][~hits['id'].mask] = hit_prof_idx[~hits['id'].mask]
        #    hit_profile['rr'][~hits['id'].mask] = hit_prof_rr[~hits['id'].mask]


        # reserve data space
        sel_slice = self.data_manager.reserve_data(
            f'{self.path}/{self.sel_dset_name}', source_slice)
        #event_tracks_slice = self.data_manager.reserve_data(
        #    f'{self.path}/{self.event_tracks_dset_name}', source_slice)
        #event_hits_slice = self.data_manager.reserve_data(
        #    f'{self.path}/{self.hit_profile_dset_name}', int((~hits['id'].mask).sum()))
        if self.is_mc:
            sel_truth_slice = self.data_manager.reserve_data(
                f'{self.path}/{self.sel_truth_dset_name}',
                source_slice)

            # write
        self.data_manager.write_data(f'{self.path}/{self.sel_dset_name}',
                                         sel_slice, sel)
        if self.is_mc:
            self.data_manager.write_data(
                f'{self.path}/{self.sel_truth_dset_name}',
                sel_truth_slice, event_true_sel)
            #self.data_manager.write_data(f'{self.path}/{self.event_tracks_dset_name}',
            #                             event_tracks_slice, event_tracks)
            #self.data_manager.write_data(f'{self.path}/{self.hit_profile_dset_name}',
            #                             event_hits_slice, hit_profile[~hits['id'].mask])
            #self.data_manager.write_ref(f'{self.path}/{self.hit_profile_dset_name}',
            #        self.hits_dset_name, np.c_[event_hits_slice, hits['id'].compressed()])

            ## calculate hit positions and charge
            #hit_q = self.larpix_gain * q['q'] # convert mV -> ke
#
            ## filter out bad channel ids
            #hit_mask = (hits['px'] != 0.0) & (hits['py'] != 0.0) & ~hit_q.mask & ~hit_drift['t_drift'].mask
            #hit_q.mask = hit_q.mask | ~hit_mask
            #hit_xyz = ma.array(np.concatenate([
            #    hits['px'][..., np.newaxis], hits['py'][..., np.newaxis], 
            #    hit_drift['z'][..., np.newaxis]], axis=-1), shrink=False,\
            #    mask=np.zeros(hits['px'].shape + (3,), dtype=bool) | hit_q.mask[...,np.newaxis] | ~hit_mask[...,np.newaxis])

        #Event charge threshold selection
        #HIP, MIP selection
        #Track fitting
        #PIDA
        #Void analysis