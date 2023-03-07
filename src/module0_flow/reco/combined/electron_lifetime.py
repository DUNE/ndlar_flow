import numpy as np
import numpy.ma as ma
import os
import scipy.optimize as optimize
import scipy.stats as stats
import scipy.interpolate as interpolate
import scipy.ndimage.filters as filters
import scipy.signal as signal

from h5flow import H5FLOW_MPI
from h5flow.core import H5FlowStage, resources

from module0_flow.util.func import condense_array


class ElectronLifetimeCalib(H5FlowStage):
    '''
        Reconstructs hit positions based on drift time and T0

        If run in "generate" mode, this module will generate an electron lifetime
        value for the given run and insert into the specified .npz file

        If run in "calibration" mode, this module will apply the electron lifetime
        calibration to each hit charge and save it to a new dataset

        Parameters:
         - ``hits_dset_name``: ``str``, path to input hits dataset
         - ``charge_dset_name``: ``str``, path to input charge dataset (must have ``"q"`` field and be 1:1 with hits dataset)
         - ``drift_dset_name``: ``str``, path to input drift dataset
         - ``mode``: ``str``, one of "generate" or "calibration"
         - ``electron_lifetime_file``: ``str``, path to .npz file to update with new electron lifetime values (only applies to "generate" mode), will name with the measured unix timestamp
         - ``tracks_dset_name``: ``str``, path to input tracks dataset (only applies to generate mode)
         - ``true_segments_dset_name``: ``str``, path to input true segments dataset for each hit (only applies to generate mode for simulation)
         - ``update_result``: ``bool``, flag to add to existing calibration file, otherwise will create a new file based on the timestamp (only applies to generate mode)
         - ``calib_dset_name``: ``str``, path to output dataset (only applies to "calibration" mode)

        Important note!! ``hits_dset_name`` and ``drift_dset_name`` do double duty and have different requirements depending on the run mode: in ``"generate"`` mode these represent the hits and drift info for the hits associated with each track, whereas in ``"calibration"`` mode these are the hits and drift info for the event. So the nominal corresponding requirements field should be::

            requirements: # generate mode
              - 'combined/tracklets'
              - name: 'track_hit_drift'
                path: ['combined/trackets', 'charge/hits', 'combined/hit_drift']
              - name: 'track_hits'
                path: ['combined/tracklets', 'charge/hits']

            requirements: # calibration mode
              - name: 'combined/hit_drift'
                path: ['charge/hits', 'combined/hit_drift']
              - 'charge/hits'

        Both ``hits_dset_name`` and ``drift_dset`` are required
        in the cache.

        Requires RunData, LArData, and Geometry resource in workflow.

        ``calib`` datatype::

            id          u4,     unique identifier
            f           f4,     calibration factor (q_calib = f * q)
            q           f4,     resulting calibrated value [mV]

        The output electron lifetime file is a .npz collection containing:

         - ``ntracks``:         number of tracks used in fit
         - ``drift_bins``:      1D array of drift time bin edges
         - ``dqdx_bins``:       1D array of dQ/dx bin edges
         - ``dedx_bins``:       1D array of dE/dx bin edges (simulation only)
         - ``timestamp``:       averaged unix time across run [s]
         - ``hist``:            2D array of bin counts for track dQ/dx
         - ``dedx_hist``:       2D array of bin counts for track true dE/dx (simulation only)
         - ``lifetime``:        best fit electron lifetime
         - ``lifetime_err``:    best fit electron lifetime error
         - ``dqdx0``:           best fit dQ/dx(t=0)
         - ``dqdx0_err``:       best fit dQ/dx(t=0) error
         - ``other_fit_p``:     1D array of power law best fit parameters
         - ``other_fit_p_err``: 1D array of power lay best fit parameter errors
         - ``dqdx_p``:          2D array of peak value extracted from each drift bin for each bootstrap sample
         - ``dqdx_p_valid``:    2D boolean array, ``True`` if peak value extraction was successful

    '''
    class_version = '0.0.0'

    calib_dtype = np.dtype([
        ('id', 'u4'),
        ('f', 'f4'),
        ('q', 'f4')
    ])

    GENERATE = 'gen'
    CALIBRATE = 'calib'

    DX = 30 # mm

    default_hits_dset_name = 'charge/hits'
    default_charge_dset_name = 'charge/hits'    
    default_drift_dset_name = 'combined/hit_drift'
    default_calib_dset_name = 'combined/q_calib'
    default_tracks_dset_name = 'combined/tracklets'
    default_true_segments_dset_name = 'mc_truth/hits/tracks'

    default_drift_bins = np.linspace(0, 188, 16) # us
    default_dqdx_bins = np.linspace(0, 90, 101) # mV/mm
    default_dedx_bins = np.linspace(0, 600, 101) # keV/mm
    default_dndx_bins = np.linspace(0, 1, 101) # 1/mm

    default_update_result = True
    
    def __init__(self, **params):
        super(ElectronLifetimeCalib, self).__init__(**params)
        # set up dataset names
        self.drift_dset_name = params.get('drift_dset_name', self.default_drift_dset_name)
        self.hits_dset_name = params.get('hits_dset_name', self.default_hits_dset_name)
        self.charge_dset_name = params.get('charge_dset_name', self.default_charge_dset_name)
        self.tracks_dset_name = params.get('tracks_dset_name', self.default_tracks_dset_name)
        self.calib_dset_name = params.get('calib_dset_name', self.default_calib_dset_name)
        self.true_segments_dset_name = params.get('true_segments_dset_name', self.default_true_segments_dset_name)

        # declare run mode
        self.mode = params['mode']
        if self.mode.lower()[:len(self.GENERATE)] == self.GENERATE:
            self.mode = self.GENERATE
        elif self.mode.lower()[:len(self.CALIBRATE)] == self.CALIBRATE:
            self.mode = self.CALIBRATE
        else:
            raise RuntimeError(f'{self.mode} not a valid run mode')

        self.electron_lifetime_file = params['electron_lifetime_file'] if self.mode == self.GENERATE else None
        self.update_result = params.get('update_result', self.default_update_result)
        self.drift_bins = np.array(params.get('drift_bins', self.default_drift_bins))
        self.dqdx_bins = np.array(params.get('dqdx_bins', self.default_dqdx_bins))
        self.dedx_bins = np.array(params.get('dedx_bins', self.default_dedx_bins))
        self.dndx_bins = np.array(params.get('dndx_bins', self.default_dndx_bins))
        self.angle_bins = np.linspace(-np.pi/2, np.pi/2, 101)

        self.dqdx_hist = np.zeros((self.drift_bins.shape[0]-1, self.dqdx_bins.shape[0]-1)) if self.mode == self.GENERATE else None
        self.dedx_hist = np.zeros((self.drift_bins.shape[0]-1, self.dedx_bins.shape[0]-1)) if self.mode == self.GENERATE else None
        self.dndx_hist = np.zeros((self.drift_bins.shape[0]-1, self.dndx_bins.shape[0]-1)) if self.mode == self.GENERATE else None
        self.zen_hist = np.zeros((self.drift_bins.shape[0]-1, self.angle_bins.shape[0]-1)) if self.mode == self.GENERATE else None
        self.anode_hist = np.zeros((self.drift_bins.shape[0]-1, self.angle_bins.shape[0]-1)) if self.mode == self.GENERATE else None        

        self.timestamp = None
        self.timestamp_n = 0

        self.ntracks = 0

    def init(self, source_name):
        super(ElectronLifetimeCalib, self).init(source_name)

        if self.mode == self.CALIBRATE:
            # create calib dset            
            self.data_manager.create_dset(self.calib_dset_name, self.calib_dtype)

            # create hit -> calib refs
            self.data_manager.create_ref(self.hits_dset_name, self.calib_dset_name)

            # set metadata
            self.data_manager.set_attrs(self.calib_dset_name,
                                        classname=self.classname,
                                        class_version=self.class_version,
                                        source_dset=source_name,
                                        hits_dset=self.hits_dset_name,
                                        charge_dset=self.charge_dset_name,
                                        drift_dset=self.drift_dset_name,
                                        mode=self.mode)

    @staticmethod
    def weighted_percentile(values, weights, p=0.5):
        ''' Finds the percentile `p` of `values` given weights `weights` along the last axis using a linear interpolation of the CDF, requires values strictly increasing along last axis '''
        cdf = np.cumsum(weights, axis=-1) / weights.sum(axis=-1, keepdims=True)

        # find local points
        i_sort = np.argsort(np.abs(cdf - p), axis=-1)[..., :2]
        cdf = np.take_along_axis(cdf, i_sort, axis=-1)
        values = np.take_along_axis(values, i_sort, axis=-1)

        # order by position
        i_sort = np.argsort(i_sort, axis=-1)
        cdf = np.take_along_axis(cdf, i_sort, axis=-1)
        values = np.take_along_axis(values, i_sort, axis=-1)

        # for locally flat, just use central value
        mask_flat = cdf[..., -1] == cdf[..., 0]
        numer = np.where(mask_flat, values[..., -1] - values[..., 0], 1)
        denom = np.where(mask_flat, cdf[..., -1] - cdf[..., 0], 1)
        step = np.where(mask_flat, p - cdf[..., 0], values[..., 1] - values[..., 0])
        
        return values[..., 0] + numer / denom * step
            
    @staticmethod
    def f_poly(dqdx, *args):
        ''' Polynomial function of arbitrary complexity (``= args[0] + args[1] * dqdx + args[2] * dqdx^2 + ...``)'''
        return sum([args[i] * (dqdx)**i for i in range(len(args))])

    @staticmethod
    def langau(dqdx, norm, mpv, eta, sigma, A):
        import pylandau
        return norm * pylandau.langau(dqdx, mpv, eta, sigma, np.abs(A))

    @staticmethod
    def f_decay(t, max_, tau, power, power_scale):
        ''' Power-law modified exponential to account for electron lifetime and signal truncation close to anode '''
        v = np.power(np.abs(t / power_scale), power)
        v = 1 / (v + 1)
        # enforce truncation constrained to 0 -> 1
        rescale_offset = v.min()
        rescale_norm = 1 - rescale_offset
        v -= rescale_offset
        v /= rescale_norm
        v[t <= 0] = 1
        return max_ * np.exp(-t/tau) * (1 - v)

    def finish(self, source_name):
        if self.mode == self.GENERATE:
            if H5FLOW_MPI:
                hists = self.comm.gather(self.dqdx_hist)
                self.dqdx_hist = np.sum(hists, axis=0)

                hists = self.comm.gather(self.dedx_hist)
                self.dedx_hist = np.sum(hists, axis=0)

                hists = self.comm.gather(self.dndx_hist)
                self.dndx_hist = np.sum(hists, axis=0)                

                hists = self.comm.gather(self.zen_hist)
                self.zen_hist = np.sum(hists, axis=0)

                hists = self.comm.gather(self.anode_hist)
                self.anode_hist = np.sum(hists, axis=0)                

                ntracks = self.comm.gather(self.ntracks)
                self.ntracks = np.sum(ntracks)

            if self.rank == 0:
                if self.update_result == True and os.path.exists(self.electron_lifetime_file):
                    self.dqdx_hist = np.load(self.electron_lifetime_file)['hist'] + self.dqdx_hist
                    self.dedx_hist = np.load(self.electron_lifetime_file)['dedx_hist'] + self.dedx_hist
                    self.dndx_hist = np.load(self.electron_lifetime_file)['dndx_hist'] + self.dndx_hist                    
                    self.zen_hist = np.load(self.electron_lifetime_file)['zen_hist'] + self.zen_hist
                    self.anode_hist = np.load(self.electron_lifetime_file)['anode_hist'] + self.anode_hist                    
                    old_ntracks = np.load(self.electron_lifetime_file)['ntracks']
                
                # set up fits
                n_poly_terms = 5
                bootstrap_count = 30
                drift_bin_center = (self.drift_bins[:-1] + self.drift_bins[1:])/2
                dqdx_bin_center = (self.dqdx_bins[:-1] + self.dqdx_bins[1:])/2
                dqdx_p = np.zeros((self.dqdx_hist.shape[0], bootstrap_count))
                dqdx_p_valid = np.zeros((self.dqdx_hist.shape[0], bootstrap_count))

                # fit each drift time bin
                for ibin in range(len(self.drift_bins)-1):
                    print(f'drift time: {self.drift_bins[1:][ibin]:0.02f}us')
                    mask = self.dqdx_hist[ibin] > 0
                    #if np.sum(mask) < 4:
                    if np.sum(self.dqdx_hist[ibin]) < 5000:
                        print(f'\t*** {ibin} not enough non-zero bins to fit ***')
                        continue

                    # scale factor to increase errors on each bootstrap fit in order to account for any systematic biases in fit
                    systematic_factor = 1

                    # calculate bootstrap errors
                    for ibstrp in range(bootstrap_count):
                        # also includes the adaptive systematic factor
                        dqdx_hist = (self.dqdx_hist[ibin] + np.random.normal(size=self.dqdx_hist[ibin].shape, loc=0, scale=systematic_factor * np.sqrt(self.dqdx_hist[ibin].clip(1,None)))).clip(0, None) 
                        try:
                            # apply boostrap statistical fluctuations assuming bins random normally distributed
                            # use +/- 1 sigma region around each peak for polynomial fit
                            #max_ = self.weighted_percentile(dqdx_bin_center, dqdx_hist, 0.84)
                            #min_ = self.weighted_percentile(dqdx_bin_center, dqdx_hist, 0.16)
                            min_ = self.weighted_percentile(dqdx_bin_center, dqdx_hist, 0.05)
                            max_ = 2 * self.weighted_percentile(dqdx_bin_center, dqdx_hist, 0.5) - min_
                            #max_ = dqdx_bin_center.max()
                            #min_ = dqdx_bin_center.min()
                            fit_mask = mask & (dqdx_bin_center >= min_) & (dqdx_bin_center <= max_) & (dqdx_hist > 0)
                            
                            # perform the polynomial fit to the peak
                            #fit_func = self.f_poly                            
                            #p0 = (dqdx_hist[fit_mask].mean(),) + (0,) * (n_poly_terms - 1)
                            fit_func = self.langau
                            p0 = (dqdx_hist.sum()/7.5, self.weighted_percentile(dqdx_bin_center, dqdx_hist, 0.5)/1.05, 1, 1.8, 0.5)
                            err = systematic_factor * np.sqrt(dqdx_hist[fit_mask]).clip(1,None)
                            p, cov = optimize.curve_fit(fit_func, dqdx_bin_center[fit_mask],
                                                        dqdx_hist[fit_mask],
                                                        p0=p0, sigma=err, bounds=([0, 0, 0, 0, 0], [np.inf, 100, 10, 10, 10]))

                            chi2 = ((dqdx_hist[fit_mask] - fit_func(dqdx_bin_center[fit_mask], *p))**2 / err**2).sum()
                            ndf = fit_mask.sum() - len(p)

                            # update the systematic factor using rolling average
                            systematic_factor = (bootstrap_count-1)/bootstrap_count * systematic_factor + 1/bootstrap_count * chi2 / ndf * systematic_factor
                            #for iparam,param in enumerate(p):
                            #    print(f'\t  {iparam}: {param:0.03f}')

                            # find peak for use in lifetime fit using a spline of the best fit polynomial
                            spline = interpolate.CubicSpline(dqdx_bin_center[fit_mask], fit_func(dqdx_bin_center[fit_mask], *p))
                            extrema = spline.derivative(1).roots(extrapolate=False)
                            maximum = extrema[np.argmax(spline(extrema))]

                            #dqdx_p[ibin,ibstrp] = maximum
                            dqdx_p[ibin,ibstrp] = p[1]
                            dqdx_p_valid[ibin,ibstrp] = True

                            print(f'\t{ibstrp} chi2/ndf = {chi2:0.02f}/{ndf:d} ({chi2/ndf:0.04f}) [f={systematic_factor:0.2f}], '
                                  f'peak = {dqdx_p[ibin,ibstrp]:0.04f} mV/mm')
                            #print(p0)
                            #print(p)

                            if ibstrp in (bootstrap_count-1, 0) and False:
                                import matplotlib.pyplot as plt
                                plt.figure()
                                plt.errorbar(dqdx_bin_center[fit_mask], dqdx_hist[fit_mask], yerr=err, color='k', fmt='.')
                                plt.plot(dqdx_bin_center[fit_mask], fit_func(dqdx_bin_center[fit_mask], *p0), label='init', color='r', ls='--')                                
                                plt.plot(dqdx_bin_center[fit_mask], spline(dqdx_bin_center[fit_mask]), label='fit', color='r')
                                plt.axvline(maximum, color='r', ls=':', label='peak')
                                plt.legend()
                                plt.xlabel('dQ/dx [mV/mm]')
                                plt.show()
                                
                        except Exception as e:
                            print(f'\t{ibstrp} ERROR:', e)

                    print(f'\tpeak: {dqdx_p[ibin].mean():0.03f} +/- {dqdx_p[ibin].std():0.03f} mV/mm')

                print('Final result')
                result = dict(
                    drift_bins=self.drift_bins,
                    dqdx_bins=self.dqdx_bins,
                    dedx_bins=self.dedx_bins,
                    dndx_bins=self.dndx_bins,
                    angle_bins=self.angle_bins,
                    timestamp=self.timestamp,
                    hist=self.dqdx_hist,
                    dedx_hist=self.dedx_hist,
                    dndx_hist=self.dndx_hist,
                    zen_hist=self.zen_hist,
                    anode_hist=self.anode_hist,
                    dqdx_p=dqdx_p,
                    dqdx_p_valid=dqdx_p_valid,
                )
                    
                print(f'total selected tracks = {self.ntracks}')
                if self.update_result == True and os.path.exists(self.electron_lifetime_file):
                    print(f'previous tracks = {old_ntracks}')
                    self.ntracks += old_ntracks
                result['ntracks'] = self.ntracks

                # fit the electron lifetime
                try:
                    # skip the last drift time bin to avoid dQ/dx for portion of track that crosses cathode
                    subset = slice(1,-1)

                    # create fit data
                    peak_fit = ma.array(dqdx_p, mask=~(dqdx_p_valid.astype(bool))).mean(axis=-1)
                    peak_err = ma.array(dqdx_p, mask=~(dqdx_p_valid.astype(bool))).std(axis=-1)                                        
                    # if polynomial fit was bad, apply huge error bars
                    peak_err = peak_err.filled(peak_fit.max())
                    peak_err[peak_err == 0] = peak_fit.max()

                    # do drift time response fit
                    lt_pnames = ('dQ/dx(0) [mV/mm]','lifetime [us]','decay power','decay scale [us]')
                    p0 = (peak_fit[1], 3e3, 2, 2)
                    p, cov = optimize.curve_fit(self.f_decay, drift_bin_center[subset],
                                                peak_fit[subset],
                                                p0=p0, sigma=peak_err[subset],
                                                bounds=[(0,)*len(p0), (np.inf,)*len(p0)],
                                                absolute_sigma=True)
                    perr = np.sqrt(np.diag(cov))
                    chi2 = ((peak_fit[subset] - self.f_decay(drift_bin_center[subset], *p))**2 / peak_err[subset]**2)
                    ndf = len(peak_fit[subset]) - len(lt_pnames)
                    print(f'included bins [us]: {drift_bin_center[subset].astype(int)}')
                    print(f'bin-wise chi2: {chi2}')
                    print(f'chi2/ndf = {chi2.sum():0.02f}/{ndf:d} ({chi2.sum()/ndf:0.04f})')

                    for iparam,p_name in enumerate(lt_pnames):
                        print(f'{p_name}: {p[iparam]:0.03f} +/- {perr[iparam]}')

                    result['lifetime'] = p[1]
                    result['lifetime_err'] = perr[1]
                    result['dqdx0'] = p[0]
                    result['dqdx0_err'] = perr[0]
                    result['other_fit_p'] = p[2:]
                    result['other_fit_p_err'] = perr[2:]
                except Exception as e:
                    print(f'ERROR: {e}')
                finally:
                    # save results to an output file
                    output_filename = (self.electron_lifetime_file[:-3]+f'{int(self.timestamp)}.npz'
                                       if self.update_result == False
                                       else self.electron_lifetime_file)
                    np.savez_compressed(output_filename, **result)
                    

    def run(self, source_name, source_slice, cache):
        super(ElectronLifetimeCalib, self).run(source_name, source_slice, cache)

        # fetch data
        events = cache[source_name]
        hits = cache[self.hits_dset_name]
        q = cache[self.charge_dset_name]['q']
        q = q.reshape(hits.shape)
        drift = cache[self.drift_dset_name]
        drift = drift.reshape(hits.shape)
        tick_size = resources['RunData'].crs_ticks

        if self.timestamp is None:
            self.timestamp = events['unix_ts'].mean()
            self.timestamp_n += events.shape[0]
        else:
            old_n = self.timestamp_n
            self.timestamp_n = events.shape[0] + self.timestamp_n
            self.timestamp = (events['unix_ts'].sum() + self.timestamp * old_n) / self.timestamp_n
        
        # apply calibration to hits
        if self.mode == self.CALIBRATE:
            v_drift = resources['LArData'].v_drift
            electron_lifetime = resources['LArData'].electron_lifetime(events['unix_ts'])[0]
            tpc_regions = np.array(resources['Geometry'].regions)
            max_drift = np.max(np.abs(np.diff(tpc_regions, axis=1))[:,:,2])
            max_drift = max_drift * 1.02
            # allows a small fudge factor near the cathode to apply the correct calibration
            # even if hits are slightly late (i.e. in the case of multiple self-triggers on
            # a channel from a track crossing the cathode)

            f = ma.masked_where(
                drift['t_drift'].mask,
                np.where(
                    (drift['t_drift'] >= 0) & (drift['t_drift'] * tick_size * v_drift <= max_drift),
                    np.exp((drift['t_drift'] * tick_size) / electron_lifetime[:,np.newaxis]),
                    1)
                )
            q = f * q

            # save data
            calib_array = np.empty(hits['id'].compressed().shape, dtype=self.calib_dtype)
            calib_array['q'] = q.compressed()
            calib_array['f'] = f.compressed()

            calib_slice = self.data_manager.reserve_data(self.calib_dset_name, len(calib_array))
            calib_array['id'] = np.r_[calib_slice]
            self.data_manager.write_data(self.calib_dset_name, calib_slice, calib_array)

            # save refs
            ref = np.c_[hits['id'].compressed(), calib_slice]
            self.data_manager.write_ref(self.hits_dset_name, self.calib_dset_name, ref)

        # fill histogram for calibration
        elif self.mode == self.GENERATE:
            if events is None or events.shape[0] == 0:
                return
            
            # grab tracks from file
            tracks = cache[self.tracks_dset_name]
            
            # apply track quality selection
            theta_anode = np.arctan2(tracks['start'][...,2] - tracks['end'][...,2], np.linalg.norm(tracks['start'][...,:2] - tracks['end'][...,:2], axis=-1))
            theta_zen = np.arctan2(np.linalg.norm(tracks['start'][...,[0,2]] - tracks['end'][...,[0,2]], axis=-1), tracks['start'][...,1] - tracks['end'][...,1])
            track_mask = tracks['length'] > 100 # mm, >10cm tracks
            track_mask = track_mask & (events['n_ext_trigs'] == 2)[:,np.newaxis] # only one light trigger
            track_mask = track_mask & (tracks['start'][...,2] * tracks['end'][...,2] > 0) # only one active TPC
            track_mask = track_mask & (theta_zen < 0.3) & (theta_anode < 0.3) # subselect angle to reduce bias

            # apply hit quality selection
            xyz = np.c_[hits['px'].ravel(), hits['py'].ravel(), drift['z'].ravel()]
            hit_mask = resources['Geometry'].in_fid(xyz) # require hits to be contained in detector (remove noise)
            hit_mask = hit_mask.reshape(hits['id'].shape) & ~hits.mask['id']
            hit_mask = hit_mask & track_mask[...,np.newaxis]

            # bin hits onto track length
            #track_dir = tracks['start'] - tracks['end']
            track_dir = tracks['trajectory'][...,0,:] - tracks['trajectory'][...,-1,:]
            track_dir /= np.linalg.norm(track_dir, keepdims=True, axis=-1).clip(1e-300,None)
            
            #hit_s = xyz.reshape(tracks.shape + (-1,3)) - tracks['end'][...,np.newaxis,:]
            hit_s = xyz.reshape(tracks.shape + (-1,3)) - tracks['trajectory'][...,np.newaxis,-1,:]
            hit_s = np.sum(hit_s * track_dir[...,np.newaxis,:], axis=-1)
            
            i_track = np.arange(np.prod(tracks.shape))
            i_track = np.broadcast_to(i_track[:,np.newaxis], i_track.shape + hits.shape[-1:])

            max_length = tracks['length'].filled(self.DX).max()
            s_bins,dx = np.linspace(0, max_length, int(np.ceil((max_length)/self.DX))+1, retstep=True)

            hit_mask = hit_mask & (hit_s > 0) & (hit_s < tracks['length'][...,np.newaxis])
            
            dq = np.histogramdd((i_track.ravel(), hit_s.ravel()),
                                weights=(q * hit_mask).ravel(),
                                bins=(np.arange(i_track[-1,-1] + 2), s_bins))[0]
            t_drift = np.histogramdd((i_track.ravel(), hit_s.ravel()),
                                     weights=(drift['t_drift'] * hit_mask * tick_size).ravel(),
                                     bins=(np.arange(i_track[-1,-1] + 2), s_bins))[0]            
            dn = np.histogramdd((i_track.ravel(), hit_s.ravel()),
                                weights=hit_mask.ravel(),
                                bins=(np.arange(i_track[-1,-1] + 2), s_bins))[0]

            # calculate dq/dx
            dqdx = dq/dx
            dqdx_mask = track_mask.ravel()[:,np.newaxis] & (dn > 0) #= self.DX / (np.sqrt(3) * resources['Geometry'].pixel_pitch))

            # exclude the end-points (which can be biased)
            dqdx_mask[:,0] = False
            np.put_along_axis(dqdx_mask, dqdx_mask.shape[-1] - 1 - np.argmax(dn[...,::-1] > 0, axis=-1)[:,np.newaxis], False, axis=-1)

            # calculate drift
            dqdx_drift = t_drift / dn.clip(1, None)

            # fill dq/dx histogram
            #self.dqdx_hist = self.dqdx_hist + np.histogramdd(
            #    (dqdx_drift[dqdx_mask].ravel(), dqdx[dqdx_mask].ravel()),
            #    bins=(self.drift_bins, self.dqdx_bins))[0]

            # estimate instead using just track information
            #track_dqdx = tracks['dq'] / np.linalg.norm(tracks['dx'], axis=-1).clip(0.1, None)
            #track_pos = (tracks['trajectory'][...,1:,:] + tracks['trajectory'][...,:-1,:])/2
            #track_z = track_pos[..., 2]
            #track_drift = np.abs(track_z) - np.array(resources['Geometry'].regions)[...,2].max()
            #track_drift = track_drift / resources['LArData'].v_drift
            #self.dqdx_hist = self.dqdx_hist + np.histogramdd(
            #    (track_drift.compressed(), track_dqdx.compressed()),
            #    bins=(self.drift_bins, self.dqdx_bins))[0]

            ## estimate instead using just hit information

            # collect all hits in event
            hits = hits.reshape(hits.shape[0:1] + (-1,))
            initial_mask = hits.mask['id'].copy()
            hits = condense_array(hits, initial_mask)
            xyz = np.c_[condense_array(xyz[...,0].reshape(initial_mask.shape), initial_mask).ravel(),
                        condense_array(xyz[...,1].reshape(initial_mask.shape), initial_mask).ravel(),
                        condense_array(xyz[...,2].reshape(initial_mask.shape), initial_mask).ravel()]
            xyz = xyz.reshape(hits.shape + (3,))

            # calculate pairwise-distance
            hit_dxyz = np.linalg.norm(xyz[...,:,np.newaxis,:] - xyz[...,np.newaxis,:,:], axis=-1)
            if hit_dxyz.size == 0:
                return
            # find largest separation between hits
            hit_imax_xyz = np.argmax(hit_dxyz * ~hits.mask['id'][...,np.newaxis,:] * (hit_dxyz < self.DX), axis=-1)
            hit_max_xyz = np.take_along_axis(xyz[...,np.newaxis,:,:], hit_imax_xyz[...,np.newaxis,np.newaxis], axis=-2)
            hit_max_dxyz = np.max(np.linalg.norm(hit_max_xyz - xyz[...,np.newaxis,:,:], axis=-1) * (hit_dxyz < self.DX), axis=-1)
            hit_max_dxyz = hit_max_dxyz + resources['Geometry'].pixel_pitch

            # only use hits within detector bulk and have good t0
            hit_mask = ~hits.mask['id']
            hit_mask = hit_mask & (events['n_ext_trigs'][:,np.newaxis] == 2)
            hit_mask = hit_mask & resources['Geometry'].in_fid(xyz.reshape(-1,3), anode_fid=self.DX, field_cage_fid=self.DX, cathode_fid=self.DX).reshape(hits.shape)

            # sum all hits in local region
            hit_dq = np.sum(hits['q'][...,np.newaxis,:] * ~hits.mask['id'][...,np.newaxis,:] * (hit_dxyz < self.DX) * (hit_dxyz > 0), axis=-1)
            hit_dn = np.sum(~hits.mask['id'][...,np.newaxis,:] * (hit_dxyz < self.DX) * (hit_dxyz > 0), axis=-1) + 1
            hit_dq += hits['q']
            hit_mask = hit_mask & (hit_dn >= np.floor(2 * self.DX / 4.434 - 1))
            hit_dq = np.where(hit_mask, hit_dq, -1)
            hit_dn = np.where(hit_mask, hit_dn, -1)

            # use central hit for drift estimate
            hit_drift = np.where(hit_mask, condense_array(drift['t_drift'].reshape(drift.shape[0],-1), initial_mask) * tick_size, -1)

            self.dqdx_hist = self.dqdx_hist + np.histogramdd(
                #(hit_drift.ravel(), hit_dq.ravel() / self.DX / 2),
                (hit_drift.ravel(), hit_dq.ravel() / hit_max_dxyz.ravel()),
                bins=(self.drift_bins, self.dqdx_bins))[0]

            self.ntracks += np.sum(dqdx_mask.any(axis=-1))

            # fill angle histograms
            self.anode_hist = self.anode_hist + np.histogramdd(
                (dqdx_drift[dqdx_mask].ravel(), np.broadcast_to(theta_anode.reshape(dqdx_mask.shape[:-1] + (1,)), dqdx_mask.shape)[dqdx_mask].ravel()),
                bins=(self.drift_bins, self.angle_bins))[0]
            self.zen_hist = self.zen_hist + np.histogramdd(
                (dqdx_drift[dqdx_mask].ravel(), np.broadcast_to(theta_zen.reshape(dqdx_mask.shape[:-1] + (1,)), dqdx_mask.shape)[dqdx_mask].ravel()),
                bins=(self.drift_bins, self.angle_bins))[0]

            self.dndx_hist = self.dndx_hist + np.histogramdd(
                #(dqdx_drift[dqdx_mask].ravel(), dn[dqdx_mask].ravel() / dx),
                (hit_drift.ravel(), hit_dn.ravel() / hit_max_dxyz.ravel()),                
                bins=(self.drift_bins, self.dndx_bins))[0]
            
            if resources['RunData'].is_mc and self.true_segments_dset_name in cache:
                #true_segments = cache[self.true_segments_dset_name].reshape(hits.shape + (-1,))
                
                # fill true dE/dx histgram
                #i_track = np.broadcast_to(i_track.reshape(true_segments.shape[:-1] + (1,)), true_segments.shape)
                #hit_s = np.broadcast_to(hit_s.reshape(true_segments.shape[:-1] + (1,)), true_segments.shape)
                #hit_mask = np.broadcast_to(hit_mask.reshape(true_segments.shape[:-1] + (1,)), true_segments.shape)
                #hit_mask = hit_mask & ~(true_segments.mask['trackID']) & (np.abs(true_segments['pdgId']) == 13)

                #dedx = np.histogramdd((i_track.ravel(), hit_s.ravel()),
                #                      weights=(true_segments['dEdx'] * hit_mask).ravel(),
                #                      bins=(np.arange(i_track.max() + 2), s_bins))[0]
                #dn = np.histogramdd((i_track.ravel(), hit_s.ravel()),
                #                      weights=hit_mask.ravel(),
                #                      bins=(np.arange(i_track.max() + 2), s_bins))[0]
                #dedx = dedx / dn.clip(1, None) # average over all true segments contributing to a hit
                #self.dedx_hist = self.dedx_hist + np.histogramdd(
                #    (dqdx_drift[dqdx_mask].ravel(), dedx[dqdx_mask].ravel()),
                #    bins=(self.drift_bins, self.dedx_bins))[0]
                pass
