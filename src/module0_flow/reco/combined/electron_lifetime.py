import numpy as np
import scipy.optimize as optimize
import scipy.stats as stats
import scipy.interpolate as interpolate
import scipy.ndimage.filters as filters
import scipy.signal as signal

from h5flow import H5FLOW_MPI
from h5flow.core import H5FlowStage, resources


class ElectronLifetimeCalib(H5FlowStage):
    '''
        Reconstructs hit positions based on drift time and T0

        If run in "generate" mode, this module will generate an electron lifetime
        value for the given run and insert into the specified .npz file

        If run in "calibration" mode, this module will apply the electron lifetime
        calibration to each hit charge and save it to a new dataset

        Parameters:
         - ``hits_dset_name``: ``str``, path to input hits dataset
         - ``drift_dset_name``: ``str``, path to input drift dataset
         - ``mode``: ``str``, one of "generate" or "calibration"
         - ``electron_lifetime_file``: ``str``, path to .npz file to update with new electron lifetime values (only applies to "generate" mode), will name with the measured unix timestamp
         - ``tracks_dset_name``: ``str``, path to input tracks dataset (only applies to generate mode)
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

         - ``drift_bins``:      1D array of drift time bin edges
         - ``dqdx_bins``:       1D array of dQ/dx bin edges
         - ``timestamp``:       averaged unix time across run [s]
         - ``hist``:            2D array of bin counts for track dQ/dx
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
    default_drift_dset_name = 'combined/hit_drift'
    default_calib_dset_name = 'combined/q_calib'
    default_tracks_dset_name = 'combined/tracklets'

    default_drift_bins = np.linspace(0,188,16) # us
    default_dqdx_bins = np.linspace(0,90,101) # mV/mm

    
    def __init__(self, **params):
        super(ElectronLifetimeCalib, self).__init__(**params)
        # set up dataset names
        self.drift_dset_name = params.get('drift_dset_name', self.default_drift_dset_name)
        self.hits_dset_name = params.get('hits_dset_name', self.default_hits_dset_name)
        self.tracks_dset_name = params.get('tracks_dset_name', self.default_tracks_dset_name)
        self.calib_dset_name = params.get('calib_dset_name', self.default_calib_dset_name)

        # declare run mode
        self.mode = params['mode']
        if self.mode.lower()[:len(self.GENERATE)] == self.GENERATE:
            self.mode = self.GENERATE
        elif self.mode.lower()[:len(self.CALIBRATE)] == self.CALIBRATE:
            self.mode = self.CALIBRATE
        else:
            raise RuntimeError(f'{self.mode} not a valid run mode')

        self.electron_lifetime_file = params['electron_lifetime_file'] if self.mode == self.GENERATE else None
        self.drift_bins = np.array(params.get('drift_bins', self.default_drift_bins))
        self.dqdx_bins = np.array(params.get('dqdx_bins', self.default_dqdx_bins))

        self.dqdx_hist = np.zeros((self.drift_bins.shape[0]-1, self.dqdx_bins.shape[0]-1)) if self.mode == self.GENERATE else None

        self.timestamp = None
        self.timestamp_n = 0


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
    def f_decay(t, max_, tau, power, power_scale):
        ''' Power-law modified exponential to account for electron lifetime and signal truncation close to anode '''
        return max_ * np.exp(-t/tau) * (1 - 1 / (1 + np.power(np.abs(t/power_scale), power)))

    def finish(self, source_name):
        if self.mode == self.GENERATE:
            if H5FLOW_MPI:
                hists = self.comm.gather(self.dqdx_hist)
                self.dqdx_hist = np.sum(hists, axis=0)

            if self.rank == 0:
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
                    if np.sum(mask) < 4:
                        print(f'\t*** {ibin} not enough non-zero bins to fit ***')
                        continue                    

                    # scale factor to increase errors on each bootstrap fit in order to account for any systematic biases in fit
                    systematic_factor = 1

                    # use +/- 1 sigma region around each peak for polynomial fit
                    max_ = self.weighted_percentile(dqdx_bin_center, self.dqdx_hist[ibin], 0.84)
                    min_ = self.weighted_percentile(dqdx_bin_center, self.dqdx_hist[ibin], 0.16)
                    fit_mask = mask & (dqdx_bin_center >= min_) & (dqdx_bin_center <= max_)

                    # calculate bootstrap errors
                    for ibstrp in range(bootstrap_count):
                        # apply boostrap statistical fluctuations assuming bins random normally distributed
                        # also includes the adaptive systematic factor
                        dqdx_hist = self.dqdx_hist[ibin] + np.random.normal(size=self.dqdx_hist[ibin].shape, loc=0, scale=np.sqrt(self.dqdx_hist[ibin].clip(1,None))) * fit_mask * systematic_factor
                        try:
                            # perform the polynomial fit to the peak
                            p0 = (dqdx_hist[fit_mask].mean(),) + (0,) * (n_poly_terms - 1)
                            err = np.sqrt(dqdx_hist[fit_mask]) * systematic_factor
                            fit_func = self.f_poly
                            p, cov = optimize.curve_fit(fit_func, dqdx_bin_center[fit_mask],
                                                        dqdx_hist[fit_mask],
                                                        p0=p0, sigma=err)

                            chi2 = ((dqdx_hist[fit_mask] - fit_func(dqdx_bin_center[fit_mask], *p))**2 / err**2).sum()
                            ndf = fit_mask.sum() - len(p)
                            print(f'\t{ibstrp} chi2/ndf = {chi2:0.02f}/{ndf:d} ({chi2/ndf:0.04f}) [f={systematic_factor:0.2f}]')

                            # update the systematic factor using rolling average
                            systematic_factor = (0.75 * systematic_factor + 0.25 * chi2 / ndf)
                            #for iparam,param in enumerate(p):
                            #    print(f'\t  {iparam}: {param:0.03f}')

                            # find peak for use in lifetime fit using a spline of the best fit polynomial
                            spline = interpolate.CubicSpline(dqdx_bin_center[fit_mask], fit_func(dqdx_bin_center[fit_mask], *p))
                            extrema = spline.derivative(1).roots(extrapolate=False)
                            maximum = extrema[np.argmax(spline(extrema))]

                            dqdx_p[ibin,ibstrp] = maximum
                            dqdx_p_valid[ibin,ibstrp] = True
                                
                        except Exception as e:
                            print('\t{ibstrp} ERROR:', e)

                    print(f'\tpeak: {dqdx_p[ibin].mean():0.03f} +/- {dqdx_p[ibin].std():0.03f} mV/mm')

                # fit the electron lifetime
                try:
                    print('Final result')
                    # skip the last drift time bin to avoid dQ/dx for portion of track that crosses cathode
                    subset = slice(0,-1)

                    # create fit data
                    peak_fit = dqdx_p.mean(axis=-1) * (dqdx_p_valid.sum(axis=-1) > 1)
                    peak_err = dqdx_p.std(axis=-1) * (dqdx_p_valid.sum(axis=-1) > 3)
                    # if polynomial fit was bad, apply huge error bars
                    peak_err[peak_err == 0] = peak_fit.max()

                    # do drift time response fit
                    lt_pnames = ('dQ/dx(0) [mV/mm]','lifetime [us]','decay power','decay scale [us]')
                    p0 = (peak_fit[1], 2e3, 2, 2)
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

                    # save results to an output file
                    np.savez_compressed(self.electron_lifetime_file[:-3]+f'{int(self.timestamp)}.npz',
                                        drift_bins=self.drift_bins,
                                        dqdx_bins=self.dqdx_bins,
                                        timestamp=self.timestamp,
                                        hist=self.dqdx_hist,
                                        lifetime=p[1],
                                        lifetime_err=perr[1],
                                        dqdx0=p[0],
                                        dqdx0_err=perr[0],
                                        other_fit_p=p[2:],
                                        other_fit_p_err=perr[2:],
                                        dqdx_p=dqdx_p,
                                        dqdx_p_valid=dqdx_p_valid,
                        )
                except Exception as e:
                    print(f'ERROR: {e}')

    def run(self, source_name, source_slice, cache):
        super(ElectronLifetimeCalib, self).run(source_name, source_slice, cache)

        # fetch data
        events = cache[source_name]
        hits = cache[self.hits_dset_name]
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
            f = np.exp((drift['t_drift'] * tick_size) / (resources['LArData'].electron_lifetime(events['unix_ts'])[0])[:,np.newaxis])
            q = f * hits['q']

            calib_array = np.empty(hits['id'].compressed().shape, dtype=self.calib_dtype)
            calib_array['q'] = q.compressed()
            calib_array['f'] = f.compressed()

            # save data
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
            track_mask = tracks['length'] > 100 # mm, >10cm tracks
            track_mask = track_mask & (events['n_ext_trigs'] == 2)[:,np.newaxis] # only one light trigger

            # apply hit quality selection
            xyz = np.c_[hits['px'].ravel(), hits['py'].ravel(), drift['z'].ravel()]
            hit_mask = resources['Geometry'].in_fid(xyz) # require hits to be contained in detector (remove noise)
            hit_mask = hit_mask.reshape(hits['id'].shape) & ~hits.mask['id']
            hit_mask = hit_mask & track_mask[...,np.newaxis]

            # bin hits onto track length
            track_dir = tracks['start'] - tracks['end']
            track_dir /= np.linalg.norm(track_dir, keepdims=True, axis=-1).clip(1e-300,None)
            
            hit_s = xyz.reshape(tracks.shape + (-1,3)) - tracks['end'][...,np.newaxis,:]
            hit_s = np.sum(hit_s * track_dir[...,np.newaxis,:], axis=-1)
            
            i_track = np.arange(np.prod(tracks.shape))
            i_track = np.broadcast_to(i_track[:,np.newaxis], i_track.shape + hits.shape[-1:])

            max_length = tracks['length'].max()
            s_bins,dx = np.linspace(0, max_length, int(np.ceil(max_length/self.DX))+1, retstep=True)
            
            dq = np.histogramdd((i_track.ravel(), hit_s.ravel()),
                                weights=(hits['q'] * hit_mask).ravel(),
                                bins=(np.arange(i_track[-1,-1] + 2), s_bins))[0]
            t_drift = np.histogramdd((i_track.ravel(), hit_s.ravel()),
                                     weights=(drift['t_drift'] * hit_mask * tick_size).ravel(),
                                     bins=(np.arange(i_track[-1,-1] + 2), s_bins))[0]            
            dn = np.histogramdd((i_track.ravel(), hit_s.ravel()),
                                weights=hit_mask.ravel(),
                                bins=(np.arange(i_track[-1,-1] + 2), s_bins))[0]

            # calculate dq/dx
            dqdx = dq/dx
            dqdx_mask = track_mask.ravel()[:,np.newaxis] & (dn > int(self.DX/(np.sqrt(3) * resources['Geometry'].pixel_pitch)))

            # exclude the end-points (which can be biased)
            dqdx_mask[:,0] = False
            np.put_along_axis(dqdx_mask, dqdx_mask.shape[-1] - 1 - np.argmax(dn[...,::-1] > 0, axis=-1)[:,np.newaxis], False, axis=-1)

            # calculate drift
            dqdx_drift = t_drift / dn.clip(1, None)

            # fill dq/dx histogram
            self.dqdx_hist = self.dqdx_hist + np.histogramdd(
                (dqdx_drift[dqdx_mask].ravel(), dqdx[dqdx_mask].ravel()),
                bins=(self.drift_bins, self.dqdx_bins))[0]
