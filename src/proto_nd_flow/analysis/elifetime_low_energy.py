import numpy as np
import h5py
import os
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from datetime import datetime
from zoneinfo import ZoneInfo
from h5flow import H5FLOW_MPI
from h5flow.core import H5FlowGenerator, resources
import multiprocessing as mp
import time
import numpy as np
from math import exp
from scipy import interpolate
from numba import njit

def make_hist(Q, q_bin_edges):
    bin_contents, _ = np.histogram(Q,bins=q_bin_edges)
    error      = np.sqrt(bin_contents)
    return bin_contents, error

@njit
def bin_t_values(t_drift, q, t_edges):
    bin_idx = np.digitize(t_drift, t_edges) - 1
    q_bin_list = []

    for i in range(len(t_edges) - 1):
        in_bin = (bin_idx == i)
        q_bin = q[in_bin]
        q_bin_list.append(q_bin)

    return q_bin_list

def compare_tbin_pdfs(lifetime, t_diff_list, q_range, q_bins_fine, q_ref_template, q_KDEs):
    diff_total = 0.0

    for i in range(1,len(q_KDEs)):
        if i == 1:
            continue

        t_diff = t_diff_list[i]
        exp_factor = exp(-t_diff / lifetime)

        pdf_ref = q_ref_template(q_bins_fine / exp_factor)
        pdf_ref /= np.trapezoid(pdf_ref)
        pdf_i = q_KDEs[i]
        #pdf_i /= np.trapezoid(pdf_i)

        diff_total += np.trapezoid((pdf_ref - pdf_i)*(pdf_ref - pdf_i)) # ISE

    return diff_total

def make_kdes(q_bin_list):
    q_KDEs = []
    for i in range(len(q_bin_list)):
        q_KDEs.append(gaussian_kde(q_bin_list[i], bw_method='scott'))
    return q_KDEs

def make_hists(q_bin_list, q_bin_edges):
    q_hists = []
    for i in range(len(q_bin_list)):
        bin_contents, error = make_hist(q_bin_list[i], q_bin_edges)
        q_hists.append((bin_contents, error))
    return q_hists

def process_sample(args):
    q, t_drift, t_bin_edges, t_diff_list, lifetime_guess, lifetime_bounds, q_range, q_bins_fine, n_q_bins, smoothing_sigma, lifetime_range, q_bins_wider_range = args
    q_bin_list = bin_t_values(t_drift, q, t_bin_edges)

    q_KDEs = make_kdes(q_bin_list)
    q_ref_template = interpolate.interp1d(q_bins_wider_range, q_KDEs[1](q_bins_wider_range), 
                                kind='cubic', bounds_error=False, 
                                fill_value='extrapolate')
    
    q_KDEs = [kde(q_bins_fine) for kde in q_KDEs]
    q_KDEs = [kde / np.trapezoid(kde) for kde in q_KDEs]

    def objective(lifetime):
        return compare_tbin_pdfs(lifetime, t_diff_list, q_range, q_bins_fine, q_ref_template, q_KDEs)

    res = minimize(
        objective,
        x0=[lifetime_guess],  # initial guess
        bounds=[(lifetime_bounds[0], lifetime_bounds[1])],
        method="Nelder-Mead"
    )
    
    return res.fun, res.x[0]  # return minimized distance and tau

def bootstrap_measurement(q, t_drift, bootstrap_samples, t_bin_edges, lifetime_range, 
                          t_diff_list, lifetime_guess, lifetime_bounds, q_range, q_bins_fine, n_q_bins, smoothing_sigma, q_bins_wider_range, n_processes=3):
    """
    Parameters:
    - q: input data array
    - t_drift: input drift times array
    - n_processes: number of processes to use (default: 4)
    """
    N = len(q)
    sample_idxs = np.random.choice(N, size=(bootstrap_samples, N), replace=True)
    
    # Create a list of arguments for each iteration
    tasks = [(q[sample_idxs[i]], t_drift[sample_idxs[i]], t_bin_edges, t_diff_list, lifetime_guess, lifetime_bounds, q_range, q_bins_fine, n_q_bins, smoothing_sigma, lifetime_range, q_bins_wider_range)#, bandwidth) 
            for i in range(bootstrap_samples)]

    with mp.Pool(processes=n_processes) as pool:
        min_distance = []
        min_lifetime = []
        with tqdm(total=bootstrap_samples, desc='Bootstrap sampling') as pbar:
            for result in pool.imap_unordered(process_sample, tasks):
                min_distance.append(result[0])
                min_lifetime.append(result[1])
                pbar.update(1)
    
    return np.array(min_distance), np.array(min_lifetime)

class ELifetimeLowEnergy(H5FlowGenerator):
    """
    This script makes measurements of electron lifetime from t0-tagged low energy events. 
    The input is CL matched low energy charge clusters created by the low-energy charge event builder
    and `low_energy_charge_light_matching.py`, it should be verified for high selection purity. 
    The script itself takes no input file, those are specified in the yaml (see below).
    
    The analysis is as follows:
        - Group charge clusters into drift time bins
            - Bin close to 0 µs, time bin 0, treated as null hypothesis of no attenuation
        - Time bin 0 is compared to every other time bin with chi squared test first by scaling charge values by
          `exp(-dt/tau)`, where `dt` is the difference in time between time bins and `tau` is the electron lifetime test value,
          build charge histograms, rescale to account for normalization, and compare with a chi^2 test. 
          Electron lifetime value with the minimum chi^2 is the measurement. 
        - Confidence intervals extracting with a Bootstrapping method and Bias-corrected and accelerated (BCa) method
            - The majority of the compute time, as the measurements needs to be repeated many times (e.g. 1000).
            - BCa accounts for the resulting bias/skewness in the resulting bootstrapped chi^2 distribution.

    Parameters:
     - ``cluster_files_list``: list of paths to hdf5 files with matched clusters (one file per line)
     - ``measurement_periods_list``: list of start and stop date/times for each measurement (one per line, start/stop are comma-separated)
            - Format: YYYY:MM:DD:HH:MM:SS
            - Example: 2024-07-08 00:00:00, 2024-07-08 12:00:00
                       2024-07-08 12:00:00, 2024-07-09 00:00:00
     - ``n_q_bins``: ``int``, number of charge bins to use in the histograms
     - ``n_t_bin``: ``int``, number of chunks to break up data points into along drift time coordinate
     - ``csa_gain``: ``float``, charge pixel gain, assumed uniform across all channels [ke-/mV]
     - ``bin_factor``: ``float``, factor to multiply by bin width to change size of bins. Defaults to 1 (no scaling).
     - ``t_range``: ``list``, range of drift times to consider in calculation [µs]. Example: [10, 180]
     - ``q_range``: ``list``, range of charge values to consider in calculation [ke-]. Example: [30, 60]
     - ``lifetime_range``: ``list``, range of lifetime values to test. Example: [500, 2000]
     - ``n_lifetime_values``: ``int``, how many lifetime values to test 
     - ``vref_mv``: ``float``, pixel vref values (detector-specific) [mV]
     - ``vcm_mv``: ``float``, pixel vcm values (detector-specific) [mV]
     - ``adc_counts``: ``int``, CSA dynamic range (e.g. 2^8)
     - ``bin_smoothing``: ``bool``, boolean to enable smoothing bin counts to account for binning effects
     - ``use_gauss_filter``: ``bool``, boolean to enable Gaussian smoothing, otherwise uses default smoothing if ``bin_smoothing`` = True
     - ``gauss_filter_sigma``: ``float``, sigma to use in Gaussian kernel for bin smoothing, if ``use_gauss_filter`` = True

    """

    class_version = '0.0.0'
    default_clusters_dset_name = 'charge/clusters_matched'
    default_tau_measurements_dset_name = 'tau_measurements'
    
    def __init__(self, **params):
        super(ELifetimeLowEnergy, self).__init__(**params)
        self.tau_measurements_dset_name = params.get('tau_measurements_dset_name', self.default_tau_measurements_dset_name)
        self.clusters_dset_name = params.get('clusters_dset_name', self.default_clusters_dset_name)
        self.clusters_dset_name = self.clusters_dset_name + '/data'
        self.cluster_files_list = params.get('cluster_files_list')
        self.measurement_periods_list = params.get('measurement_periods_list')
        self.n_q_bins = params.get('n_q_bins')
        self.n_t_bins = params.get('n_t_bins')
        self.csa_gain = params.get('csa_gain')
        self.bin_factor = params.get('bin_factor', 1)
        self.t_range = params.get('t_range')
        self.q_range = params.get('q_range')
        lifetime_range = params.get('lifetime_range')
        lifetime_bin_size = params.get('lifetime_bin_size')
        self.vref_mv = params.get('vref_mv')
        self.vcm_mv = params.get('vcm_mv')
        self.adc_counts = params.get('adc_counts')
        self.lifetime_range = np.arange(lifetime_range[0], lifetime_range[1], lifetime_bin_size)
        self.bootstrap_samples = params.get('bootstrap_samples')
        self.lifetime_guess = params.get('lifetime_guess')
        self.lifetime_bounds = params.get('lifetime_bounds')
        self.q_bins_fine = np.arange(self.q_range[0], self.q_range[1], 0.1)
        self.clusters_data_path = params.get('clusters_data_path')
        self.data_type = params.get('data_type')
        self.smoothing_sigma = params.get('smoothing_sigma')
        self.bandwidth = params.get('kde_bandwidth')
        self.kde_bandwidth_path = params.get('kde_bandwidth_path')
        self.rand_seed = params.get('rand_seed')
        if self.rand_seed < 0:
            self.rand_seed = int(time.time())
        np.random.seed(seed=self.rand_seed)
        self.q_bin_edges, self.q_bin_centers, self.t_bin_edges, self.t_bin_centers = self.get_bins()
        q_bins = np.sum((self.q_bin_centers > self.q_range[0]) & ((self.q_bin_centers < self.q_range[1])))
        self.tau_measurements_dtype = np.dtype([
                ('id', '<i2'),
                ('tau_measurement_with_scan', '<f4'), 
                ('tau_measurement_distances', ('<f4', len(self.lifetime_range))), 
                ('tau_measurement_with_minimization', '<f4'), 
                ('bt_tau_lower_bound', '<f4'), 
                ('bt_tau_median', '<f4'), 
                ('bt_tau_upper_bound', '<f4'),
                ('unix_start', '<i8'), 
                ('unix_stop', '<i8'), 
                ('true_tau', '<f4'),
                ('t_bin_centers', ('<f4', self.n_t_bins+1)),
                ('bt_lifetimes', ('<f4', self.bootstrap_samples)), 
                ('bt_distances', ('<f4', self.bootstrap_samples)), 
                ('tau_test_values', ('<f4', len(self.lifetime_range))), 
                ('spline', ('<f4', len(self.lifetime_range))), 
                ('q_bins_fine', ('<f4', len(self.q_bins_fine))), 
                ('KDEs', ('<f4', (self.n_t_bins, q_bins))), 
                ('t_bin_hists', ('<f4', (self.n_t_bins, self.n_q_bins))), 
                ('t_bin_hists_err', ('<f4', (self.n_t_bins, self.n_q_bins))),
                ('q_bin_centers', ('<f4', self.n_q_bins)), 
                ('n_clusters_over_20ke', '<i4'), 
                ('n_clusters', '<i4'), 
                ('total_time', '<f4'), 
                ('rate', '<f4')])
        
    def init(self):
        super(ELifetimeLowEnergy, self).init()
        
        # initialize data objects
        self.data_manager.create_dset(self.tau_measurements_dset_name, dtype=self.tau_measurements_dtype)

        self.data_manager.set_attrs(self.tau_measurements_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version)

        if self.cluster_files_list != '':
            if not os.path.exists(self.cluster_files_list):
                raise Exception(f'Cluster files list does not exist: {self.cluster_files_list}')
            with open(self.cluster_files_list, 'r') as file:
                self.cluster_files = [line.strip() for line in file]
            if not len(self.cluster_files):
                raise Exception(f'No files found in cluster files list: {self.cluster_files_list}')
        if self.measurement_periods_list != '':
            if not os.path.exists(self.measurement_periods_list):
                raise Exception(f'Measurement periods list does not exist: {self.measurement_periods_list}')
        if self.kde_bandwidth_path != '':
            if not os.path.exists(self.kde_bandwidth_path):
                raise Exception(f'KDE bandwidth list does not exist: {self.kde_bandwidth_path}')
            
        self.meas_start_unix, self.meas_stop_unix = [], []
        self.meas_clusters, self.meas_true_tau = [], []
        self.kde_bandwidth = []
        self.clusters_data = []
        if self.data_type == 'data':
            with open(self.measurement_periods_list, 'r') as file:
                for line in file:
                    line = line.split(',')
                    if len(line) != 2:
                        raise Exception('Error reading lines of measurement periods list, check formatting')
                    start_t = line[0].strip().replace(':', '').replace('-', '').replace(' ', '')
                    stop_t = line[1].strip().replace(':', '').replace('-', '').replace(' ', '')
                    start_t = datetime.strptime(start_t, "%Y%m%d%H%M%S").replace(tzinfo=ZoneInfo("America/Chicago")).timestamp()
                    stop_t = datetime.strptime(stop_t, "%Y%m%d%H%M%S").replace(tzinfo=ZoneInfo("America/Chicago")).timestamp()
                    self.meas_start_unix.append(start_t)
                    self.meas_stop_unix.append(stop_t)
            print(f"{self.meas_start_unix=}")
            print(f"{self.meas_stop_unix=}")
            
            self.clusters_data = np.load(self.clusters_data_path)
            for timestamp in np.unique(self.clusters_data['unix_ts']):
                utc_dt = datetime.fromtimestamp(timestamp, tz=ZoneInfo("America/Chicago"))
                print(utc_dt)
        elif self.data_type == 'MC':
            clusters_data = np.load(self.clusters_data_path)
            ci = 0
            for key in list(clusters_data.keys()):
                if 'cluster' in key:
                    self.meas_clusters.append(clusters_data[key])
                    self.meas_true_tau.append(clusters_data['tau'][ci])
                    ci += 1

        self.n_iterations = len(self.meas_start_unix) if self.data_type == 'data' else len(self.meas_clusters)
        #if not self.data_type == 'data':
        #    self.kde_bandwidth = [self.bandwidth]*len(self.meas_clusters)
        self.iteration = 0
        
        self.t_diff_list = [abs(t_bin - self.t_bin_centers[0]) for t_bin in self.t_bin_centers]
        self.q_bins_wider_range = np.arange(0, self.q_range[1], 0.5)

        self.current_cluster_q = []
        self.current_cluster_t_drift = []

    def finish(self):
        super(ELifetimeLowEnergy, self).finish()
        print('finished')
        #self.input_fh.close()

    def next(self):
        """
        Process one measurement according to measurement start/stop times.
        """
        if self.iteration >= self.n_iterations:
            print('return empty')
            return H5FlowGenerator.EMPTY
        print(f'iteration = {self.iteration}')

        #kde_bandwidth = self.kde_bandwidth[self.iteration]
        if self.data_type == 'data':
            measurement_unix_span = (self.meas_start_unix[self.iteration], self.meas_stop_unix[self.iteration])
            time_mask = (self.clusters_data['unix_ts'] > measurement_unix_span[0]) & (self.clusters_data['unix_ts'] < measurement_unix_span[1]) 
            self.current_cluster_q = self.clusters_data['Q'][time_mask]
            try:
                self.current_cluster_t_drift = self.clusters_data['t_drift'][:,1][time_mask]
            except:
                self.current_cluster_t_drift = self.clusters_data['t_drift'][time_mask]
        else:
            self.current_cluster_q = self.meas_clusters[self.iteration]['Q']
            try:
                self.current_cluster_t_drift = self.meas_clusters[self.iteration]['t_drift'][:,1]
            except:
                self.current_cluster_t_drift = self.meas_clusters[self.iteration]['t_drift']

            true_tau = self.meas_true_tau[self.iteration]
        print(f"Found {len(self.current_cluster_q)} clusters in measurement range")
        #print(self.current_cluster_t_drift)
        clusters_indices = np.arange(len(self.current_cluster_q))
        fraction = 1.0
        n_samples = int(len(clusters_indices) * fraction)
        selected_indices = np.random.choice(clusters_indices, size=n_samples, replace=False)
        self.current_cluster_q = self.current_cluster_q[selected_indices]
        self.current_cluster_t_drift = self.current_cluster_t_drift[selected_indices]
        n_clusters = len(self.current_cluster_q[self.current_cluster_q > 30])

        print(f"Found {len(self.current_cluster_q[self.current_cluster_q > 30])} clusters in measurement range w/ greater than 30ke-")

        # make primary measurement w/ lifetime scan
        q_bin_list = bin_t_values(
            self.current_cluster_t_drift,
            self.current_cluster_q,
            self.t_bin_edges
        )
        
        q_hists = make_hists(q_bin_list, self.q_bin_edges)
        t_bin_hists = []
        t_bin_hists_err = []
        for q_hist in q_hists:
            bin_contents = q_hist[0]
            bin_errors = q_hist[1]
            bins_mask = (self.q_bin_centers > self.q_range[0]) & (self.q_bin_centers < self.q_range[1])
            norm_factor = np.sum(bin_contents[bins_mask])
            bin_contents = bin_contents / norm_factor
            bin_errors = bin_errors / norm_factor
            t_bin_hists.append(bin_contents)
            t_bin_hists_err.append(bin_errors)

        q_KDEs = make_kdes(q_bin_list)
        q_ref_template = interpolate.interp1d(self.q_bins_wider_range, q_KDEs[1](self.q_bins_wider_range), 
                                  kind='cubic', bounds_error=False, 
                                  fill_value='extrapolate')
        q_KDEs_1 = [kde(self.q_bins_fine) for kde in q_KDEs]
        q_KDEs_1 = [kde / np.trapezoid(kde) for kde in q_KDEs_1]
        q_KDEs_2 = [kde(self.q_bin_centers[bins_mask]) for kde in q_KDEs]
        
        distance_metric_list = []
        for lifetime in self.lifetime_range:
            distance_metric_list.append(compare_tbin_pdfs(lifetime, self.t_diff_list, self.q_range, self.q_bins_fine, q_ref_template, q_KDEs_1))
        
        primary_lifetime_measurement = self.lifetime_range[np.argmin(distance_metric_list)]

        # make primary measurement w/ minimization
        args = (self.current_cluster_q, self.current_cluster_t_drift, self.t_bin_edges, 
                                          self.t_diff_list, self.lifetime_guess, self.lifetime_bounds, self.q_range, self.q_bins_fine, self.n_q_bins, self.smoothing_sigma, self.lifetime_range, self.q_bins_wider_range)
        _, minimized_tau = process_sample(args)

        dist_min_bootstraps, bootstrap_lifetimes = bootstrap_measurement(self.current_cluster_q, self.current_cluster_t_drift, self.bootstrap_samples, 
                                                    self.t_bin_edges, self.lifetime_range, self.t_diff_list, 
                                                    self.lifetime_guess, self.lifetime_bounds, self.q_range, self.q_bins_fine, self.n_q_bins, self.smoothing_sigma, self.q_bins_wider_range, n_processes=3)

        tau_measurement_data = np.zeros(1, dtype=self.tau_measurements_dtype)
        slice = self.data_manager.reserve_data(self.tau_measurements_dset_name, 1)
        tau_measurement_data['id'] = slice.start + self.iteration 
        tau_measurement_data['tau_measurement_with_scan'] = primary_lifetime_measurement 
        tau_measurement_data['tau_measurement_distances'] = np.array(distance_metric_list)
        tau_measurement_data['tau_measurement_with_minimization'] = minimized_tau
        print(f'{minimized_tau=}')
        tau_measurement_data['bt_tau_lower_bound'] = np.percentile(bootstrap_lifetimes, 16)
        tau_measurement_data['bt_tau_median'] = np.median(bootstrap_lifetimes)
        tau_measurement_data['bt_tau_upper_bound'] = np.percentile(bootstrap_lifetimes, 84)
        if self.data_type == 'data':
            tau_measurement_data['unix_start'] = np.min(self.clusters_data['unix_ts'][time_mask]) #measurement_unix_span[0]
            tau_measurement_data['unix_stop'] = np.max(self.clusters_data['unix_ts'][time_mask]) #measurement_unix_span[1]
        tau_measurement_data['t_bin_centers'] = self.t_bin_centers
        tau_measurement_data['q_bins_fine'] = self.q_bins_fine
        tau_measurement_data['bt_lifetimes'] = bootstrap_lifetimes
        tau_measurement_data['bt_distances'] = dist_min_bootstraps
        tau_measurement_data['tau_test_values'] = self.lifetime_range
        if self.data_type == 'data':
            unique_unix_ts = np.unique(self.clusters_data['unix_ts'][time_mask])
            unique_unix_ts_diff = np.abs(np.diff(unique_unix_ts.astype('int')))
            total_time_seconds = np.sum(unique_unix_ts_diff[unique_unix_ts_diff == 1])
            tau_measurement_data['n_clusters'] = len(self.clusters_data['unix_ts'][time_mask])
            tau_measurement_data['total_time'] = total_time_seconds
            tau_measurement_data['rate'] = len(self.clusters_data['unix_ts'][time_mask]) / total_time_seconds 
        if self.data_type == 'MC':
            tau_measurement_data['true_tau'] = true_tau
        tau_measurement_data['KDEs'] = np.array(q_KDEs_2)
        tau_measurement_data['t_bin_hists'] = np.array(t_bin_hists)
        tau_measurement_data['t_bin_hists_err'] = np.array(t_bin_hists_err)
        tau_measurement_data['q_bin_centers'] = self.q_bin_centers
        tau_measurement_data['n_clusters_over_20ke'] = n_clusters
        self.data_manager.write_data(self.tau_measurements_dset_name, slice, tau_measurement_data)
        
        self.iteration += 1

    def get_bins(self):
        """
        Returns charge bin edges/centers and drift time bin edges/centers.
        Charge bins are automatically set to be as small as the approximate LSB with assumed uniform CSA gain.

        Parameters:
            - ``n_q_bins`` : ``int``, number of charge bins, does not control size of bins
            - ``q_range``: ``tuple``, charge range to consider in measurement [ke-]
            - ``n_t_bins``: ``int``, number of t_drift bins
            - ``t_range``: ``tuple``, t_drift range to consider in measurement [µs]. Example: (10, 180) 
            - ``vref_mv``: ``float``, pixel vref voltage [mV]
            - ``vcm_mv``: ``float``, pixel vcm voltage [mV]
            - ``adc_counts``: ``int``, CSA dynamic range (e.g. 2^8)
            - ``csa_gain``: ``float``, CSA gain [ke-/mV]
            - ``bin_factor``: ``float``, optional, factor to scale binsize by, if you want to make bins larger.

        Returns:
            - ``q_bin_edges``: ``array``, edges of charge bins 
            - ``q_bin_centers``: ``array``, centers of charge bins
            - ``t_bin_edges``: ``array``, edges of drift time bins 
            - ``t_bin_centers``: ``array``, centers of drift time bins
            
        """
        LSB = (self.vref_mv - self.vcm_mv)/self.adc_counts
        bin_width = LSB * self.csa_gain 
        
        q_bin_edges = np.linspace(-0.5*bin_width, self.n_q_bins * bin_width, self.n_q_bins + 1)
        #q_bin_edges = q_bin_edges[(q_bin_edges > self.q_range[0]) & (q_bin_edges < self.q_range[1])]
        q_bin_centers = 0.5 * (q_bin_edges[1:] + q_bin_edges[:-1])
        step = (self.t_range[1] - self.t_range[0])/self.n_t_bins
        t_bin_edges = np.arange(self.t_range[0], self.t_range[1]+step, step)
        t_bin_centers = t_bin_edges + (t_bin_edges[1]-t_bin_edges[0])/2 #0.5 * (t_bin_edges[:-1] + t_bin_edges[1:])
        return q_bin_edges, q_bin_centers, t_bin_edges, t_bin_centers
    