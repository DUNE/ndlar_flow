import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import scipy
from scipy.integrate import simps
from scipy.stats import poisson, norm
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
import yaml

SAMPLE_WIDTH_NS = 16
N_PHOTOELECTRONS = 2

#def remove_outliers(histogram):
#    shift_1 = histogram[1:]
#    shift_2 = histogram[:-1]
#    new_hist = shift_1 + shift_2
'''
def clean_histogram(hist, zero_run_threshold=10):
    hist = np.asarray(hist)
    zero_mask = (hist == 0)

    # Identify consecutive runs of zeros
    runs = np.flatnonzero(np.diff(np.concatenate(([0], zero_mask.view(np.int8), [0]))))
    run_starts = runs[::2]
    run_ends = runs[1::2]

    to_remove = np.zeros_like(hist, dtype=bool)

    for start, end in zip(run_starts, run_ends):
        run_length = end - start
        if run_length >= zero_run_threshold:
            # Mark the zero run
            to_remove[start:end] = True

            # Optionally remove one bin before and after the zero run if non-zero
            if start > 0 and hist[start - 1] <= 1:
                to_remove[start - 1] = True
            if end < len(hist) and hist[end] <= 1:
                to_remove[end] = True

    # Keep only the bins not marked for removal
    cleaned_hist = hist[~to_remove]
    #print('clean hist:', len(cleaned_hist))
    cleaned_indices = np.where(~to_remove)[0]
    #print('clean idx:', len(cleaned_indices))

    return cleaned_hist, cleaned_indices
'''

def clean_histogram(hist, zero_run_threshold=10):
    hist = np.asarray(hist)
    zero_mask = (hist == 0)

    # Identify leading zero run
    leading_zeros = 0
    for count in zero_mask:
        if count:
            leading_zeros += 1
        else:
            break

    # Identify trailing zero run
    trailing_zeros = 0
    for count in zero_mask[::-1]:
        if count:
            trailing_zeros += 1
        else:
            break

    to_keep = np.ones_like(hist, dtype=bool)

    # Mask leading zeros if long enough
    if leading_zeros >= zero_run_threshold:
        to_keep[:leading_zeros] = False

    # Mask trailing zeros if long enough
    if trailing_zeros >= zero_run_threshold:
        to_keep[-trailing_zeros:] = False

    cleaned_hist = hist[to_keep]
    cleaned_indices = np.where(to_keep)[0]
    return cleaned_hist, cleaned_indices


def fit_pedestal(hist_vals, bin_edges, fixed_amp=None, fixed_mu=None):
    centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Use defaults if not supplied
    if fixed_amp is None:
        fixed_amp = max(hist_vals)
        #print('Ped Amplitude:', fixed_amp)
    if fixed_mu is None:
        fixed_mu = np.average(centers, weights=hist_vals[:-1])
        #print('Ped Mu:', fixed_mu)

    # Model with fixed amplitude and mean
    def fixed_gaussian(x, sigma):
        weight = fixed_amp * (np.sqrt(2*np.pi)*sigma)
        return weight * norm.pdf(x, fixed_mu, sigma)

    # Initial guess for sigma
    var = np.sum((centers - fixed_mu)**2) / len(centers)
    sigma0 = np.sqrt(var)

    # Fit only sigma
    popt, pcov = curve_fit(fixed_gaussian, centers, hist_vals[:-1], p0=[sigma0])
    #print('p0 Pedestal:', popt)

    return fixed_amp, fixed_mu, popt[0]

def spe_model(q, A1, mu, q_spe, sigma_spe, A_lump, mu_lump, sigma_lump, A0, mu0, sigma0, q0, n_max=3):
  q = np.asarray(q)
  total = np.zeros_like(q, dtype=np.float64)
  components = []
  lobes = []

  for n in range(n_max + 1):
    if n == 0:
      # Pedestal (0 charge peak) with freely floating width and amplitude
      weight = A0 * (np.sqrt(2*np.pi)*sigma0)  # Freely floating amplitude for pedestal
      sigma_n = sigma0  # Freely floating width for pedestal
      mean = mu0  # Initialize mean for the pedestal
      component = weight * norm.pdf(q, loc=mean, scale=sigma_n)
      total += component
      components.append(component)
      #print(str(n)+' has run through')
    else:
      mean = q0 + (n - 1) * q_spe
      sigma_n = np.sqrt(sigma0**2 + (n * sigma_spe**2))
      weight = A1 * (np.sqrt(2*np.pi)*sigma_n) * poisson.pmf(n-1, mu)#* np.exp(-(n-1) / mu)
      weight_2 = weight * A_lump #* poisson.pmf(n-1, mu)
      mean_2 = q0 + ((n - 1) * q_spe) + mu_lump
      sigma_n_2 = sigma_lump
    # Gaussian distribution for each component
      component = weight * norm.pdf(q, loc=mean, scale=sigma_n) #+ weight_2 * norm.pdf(q, loc=mean_2, scale=sigma_n_2)
      component_2 = weight_2 * norm.pdf(q, loc=mean_2, scale=sigma_n_2)
      #print(str(n)+' has run through')
      total += component
      components.append(component)
      total += component_2
      lobes.append(component_2)
    #print(str(n)+' has run through')

  return total, components, lobes

def spe_peak(q, q0, A1, mu, q_spe, sigma_spe, sigma0, n_max=3):
    q = np.asarray(q)
    total = np.zeros_like(q, dtype=np.float64)
    components = []

    for n in range(n_max + 1):
        mean = q0 + (n * q_spe)
        sigma_n = np.sqrt(sigma0**2 + ((n+1) * sigma_spe**2))
        weight = A1 * (np.sqrt(2*np.pi)*sigma_n) * poisson.pmf(n, mu)#* np.exp(-(n-1) / mu)
        # Gaussian distribution for each component
        component = weight * norm.pdf(q, loc=mean, scale=sigma_n)

        total += component
        components.append(component)

    return total, components

# Fit SPE spectrum using the model
def fit_lump_spectrum(hist, bins, npe=N_PHOTOELECTRONS, A0=None, mu0=None, sigma0=None, q0=None, p0=None):
    centers = 0.5 * (bins[1:] + bins[:-1])
    #print('entering SPE fit')
    #print('Lower Bounds:',[p0[0]-100, 0.05, q0*0.7, 1, 1, 0.1*q0, 1.1*p0[3]])
    #print('Upper Bounds:',[p0[0]+100, 5, q0*1.5, 1e3, p0[0]*0.5, 0.5*q0, 1e3])
    popt, pcov = curve_fit(
        lambda q, A1, mu, q_spe, sigma_spe, A_lump, mu_lump, sigma_lump: spe_model(q, A1, mu, q_spe, sigma_spe, A_lump, mu_lump, sigma_lump, A0, mu0, sigma0, q0, npe)[0],
        centers,
        hist[:-1],
        p0=p0,
        bounds=([p0[0]-(0.25*p0[0]), 0.05, q0*0.7, 1, 0.01, 0.2*q0, 1], [p0[0]+(0.25*p0[0]), 5, q0*1.3, 1e3, 1.1, 0.7*q0, 1e3])
    )
    return popt, pcov, centers

# Fit SPE peak only
def fit_spe_spectrum(hist, bins, sigma0=None, npe=N_PHOTOELECTRONS, p0=None):
    centers = 0.5 * (bins[1:] + bins[:-1])
    #print('entering SPE fit')
    popt, pcov = curve_fit(
        lambda q, q0, A1, mu, q_spe, sigma_spe: spe_peak(q, q0, A1, mu, q_spe, sigma_spe, sigma0, npe)[0],
        centers,
        hist[:-1],
        p0=p0,
        bounds=([p0[0]-(0.25*p0[0]), p0[1]-(0.25*p0[1]), 0.1, p0[0]*0.7, 1], [p0[0]+(0.25*p0[0]), p0[1]+(0.25*p0[1]), 5, p0[0]*1.5, 1e3])
    )
    return popt, pcov, centers


# Chi-squared per degree of freedom
def chi_squared(y_obs, y_fit, dof=None, min_expected=5):
    # Only use bins with decent statistics
    mask = y_fit >= min_expected
    if np.sum(mask) < 2:
        return np.inf  # Not enough points

    residuals = y_obs[mask] - y_fit[mask]
    chi2 = np.sum(residuals**2 / y_fit[mask])
    if dof is None:
        dof = np.sum(mask) - 1  # default to N - 1
    return chi2 / dof


def analyze_spe(file_path, output_file):
    #os.makedirs(output_dir, exist_ok=True)
    results = []

    # Configuration
    #PEDESTAL_RANGE = (0, 50)
    #SIGNAL_RANGE = (60, 85)
    N_PHOTOELECTRONS = 3
    BINS = 200
    pdf_filname = output_file
    gain_outputs = np.zeros((8,64,2), dtype=np.float64)
    #with h5py.File(file_path, "r") as f
    waveform_data = np.load(file_path)
    waveforms = waveform_data[list(waveform_data.keys())[0]]
    with open('/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/AFIViewer/WaveformCalib_Mod0Estimates.yaml', 'r') as file:
        data = yaml.safe_load(file)

    # Access the gain dictionary
    gain_dict = data['params']['gain']

    with PdfPages(pdf_filname) as pdf_output:
        _, nadcs, nchans, _ = waveforms.shape
        for adc in range(nadcs):
        #for adc in range(4):
            for ch in range(nchans):
        #    for ch in range(6):
        #adc = 4
        #ch = 57
                wfs_v1 = waveforms[:,adc,ch,:]
                charge_mask = (np.mean(wfs_v1, axis=-1)!=0)
                if len(wfs_v1[charge_mask==1]) > 0:
                    if adc < 2:
                        bin_width = 10
                        qspe_est = 200
                    else:
                        bin_width = 50
                        qspe_est = 2500
                    wfs = wfs_v1[charge_mask==1]
                    del wfs_v1, charge_mask
                    #ped_integrals = np.array([integrate_region(wf, *PEDESTAL_RANGE) for wf in wfs])
                    #sig_integrals = np.array([integrate_region(wf, *SIGNAL_RANGE) for wf in wfs])
                    ped_integrals = np.trapz(wfs[:,0:50], axis=-1)
                    sig_integrals = np.trapz(wfs[:,60:85], axis=-1)
                    sig_subtracted = sig_integrals - (np.mean(wfs[:,0:50], axis=-1)*25)
                    del sig_integrals
                    combo_integrals = np.concatenate((sig_subtracted, ped_integrals), axis=0)

                    # Fit pedestal
                    #bin_width = 30
                    ped_min = np.min(ped_integrals)
                    ped_max = np.max(ped_integrals)
                    if ped_max > 1.2e4:
                        ped_max = 1.2e4
                    ped_bins_pre = np.arange(start=ped_min, stop=ped_max + bin_width, step=bin_width)
                    ped_hist_1, ped_bins_1 = np.histogram(ped_integrals, bins=ped_bins_pre) # define a bin width instead
                    ped_hist, ped_bins_idx = clean_histogram(ped_hist_1, zero_run_threshold=5)
                    ped_bins = ped_bins_1[ped_bins_idx]
                    del ped_hist_1, ped_bins_1
                    #print('ped_hist', ped_hist)
                    del ped_integrals
                    try:
                        A0, mu0, sigma0 = fit_pedestal(ped_hist, ped_bins)
                        del ped_bins
                    except: 
                        A0, mu0, sigma0 = 1, 0, 1
                        #A0_est, q0_est, sigma0_est = np.max(ped_hist), popt_ped[1], np.clip(popt_ped[2], 1e-2, 100)
                        #A0_est = 1200
                        #mu0_est = -100
                        #sigma0_est = 500
                    del ped_hist

                    # Fit signal spectrum
                    #sig_hist, sig_bins = np.histogram(sig_subtracted, bins=BINS)
                    sig_min = np.min(sig_subtracted)
                    sig_max = np.max(sig_subtracted)
                    if sig_max > 1.2e4:
                        sig_max = 1.2e4
                    sig_bins_pre = np.arange(start=sig_min, stop=sig_max + bin_width, step=bin_width)
                    sig_hist_1, sig_bins_1 = np.histogram(sig_subtracted, bins=sig_bins_pre)
                    sig_hist, sig_bins_idx = clean_histogram(sig_hist_1, zero_run_threshold=5)
                    sig_bins = sig_bins_1[sig_bins_idx]
                    del sig_hist_1, sig_bins_1
                    #print('Sig Hist:', sig_hist)
                    #del sig_subtracted
                    A1 = max(sig_hist)
                    #A1_est = 900 
                    q0 = sig_bins[np.argmax(sig_hist)] #- mu0
                    mu_est = 0.5 #2400 #sig_bins[np.argmax(sig_hist)]
                    #qspe_est = 500
                    sigmaspe_est = 1
                    #initial_guess = [A0_est, mu0_est, sigma0_est, A1_est, mu, qspe_est, sigmaspe_est]
                    #initial_guess = [A0, mu0, sigma0, q0, A1, mu_est, qspe_est, sigmaspe_est]
                    initial_guess = [q0, A1, mu_est, qspe_est, sigmaspe_est]
                    #print('Initial Guess:',initial_guess)
                    try:
                        popt_spe_1, pcov_spe_1, centers_1 = fit_spe_spectrum(sig_hist, sig_bins, sigma0=sigma0, npe=N_PHOTOELECTRONS, p0=initial_guess)
                        new_npe = N_PHOTOELECTRONS
                    except:
                        popt_spe_1 = [q0, A1, mu_est, q0, sigmaspe_est]
                        new_npe = 2
                    #print('Made it through First pass')
                    #fit_vals_1, components_1 = spe_peak(centers_1, *popt_spe_1)
                    del sig_hist, sig_bins
                    mu_lump = 0.3*popt_spe_1[0]
                    sigma_lump = 1.4*popt_spe_1[4]
                    A_lump = 0.5
                    second_guess = [A1, popt_spe_1[2], popt_spe_1[3], sigmaspe_est, A_lump, mu_lump, sigma_lump]
                    bounds = [second_guess[0]-(0.25*second_guess[0]), 0.05, q0*0.7, 1, 0.01, 0.2*q0, 1.3*second_guess[3]], [second_guess[0]+(0.25*second_guess[0]), 5, q0*1.3, 1e3, 1.1, 0.7*q0, 1e3]
                    #print('Second Guess:', second_guess)
                    #del A0_est, mu0_est, sigma0_est, A1_est, mu, qspe_est, sigmaspe_est
                    del qspe_est, sigmaspe_est
                    combo_min = np.min(combo_integrals)
                    combo_max = np.max(combo_integrals)
                    if combo_max > 1.2e4:
                        combo_max = 1.2e4
                    combo_bins_pre = np.arange(start=combo_min, stop=combo_max + bin_width, step=bin_width)
                    combo_hist_1, combo_bins_1 = np.histogram(combo_integrals, bins=combo_bins_pre)
                    combo_hist, combo_bins_idx = clean_histogram(combo_hist_1, zero_run_threshold=5)
                    combo_bins = combo_bins_1[combo_bins_idx]
                    print('Combo_hist bins:', np.unique(combo_bins[1:] - combo_bins[:-1]))
                    del combo_hist_1, combo_bins_1
                    #del combo_integrals
                #params = {
                #"A0": A0,  # Freely floating amplitude for pedestal
                #"q0": q0,
                #"sigma0": sigma0,  # Freely floating width for pedestal
                #"A": A1,  # Freely floating amplitude for pedestal
                #"mu": 0.5,
                #"q_spe": 2500.0,
                #"sigma_spe": 10,
                #"n_max": 3
                #}
                #q = np.linspace(0, 10000, 1000)
                #fit_vals, components = spe_model(combo_bins, **params)
                    try: 
                        popt_spe, pcov_spe, centers = fit_lump_spectrum(combo_hist, combo_bins, npe=(new_npe+1), A0=A0, mu0=mu0, sigma0=sigma0, q0=q0,  p0=second_guess)
                        #print('Made it through second pass')
                    #popt_spe, pcov_spe, centers = fit_spe_spectrum(combo_hist, combo_bins, npe=(N_PHOTOELECTRONS+1), p0=initial_guess)
                    #del initial_guess, pcov_spe, combo_bins
                        fit_vals, components, lobes = spe_model(centers, A0=A0, mu0=mu0, sigma0=sigma0, q0=q0, n_max=new_npe, *popt_spe)
                        #print('Made it through the fit')
                        chi2_ndf = chi_squared(combo_hist[:-1], fit_vals, len(combo_hist[:-1]) - len(popt_spe))
                        gain_outputs[adc,ch,0] += np.float64(popt_spe[2]/4)
                        gain_outputs[adc,ch,1] += np.float64(chi2_ndf)
                        
                        # Plot results
                        #plt.figure(figsize=(10, 5))
                        fig, axs = plt.subplots(3, 1, figsize=(10, 12),gridspec_kw={'height_ratios': [2, 1, 1]})
                    # --- First subplot: SPE fit histogram ---
                        axs[0].step(combo_bins, combo_hist, where='mid', label="Signal Data", color="black")
                        colors = plt.cm.tab20(np.linspace(0, 1, len(components)))
                        for i, comp in enumerate(components):
                            axs[0].plot(combo_bins[:-1], comp, '--', color=colors[i], label=f'{i} PE')
                        for i, lobe in enumerate(lobes):
                            axs[0].plot(combo_bins[:-1], lobe, ':', color=colors[i+1], label=f'{i} PE Lobe')
                        axs[0].plot(centers, fit_vals, 'g--', label=f"SPE Fit\nμ={popt_spe[1]:.2f}, Gain={popt_spe[2]:.2f}, χ²/ndf={chi2_ndf:.2f}")
                        axs[0].set_title(f"SPE Fit for ADC {adc}, Channel {ch}")
                        axs[0].set_xlabel("Integrated Charge [a.u.]")
                        axs[0].set_ylabel("Counts")
                        axs[0].legend()
                        axs[0].grid(True)

                        # --- Second subplot: Overlaid waveforms ---
                        for wf in wfs:  # optional: show only first 100
                            axs[1].plot(wf, alpha=0.3, color='blue', rasterized=True)
                        axs[1].set_title(f"ADC {adc}, Channel {ch}: Dark Count Waveforms")
                        axs[1].set_xlabel("Sample")
                        axs[1].set_ylabel("Raw ADC Counts")
                        axs[1].grid(True)

                        gain_integrals = (sig_subtracted) / np.int32(popt_spe[2])
                        #gain_integrals = sig_subtracted / np.int32(popt_spe[2])
                        og_gain = gain_dict.get(adc, {}).get(ch, 0.0)
                        #gain_og_int = ((sig_subtracted - np.int32(popt_spe[1])) / 4) * og_gain
                        gain_og_int = (sig_subtracted / 4) * og_gain
                        #bin_min = 0
                        #bin_max = 10
                        # Create shared bin edges
                        #bins = np.linspace(bin_min, bin_max + 0.1, 0.1)
                        bin_width = 0.1
                        data_min = 0
                        data_max = 10
                        bins = np.arange(np.floor(data_min), np.ceil(data_max) + bin_width, step=bin_width)
                        for k in range(9):
                            axs[2].axvline(x=k+1, linewidth=3, color='yellowgreen', alpha=0.5)
                        # Plot both histograms using identical bins on axs[2]
                        axs[2].hist(gain_og_int, bins=bins, histtype='stepfilled', label='LED Calibrated', alpha=0.3, color="mediumblue")
                        axs[2].hist(gain_integrals, bins=bins, histtype='step', label='DC Calibrated', color='black')
                        #gain_hist, _ = np.histogram(gain_integrals, bins=bins)
                        #og_gain_hist, _ = np.histogram(gain_og_int, bins=bins)
                        #centers = 0.5 * (bins[:-1] + bins[1:])
                        #axs[2].fill_between(centers, og_gain_hist, step="mid", alpha=0.3, color="mediumblue", label="LED Calibrated")
                        #axs[2].hist(og_gain_bins, og_gain_hist, color="mediumblue", histtype='stepfilled', alpha=0.3, label="LED Calibrated")
                        #axs[2].step(og_gain_bins[:-1], og_gain_hist, where='mid', label="LED Calibrated", color="mediumblue")
                        #axs[2].step(centers, gain_hist, step="mid", color="black", label="DC Calibrated")
                        axs[2].set_title(f"ADC {adc}, Channel {ch}: Integral of Calibrated Dark Counts")
                        axs[2].set_xlabel("Integral [p.e.]")
                        axs[2].set_ylabel("Frequency [log]")
                        #axs[2].set_yscale('log')
                        axs[2].grid(True)
                        axs[2].legend()
                        
                        plt.tight_layout()
                        pdf_output.savefig(fig)
                        plt.close(fig)

                    except:
                        print(f"Fit failed for ADC {adc} CH {ch}")
                        fig, axs = plt.subplots(2, 1, figsize=(10, 6),gridspec_kw={'height_ratios': [1, 1]})
                        for wf in wfs:
                            axs[0].plot(wf, alpha=0.3, color='blue', rasterized=True)
                        axs[0].set_title(f"ADC {adc}, Channel {ch}: Fit Failure on Dark Counts")
                        axs[0].set_xlabel("Sample")
                        axs[0].set_ylabel("Raw ADC Counts")
                        axs[0].grid(True)
                        try: 
                            data_min = np.sort(combo_integrals)[10]
                            data_max = np.sort(combo_integrals)[-10]
                            bins = np.arange(np.floor(data_min), np.ceil(data_max) + bin_width, step=bin_width)
                            axs[1].hist(combo_integrals, bins=bins, histtype='step', label='Pedestal and Signal Integrals', color='black')
                            axs[1].set_title(f"ADC {adc}, Channel {ch}: Fit Failure on Dark Counts")
                            axs[1].set_xlabel("Integral")
                            axs[1].set_ylabel("Frequency")
                            axs[1].grid(True)

                            plt.tight_layout()
                            pdf_output.savefig(fig)
                            plt.close(fig)

                            second_guess_rounded = [round(val, 2) for val in second_guess]
                            lower_bounds_rounded = [round(val, 2) for val in bounds[0]]
                            upper_bounds_rounded = [round(val, 2) for val in bounds[1]]

                            # Create text lines
                            lines = []
                            for idx, (val, low, high) in enumerate(zip(second_guess_rounded, lower_bounds_rounded, upper_bounds_rounded)):
                                line = f"Param {idx+1}: {low} < **{val}** < {high}"
                                lines.append(line)
                            # Plot the text box
                            fig, ax = plt.subplots(figsize=(6, len(lines)*0.5 + 1))
                            ax.axis('off')

                            for i, line in enumerate(lines):
                                ax.text(0.05, 1 - (i + 1) * 0.1, line, fontsize=12, verticalalignment='top')

                            plt.tight_layout()
                            pdf_output.savefig(fig)
                            plt.close(fig)

                        except:
                            data_min = np.min(combo_integrals)
                            data_max = np.max(combo_integrals)
                            bins = np.arange(np.floor(data_min), np.ceil(data_max) + bin_width, step=bin_width)
                            axs[1].hist(combo_integrals, bins=bins, histtype='step', label='Pedestal and Signal Integrals', color='black')
                            axs[1].set_title(f"ADC {adc}, Channel {ch}: Fit Failure on Dark Counts")
                            axs[1].set_xlabel("Integral")
                            axs[1].set_ylabel("Frequency")
                            axs[1].grid(True)

                            plt.tight_layout()
                            pdf_output.savefig(fig)
                            plt.close(fig)

                        #del initial_guess, combo_hist, combo_bins
                else:
                    print(f"Inactive or Dead for ADC {adc} CH {ch}")

    del waveform_data, waveforms
    return gain_outputs

def main(input_file, output_file):

   gain_values = analyze_spe(input_file, output_file)
   np.savez('/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/AFIViewer/Output_06/filt_v01_gain_chi2.npz', data=gain_values)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', default=None, required=True, type=str, \
                        help='''string corresponding to the path of the doctored .npz files to be considered''')
    parser.add_argument('-o', '--output_file', default=None, required=True, type=str, \
                        help='''string corresponding to desired output .pdf path and name''')
    args = parser.parse_args()
    main(**vars(args))

  