import argparse
import json
import numpy as np
from math import ceil, floor, sqrt
import h5py

import h5flow
#from h5flow.data import dereference, dereference_chain

from ROOT import TH1D, TH2D, TFile, TCanvas, TF1

from pixelQ import get_pixels_yz, build_pixel_lookup

def pixelFit(h, highstat=False):    
    f = TF1("f", "landau", 15.0, 120.0)
    f.SetParameters(50.0, 40.0 , 5)
    #f.SetParLimits(3, 0, 100) #Keeping this parameter positive makes the fit more stable
    #f.SetParLimits(1, 0, 100)
        
    c1 = TCanvas("c1", "c1", 800, 1000)

    # Fit and draw result of the fit
    #do chi-square for hight stat, negative log likelihood for low stats.
    if highstat:
        h.Fit("f", "R")
    else:
        h.Fit("f","LQRE") #L for likelihood fit, Q to suppress printout, R to use fit range, E to get fit errors

    mpv = f.GetMaximumX()
    ndf = f.GetNDF()

    mpverr = f.GetParError(1)

    chi2 = f.GetChisquare() 
    #c1.SaveAs("fitConvolution.png")

    return mpv, chi2, chi2/ndf, mpverr
    
def gaussFit(h):
    f2 = TF1("f2", "gaus", 20.0, 80.0)
    f2.SetParameters(500.0, 40.0 , 5)
    f2.ReleaseParameter(2)
    #f.SetParLimits(3, 0, 100) #Keeping this parameter positive makes the fit more stable
    #f.SetParLimits(1, 0, 100)
        
    c1 = TCanvas("c1", "c1", 800, 1000)

    # Fit and draw result of the fit
    h.Fit("f2", "R")

    #c1.SaveAs("fitConvolution.png")


io_group = 3
is_mc = False
if is_mc:
    f_name = '/global/homes/l/lzazueta/rockmuon_MCMinirun65_pcainfo.hdf5'

    #file_name = "/pscratch/sd/l/lzazueta/pixelQ_iogroup3_datav11_64_withdx_withfit_v46.root"
    file_name = "/pscratch/sd/l/lzazueta/pixelQ_iogroup3_mcmr65_1000_v503.root"

else:
    #f_name = '/global/homes/l/lzazueta/rockmuon_Datafilterv2_july8.hdf5'
    #f_name = '/global/homes/l/lzazueta/rockmuon_Datav10_160f.hdf5'
    f_name = '/global/homes/l/lzazueta/rockmuon_Datav11_64files.hdf5'

    file_name = "/pscratch/sd/l/lzazueta/pixelQ_iogroup3_datav11_64_v503.root"


f_manager = h5flow.data.H5FlowDataManager(f_name, 'r')
positions = get_pixels_yz( f_manager, io_group )
lookup = build_pixel_lookup( f_manager, io_group )

if io_group in (5, 6):
    pixel_pitch = 0.387975
    npix_y, npix_z = 320, 160
else:
    pixel_pitch = 0.4434
    npix_y, npix_z = 280, 140
ymax, zmax = np.amax(positions, 0) + 0.44
ymin, zmin = np.amin(positions, 0)
npix = npix_y * npix_z

fit_file = TFile("/pscratch/sd/l/lzazueta/fitted_hist_landau_data_v503_test.root", "RECREATE")

hgains = TH1D('gains', 'Gain correction', 100, 0.5, 1.8 )
hmpvs = TH1D("mpvs", "pixel MPVs ke/cm", 100, 20, 80 )
hhighmpv = TH2D('highmpv', 'investigation of high mpv', npix_z, zmin, zmax, npix_y, ymin, ymax )
hmpvs2d = TH2D('mpvs2d', 'MPV on the anode', npix_z, zmin, zmax, npix_y, ymin, ymax )
hmpverr2d = TH2D('mpverr2d', 'MPV error on the anode', npix_z, zmin, zmax, npix_y, ymin, ymax )
hmvp_thresholds = TH2D('mpv_thresholds', 'MPV vs Thresholds', 100, 0, 20, 1000, 20, 70 )
hmvp_vs_chi2 = TH2D('mpv_chi2', 'MPV vs Chi2', 100, 20, 80, 100, 0, 80 )
hmvp_chi2ndf = TH2D('mpv_chi2ndf', 'MPV vs Chi2/ndf', 100, 20, 80, 100, 0, 2 )

hentries = TH1D('entries', 'Entries per hpixel', 100, 0, 100 )

#hthreshold = TH1D('thresholds', 'Thresholds', 100, 0, 20)
#for t in thresholds.values():
#    hthreshold.Fill(t)

#load histograms from root file
file_in = TFile.Open(file_name)

hist = []
for pixnum in range(npix):
    hist.append(file_in.Get("dqdx" + str(pixnum) ))
    hentries.Fill( hist[pixnum].Integral() )

hsegdqdx = file_in.Get("segdqdx")
if hsegdqdx is None:
    print("Error: histogram segdqdx not found in file ", file_name)
    exit(1)
hsegdqdx.Print()
mpv_fit = hsegdqdx.GetFunction("f").GetMaximumX()
print("MPV from fit is ", mpv_fit)

#fit segment dqdx
#mpv = hsegdqdx.GetXaxis().GetBinCenter( hsegdqdx.GetMaximumBin() )
chi2_list = np.ones((npix_y*npix_z, 1 ))
chi2_list2 = np.ones((npix_y*npix_z, 1 ))
mpverr_list = np.ones((npix_y*npix_z, 1 ))
gains = np.zeros((npix_y*npix_z, 1 ) )
for pix in range(npix):
    if hist[pix].Integral() < 10:
        continue
        #gains[pix] = 1.
    else:
        #gains[pix], chi2_list[pix] = langau_fit(hist[pix], highstat=False) 
        gains[pix], chi2_list[pix], chi2_list2[pix], mpverr_list[pix] = pixelFit(hist[pix], highstat=False) 

    if pix  % 1000 == 0:
        print(pix)

mpvs = gains
gains = gains / mpv_fit

gains_by_networkid = {}
for pix in range(npix):
    if pix not in networkids:
        continue
    g = gains[pix]
    if not np.isfinite(g).all():
        continue
    gains_by_networkid[str(networkids[pix])] = float(g.item())

with open("/pscratch/sd/l/lzazueta/gains_by_networkid.json", "w") as jf:
    json.dump(gains_by_networkid, jf, indent=2, sort_keys=True)

#create a file to save the pixel number and parameters
with open("pixel_parameters.txt", "w") as f:
    f.write("Pixel_Index\tMPV_ke_per_cm\tGain_Correction\tthreshold\tEntries\tChi2\n")


    for pix in range(npix):
        #if gains[pix] > 1.5:
        #    print("Pixel ", positions[pix][1], positions[pix][0], " gain is " + str(gains[pix]))

        hmpvs.Fill(mpvs[pix].item())
        if gains[pix] < 0.1:
            continue

        else:
            g = gains[pix].item()

            if g > 1.190 and g < 1.196: 
                f.write(f"{pix}\t{mpvs[pix].item()}\t{g}\t{1.0}\t{hist[pix].Integral()}\t{chi2_list[pix].item()}\n")
                hhighmpv.Fill(positions[pix][1], positions[pix][0] )
            
            hgains.Fill(g)
            '''
            if hist[pix].Integral() >= 10 and hist[pix].Integral() <= 24:
                hgains_bin1.Fill(g)
            elif hist[pix].Integral() > 24 and hist[pix].Integral() <= 39:
                hgains_bin2.Fill(g)
            elif hist[pix].Integral() > 39:
                hgains_bin3.Fill(g)
            '''
            #print("threshold idx for pixel ", pix, " is ", threshold_idx[pix], " gain is ", g)
            #if threshold_idx[pix] is not None:
            #    hmvp_thresholds.Fill(threshold_idx[pix], mpvs[pix].item())

            hmvp_vs_chi2.Fill(mpvs[pix].item(), chi2_list[pix].item() )
            hmvp_chi2ndf.Fill(mpvs[pix].item(), chi2_list2[pix].item() )

            hmpvs2d.Fill(positions[pix][1], positions[pix][0], mpvs[pix].item() )
            hmpverr2d.Fill(positions[pix][1], positions[pix][0], mpverr_list[pix].item() )

gaussFit(hmpvs)


fit_file.cd()
for h in hist:
    h.Write()
hhighmpv.Write()
#hthreshold.Write()
hmvp_vs_chi2.Write()
hmvp_chi2ndf.Write()
hmpvs.Write()
hmpvs2d.Write()
hmpverr2d.Write()
hgains.Write()
hentries.Write()

fit_file.Close()
