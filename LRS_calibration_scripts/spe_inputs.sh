#!/usr/bin/env bash

# By default, run on the host's venv
module unload python 2>/dev/null
module load python/3.11

inFile=/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/FiltFiles_v03_wvfms.npz
#inFile=/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/FSD_DC_FiltFiles_v01_wvfms.npz

outFile=/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/AFIViewer/Output_07/FiltChans_Amp_v04_3plots.pdf
#outFile=/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/AFIViewer/Output_FSD_01/FSD_DC_FiltChans_PDF_v01.pdf

echo "Starting Run Attempt"

#python3 sipm_spe_analysis_v2.py --i $inFile --o $outFile
python3 sipm_spe_amplitude.py --i $inFile --o $outFile
#python3 sipm_spe_analysis_FSD.py --i $inFile --o $outFile
        
echo "Finished Run Attempt"