#!/usr/bin/env bash

# By default, run on the host's venv
module unload python 2>/dev/null
module load python/3.11

#N_files=200

# 2x2: you need the file location for your output (which is isolated noise-only and darkcount pulse waveforms)
#output_file=/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/FiltFiles_v03_wvfms.npz
# FSD: 
#output_file=/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/FSD_DC_FiltFiles_v01_wvfms.npz

echo "Starting Run Attempt"
#2x2, run this from the directory where your script lives
python3 Step2_pythonScript.py --o $output_file
FSD
#python3 Step3_FSD_pythonScript.py --o $output_file
        
echo "Finished Run Attempt"
