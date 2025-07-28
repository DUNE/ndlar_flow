#!/usr/bin/env bash

# By default, run on the host's venv
module unload python 2>/dev/null
module load python/3.11

#N_files=200

#output_file=/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/FiltFiles_v03_wvfms.npz
output_file=/global/cfs/cdirs/dune/users/ajwhite/2x2_LRS_DataAssess/2025_Calibration/FSD_DC_FiltFiles_v01_wvfms.npz

echo "Starting Run Attempt"

#python3 Step2_pythonScript.py --o $output_file
python3 Step3_FSD_pythonScript.py --o $output_file
        
echo "Finished Run Attempt"