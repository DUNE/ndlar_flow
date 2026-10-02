#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -A dune
#SBATCH -t 2:00:0

srun parallel --jobs 2 ./run_muon_selection_data.sh {} :::: filelist.txt 