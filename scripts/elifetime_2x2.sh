#!/usr/bin/env bash

workflow=yamls/proto_nd_flow/workflows/analysis/elifetime_low_energy.yaml

#cd /global/cfs/cdirs/dune/users/sfogarty/ndlar_flow_low_energy/ndlar_flow/scripts/low_energy_scripts/2x2/ndlar_flow_low_energy_2/ndlar_flow/
cd ..
source ndlar_flow.venv/bin/activate
pip install .

OUTFILE=test2/kde_elifetime_estimate_2x2_2025_30ke_to_60ke_run2_radon.hdf5
rm -f $OUTFILE
h5flow -c $workflow -o $OUTFILE
