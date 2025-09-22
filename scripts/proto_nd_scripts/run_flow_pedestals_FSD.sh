#!/usr/bin/env bash

dir=/global/cfs/cdirs/dune/www/data/FSD/nearline/packet/CRS/pedestal_11Nov2024/PRC_256
INFILE=/global/cfs/cdirs/dune/www/data/FSD/nearline/packet/CRS/pedestal_11Nov2024/PRC_256/packet-pedestal-2024_11_12_03_08_18_CET.h5
FILES=(
	"packet-pedestal-2024_11_12_03_08_18_CET.h5"
    "packet-pedestal-2024_11_12_03_11_20_CET.h5"
    "packet-pedestal-2024_11_12_03_14_21_CET.h5"
	"packet-pedestal-2024_11_12_03_17_22_CET.h5"
    "packet-pedestal-2024_11_12_03_20_23_CET.h5"
	)

OUTFILE_1=/global/cfs/cdirs/dune/users/sfogarty/flow_LowEnergy/FSD/pedestal_tests/FSD_pedestals_20241112_temp.FLOW.hdf5
OUTFILE_2=/global/cfs/cdirs/dune/users/sfogarty/flow_LowEnergy/FSD/pedestal_tests/FSD_pedestals_20241112.FLOW.hdf5
workflow_1=yamls/fsd_flow/workflows/charge/make_pedestal_hist.yaml
workflow_2=yamls/fsd_flow/workflows/charge/generate_pedestal_json.yaml

list_length=${#FILES[@]}
for ((i = 0; i < list_length; i++)); do
    INFILE=${dir}/${FILES[i]}
    h5flow -c $workflow_1 -i $INFILE -o $OUTFILE_1
done

h5flow -c $workflow_2 -i $OUTFILE_1 -o $OUTFILE_2
