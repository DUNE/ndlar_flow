#!/usr/bin/env bash

dir=/global/cfs/cdirs/dune/www/data/2x2/nearline/packet/commission/June2024/noise_assessments_06_05
FILES=(
	"packet-cold-pedestal-2024_06_05_08_28_19_CDT.h5"
	)

OUTFILE_1=/global/cfs/cdirs/dune/users/sfogarty/flow_LowEnergy/2x2/pedestal_tests/2x2_pedestals_20240605_temp.FLOW.hdf5
OUTFILE_2=/global/cfs/cdirs/dune/users/sfogarty/flow_LowEnergy/2x2/pedestal_tests/2x2_pedestals_20240605.FLOW.hdf5
workflow_1=yamls/proto_nd_flow/workflows/charge/make_pedestal_hist.yaml
workflow_2=yamls/proto_nd_flow/workflows/charge/generate_pedestal_json.yaml

list_length=${#FILES[@]}
for ((i = 0; i < list_length; i++)); do
    INFILE=${dir}/${FILES[i]}
    h5flow -c $workflow_1 -i $INFILE -o $OUTFILE_1
done

h5flow -c $workflow_2 -i $OUTFILE_1 -o $OUTFILE_2
