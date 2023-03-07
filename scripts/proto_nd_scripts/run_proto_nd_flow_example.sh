#!/bin/bash
# Runs proto_nd_flow on an example file.
# Before using this script, use
# >> source get_proto_nd_input.sh
# to download all the necessary inputs into the correct directories
#

#INPUT_FILE=$1
INPUT_FILE='data/proto_nd_flow/sim2x2_challenge_5xNuMIspills_ME_65E12POTperSpill_larndsim_mockdata_v1.h5'

# below is the same file with mc information included
#INPUT_FILE='/home/kwood/research/dune/2x2/data/simulation_challenge_26Oct2022/sim2x2_challenge_5xNuMIspills_ME_65E12POTperSpill_v2.larndsim.h5'
OUTPUT_FILE=${INPUT_FILE//.h5/.proto_nd_flow.h5}

# for running on a login node
H5FLOW_CMD='h5flow'
# for running on a single compute node with 32 cores
#H5FLOW_CMD='srun -n32 h5flow'

# run all stages
WORKFLOW1='yamls/proto_nd_flow/workflows/charge/charge_event_building.yaml'
WORKFLOW2='yamls/proto_nd_flow/workflows/charge/charge_event_reconstruction.yaml'
#WORKFLOW3='h5flow_yamls/workflows/combined/combined_reconstruction.yaml'

# assumes this is being run from scripts/proto_nd_scripts

HERE=`pwd`
cd ../../


if [ -e $OUTPUT_FILE ]; then
    rm -i $OUTPUT_FILE
fi

#$H5FLOW_CMD -c $WORKFLOW1 $WORKFLOW2 $WORKFLOW3 -i $INPUT_FILE -o $OUTPUT_FILE
$H5FLOW_CMD -c $WORKFLOW1 $WORKFLOW2 -i $INPUT_FILE -o $OUTPUT_FILE
#$H5FLOW_CMD -c $WORKFLOW1 -i $INPUT_FILE -o $OUTPUT_FILE

echo "Done!"
echo "Output can be found at $OUTPUT_FILE"

cd ${HERE}

