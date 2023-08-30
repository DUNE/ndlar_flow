#!/bin/bash
# Runs module1_flow on an example file.
#

INPUT_FILE=$1

OUTPUT_DIR=`pwd` #!!! change me
OUTPUT_NAME=(${INPUT_FILE//"/"/ })
OUTPUT_NAME=${OUTPUT_NAME[-1]}
OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}"
OUTPUT_FILE=${OUTPUT_FILE//.h5/.module1_flow.h5}
echo ${OUTPUT_FILE}

# for running on a login node
H5FLOW_CMD='h5flow'
# for running on a single compute node with 32 cores
#H5FLOW_CMD='srun -n32 h5flow'

# run all stages
WORKFLOW1='yamls/module1_flow/workflows/charge/charge_event_building.yaml'
WORKFLOW2='yamls/module1_flow/workflows/charge/charge_event_reconstruction.yaml'
WORKFLOW3='yamls/module1_flow/workflows/combined/combined_reconstruction.yaml'
WORKFLOW4='yamls/module1_flow/workflows/charge/prompt_calibration.yaml'
WORKFLOW5='yamls/module1_flow/workflows/charge/final_calibration.yaml'

HERE=`pwd`
#cd ndlar_flow
# assumes this is being run from ndlar_flow/scripts/proto_nd_flow:
cd ../../

# avoid being asked if we want to overwrite the file if it exists.
# this is us answering "yes".
if [ -e $OUTPUT_FILE ]; then
    rm -i $OUTPUT_FILE
fi

$H5FLOW_CMD -c $WORKFLOW1 $WORKFLOW2 $WORKFLOW3 $WORKFLOW4 $WORKFLOW5 -i $INPUT_FILE -o $OUTPUT_FILE

echo "Done!"
echo "Output can be found at $OUTPUT_FILE"

cd ${HERE}

