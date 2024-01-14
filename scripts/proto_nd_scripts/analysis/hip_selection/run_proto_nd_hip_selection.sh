#!/bin/bash
# Runs proto_nd_flow HIP selection on an example file.
# Before using this script, use
# >> source get_proto_nd_input.sh
# to download all the necessary inputs into the correct directories
#
INPUT_FILE=$1

OUTPUT_DIR=`pwd`
OUTPUT_NAME=(${INPUT_FILE//"/"/ })
OUTPUT_NAME=${OUTPUT_NAME[-1]}
OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}"
OUTPUT_FILE=${OUTPUT_FILE//.h5/.proto_nd_flow.HIP_SEL.h5}
echo ${OUTPUT_FILE}

# for running on a login node
H5FLOW_CMD='h5flow'
# for running on a single compute node with 32 cores
#H5FLOW_CMD='srun -n32 h5flow'

# run all stages
WORKFLOW1='yamls/proto_nd_flow/workflows/analysis/hip_sel_workflow.yaml'

HERE=`pwd`
#cd ndlar_flow
# assumes this is being run from ndlar_flow/scripts/proto_nd_flow/analysis/hip_selection/:
cd ../../../../

# avoid being asked if we want to overwrite the file if it exists.
# this is us answering "yes".
if [ -e $OUTPUT_FILE ]; then
    rm -i $OUTPUT_FILE
fi

$H5FLOW_CMD -c $WORKFLOW1 -i $INPUT_FILE -o $OUTPUT_FILE

echo "Done!"
echo "Output can be found at $OUTPUT_FILE"

cd ${HERE}

