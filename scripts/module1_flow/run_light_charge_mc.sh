#!/bin/bash

module load python
# assumes this is being run from ndlar_flow/scripts/module1_flow and that your flow.venv is
#         in 2x2_sim/run-ndlar-flow
source ../../../flow.venv/bin/activate

set -o errexit

LIGHT_INPUT_FILE=$1
CHARGE_INPUT_FILE=$1

LIGHT_OUTPUT_NAME=(${LIGHT_INPUT_FILE//"/"/ })
LIGHT_OUTPUT_NAME=${LIGHT_OUTPUT_NAME[-1]}

OUTPUT_DIR=${SCRATCH}
OUTPUT_NAME=(${CHARGE_INPUT_FILE//"/"/ })
OUTPUT_NAME=${OUTPUT_NAME[-1]}
OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}"
OUTPUT_FILE=${OUTPUT_FILE//.h5/.module1_flow.h5}
echo ${OUTPUT_FILE}

H5FLOW_CMD='h5flow'

HERE=`pwd`
#cd ndlar_flow
# assumes this is being run from ndlar_flow/scripts/module1_flow:
cd ../../

# avoid being asked if we want to overwrite the file if it exists.
# this is us answering "yes".
if [ -e $OUTPUT_FILE ]; then
    rm $OUTPUT_FILE
fi

#WORKFLOW1='yamls/module1_flow/workflows/light/light_event_building_mc.yaml'
#WORKFLOW2='yamls/module1_flow/workflows/light/light_event_reconstruction.yaml'

#$H5FLOW_CMD -c $WORKFLOW1 $WORKFLOW2 -i $LIGHT_INPUT_FILE -o $OUTPUT_FILE

WORKFLOW3='yamls/module1_flow/workflows/charge/charge_event_building.yaml'
WORKFLOW4='yamls/module1_flow/workflows/charge/charge_event_reconstruction.yaml'
WORKFLOW5='yamls/module1_flow/workflows/combined/combined_reconstruction.yaml'
WORKFLOW6='yamls/module1_flow/workflows/charge/prompt_calibration_mc.yaml'
WORKFLOW7='yamls/module1_flow/workflows/charge/final_calibration.yaml'

$H5FLOW_CMD -c $WORKFLOW3 $WORKFLOW4 $WORKFLOW5 $WORKFLOW6 $WORKFLOW7 -i $CHARGE_INPUT_FILE -o $OUTPUT_FILE

#WORKFLOW8='yamls/module1_flow/workflows/charge/charge_light_assoc.yaml'

#$H5FLOW_CMD -c $WORKFLOW8 -i $OUTPUT_FILE -o $OUTPUT_FILE

echo "Done!"

FINALOUTDIR=$2
mkdir -p $FINALOUTDIR
echo "Move file to $FINALOUTDIR"
mv $OUTPUT_FILE $FINALOUTDIR
echo "Output can be found at $FINALOUTDIR"

cd ${HERE}
