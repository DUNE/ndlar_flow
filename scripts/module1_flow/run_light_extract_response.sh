#!/bin/bash

set -o errexit

module load python
source ~/reflow/flow.venv/bin/activate

LIGHT_INPUT_FILE='/global/cfs/cdirs/dune/www/data/Module1/LRS/SingleModule_Jan22/0cd913fb_20220211_074023.data'

OUTPUT_FILE=$PWD/testout.h5

H5FLOW_CMD='h5flow'

HERE=`pwd`
#cd ndlar_flow
# assumes this is being run from ndlar_flow/scripts/module1_flow:
cd ../../

WORKFLOW1='yamls/module1_flow/workflows/light/light_event_building_adc64.yaml'
WORKFLOW2='yamls/module1_flow/workflows/light/light_extract_response.yaml'

$H5FLOW_CMD -c $WORKFLOW1 $WORKFLOW2 -i $LIGHT_INPUT_FILE -o $OUTPUT_FILE

echo "Done!"

cd ${HERE}
