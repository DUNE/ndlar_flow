LIGHT_INPUT_FILE=$1
CHARGE_INPUT_FILE=$2

LIGHT_OUTPUT_NAME=(${LIGHT_INPUT_FILE//"/"/ })
LIGHT_OUTPUT_NAME=${LIGHT_OUTPUT_NAME[-1]}

OUTPUT_DIR=`pwd` #!!! change me
OUTPUT_NAME=(${CHARGE_INPUT_FILE//"/"/ })
OUTPUT_NAME=${OUTPUT_NAME[-1]}
OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}"
OUTPUT_FILE=${OUTPUT_FILE//.h5/_${LIGHT_OUTPUT_NAME}.module1_flow.h5}
echo ${OUTPUT_FILE}

H5FLOW_CMD='h5flow'

HERE=`pwd`
#cd ndlar_flow
# assumes this is being run from ndlar_flow/scripts/module1_flow:
cd ../../

# avoid being asked if we want to overwrite the file if it exists.
# this is us answering "yes".
if [ -e $OUTPUT_FILE ]; then
    rm -i $OUTPUT_FILE
fi

WORKFLOW1='yamls/module1_flow/workflows/light/light_event_building_adc64.yaml'
WORKFLOW2='yamls/module1_flow/workflows/light/light_event_reconstruction.yaml'

$H5FLOW_CMD -c $WORKFLOW1 $WORKFLOW2 -i $LIGHT_INPUT_FILE -o $OUTPUT_FILE

WORKFLOW1='yamls/module1_flow/workflows/charge/charge_event_building.yaml'
WORKFLOW2='yamls/module1_flow/workflows/charge/charge_event_reconstruction.yaml'
WORKFLOW3='yamls/module1_flow/workflows/combined/combined_reconstruction.yaml'
WORKFLOW4='yamls/module1_flow/workflows/charge/prompt_calibration.yaml'
WORKFLOW5='yamls/module1_flow/workflows/charge/final_calibration.yaml'

$H5FLOW_CMD -c $WORKFLOW1 $WORKFLOW2 $WORKFLOW3 $WORKFLOW4 $WORKFLOW5 -i $CHARGE_INPUT_FILE -o $OUTPUT_FILE

echo "Done!"
echo "Output can be found at $OUTPUT_FILE"

cd ${HERE}
