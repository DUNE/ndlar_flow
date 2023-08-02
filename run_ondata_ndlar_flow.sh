#!/usr/bin/env bash

cd $HOME/2x2_sim/run-ndlar-flow

module load python

source flow.venv/bin/activate

if [[ "$NERSC_HOST" == "cori" ]]; then
    export HDF5_USE_FILE_LOCKING=FALSE
fi

#echo "Charge Date tag is $DATETAG_C"
#echo "Light Date tag is $DATETAG_L"

inDir_C=/global/cfs/cdirs/dune/www/data/Module1/TPC12/dataRuns/packetData
inDir_L=/global/cfs/cdirs/dune/www/data/Module1/LRS/SingleModule_Jan22

outDir=$PSCRATCH/output/$DATETAG_L
mkdir -p $outDir

outName=$DATETAG_L.flowed
#echo "outName is $outName"

timeFile=$outDir/TIMING/$outName.time
mkdir -p "$(dirname "$timeFile")"
timeProg=/usr/bin/time

run() {
    echo RUNNING "$@"
    time "$timeProg" --append -f "$1 %P %M %E" -o "$timeFile" "$@"
}

inFile_C=$inDir_C/packet_${DATETAG_C}_CET.h5
inFile_L=$inDir_L/0cd913fb_${DATETAG_L}.data

echo "Input Charge File: "$inFile_C
echo "Input Light File: "$inFile_L

flowOutDir=$outDir/FLOW
mkdir -p $flowOutDir

outFile_L=$flowOutDir/light_${outName}.h5
outFile_C=$flowOutDir/charge_${outName}.h5
outFile=$flowOutDir/combined_${outName}.h5
rm -f "$outFile"

echo "Output Combined File: "$outFile

workflow1_L='yamls/module0_flow/workflows/light/light_event_building_adc64.yaml'
workflow2_L='yamls/module0_flow/workflows/light/light_extract_response.yaml'
workflow3_L='yamls/module0_flow/workflows/light/light_event_reconstruction.yaml'

workflow1_C='yamls/module0_flow/workflows/charge/charge_event_building.yaml'
workflow2_C='yamls/module0_flow/workflows/charge/charge_event_reconstruction.yaml'
workflow3_C='yamls/module0_flow/workflows/charge/charge_light_association.yaml'
workflow4_C='yamls/module0_flow/workflows/combined/combined_reconstruction.yaml'

cd ndlar_flow
pip install adc64format

#h5flow --nompi -c $workflow1_L $workflow2_L \
#    -i $inFile_L -o $outFile_L

#h5flow --nompi -c $workflow3_L \
#    -i $outFile_L -o $outFile

#h5flow --nompi -c $workflow1_C $workflow2_C $workflow3_C $workflow4_C \
h5flow --nompi -c $workflow1_C $workflow2_C \
    -i $inFile_C -o $outFile_C	
