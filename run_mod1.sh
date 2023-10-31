#!/bin/bash

workflow1='/global/cfs/cdirs/dune/users/kwresilo/ndlar_flow/yamls/module1_flow/workflows/charge/charge_event_building.yaml'
workflow2='/global/cfs/cdirs/dune/users/kwresilo/ndlar_flow/yamls/module1_flow/workflows/charge/charge_event_reconstruction.yaml'
workflow3='/global/cfs/cdirs/dune/users/kwresilo/ndlar_flow/yamls/module1_flow/workflows/light/light_event_building_adc64.yaml'
workflow4='/global/cfs/cdirs/dune/users/kwresilo/ndlar_flow/yamls/module1_flow/workflows/charge/charge_light_assoc.yaml'

charge_data='/global/cfs/cdirs/dune/www/data/Module1/TPC12/dataRuns/packetData/packet_2022_02_08_12_48_18_CET.h5'
light_data='/global/cfs/cdirs/dune/www/data/Module1/LRS/SingleModule_Jan22/0cd9415c_20220208_124818.data'

output_dir='/global/cfs/cdirs/dune/users/kwresilo/data/module1/' 
run='20220208_124818'
outfile='_matched_charge_ts_corrected.h5'

#echo "RUNNING CHARGE RECO"
#run charge building and reco
h5flow --nompi -c $workflow1 $workflow2\
    -i ${charge_data} -o ${output_dir}${run}${outfile}

echo "RUNNING LIGHT RECO"
#run light reco
h5flow --nompi -c ${workflow3}\
    -i ${light_data}\
    -o ${output_dir}${run}${outfile}

echo "RUNNING MATCHING"
#run CL matching
h5flow -c ${workflow4}\
    -i ${output_dir}${run}${outfile}\
    -o ${output_dir}${run}${outfile}
