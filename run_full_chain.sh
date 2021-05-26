#!/bin/bash

output_file=$1

input_datalog='/Volumes/storage/module0_data/reference_run/datalog_2021_04_10_04_21_27_CEST.h5'
input_rwf='/Volumes/storage/module0_data/reference_run/rwf_20210410_042130.data.root'

# datalog_subselection='-e 5000000'
datalog_subselection=''
# rwf_subselection='-e 500000'
rwf_subselection=''

# charge event building
rm -rf ${output_file//.h5/.cev.h5}
mpiexec h5flow -v -c h5flow_yamls/charge_event_building.yaml -i $input_datalog -o ${output_file//.h5/.cev.h5} $datalog_subselection

# light event building
cp -f ${output_file//.h5/.cev.h5} ${output_file//.h5/.cev.lev.h5}
mpiexec h5flow -v -c h5flow_yamls/light_event_building.yaml -i $input_rwf -o ${output_file//.h5/.cev.lev.h5} $rwf_subselection

# charge -> light association
rm -rf ${output_file}
mpiexec h5flow -v -c h5flow_yamls/charge_light_assoc.yaml -i ${output_file//.h5/.cev.lev.h5} -o ${output_file}

# compress datasets
rm -rf ${output_file//.h5/.gz.h5}
h5repack -f GZIP=6 $output_file ${output_file//.h5/.gz.h5}

# remove temporary files
rm -rf ${output_file} ${output_file//.h5/.cev.lev.h5} ${output_file//.h5/.cev.h5}
