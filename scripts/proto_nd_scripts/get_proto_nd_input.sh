#!/bin/bash

#DATA_DIR=$1
DATA_DIR="../../data/proto_nd_flow/"

HERE=`pwd`

cd ${DATA_DIR}

# tile layout describing a *single* module (fix me)
curl -O https://portal.nersc.gov/project/dune/data/2x2/simulation/kwood_dev/proto_nd_flow_inputs/multi_tile_layout-2.4.16.yaml 

# 2x2 detector description
curl -O https://portal.nersc.gov/project/dune/data/2x2/simulation/kwood_dev/proto_nd_flow_inputs/2x2.yaml

# place holder for run list
curl -O https://portal.nersc.gov/project/dune/data/2x2/simulation/kwood_dev/proto_nd_flow_inputs/runlist-2x2-mcexample.txt

# place holder for light system geometry description
curl -O https://portal.nersc.gov/project/dune/data/2x2/simulation/kwood_dev/proto_nd_flow_inputs/light_module_desc-0.0.0.yaml

cd ${HERE}
