#!/bin/bash

#DATA_DIR=$1
DATA_DIR="../../data/ndlar_flow/"
mkdir -p $DATA_DIR

HERE=`pwd`

cd ${DATA_DIR}

# ndlar layout describing a *single* module (fix me)
curl -O https://raw.githubusercontent.com/DUNE/larnd-sim/develop/larndsim/pixel_layouts/multi_tile_layout-3.0.40.yaml

# ndlar detector description
curl -O https://raw.githubusercontent.com/DUNE/larnd-sim/nd-production-v02.00/larndsim/detector_properties/ndlar-module.yaml

# Placeholder
curl -O https://portal.nersc.gov/project/dune/data/2x2/simulation/kwood_dev/proto_nd_flow_inputs/runlist-2x2-mcexample.txt

cd ${HERE}
