#!/bin/bash

#DATA_DIR=$1
DATA_DIR="../../data/proto_nd_flow/"

HERE=`pwd`

cd ${DATA_DIR}

# tile layout describing modules 0/1/3 and module 2, respectively
curl -O https://raw.githubusercontent.com/DUNE/larnd-sim/develop/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml
curl -O https://raw.githubusercontent.com/DUNE/larnd-sim/develop/larndsim/pixel_layouts/multi_tile_layout-2.5.16_v4.yaml

# 2x2 detector description
curl -O https://raw.githubusercontent.com/DUNE/larnd-sim/develop/larndsim/detector_properties/2x2.yaml

cd ${HERE}
