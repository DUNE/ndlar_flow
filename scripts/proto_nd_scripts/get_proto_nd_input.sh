#!/bin/bash

#DATA_DIR=$1
DATA_DIR="../../data/proto_nd_flow/"
DATA_DIR_FSD="../../data/fsd_flow/"

HERE=`pwd`

cd ${DATA_DIR}

# tile layout describing a *single* module (fix me)
#curl -O https://raw.githubusercontent.com/DUNE/larnd-sim/MiniRun6-v1/larndsim/pixel_layouts/multi_tile_layout-2.4.16.yaml
#curl -O https://raw.githubusercontent.com/DUNE/larnd-sim/MiniRun6-v1/larndsim/pixel_layouts/multi_tile_layout-2.5.16.yaml
#curl -O https://raw.githubusercontent.com/DUNE/larnd-sim/develop/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v3.yaml
#curl -O https://raw.githubusercontent.com/DUNE/larnd-sim/develop/larndsim/pixel_layouts/multi_tile_layout-2.5.16_v3.yaml
curl -O https://raw.githubusercontent.com/DUNE/larnd-sim/develop/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml
curl -O https://raw.githubusercontent.com/DUNE/larnd-sim/develop/larndsim/pixel_layouts/multi_tile_layout-2.5.16_v4.yaml

# 2x2 detector description
curl -O https://raw.githubusercontent.com/DUNE/larnd-sim/MiniRun6-v1/larndsim/detector_properties/2x2.yaml

# place holder for light system geometry description
curl -O https://portal.nersc.gov/project/dune/data/2x2/simulation/kwood_dev/proto_nd_flow_inputs/light_module_desc-0.0.0.yaml

#Download 2x2 and FSD databases
URL="https://dbdata0vm.fnal.gov:9443/dune_runcon_prod/get"


echo "Querying 2x2 electron lifetime data..."
curl -s "$URL" \
   --get \
   --data-urlencode "folder=neardet2x2.elifetime" \
   --data-urlencode "t0=0" \
   --data-urlencode "t1=1000000000000" \
   --data-urlencode "data_type=2x2_data" \
   --data-urlencode "format=json" \
 | jq '
     .rows
     | map({ (.tv|tostring): [ .elifetime, .err_elifetime ] })
     | add
 ' > 2x2_electron_lifetime_db.json



echo "Querying 2x2 pedestal data..."

curl -s "$URL" \
  --get \
  --data-urlencode "folder=neardet2x2.pedestal_test" \
  --data-urlencode "t0=0" \
  --data-urlencode "t1=1000000000000" \
  --data-urlencode "data_type=2x2_data" \
  --data-urlencode "format=json" \
| jq '
  .rows
  | group_by(.tv)
  | map({
      (.[0].tv|tostring):
        ( map({ (.channel|tostring): { "pedestal_mv": .pedestal } }) | add )
    })
  | add
' > 2x2_pedestal_db.json


echo "Querying 2x2 gain data..."

curl -s "$URL" \
  --get \
  --data-urlencode "folder=neardet2x2.gain" \
  --data-urlencode "t0=0" \
  --data-urlencode "t1=1000000000000" \
  --data-urlencode "data_type=2x2_data" \
  --data-urlencode "format=json" \
| jq '
  .rows
  | group_by(.tv)
  | map({
      (.[0].tv|tostring):
        ( map({ (.channel|tostring): { "gain": .pedestal } }) | add )
    })
  | add
' > 2x2_gain_db.json





cd $DATA_DIR_FSD


echo "Querying FSD electron lifetime data..."
curl -s "$URL" \
  --get \
  --data-urlencode "folder=neardet2x2.elifetime" \
  --data-urlencode "t0=0" \
  --data-urlencode "t1=1000000000000" \
  --data-urlencode "data_type=FSD_data" \
  --data-urlencode "format=json" \
| jq '
    .rows
    | map({ (.tv|tostring): [ .elifetime, .err_elifetime ] })
    | add
' > FSD_electron_lifetime_db.json



echo "Querying FSD pedestal data..."

curl -s "$URL" \
  --get \
  --data-urlencode "folder=neardet2x2.pedestal_test" \
  --data-urlencode "t0=0" \
  --data-urlencode "t1=1000000000000" \
  --data-urlencode "data_type=FSD_data" \
  --data-urlencode "format=json" \
| jq '
  .rows
  | group_by(.tv)
  | map({
      (.[0].tv|tostring):
        ( map({ (.channel|tostring): { "pedestal_mv": .pedestal } }) | add )
    })
  | add
' > FSD_pedestal_db.json


echo "Querying FSD gain data..."

curl -s "$URL" \
  --get \
  --data-urlencode "folder=neardet2x2.gain" \
  --data-urlencode "t0=0" \
  --data-urlencode "t1=1000000000000" \
  --data-urlencode "data_type=FSD_data" \
  --data-urlencode "format=json" \
| jq '
  .rows
  | group_by(.tv)
  | map({
      (.[0].tv|tostring):
        ( map({ (.channel|tostring): { "gain": .pedestal } }) | add )
    })
  | add
' > FSD_gain_db.json

cd ${HERE}
