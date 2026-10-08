#!/usr/bin/env bash

set -euo pipefail

input=/global/cfs/cdirs/dune/www/data/2x2/simulation/productions/MiniRun6.5_1E19_RHC/MiniRun6.5_1E19_RHC.larnd/LARNDSIM/0000000/MiniRun6.5_1E19_RHC.larnd.0000123.LARNDSIM.hdf5

workflows_charge=(
    yamls/proto_nd_flow/workflows/charge/charge_event_building_mc.yaml
    yamls/proto_nd_flow/workflows/charge/charge_event_reconstruction_mc.yaml
    yamls/proto_nd_flow/workflows/combined/combined_reconstruction_mc.yaml
    yamls/proto_nd_flow/workflows/charge/prompt_calibration_mc.yaml
    yamls/proto_nd_flow/workflows/charge/filtered_calibration_mc.yaml
)

workflows_light=(
    yamls/proto_nd_flow/workflows/light/light_event_building_mc.yaml
    yamls/proto_nd_flow/workflows/light/light_event_reconstruction_mc.yaml
)

workflows_match=(
    yamls/proto_nd_flow/workflows/charge/charge_light_assoc_mc.yaml
)


mkdir -p end2end_outputs
output=end2end_outputs/$(basename "$input" .LARNDSIM.hdf5).FLOW.hdf5
output=$(realpath "$output")
rm -f "$output"

cd "$(dirname "${BASH_SOURCE[0]}")"/../..

/usr/bin/time --append -f "charge %P %M %E" -o "$output.time" \
    h5flow -i "$input" -o "$output" -c "${workflows_charge[@]}"

/usr/bin/time --append -f "light %P %M %E" -o "$output.time" \
    h5flow -i "$input" -o "$output" -c "${workflows_light[@]}"

/usr/bin/time --append -f "match %P %M %E" -o "$output.time" \
    h5flow -i "$output" -o "$output" -c "${workflows_match[@]}"
