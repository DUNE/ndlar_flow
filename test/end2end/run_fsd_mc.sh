#!/usr/bin/env bash

set -euo pipefail

input=/global/cfs/cdirs/dunepro/people/cuddandr/FSD_CosmicRun4_beta/run-larnd-sim/FSD_CosmicRun4.larnd/LARNDSIM/0000000/FSD_CosmicRun4.larnd.0000123.LARNDSIM.hdf5

workflows_charge=(
    yamls/fsd_flow/workflows/charge/charge_event_building_mc.yaml
    yamls/fsd_flow/workflows/charge/charge_event_reconstruction_mc.yaml
    yamls/fsd_flow/workflows/combined/combined_reconstruction_mc.yaml
    yamls/fsd_flow/workflows/charge/prompt_calibration_mc.yaml
    yamls/fsd_flow/workflows/charge/filtered_calibration_mc.yaml
)

workflows_light=(
    yamls/fsd_flow/workflows/light/light_event_building_mc.yaml
    yamls/fsd_flow/workflows/light/light_event_reconstruction_mc.yaml
)

workflows_match=(
    yamls/fsd_flow/workflows/charge/charge_light_assoc_mc.yaml
)


mkdir -p end2end_outputs
output=end2end_outputs/$(basename "$input" .LARNDSIM.hdf5).FLOW.hdf5
output=$(realpath "$output")
rm -f "$output"

cd "$(dirname "${BASH_SOURCE[0]}")"/../..

/usr/bin/time --append -f "charge %P %M %E" -o "$output.time" \
    h5flow -i "$input" -o "$output" -c "${workflows_charge[@]}"

## temporarily skip light
## need to respin larnd-sim (or add logic to split light_dat_allmodules)

# /usr/bin/time --append -f "light %P %M %E" -o "$output.time" \
#     h5flow -i "$input" -o "$output" -c "${workflows_light[@]}"

# /usr/bin/time --append -f "match %P %M %E" -o "$output.time" \
#     h5flow -i "$output" -o "$output" -c "${workflows_match[@]}"
