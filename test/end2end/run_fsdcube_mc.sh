#!/usr/bin/env bash

set -euo pipefail

input=/global/cfs/cdirs/dune/www/data/2x2/people/mkramer/fsd_cube_showers/prc2_r4_patch_noDrop/showers.prc2_r4_patch_noDrop.00.LARND.hdf5

workflows_charge=(
    yamls/fsdcube_flow/workflows/charge/charge_event_building_mc.yaml
    yamls/fsdcube_flow/workflows/charge/charge_event_reconstruction_mc.yaml
    yamls/fsdcube_flow/workflows/combined/combined_reconstruction_mc.yaml
    yamls/fsdcube_flow/workflows/charge/prompt_calibration_mc.yaml
    yamls/fsdcube_flow/workflows/charge/filtered_calibration_mc.yaml
)

mkdir -p end2end_outputs
output=end2end_outputs/$(basename "$input" .LARND.hdf5).FLOW.hdf5
output=$(realpath "$output")
rm -f "$output"

cd "$(dirname "${BASH_SOURCE[0]}")"/../..

/usr/bin/time --append -f "charge %P %M %E" -o "$output.time" \
    h5flow -i "$input" -o "$output" -c "${workflows_charge[@]}"
