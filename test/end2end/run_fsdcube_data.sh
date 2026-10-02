#!/usr/bin/env bash

set -euo pipefail

input_charge=/global/cfs/cdirs/dune/www/data/2x2/people/mkramer/fsd_cube_showers/prc2.packet-cosmics-2026_02_28_13_58_02_PST.hdf5

workflows_charge=(
    yamls/fsdcube_flow/workflows/charge/charge_event_building_data.yaml
    yamls/fsdcube_flow/workflows/charge/charge_event_reconstruction_data.yaml
    yamls/fsdcube_flow/workflows/combined/combined_reconstruction_data.yaml
    yamls/fsdcube_flow/workflows/charge/prompt_calibration_data.yaml
    yamls/fsdcube_flow/workflows/charge/filtered_calibration_data.yaml
)

mkdir -p end2end_outputs
output=end2end_outputs/$(basename "$input_charge" .h5).FLOW.hdf5
output=$(realpath "$output")
rm -f "$output"

cd "$(dirname "${BASH_SOURCE[0]}")"/../..

/usr/bin/time --append -f "charge %P %M %E" -o "$output.time" \
    h5flow -i "$input_charge" -o "$output" -c "${workflows_charge[@]}"
