#!/usr/bin/env bash

set -euo pipefail

input=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-larnd-sim/MiniProdN5p1_NDComplex_FHC.larnd.full.sanddrift/LARNDSIM/0000000/MiniProdN5p1_NDComplex_FHC.larnd.full.sanddrift.0000123.LARNDSIM.hdf5

workflows_charge=(
    yamls/ndlar_flow/workflows/charge/charge_event_building_mc.yaml
    yamls/ndlar_flow/workflows/charge/charge_event_reconstruction_mc.yaml
    yamls/ndlar_flow/workflows/combined/combined_reconstruction_mc.yaml
    yamls/ndlar_flow/workflows/charge/prompt_calibration_mc.yaml
    yamls/ndlar_flow/workflows/charge/merged_calibration_mc.yaml
)

workflows_light=(
    yamls/ndlar_flow/workflows/light/light_event_building_mc.yaml
    yamls/ndlar_flow/workflows/light/light_event_reconstruction_mc.yaml
)

workflows_match=(
    yamls/ndlar_flow/workflows/charge/charge_light_assoc_mc.yaml
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
