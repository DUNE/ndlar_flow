#!/usr/bin/env bash

set -euo pipefail

input_charge=/global/cfs/cdirs/dune/www/data/FSD/nearline/packet/CRS/cosmics/07Nov2024/packet-0020113-2024_11_08_03_01_12_CET.h5

inputs_light=(
    /global/cfs/cdirs/dune/www/data/FSD/LRS/data_bin_08/mpd_run_data_rctl_243_p113.data
    /global/cfs/cdirs/dune/www/data/FSD/LRS/data_bin_08/mpd_run_data_rctl_243_p114.data
)

workflows_charge=(
    yamls/fsd_flow/workflows/charge/charge_event_building_data.yaml
    yamls/fsd_flow/workflows/charge/charge_event_reconstruction_data.yaml
    yamls/fsd_flow/workflows/combined/combined_reconstruction_data.yaml
    yamls/fsd_flow/workflows/charge/prompt_calibration_data.yaml
    yamls/fsd_flow/workflows/charge/final_calibration_data.yaml
)

workflows_light_evb=(
    yamls/fsd_flow/workflows/light/light_event_building_mpd.yaml
)

workflows_light_reco=(
    yamls/fsd_flow/workflows/light/light_event_reconstruction_data.yaml
)

workflows_match=(
    yamls/fsd_flow/workflows/charge/charge_light_assoc_data.yaml
)

mkdir -p end2end_outputs
output=end2end_outputs/$(basename "$input_charge" .h5).FLOW.hdf5
output=$(realpath "$output")
rm -f "$output"

cd "$(dirname "${BASH_SOURCE[0]}")"/../..

/usr/bin/time --append -f "charge %P %M %E" -o "$output.time" \
    h5flow -i "$input_charge" -o "$output" -c "${workflows_charge[@]}"

## temporarily skip light
## missing light/events/rms during reco

# for input_light in "${inputs_light[@]}"; do
#     /usr/bin/time --append -f "light_evb %P %M %E" -o "$output.time" \
#         h5flow -i "$input_light" -o "$output" -c "${workflows_light_evb[@]}"
# done

# /usr/bin/time --append -f "light_reco %P %M %E" -o "$output.time" \
#     h5flow -i "$output" -o "$output" -c "${workflows_light_reco[@]}"

# /usr/bin/time --append -f "match %P %M %E" -o "$output.time" \
#     h5flow -i "$output" -o "$output" -c "${workflows_match[@]}"
