#!/usr/bin/env bash

set -euo pipefail

input_charge=/global/cfs/cdirs/dune/www/data/2x2/nearline/packet/beam/july8_2024/nominal_hv/packet-0050017-2024_07_09_00_14_34_CDT.h5

inputs_light=(
    /global/cfs/cdirs/dune/www/data/2x2/LRS/data_bin003/mpd_run_hvramp_rctl_104_p126.data
    /global/cfs/cdirs/dune/www/data/2x2/LRS/data_bin003/mpd_run_hvramp_rctl_104_p127.data
)

workflows_charge=(
    yamls/proto_nd_flow/workflows/charge/charge_event_building_data.yaml
    yamls/proto_nd_flow/workflows/charge/charge_event_reconstruction_data.yaml
    yamls/proto_nd_flow/workflows/combined/combined_reconstruction_data.yaml
    yamls/proto_nd_flow/workflows/charge/prompt_calibration_data.yaml
    yamls/proto_nd_flow/workflows/charge/filtered_calibration_data.yaml
)

workflows_light_evb=(
    yamls/proto_nd_flow/workflows/light/light_event_building_mpd.yaml
)

workflows_light_reco=(
    yamls/proto_nd_flow/workflows/light/light_event_reconstruction_data.yaml
)

workflows_match=(
    yamls/proto_nd_flow/workflows/charge/charge_light_assoc_data.yaml
)

mkdir -p end2end_outputs
output=end2end_outputs/$(basename "$input_charge" .h5).FLOW.hdf5
output=$(realpath "$output")
rm -f "$output"

cd "$(dirname "${BASH_SOURCE[0]}")"/../..

/usr/bin/time --append -f "charge %P %M %E" -o "$output.time" \
    h5flow -i "$input_charge" -o "$output" -c "${workflows_charge[@]}"

for input_light in "${inputs_light[@]}"; do
    /usr/bin/time --append -f "light_evb %P %M %E" -o "$output.time" \
        h5flow -i "$input_light" -o "$output" -c "${workflows_light_evb[@]}"
done

/usr/bin/time --append -f "light_reco %P %M %E" -o "$output.time" \
    h5flow -i "$output" -o "$output" -c "${workflows_light_reco[@]}"

/usr/bin/time --append -f "match %P %M %E" -o "$output.time" \
    h5flow -i "$output" -o "$output" -c "${workflows_match[@]}"
