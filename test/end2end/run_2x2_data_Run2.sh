#!/usr/bin/env bash

set -euo pipefail

input_charge=/global/cfs/cdirs/dune/www/data/2x2/nearline_run2/packet/ColdOperations/data/2025_Operations_Cold/source/AmBe_1107/packet-0060100-2025_11_10_17_01_27_CST.h5

inputs_light=(
    /global/cfs/cdirs/dune/www/data/2x2/LRS_run2/source_ambe_bin3/two_trig_109us_period/mpd_run_data_rctl_633_p89.data
    /global/cfs/cdirs/dune/www/data/2x2/LRS_run2/source_ambe_bin3/two_trig_109us_period/mpd_run_data_rctl_633_p90.data
    /global/cfs/cdirs/dune/www/data/2x2/LRS_run2/source_ambe_bin3/two_trig_109us_period/mpd_run_data_rctl_633_p91.data
    /global/cfs/cdirs/dune/www/data/2x2/LRS_run2/source_ambe_bin3/two_trig_109us_period/mpd_run_data_rctl_633_p92.data
    /global/cfs/cdirs/dune/www/data/2x2/LRS_run2/source_ambe_bin3/two_trig_109us_period/mpd_run_data_rctl_633_p93.data
    /global/cfs/cdirs/dune/www/data/2x2/LRS_run2/source_ambe_bin3/two_trig_109us_period/mpd_run_data_rctl_633_p94.data
    /global/cfs/cdirs/dune/www/data/2x2/LRS_run2/source_ambe_bin3/two_trig_109us_period/mpd_run_data_rctl_633_p95.data
    /global/cfs/cdirs/dune/www/data/2x2/LRS_run2/source_ambe_bin3/two_trig_109us_period/mpd_run_data_rctl_633_p96.data
    /global/cfs/cdirs/dune/www/data/2x2/LRS_run2/source_ambe_bin3/two_trig_109us_period/mpd_run_data_rctl_633_p97.data
    /global/cfs/cdirs/dune/www/data/2x2/LRS_run2/source_ambe_bin3/two_trig_109us_period/mpd_run_data_rctl_633_p98.data
    /global/cfs/cdirs/dune/www/data/2x2/LRS_run2/source_ambe_bin3/two_trig_109us_period/mpd_run_data_rctl_633_p99.data
)

workflows_charge=(
    yamls/proto_nd_flow/workflows/charge/charge_event_building_data_Run2.yaml
    yamls/proto_nd_flow/workflows/charge/charge_event_reconstruction_data_Run2.yaml
    yamls/proto_nd_flow/workflows/combined/combined_reconstruction_data.yaml
    yamls/proto_nd_flow/workflows/charge/prompt_calibration_data_Run2.yaml
    yamls/proto_nd_flow/workflows/charge/final_calibration_data_Run2.yaml
)

workflows_light_evb=(
    yamls/proto_nd_flow/workflows/light/light_event_building_mpd_Run2.yaml
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
