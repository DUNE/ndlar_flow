#!/bin/bash
# run ndlar_flow sum_hits and hitfinder on light data

#filelist=/global/homes/s/sfogarty/ndlar_flow_low_energy/ndlar_flow/charge_files_module0.txt
filelist=/sdf/home/s/sfogarty/ndlar_flow/scripts/MC_larndsim_filelist.txt
dir=/sdf/data/neutrino/sfogarty/larndsim_files
#ID=$((SLURM_NODEID*SLURM_NTASKS_PER_NODE + SLURM_LOCALID))
#ID=0
#filename_base=$(sed -n $((ID + 1))p ${filelist})
#filename_base=$(basename "$filepath" .FLOW.hdf5)
#workflow=yamls/proto_nd_flow/workflows/light/light_event_reconstruction_LowEnergy.yaml
workflow_1=yamls/proto_nd_flow/workflows/charge/charge_event_building_LowEnergy_ChargeOnly.yaml
workflow_2=yamls/proto_nd_flow/workflows/charge/charge_event_reconstruction_LowEnergy_ChargeOnly.yaml

files=("larndsim_Th232_gammas_10000_1.h5" "larndsim_Th232_gammas_10000_2.h5" "larndsim_Th232_gammas_10000_3.h5" "larndsim_Th232_gammas_10000_4.h5" "larndsim_Th232_gammas_10000_5.h5" "larndsim_Th232_gammas_10000_6.h5" "larndsim_Th232_gammas_10000_7.h5" "larndsim_Th232_gammas_10000_8.h5")

#INDIR=/global/cfs/cdirs/dune/www/data/Module0/TPC1+2/dataRuns/packetData
#OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/flow_LowEnergy/2x2/LowThreshold_July12/CLMatched
#OUTDIR=/pscratch/sd/s/sfogarty/module0_low_energy/charge
#OUTFILE=${OUTDIR}/module0_charge_reco_${filename_base}.FLOW.hdf5
cd /sdf/home/s/sfogarty/ndlar_flow
#pip install .
for filename in "${files[@]}"; do
    #filename=$(sed -n $((i + 1))p ${filelist})
    echo "input filename: ${dir}/${filename}"
    #filepath_i=${dir}/packet-0050040-2024_07_12_16_38_32_CDT.h5
    filepath_i=${dir}/${filename}
    #filepath_i=/global/cfs/cdirs/dune/users/sfogarty/flow_LowEnergy/2x2/LowThreshold_July12/CLMatched/packet-0050040-2024_07_12_16_38_32_CDT.FLOW.h5
    #filepath_o=/sdf/data/neutrino/sfogarty/flow_files/flow_clusters_${filename}
    filepath_o=flow_clusters_${filename}
    #h5flow -c $workflow_1 $workflow_2 $workflow_3 $workflow_4 -i $filepath_i -o $filepath_o
    h5flow -c $workflow_1 -i $filepath_i -o $filepath_o

done
