#!/bin/bash

skip_groups=()

for i in $(seq 1 8); do
    if [[ " ${skip_groups[*]} " =~ " $i " ]]; then
        continue
    fi
    python pixelQ.py --n_files 342 --io_group "$i"
done
