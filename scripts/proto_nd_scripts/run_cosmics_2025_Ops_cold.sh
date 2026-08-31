#!/bin/bash

for file in /global/cfs/cdirs/dune/www/data/2x2/nearline_run2/flowed_charge/ColdOperations/data/2025_Operations_Cold/Commission/Cosmics_1010/*.hdf5; do
	./run_muon_selection_data.sh $file; done

