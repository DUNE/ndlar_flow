#!/bin/bash
# Uses h5repack to compress datasets within files
#
# Will convert files from *.h5 -> *.gz.h5 and delete original
# if compression successful
#
# Usage:
#   ./compress_files.sh <nparallel> <file 1> <file 2>
#
nproc=$1
shift

for file in "$@"; do
    if [[ $file = *.h5 ]]; then
	if [[ $file = *.gz.h5 ]]; then
	    continue
	fi
	echo $(basename $file)
	time h5repack -f GZIP=9 $file ${file//.h5/.gz.h5} && rm -fv $file &
    fi

    pids=( $(jobs -p) )
    if [ "${#pids[@]}" -ge $nproc ]; then
	echo "Waiting for jobs to finish..."
	wait -n
    fi
done

wait
