#!/bin/bash
# Uses h5repack to compress datasets within files
#
# Will convert files from *.h5 -> *.gz.h5 and delete original
# if compression successful
#
# Usage:
#   ./compress_files.sh <nparallel> <suffix> <file 1> <file 2> ...
#
nproc=$1
shift
suffix=$1
shift

workdir=$SCRATCH/compress_files/

for file in "$@"; do
    if [[ $file = *$suffix ]]; then
	if [[ $file = *.gz.h5 ]]; then
	    continue
	fi
	echo $(basename $file)
	infile=$workdir/$(basename $file)
	outfile=${file//${suffix}/.gz.h5}    
	tempfile=$workdir/$(basename ${outfile})
	if [[ ! -e $infile ]]; then
	    cp -v $file $infile
	fi
	time h5repack -f GZIP=9 $infile $tempfile && mv -fv $tempfile $outfile && rm -fv $infile && rm -fv $file &
    fi

    pids=( $(jobs -p) )
    if [ "${#pids[@]}" -ge $nproc ]; then
	echo "Waiting for jobs to finish..."
	wait -n
    fi
done

wait
