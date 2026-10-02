#!/bin/bash

# Check if the directory is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

DIRECTORY=$1

# Check if the provided argument is a valid directory
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: $DIRECTORY is not a valid directory."
    exit 1
fi

# Loop through all files in the directory
for FILE in "$DIRECTORY"/packet-0050017-2024_07_10_0*.hdf5; do
    if [ -f "$FILE" ]; then
        echo "Processing $FILE..."
        ./run_muon_selection_data.sh "$FILE"
    fi
done

echo "All files processed."