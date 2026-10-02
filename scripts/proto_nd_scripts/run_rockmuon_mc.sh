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
for FILE in "$DIRECTORY"/MiniRun6.5_1E19_RHC.flow.00006*; do
    if [ -f "$FILE" ]; then
        echo "Processing $FILE..."
        ./run_muon_selection_MC.sh "$FILE"
    fi
done

echo "All files processed."