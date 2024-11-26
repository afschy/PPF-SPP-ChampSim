#!/bin/bash

# Paths to the necessary folders
bin_folder="./bin"
trace_folder="./Trace Files"

# Array of binaries
binaries=("spp" "ppf")

# Check if the bin and Trace files directories exist
if [ ! -d "$bin_folder" ]; then
    echo "Error: bin folder not found!"
    exit 1
fi

if [ ! -d "$trace_folder" ]; then
    echo "Error: Trace files folder not found!"
    exit 1
fi

# Iterate over each subfolder in the Trace files folder
for subfolder in "$trace_folder"/*; do
    # Check if it's a directory
    if [ -d "$subfolder" ]; then
        echo "Processing subfolder: $subfolder"
        
        # Extract the <number.name> part from the subfolder name
        folder_base=$(basename "$subfolder")

        # Iterate over each .xz file in the current subfolder
        for trace_file in "$subfolder"/*.xz; do
            # Check if the file exists (to handle empty folders)
            if [ -e "$trace_file" ]; then
                # Extract the <instruction> part from the filename
                instructions=$(basename "$trace_file" | sed -n 's/.*-\(.*\)B\.champsimtrace\.xz/\1/p')
                
                # Iterate over each binary
                for binary in "${binaries[@]}"; do
                    # Construct the path to the binary executable
                    binary_path="$bin_folder/$binary"
                    
                    # Check if the binary exists
                    if [ ! -f "$binary_path" ]; then
                        echo "Error: Binary $binary not found in $bin_folder!"
                        continue
                    fi

                    # Construct the output JSON file name
                    json_output="$subfolder/$instructions.$binary.json"
                    
                    # Run the simulator synchronously (one at a time)
                    echo "Running $binary on $trace_file, output: $json_output"
                    "$binary_path" --warmup-instructions 200000000 --simulation-instructions 500000000 "$trace_file" --json "$json_output"
                    
                    # Wait for the current process to complete before proceeding
                    wait
                done
            else
                echo "No .xz files found in $subfolder."
            fi
        done
    else
        echo "$subfolder is not a directory."
    fi
done

echo "All processes have completed."
