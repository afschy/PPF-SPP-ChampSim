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

# Function to run a single simulation
run_simulation() {
    local binary="$1"
    local trace_file="$2"
    local subfolder="$3"
    local instructions="$4"

    local binary_path="$bin_folder/$binary"
    local json_output="$subfolder/$instructions.$binary.json"

    if [ -f "$binary_path" ]; then
        echo "Running $binary on $trace_file, output: $json_output"
        "$binary_path" --warmup-instructions 200000000 --simulation-instructions 500000000 "$trace_file" --json "$json_output"
    else
        echo "Error: Binary $binary not found in $bin_folder!"
    fi
}

export -f run_simulation

# Prepare parallel input
parallel_jobs=()
for subfolder in "$trace_folder"/*; do
    if [ -d "$subfolder" ]; then
        echo "Processing subfolder: $subfolder"

        folder_base=$(basename "$subfolder")
        for trace_file in "$subfolder"/*.xz; do
            if [ -e "$trace_file" ]; then
                instructions=$(basename "$trace_file" | sed -n 's/.*-\(.*\)B\.champsimtrace\.xz/\1/p')
                for binary in "${binaries[@]}"; do
                    parallel_jobs+=("$binary $trace_file $subfolder $instructions")
                done
            else
                echo "No .xz files found in $subfolder."
            fi
        done
    else
        echo "$subfolder is not a directory."
    fi
done

# Run jobs in parallel
printf "%s\n" "${parallel_jobs[@]}" | parallel -j $(nproc) --colsep ' ' run_simulation

echo "All processes have completed."
