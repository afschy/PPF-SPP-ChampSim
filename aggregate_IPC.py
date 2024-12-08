import os
import json

# Paths to the root folders
trace_files_root = "./Trace files"
binaries = ["no", "spp", "ppf", "nn"]

# Function to calculate aggregate IPC
def calculate_aggregate_ipc(folder_path, binary):
    simpoints_file = os.path.join(folder_path, "simpoints.out")
    weights_file = os.path.join(folder_path, "weights.out")

    # Read numbers and weights
    try:
        with open(simpoints_file, "r") as f:
            simpoints = [int(line.strip()) for line in f.readlines()]
        with open(weights_file, "r") as f:
            weights = [float(line.strip()) for line in f.readlines()]
    except FileNotFoundError:
        print(f"Missing simpoints.out or weights.out in {folder_path}")
        return None

    # Ensure simpoints and weights match
    if len(simpoints) != len(weights):
        print(f"Mismatch in lengths of simpoints and weights in {folder_path}")
        return None

    # Map simpoints to weights
    weights_map = dict(zip(simpoints, weights))

    # Process JSON files
    total_weighted_ipc = 0
    total_weight = 0

    for simpoint in simpoints:
        json_file = os.path.join(folder_path, f"{simpoint}.{binary}.json")
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                data = json.load(f)
                core_data = data[0]["roi"]["cores"][0]
                instructions = core_data["instructions"]
                cycles = core_data["cycles"]
                ipc = instructions / cycles

                weight = weights_map[simpoint]
                total_weighted_ipc += ipc * weight
                total_weight += weight
        else:
            print(f"JSON file for {simpoint} not found for {binary} in {folder_path}")

    if total_weight == 0:
        return None
    return total_weighted_ipc / total_weight

# Main function to iterate through folders and binaries
def process_all_folders():
    for folder in os.listdir(trace_files_root):
        folder_path = os.path.join(trace_files_root, folder)
        if not os.path.isdir(folder_path):
            continue

        print(f"Processing folder: {folder}")
        output_file = os.path.join(folder_path, "aggregate_ipc.txt")
        with open(output_file, "w") as output:
            for binary in binaries:
                print(f"  Calculating for binary: {binary}")
                aggregate_ipc = calculate_aggregate_ipc(folder_path, binary)
                if aggregate_ipc is not None:
                    output.write(f"{binary}: {aggregate_ipc:.6f}\n")
                else:
                    output.write(f"{binary}: Error in calculation\n")
        print(f"Results saved to {output_file}")

# Run the script
if __name__ == "__main__":
    process_all_folders()

