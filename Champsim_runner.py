import subprocess
import glob
import re
import os

# Paths to binaries and dataset
bin_folder = '/home/afschy/PPF-SPP-ChampSim/bin'
data_folder = '/mnt/ssd/A/Thesis/champsim_dataset/'

# List of binaries to execute
binaries = ['no', 'spp', 'ppf', 'nn']

# Generate the list of all subfolders in the dataset directory
folder_list = [x for x in glob.glob(os.path.join(data_folder, '*')) if os.path.isdir(x)]

# List to store running processes
proclist = []

# Iterate over each folder
for folder in folder_list:
    print(f"Processing folder: {folder}")
    file_list = glob.glob(os.path.join(folder, '*.xz'))
    for file in file_list:
        # Extract the instruction count from the file name
        filenum = re.search(r'-(.+?)B', file).group(1)

        # Iterate over each binary and create the JSON output for each
        for binary_name in binaries:
            binary_path = os.path.join(bin_folder, binary_name)
            json_output = os.path.join(folder, f"{filenum}.{binary_name}.json")
            
            # Build the command
            command = f"{binary_path} --warmup-instructions 20000000 --simulation-instructions 100000000 {file} --json {json_output}"
            
            # Start the process and add it to the list
            print(f"Running: {command}")
            proclist.append(subprocess.Popen(command, shell=True))

# Wait for all processes to complete
for proc in proclist:
    proc.wait()

print("All processes have completed.")
