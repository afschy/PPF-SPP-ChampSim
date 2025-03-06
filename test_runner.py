import subprocess, glob, re

bin_folder = '/home/afschy/PPF-SPP-ChampSim/bin'
bin_name_list = ['nn_tiny']

data_folder = '/mnt/ssd/A/Thesis/champsim_dataset/'

folder_num_list = input()
folder_num_list = [int(x) for x in folder_num_list.split()]

folder_list = []
for num in folder_num_list:
    for x in glob.glob(data_folder + str(num) + '*', recursive=False):
        folder_list.append(x)

proclist = []
list_size = 0

for bin_name in bin_name_list:
    binary = bin_folder + '/' + bin_name
    for folder in folder_list:
        file_list = glob.glob(folder + '/*.xz')
        for file in file_list:
            filenum = re.search('-(.+?)B', file).group(1)
            json_output = folder + '/' + filenum + '.' + bin_name + '.json'
            # command = binary + ' --warmup-instructions 20000000 --simulation-instructions 100000000 ' + file + ' --json ' + folder + '/' + filenum + '.' + bin_name + '.json ' + '> /dev/null'
            command = f"{binary} --warmup-instructions 200000000 --simulation-instructions 1000000000 {file} --json {json_output} > {folder}/{filenum}.{bin_name}.txt"
            proclist.append(subprocess.Popen(command, shell=True))
            list_size += 1

completed = 0
print(f"Starting {list_size} processes.")
for proc in proclist:
    proc.wait()
    completed += 1
    print(f"Completed {completed}/{list_size}")