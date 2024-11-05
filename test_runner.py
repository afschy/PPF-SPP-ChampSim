import sys, subprocess, glob, re

bin_folder = '/home/afschy/PPF-SPP-ChampSim/bin'
bin_name = 'ppf_og'
binary = bin_folder + '/' + bin_name

data_folder = '/mnt/ssd/A/Thesis/champsim_dataset/'

folder_num_list = input()
folder_num_list = [int(x) for x in folder_num_list.split()]

folder_list = []
for num in folder_num_list:
    for x in glob.glob(data_folder + str(num) + '*', recursive=False):
        folder_list.append(x)

proclist = []
for folder in folder_list:
    file_list = glob.glob(folder + '/*.xz')
    for file in file_list:
        filenum = re.search('-(.+?)B', file).group(1)
        command = binary + ' --warmup-instructions 20000000 --simulation-instructions 100000000 ' + file + ' --json ' + folder + '/' + filenum + '.' + bin_name + '.json ' + '> /dev/null'
        proclist.append(subprocess.Popen(command, shell=True))

for proc in proclist:
    proc.wait()