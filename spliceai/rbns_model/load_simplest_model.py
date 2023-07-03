'''
load the RBNS data
'''

import csv
import re
import h5py
import urllib3
import urllib
import gzip
from io import StringIO
import requests
import numpy as np
import time

from utils_rbns import create_datapoints

rbns_data = 'metadata_rbns.csv'
encode_dir = 'encode/'
data_dir = 'data/'
protein_idx_map_txt = 'protein_idx_map.txt'

# PROTEIN_NAME = []
# SEQ = []
# CONCENTRATION = []
# PROTEIN_IDX = []
X_train = [] # seq
Y_train = [] # label, 0 for the background data
F_train = [] # concentration
L_train = [] # read length
X_test = []
Y_test = []
F_test = []
L_test = []

protein_dict = {}
protein_idx = 0
step = -1
protein_list = list()

pentamer_protein_acc_dict = {}
pentamer_protein_control_dict = {}

enrichment_dict = {}
read_concentration_time = time.time()
with open(rbns_data, 'r') as f:
    f.readline()
    for line in f:
        data = re.split(',|\n', line)[:-1]
        output_type = data[4]
        expr_target = data[7]
        # if expr_target == 'RBFOX2-human':
        #     print(f"expr_target, {expr_target}")
        #     print(f"output_type, {output_type}")
        url = data[-9]

        if output_type == 'enrichment':
            # print('start getting', url)
            response = urllib.request.urlopen(url)
            enrichment_data = response
            n = 0
            for data_line in enrichment_data:
                decode_line = data_line.decode("utf-8")
                # if expr_target == 'RBFOX2-human':
                #     print(f"decode_line: {decode_line}")
                if n == 0:
                    tmp_concentration_list = re.split('\t|\n', decode_line)[:-1]
                    if 'nM' in decode_line:
                        concentration_list = [float(value.split(' ')[0]) for value in tmp_concentration_list[1:]]
                    else:
                        concentration_list = [float(value) for value in tmp_concentration_list[1:]]
                    print(f"concentration_list: {concentration_list}")
                    if len(concentration_list) == 0:
                        print(f"decode line: {decode_line[:-1]}")
                    
                    # if expr_target == 'RBFOX2-human':
                    #     print(f"decode_line: {decode_line}")
                    # if expr_target == 'RBFOX2-human':
                    #     print(f"concentration_list: {concentration_list}")

                if n == 1:
                    # if expr_target == 'RBFOX2-human':
                    #     print(decode_line)
                    enrichment_list = re.split('\t|\n', decode_line)[:-1]
                    x_mer = enrichment_list[0]
                    if len(x_mer) != 5:
                        break
                    # if expr_target == 'RBFOX2-human':
                    #     print(f"enrichment_list: {enrichment_list}")
                    number = enrichment_list[1:]
                    max_number = -1
                    arg_max = 0
                    for idx, value in enumerate(number):
                        number[idx] = float(number[idx])
                        if number[idx] > max_number:
                            arg_max = idx
                            max_number = number[idx]
                    if '*' in x_mer:
                        x_mer = x_mer.replace('*', '')
                    best_concentration = concentration_list[arg_max]
                    enrichment_dict[expr_target] = (best_concentration, x_mer, max_number) # concentration and highest enrichment pentarmer
                    # if expr_target == 'RBFOX2-human':
                    #     print(f"best concentration: {best_concentration}")
                    # if expr_target == 'RBFOX2-human':
                    #     print(enrichment_list)
                    break
                n += 1
    
# select the protein with top concentration
print(f"enrichment_dict: length: {len(enrichment_dict)} \n {enrichment_dict}")

with open(f"enrichment_pentamer.txt", 'w') as f:
    for target, info in sorted(enrichment_dict.items()):
        f.write(f"{target}, {info[0]}, {info[1]}, {info[2]}\n")
    f.close()
exit(0)
print(f"calculate enrichment time: {time.time() - read_concentration_time}")
# exit(0)
# exit(0)


with open(rbns_data, 'r') as f:
    f.readline()
    i = 0
    flag = 0
    for line in f:
        # print(f"line: {line}")
        data = re.split(',|\n', line)[:-1]

        file_format = data[1]
        expr_target = data[7]
        protein_concentration = data[9].split(' ')[0]
        size = data[10]
        read_length = data[13]
        url = data[25]

        if file_format != 'fastq': #  or len(expr_target) <= 0:
            continue

        if expr_target in protein_dict:
            cur_idx = protein_dict[expr_target]
        else:
            protein_idx += 1
            cur_idx = protein_idx
            protein_dict[expr_target] = protein_idx

        # filter the case where the data is control data and length is larger than 20
        if expr_target == '' and int(read_length) > 20:
            continue
        
        if expr_target == '':
            if flag == 1:
                continue
            print(f"current: control data, cur_idx: {cur_idx}")
            flag = 1
            pass
        else:
            # print(f"target: {enrichment_dict[expr_target]}; current: {protein_concentration}")
            if expr_target in enrichment_dict:
                if enrichment_dict[expr_target][0] != float(protein_concentration):
                    continue
            else:
                continue
        
        start_time = time.time()
        response = urllib.request.urlopen(url)
        # print('requesting time -- %s s --' % (time.time() - start_time))
        control_data = gzip.GzipFile(fileobj=response)
        # print(f"read length: {read_length}")
        # if expr_target == 'RBFOX2-human':
        #     print(f"HERE!: {expr_target}, l: {read_length}")
        #     exit(0)
        # print(f"READ: {expr_target}, {protein_concentration}")

        n = 0

        
        count = 0
        for data_line in control_data:
            if n > 15000: 
                break

            if n%4 == 1:
                # print('here:', data_line)
                # print(cur_idx, expr_target, data_line, protein_concentration)
                seq = data_line.decode("utf-8")[:-1]
                if expr_target == '':
                    for target in enrichment_dict:
                        highest_enrichment_pentamar = enrichment_dict[target][1]
                        if highest_enrichment_pentamar in seq:
                            if target in pentamer_protein_control_dict:
                                pentamer_protein_control_dict[target] += 1
                            else:
                                pentamer_protein_control_dict[target] = 1
                else:
                    highest_enrichment_pentamar = enrichment_dict[expr_target][1]
                    if highest_enrichment_pentamar in seq:
                        if expr_target in pentamer_protein_acc_dict:
                            pentamer_protein_acc_dict[expr_target] += 1
                        else:
                            pentamer_protein_acc_dict[expr_target] = 1
                count += 1

            n += 1
        # print(f"after one expr: {expr_target}")
        # print(f"pentamer_protein_control_dict: {pentamer_protein_control_dict}")
        # print(f"pentamer_protein_acc_dict: {pentamer_protein_acc_dict}")
        
        if expr_target == '':
            for target in pentamer_protein_control_dict:
                pentamer_protein_control_dict[target] = pentamer_protein_control_dict[target] * 1.0 / count
        else:
            pentamer_protein_acc_dict[expr_target] = pentamer_protein_acc_dict[expr_target] * 1.0 / count
        i += 1
        # break

f.close()

with open(f"pentamer_protein_simple_control_dict.txt", 'w') as f:
    for target, acc in sorted(pentamer_protein_control_dict.items()):
        f.write(f"{target}, {acc}\n")
    f.close()

with open(f"pentamer_protein_simple_protein_dict.txt", 'w') as f:
    for target, acc in sorted(pentamer_protein_acc_dict.items()):
        f.write(f"{target}, {acc}\n")

# print('protein', protein_dict)

# with open(protein_idx_map_txt, 'w') as protein_map_f:
#     protein_map_f.write(str(protein_dict))
# protein_map_f.close()

# if len(X_train) <= 0:
#     pass
# else:
#     h5f = h5py.File(data_dir + 'rbns' + '_' + 'train' + '.h5', 'a')
#     h5f['X'].resize((h5f['X'].shape[0] + np.asarray(X_train).shape[0]), axis=0)
#     h5f['X'][-np.asarray(X_train).shape[0]:] = np.asarray(X_train)
#     h5f['Y'].resize((h5f['Y'].shape[0] + np.asarray(Y_train).shape[0]), axis=0)
#     h5f['Y'][-np.asarray(Y_train).shape[0]:] = np.asarray(Y_train)
#     h5f['F'].resize((h5f['F'].shape[0] + np.asarray(F_train).shape[0]), axis=0)
#     h5f['F'][-np.asarray(F_train).shape[0]:] = np.asarray(F_train)
#     h5f['L'].resize((h5f['L'].shape[0] + np.asarray(L_train).shape[0]), axis=0)
#     h5f['L'][-np.asarray(L_train).shape[0]:] = np.asarray(L_train)
#     h5f.close()

#     h5f = h5py.File(data_dir + 'rbns' + '_' + 'test' + '.h5', 'a')
#     h5f['X'].resize((h5f['X'].shape[0] + np.asarray(X_test).shape[0]), axis=0)
#     h5f['X'][-np.asarray(X_test).shape[0]:] = np.asarray(X_test)
#     h5f['Y'].resize((h5f['Y'].shape[0] + np.asarray(Y_test).shape[0]), axis=0)
#     h5f['Y'][-np.asarray(Y_test).shape[0]:] = np.asarray(Y_test)
#     h5f['F'].resize((h5f['F'].shape[0] + np.asarray(F_test).shape[0]), axis=0)
#     h5f['F'][-np.asarray(F_test).shape[0]:] = np.asarray(F_test)
#     h5f['L'].resize((h5f['L'].shape[0] + np.asarray(L_test).shape[0]), axis=0)
#     h5f['L'][-np.asarray(L_test).shape[0]:] = np.asarray(L_test)
#     h5f.close()




# test
# h5f = h5py.File('rbns.h5', 'r')
# X = h5f['X'][:]
# Y = h5f['Y'][:]
# F = h5f['F'][:]
# h5f.close()

# print(X.shape)

# for i in range(X.shape[0]):
#     print(X[i])
#     print(Y[i])
#     print(F[i])

        
        




        




    # print(file_format, expr_target, protein_concentration, size, read_length, url)
        
        
