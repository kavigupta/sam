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
data_dir = 'data_large/' # 'data/'
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
                    best_concentration = concentration_list[arg_max]
                    enrichment_dict[expr_target] = best_concentration
                    # if expr_target == 'RBFOX2-human':
                    #     print(f"best concentration: {best_concentration}")
                    # if expr_target == 'RBFOX2-human':
                    #     print(enrichment_list)
                    break
                n += 1
    
# select the protein with top concentration
print(f"enrichment_dict: length: {len(enrichment_dict)} \n {enrichment_dict}")
print(f"calculate enrichment time: {time.time() - read_concentration_time}")
# exit(0)
# exit(0)


with open(rbns_data, 'r') as f:
    f.readline()
    i = 0
    for line in f:
        # print(f"line: {line}")
        data = re.split(',|\n', line)[:-1]

        file_format = data[1]
        expr_target = data[7]
        protein_concentration = data[9].split(' ')[0]
        size = data[10]
        read_length = data[13]
        url = data[25]
        # print(f"file_format: {file_format}, url:{url}")
        # exit(0)
        # print(expr_target)

        if file_format != 'fastq': #  or len(expr_target) <= 0:
            continue

        if expr_target in protein_dict:
            cur_idx = protein_dict[expr_target]
        else:
            protein_idx += 1
            cur_idx = protein_idx
            protein_dict[expr_target] = protein_idx
        # or float(protein_concentration) <= 0.0s
        # continue
        if i <= step:
            i += 1
            continue

        # filter the case where the data is control data and length is larger than 20
        if expr_target == '' and int(read_length) > 20:
            continue
        
        if expr_target == '':
            print(f"current: control data, cur_idx: {cur_idx}")
            pass
        else:
            if expr_target in enrichment_dict:
                if enrichment_dict[expr_target] != float(protein_concentration):
                    continue
            else:
                continue
        
        start_time = time.time()
        response = urllib.request.urlopen(url)
        # print('requesting time -- %s s --' % (time.time() - start_time))
        control_data = gzip.GzipFile(fileobj=response)

        n = 0
        for data_line in control_data:
            # print(data_line)
            if n > 44000: 
                # print(n)
                break

            if n%4 == 1:
                data, label, concentration, l = create_datapoints(data_line, cur_idx, protein_concentration, read_length)
                # print(n)
                if n < 40000:
                    X_train.append(data)
                    Y_train.append(label)
                    F_train.append(concentration)
                    L_train.append(l)

                elif n < 44000:
                    X_test.append(data)
                    Y_test.append(label)
                    F_test.append(concentration)
                    L_test.append(l)
            
                if n > 43990 or l == 0:
                    print(i, protein_idx, size, data_line, label, concentration, l)
            
            # if n > 50000000: break

            n += 1

        if i == 0:
            # print('create h5 file!')
            h5f = h5py.File(data_dir + 'rbns' + '_' + 'train' + '.h5', 'w')
            # print(np.asarray(X_train).shape)
            h5f.create_dataset('X', data=np.asarray(X_train), maxshape=(None, None, 4))
            h5f.create_dataset('Y', data=np.asarray(Y_train), maxshape=(None, ))
            h5f.create_dataset('F', data=np.asarray(F_train), maxshape=(None, ))
            h5f.create_dataset('L', data=np.asarray(L_train), maxshape=(None, ))
            h5f.close()

            h5f = h5py.File(data_dir + 'rbns' + '_' + 'test' + '.h5', 'w')
            # print(np.asarray(X_test).shape)
            h5f.create_dataset('X', data=np.asarray(X_test), maxshape=(None, None, 4))
            h5f.create_dataset('Y', data=np.asarray(Y_test), maxshape=(None, ))
            h5f.create_dataset('F', data=np.asarray(F_test), maxshape=(None, ))
            h5f.create_dataset('L', data=np.asarray(L_test), maxshape=(None, ))
            h5f.close()
            # print('finish writing h5')

            h5f = h5py.File(data_dir + 'rbns' + '_' + 'train' + '.h5', 'r')
            L = h5f['L'][:]
            h5f.close()
            l_set = set()
            for idx in range(L.shape[0]):
                l = L[idx]
                if l in l_set:
                    continue
                else:
                    l_set.add(l)
            print('l set:', l_set)

            X_train = []
            Y_train = []
            F_train = []
            L_train = []
            X_test = []
            Y_test = []
            F_test = []
            L_test = []

        elif i % 5 == 0:
            # print('L_train to write', L_train)
            h5f = h5py.File(data_dir + 'rbns' + '_' + 'train' + '.h5', 'a')
            h5f['X'].resize((h5f['X'].shape[0] + np.asarray(X_train).shape[0]), axis=0)
            h5f['X'][-np.asarray(X_train).shape[0]:] = np.asarray(X_train)
            h5f['Y'].resize((h5f['Y'].shape[0] + np.asarray(Y_train).shape[0]), axis=0)
            h5f['Y'][-np.asarray(Y_train).shape[0]:] = np.asarray(Y_train)
            h5f['F'].resize((h5f['F'].shape[0] + np.asarray(F_train).shape[0]), axis=0)
            h5f['F'][-np.asarray(F_train).shape[0]:] = np.asarray(F_train)
            h5f['L'].resize((h5f['L'].shape[0] + np.asarray(L_train).shape[0]), axis=0)
            h5f['L'][-np.asarray(L_train).shape[0]:] = np.asarray(L_train)
            h5f.close()

            h5f = h5py.File(data_dir + 'rbns' + '_' + 'test' + '.h5', 'a')
            h5f['X'].resize((h5f['X'].shape[0] + np.asarray(X_test).shape[0]), axis=0)
            h5f['X'][-np.asarray(X_test).shape[0]:] = np.asarray(X_test)
            h5f['Y'].resize((h5f['Y'].shape[0] + np.asarray(Y_test).shape[0]), axis=0)
            h5f['Y'][-np.asarray(Y_test).shape[0]:] = np.asarray(Y_test)
            h5f['F'].resize((h5f['F'].shape[0] + np.asarray(F_test).shape[0]), axis=0)
            h5f['F'][-np.asarray(F_test).shape[0]:] = np.asarray(F_test)
            h5f['L'].resize((h5f['L'].shape[0] + np.asarray(L_test).shape[0]), axis=0)
            h5f['L'][-np.asarray(L_test).shape[0]:] = np.asarray(L_test)
            h5f.close()

            h5f = h5py.File(data_dir + 'rbns' + '_' + 'train' + '.h5', 'r')
            L = h5f['L'][:]
            h5f.close()
            l_set = set()
            for idx in range(L.shape[0]):
                l = L[idx]
                if l in l_set:
                    continue
                else:
                    l_set.add(l)
            print('l set:', l_set)


            X_train = []
            Y_train = []
            F_train = []
            L_train = []
            X_test = []
            Y_test = []
            F_test = []
            L_test = []

        i += 1
        # break

f.close()
# print('protein', protein_dict)

with open(protein_idx_map_txt, 'w') as protein_map_f:
    protein_map_f.write(str(protein_dict))
protein_map_f.close()

if len(X_train) <= 0:
    pass
else:
    h5f = h5py.File(data_dir + 'rbns' + '_' + 'train' + '.h5', 'a')
    h5f['X'].resize((h5f['X'].shape[0] + np.asarray(X_train).shape[0]), axis=0)
    h5f['X'][-np.asarray(X_train).shape[0]:] = np.asarray(X_train)
    h5f['Y'].resize((h5f['Y'].shape[0] + np.asarray(Y_train).shape[0]), axis=0)
    h5f['Y'][-np.asarray(Y_train).shape[0]:] = np.asarray(Y_train)
    h5f['F'].resize((h5f['F'].shape[0] + np.asarray(F_train).shape[0]), axis=0)
    h5f['F'][-np.asarray(F_train).shape[0]:] = np.asarray(F_train)
    h5f['L'].resize((h5f['L'].shape[0] + np.asarray(L_train).shape[0]), axis=0)
    h5f['L'][-np.asarray(L_train).shape[0]:] = np.asarray(L_train)
    h5f.close()

    h5f = h5py.File(data_dir + 'rbns' + '_' + 'test' + '.h5', 'a')
    h5f['X'].resize((h5f['X'].shape[0] + np.asarray(X_test).shape[0]), axis=0)
    h5f['X'][-np.asarray(X_test).shape[0]:] = np.asarray(X_test)
    h5f['Y'].resize((h5f['Y'].shape[0] + np.asarray(Y_test).shape[0]), axis=0)
    h5f['Y'][-np.asarray(Y_test).shape[0]:] = np.asarray(Y_test)
    h5f['F'].resize((h5f['F'].shape[0] + np.asarray(F_test).shape[0]), axis=0)
    h5f['F'][-np.asarray(F_test).shape[0]:] = np.asarray(F_test)
    h5f['L'].resize((h5f['L'].shape[0] + np.asarray(L_test).shape[0]), axis=0)
    h5f['L'][-np.asarray(L_test).shape[0]:] = np.asarray(L_test)
    h5f.close()




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
        
        
