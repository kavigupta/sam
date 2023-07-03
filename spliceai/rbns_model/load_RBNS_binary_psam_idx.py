'''
load the RBNS data
'''

import csv
import re
import h5py
import urllib
import gzip
from io import StringIO
import requests
import numpy as np
import time
import os

from utils_rbns import create_datapoints
from constants import get_args
from map_protein_name import get_map_protein

rbns_data = 'metadata_rbns.csv'
# 'data/'
# protein_idx_map_txt = 'protein_idx_map.txt'

def generate_dataset(target_protein, each_class_train_size, data_dir, map_index, enrichment_dict):

    path = os.path.join(data_dir, f"protein_{target_protein}")
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    path += '/'

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

    with open(rbns_data, 'r') as f:
        f.readline()
        i = 0
        control_flag = 0
        for line in f:
            # print(f"line: {line}")
            data = re.split(',|\n', line)[:-1]

            file_format = data[1]
            expr_target = data[7]
            protein_concentration = data[9].split(' ')[0]
            size = data[10]
            # read_length = data[13]
            read_length = data[10].split('-')[0]
            if read_length == "45":
                print(data[10], data)
                exit(0)
            url = data[25]

            if file_format != 'fastq': #  or len(expr_target) <= 0:
                continue

            if expr_target in protein_dict:
                cur_idx = protein_dict[expr_target]
            else:
                protein_idx += 1
                cur_idx = protein_idx
                protein_dict[expr_target] = protein_idx
            
            if expr_target == '':
                pass
            elif int(protein_idx) in map_index and target_protein in map_index[int(protein_idx)]:
                pass
            else:
                continue

            # filter the case where the data is control data and length is larger than 20
            if expr_target == '' and int(read_length) > 20:
                continue
            
            if expr_target == '':
                if control_flag == 1:
                    continue
                control_flag = 1
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
                if n > each_class_train_size * 4.4: 
                    break

                if n%4 == 1:
                    data, label, concentration, l = create_datapoints(data_line, cur_idx, protein_concentration, read_length)
                    if expr_target == '':
                        label = 0 # control data
                    else:
                        label = 1 # protein data
                    if n < each_class_train_size * 4:
                        X_train.append(data)
                        Y_train.append(label)
                        F_train.append(concentration)
                        L_train.append(l)

                    elif n < each_class_train_size * 4.4:
                        X_test.append(data)
                        Y_test.append(label)
                        F_test.append(concentration)
                        L_test.append(l)
                n += 1

            if i == 0:
                # print('create h5 file!')
                h5f = h5py.File(path + 'rbns' + '_' + 'train' + '.h5', 'w')
                # print(np.asarray(X_train).shape)
                h5f.create_dataset('X', data=np.asarray(X_train), maxshape=(None, None, 4))
                h5f.create_dataset('Y', data=np.asarray(Y_train), maxshape=(None, ))
                h5f.create_dataset('F', data=np.asarray(F_train), maxshape=(None, ))
                h5f.create_dataset('L', data=np.asarray(L_train), maxshape=(None, ))
                h5f.close()

                h5f = h5py.File(path + 'rbns' + '_' + 'test' + '.h5', 'w')
                # print(np.asarray(X_test).shape)
                h5f.create_dataset('X', data=np.asarray(X_test), maxshape=(None, None, 4))
                h5f.create_dataset('Y', data=np.asarray(Y_test), maxshape=(None, ))
                h5f.create_dataset('F', data=np.asarray(F_test), maxshape=(None, ))
                h5f.create_dataset('L', data=np.asarray(L_test), maxshape=(None, ))
                h5f.close()
                # print('finish writing h5')

                X_train = []
                Y_train = []
                F_train = []
                L_train = []
                X_test = []
                Y_test = []
                F_test = []
                L_test = []

            else:
                # print('L_train to write', L_train)
                h5f = h5py.File(path + 'rbns' + '_' + 'train' + '.h5', 'a')
                h5f['X'].resize((h5f['X'].shape[0] + np.asarray(X_train).shape[0]), axis=0)
                h5f['X'][-np.asarray(X_train).shape[0]:] = np.asarray(X_train)
                h5f['Y'].resize((h5f['Y'].shape[0] + np.asarray(Y_train).shape[0]), axis=0)
                h5f['Y'][-np.asarray(Y_train).shape[0]:] = np.asarray(Y_train)
                h5f['F'].resize((h5f['F'].shape[0] + np.asarray(F_train).shape[0]), axis=0)
                h5f['F'][-np.asarray(F_train).shape[0]:] = np.asarray(F_train)
                h5f['L'].resize((h5f['L'].shape[0] + np.asarray(L_train).shape[0]), axis=0)
                h5f['L'][-np.asarray(L_train).shape[0]:] = np.asarray(L_train)
                h5f.close()

                h5f = h5py.File(path + 'rbns' + '_' + 'test' + '.h5', 'a')
                h5f['X'].resize((h5f['X'].shape[0] + np.asarray(X_test).shape[0]), axis=0)
                h5f['X'][-np.asarray(X_test).shape[0]:] = np.asarray(X_test)
                h5f['Y'].resize((h5f['Y'].shape[0] + np.asarray(Y_test).shape[0]), axis=0)
                h5f['Y'][-np.asarray(Y_test).shape[0]:] = np.asarray(Y_test)
                h5f['F'].resize((h5f['F'].shape[0] + np.asarray(F_test).shape[0]), axis=0)
                h5f['F'][-np.asarray(F_test).shape[0]:] = np.asarray(F_test)
                h5f['L'].resize((h5f['L'].shape[0] + np.asarray(L_test).shape[0]), axis=0)
                h5f['L'][-np.asarray(L_test).shape[0]:] = np.asarray(L_test)
                h5f.close()

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

    # with open(protein_idx_map_txt, 'w') as protein_map_f:
    #     protein_map_f.write(str(protein_dict))
    # protein_map_f.close()

    if len(X_train) <= 0:
        pass
    else:
        h5f = h5py.File(path + 'rbns' + '_' + 'train' + '.h5', 'a')
        h5f['X'].resize((h5f['X'].shape[0] + np.asarray(X_train).shape[0]), axis=0)
        h5f['X'][-np.asarray(X_train).shape[0]:] = np.asarray(X_train)
        h5f['Y'].resize((h5f['Y'].shape[0] + np.asarray(Y_train).shape[0]), axis=0)
        h5f['Y'][-np.asarray(Y_train).shape[0]:] = np.asarray(Y_train)
        h5f['F'].resize((h5f['F'].shape[0] + np.asarray(F_train).shape[0]), axis=0)
        h5f['F'][-np.asarray(F_train).shape[0]:] = np.asarray(F_train)
        h5f['L'].resize((h5f['L'].shape[0] + np.asarray(L_train).shape[0]), axis=0)
        h5f['L'][-np.asarray(L_train).shape[0]:] = np.asarray(L_train)
        h5f.close()

        h5f = h5py.File(path + 'rbns' + '_' + 'test' + '.h5', 'a')
        h5f['X'].resize((h5f['X'].shape[0] + np.asarray(X_test).shape[0]), axis=0)
        h5f['X'][-np.asarray(X_test).shape[0]:] = np.asarray(X_test)
        h5f['Y'].resize((h5f['Y'].shape[0] + np.asarray(Y_test).shape[0]), axis=0)
        h5f['Y'][-np.asarray(Y_test).shape[0]:] = np.asarray(Y_test)
        h5f['F'].resize((h5f['F'].shape[0] + np.asarray(F_test).shape[0]), axis=0)
        h5f['F'][-np.asarray(F_test).shape[0]:] = np.asarray(F_test)
        h5f['L'].resize((h5f['L'].shape[0] + np.asarray(L_test).shape[0]), axis=0)
        h5f['L'][-np.asarray(L_test).shape[0]:] = np.asarray(L_test)
        h5f.close()


if  __name__ == "__main__":
    args = get_args()
    # target_protein = args.protein_test
    each_class_train_size = args.each_class_train_size
    data_dir = args.data_dir

    map_index = get_map_protein()
    enrichment_dict = {}
    with open('intermediate_data/enrichment_pentamer.txt', 'r') as f:
        for line in f:
            content = line.split(', ')
            enrichment_dict[content[0]] = float(content[1])
        f.close()
    # print(enrichment_dict)

    # print(map_index)

    for i in range(81):
        if i < 2:
            continue
        generate_dataset(target_protein=i, each_class_train_size=each_class_train_size, data_dir=data_dir, map_index=map_index, enrichment_dict=enrichment_dict)
        print(f"Generatio of {i}-th protein Done!")



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
        
        
