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
                if n < 1:
                    n += 1
                    continue
                # if expr_target == 'RBFOX2-human':
                #     print(decode_line)
                enrichment_list = re.split('\t|\n', decode_line)[:-1]
                x_mer = enrichment_list[0]
                if len(x_mer) != 5:
                    break

                if n == 1:
                    enrichment_dict[expr_target] = [x_mer] # concentration and highest enrichment pentarmer
                else:
                    enrichment_dict[expr_target].append(x_mer)

                n += 1
    

with open(f"dataset_analysis.txt", 'w') as f:
    for target, l in sorted(enrichment_dict.items()):
        f.write(f"{target}, ")
        for x_mer in l:
            f.write(f"{x_mer}, ")
        f.write(f"\n")
    f.close()
