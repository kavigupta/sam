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

protein_idx = 0
protein_list = list()
protein_dict = {}

with open(rbns_data, 'r') as f:
    f.readline()
    i = 0
    for line in f:
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
            protein_list.append(expr_target)

f = open('protein_rbns_name.txt', 'w')
for protein_name in protein_list:
    f.write(protein_name.split('-')[0] + '\n')
f.close()
