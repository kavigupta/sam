
import numpy as np
import re

IN_MAP = np.asarray(
    [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
)
# One-hot encoding of the inputs: 0 is for padding, and 1, 2, 3, 4 correspond
# to A, C, G, T respectively.

in_map_list = IN_MAP.tolist()
base_map_dict = {
        0: 'N', 
        1: 'A',
        2: 'C',
        3: 'G',
        4: 'T',
    }

def create_datapoints(seq, protein_idx, concentration, read_length):
    
    seq = seq.decode("utf-8")[:-1]
    Y = protein_idx
    F = int(concentration)
    L = int(read_length)

    length = len(seq)
    seq = seq + 'N' * (50 - length)
    seq = seq.upper().replace("A", "1").replace("C", "2")
    seq = seq.replace("G", "3").replace("T", "4").replace("N", "0")

    X = np.asarray(list(map(int, list(seq))))
    X = IN_MAP[X.astype("int8")]

    if F == 0:
        F = 20 # use a arbitraty value
        Y = 0

    return X, Y, F, L


def list2seq(X):
    seq = ''
    for x in X.tolist():
        idx = in_map_list.index(x)
        seq += base_map_dict[idx]
    return seq

