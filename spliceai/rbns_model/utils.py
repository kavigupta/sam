import torch
import numpy as np

from sklearn.metrics import average_precision_score

import itertools
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

def batched(m, x, bs):
    import torch

    with torch.no_grad():
        ys = []
        for i in range((x.shape[0] + bs - 1) // bs):
            y = m(torch.tensor(x[i * bs : i * bs + bs]).cuda()).detach().cpu().numpy()
            ys.append(y)
        return np.concatenate(ys)


def clip_datapoints(X, Y, CL, N_GPUS, *, CL_max):
    # This function is necessary to make sure of the following:
    # (i) Each time model_m.fit is called, the number of datapoints is a
    # multiple of N_GPUS. Failure to ensure this often results in crashes.
    # (ii) If the required context length is less than CL_max, then
    # appropriate clipping is done below.
    # Additionally, Y is also converted to a list (the .h5 files store
    # them as an array).

    rem = X.shape[0] % N_GPUS
    clip = (CL_max - CL) // 2

    if rem != 0 and clip != 0:
        return X[:-rem, clip:-clip], [Y[t][:-rem] for t in range(1)]
    elif rem == 0 and clip != 0:
        return X[:, clip:-clip], [Y[t] for t in range(1)]
    elif rem != 0 and clip == 0:
        return X[:-rem], [Y[t][:-rem] for t in range(1)]
    else:
        return X, [Y[t] for t in range(1)]


def clip_datapoint(x, *, CL, CL_max):
    clip = (CL_max - CL) // 2
    if clip == 0:
        return x
    assert clip > 0
    return x[clip:-clip]


def modify_sl(x, y, SL):
    if y.shape[0] < SL:
        return [x], [y]
    CL = x.shape[0] - y.shape[0]
    # if y.shape[0] != SL:
    #     print(y.shape[0], SL)
    assert y.shape[0] % SL == 0, "the SL must be a factor of the SL from the data"
    chunks = y.shape[0] // SL
    xs = []
    ys = []
    for c in range(chunks):
        start, end = c * SL, (c + 1) * SL
        xs.append(x[start : end + CL])
        ys.append(y[start:end])
    return xs, ys


def print_topl_statistics(
    y_true, y_pred, quiet=False, lengths=[0.5, 1, 2, 4], compute_auprc=True
):
    # Prints the following information: top-kL statistics for k=0.5,1,2,4,
    # auprc, thresholds for k=0.5,1,2,4, number of true splice sites.

    idx_true = np.nonzero(y_true == 1)[0]

    topkl_accuracy = {}
    threshold = {}

    if quiet:
        lengths = [1]

    for top_length in lengths:

        kl = int(top_length * len(idx_true))

        idx_pred = np.argpartition(y_pred, -kl)[-kl:]
        threshold[top_length] = y_pred[idx_pred[0]]

        intersection_size = np.size(np.intersect1d(idx_true, idx_pred))
        max_intersection_size = min(len(idx_pred), len(idx_true))
        if max_intersection_size == 0:
            topkl_accuracy[top_length] = np.nan
        else:
            topkl_accuracy[top_length] = intersection_size / max_intersection_size

    if not quiet:
        print(f"N = {len(y_pred)}")
        for length in lengths:
            print(
                f"l = {length}, top-kl acc = {topkl_accuracy[length]:.4f}, threshold = {threshold[length]:.4f}"
            )
        if compute_auprc:
            auprc = average_precision_score(y_true, y_pred)
            print(f"auprc = {auprc}")

    return topkl_accuracy[1]


def prod(x):
    res = 1
    for i in x:
        res *= i
    return res


def permute(repeat=5):
    permute_dict = dict()
    for i in itertools.product(['A', 'C', 'T', 'G'], repeat=repeat):
        seq = ''.join(i)
        permute_dict[seq] = list()
    return permute_dict


def permute_l(repeat=5):
    permute_list = list()
    for i in itertools.product(['A', 'C', 'T', 'G'], repeat=repeat):
        seq = ''.join(i)
        permute_list.append(seq)
    return permute_list


def count_str(permute, l_array):
    cnt = 0
    for l in l_array:
        str_l = list2seq(l)
        cnt += str_l.count(permute)
    return cnt


def extract_dist(permute_dict, tp_list, fp_list, tn_list, fn_list):
    for permute in permute_dict:
        tp_cnt = count_str(permute, tp_list)
        fp_cnt = count_str(permute, fp_list)
        tn_cnt = count_str(permute, tn_list)
        fn_cnt = count_str(permute, fn_list)
        permute_dict[permute] = [tp_cnt, fp_cnt, tn_cnt, fn_cnt]
    
    return permute_dict


def write_permute_res(permute_dict, path, acc=None):
    f = open(path, 'w')
    f.write(f"{acc}, TP, FP, TN, FN\n")
    for permute in sorted(permute_dict):
        l = permute_dict[permute]
        f.write(f"{permute}, {l[0]}, {l[1]}, {l[2]}, {l[3]}\n")
    f.close()







