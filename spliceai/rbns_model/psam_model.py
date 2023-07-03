import h5py
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os

from random import shuffle
import itertools, operator

from load_psam_motif import *
from map_protein_name import *

import random

from utils import (
    extract_dist,
    list2seq,
    permute,
)

class MotifModel(nn.Module):

    def __init__(self, motifs, num_proteins=104, protein=1):
        super(MotifModel, self).__init__()
        self.motifs = motifs # read_motifs()
        self.protein_idx = protein

    def forward(self, input, mc, l, prod=True): # X, concentration
        # print('input shape', input.shape)
        # print('input', input)
        mc = mc.unsqueeze(1)
        output = torch.tensor(motifs_for(self.motifs, input)).cuda()

        # output = F.normalize(output, p=2, dim=1)
        output[output > 1] = 1
        control_prediction = output[:, :, self.protein_idx: self.protein_idx+1]
        return control_prediction, mc

        # if prod:
        #     control_prediction = torch.prod(1-control_prediction, dim=1)
        #     target_prediction = 1 - control_prediction
        #     target_prediction = mc / (mc + 1/target_prediction)
        #     # output = target_prediction.squeeze(1)
        # else:
        #     # print(control_prediction.shape)
        #     target_prediction, _ = torch.max(control_prediction, dim=1)
        #     # print(target_prediction.shape)
        #     target_prediction = mc / (mc + 1/target_prediction)
        #     # output = target_prediction
        # # # print(f"occupancy: {occupancy.shape}")
        # # # exit(0)

        # # # print(f"target_prediction: {target_prediction}")
        # output = target_prediction.squeeze(1)
        # # # exit(0)

        # # # return prob # output
        # # # print('pred  shape', psam_prediction.shape)
        # # # return psam_prediction
        # # # print(f"output shape: {output.shape}")
        # # # exit(0)
        # return output


class MotifModelDataset(torch.utils.data.IterableDataset):
    def __init__(self, path):
        self.path = path
        data = h5py.File(path, "r")
        self._l = self.dsize(data)
    
    def __len__(self):
        return self._l
    
    @staticmethod
    def dsize(data):
        L = data["L"][:]
        return L.shape[0]
    
    @staticmethod
    def ranked_size(data):
        i_s = list()
        idx_s = list(range(MotifModelDataset.dsize(data)))
        for i in idx_s:
            l = data["L"][i]
            i_s.append([i, l])
        return i_s
    
    @staticmethod
    def shuffle_group_by_l(data, size=500):
        # group i_s with batch_size: 500/1000, then shuffle i_s in the unit of groups

        i_s = MotifModelDataset.ranked_size(data)
        for idx, unit in enumerate(i_s):
            i_s[idx].append(int(idx / size))

        groups = [list(g) for _, g in itertools.groupby(i_s, operator.itemgetter(2))]
        shuffle(groups)
        # print(groups)
        i_s = [item[0] for group in groups for item in group]
        # print(i_s)

        return i_s
        
    def __iter__(self):
        data = h5py.File(self.path, "r")

        # i_s = list(range(MotifModelDataset.dsize(data)))
        # shuffle(i_s)
        
        i_s = MotifModelDataset.shuffle_group_by_l(data)

        for i in i_s:
            X = data["X"][i]
            Y = data["Y"][i]
            F = data["F"][i]
            L = data["L"][i]

            yield X[:L].astype(np.float32), Y.astype(np.long), F.astype(np.float32), L.astype(np.float32)
    
    def loader(self, batch_size):
        l = torch.utils.data.DataLoader(self, num_workers=0, batch_size=batch_size)
        return l


def equal_length(x):
    unique_x = torch.unique(x)
    res = unique_x.shape[0]
    if res == 1:
        return True
    else:
        return False


def evaluate_model(m, d, map_index, threshold_list, limit=float("inf"), bs=128, pbar=lambda x: x, num_proteins=2, quiet=True, evaluate_result=False):
    count = 0
    threshold_dict = dict()
    for threshold in threshold_list:
        threshold_dict[threshold] = {
            "total_correct": 0,
            "total_seq": 0, 
            "class_correct": list(0.0 for i in range(num_proteins)),
            "class_total": list(0.0 for i in range(num_proteins))
        }
    
    if evaluate_result:
        permute_dict = permute(repeat=5)
        # each list include number of TP, FP, TN, FN
        tp_list, fp_list, tn_list, fn_list = list(), list(), list(), list()

    try:
        m.eval()
        dataset = d.loader(bs)

        for i, (x, y, f, l) in enumerate(dataset):

            x = x.cuda()
            y = y.cuda()
            f = f.cuda()
            l = l.cuda()

            y = y.unsqueeze(1)
            with torch.no_grad():
                real_yp, mc = m(x, f, l)

                for threshold in threshold_dict:
                    yp = real_yp
                    # yp[yp < threshold] = 0
                    yp = torch.prod(1-yp, dim=1)
                    yp = 1 - yp
                    yp = mc / (mc + 1/yp)
                    yp = yp.squeeze(1)

                    predicted = yp > threshold
                    predicted_label = predicted.cpu().numpy().tolist()
                    y = torch.squeeze(y)

                    for idx, value in enumerate(predicted_label):
                        y_label = y[idx].data.item() # map_index[y[idx].data.item()]
                        # print(f"y_label: {y_label}")
                        # if len(y_label) > 1:
                        #     print(y_label)
                        yp_label = predicted[idx].data.item()
                        # print(y_label, yp_label)
                        if yp_label == y_label:
                            threshold_dict[threshold]["total_correct"] += 1
                            threshold_dict[threshold]["class_correct"][yp_label] += 1
                        # for sub_y_label in y_label:
                        threshold_dict[threshold]["class_total"][y_label] += 1
                        threshold_dict[threshold]["total_seq"] += 1
                        if evaluate_result:
                            if yp_label == y_label:
                                if y_label == 1.0:
                                    tp_list.append(x[idx].detach().cpu())
                                else:
                                    tn_list.append(x[idx].detach().cpu())
                            else:
                                if y_label == 1.0:
                                    fp_list.append(x[idx].detach().cpu())
                                else:
                                    fn_list.append(x[idx].detach().cpu())

            count += bs
            if count >= limit:
                break
    finally:
        m.train()

    # print(len(tp_list), len(fp_list))

    for threshold in threshold_dict:
        total_correct = threshold_dict[threshold]["total_correct"]
        total_seq = threshold_dict[threshold]["total_seq"]
        class_correct = threshold_dict[threshold]["class_correct"]
        class_total = threshold_dict[threshold]["class_total"]

        acc = total_correct * 1.0 / total_seq
        if not quiet:
            print(f"Accuracy: {acc}")

            class_acc = list()
            for i in range(num_proteins):
                if class_total[i] > 0:
                    class_acc.append(class_correct[i] * 1.0 / class_total[i])
                else:
                    class_acc.append(0.0)

            with open('binary_all_acc_tmp.txt', 'a') as f:
                f.write(f"--threshold: {threshold}\n")
                f.write(f"accuracy: {acc}; class correct: {class_correct}; class_total: {class_total}\n")
                f.close()
    
    if evaluate_result:
        permute_dict = extract_dist(permute_dict, tp_list, fp_list, tn_list, fn_list)
        return acc, permute_dict

    return acc


def load_model(folder, step=None):
    # return None, torch.load(folder)
    
    if os.path.isfile(folder):
        return None, torch.load(folder)
    model_dir = os.path.join(folder, "model")
    if not os.path.exists(model_dir):
        return None, None
    if step is None and os.listdir(model_dir):
        step = max(os.listdir(model_dir), key=int)
    path = os.path.join(model_dir, str(step))
    if not os.path.exists(path):
        return None, None
    return int(step), torch.load(path)


def save_model(model, folder, step):
    path = os.path.join(folder, "model", str(step))
    try:
        os.makedirs(os.path.dirname(path))
    except FileExistsError:
        pass
    torch.save(model, path)



        

