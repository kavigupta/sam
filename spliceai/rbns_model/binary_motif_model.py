import h5py
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import torch.nn as nn
import torch.nn.functional as F
import os
from utils import (
    extract_dist,
    list2seq,
    permute,
)

from random import shuffle
import itertools, operator

from tqdm import tqdm
# from load_psam_motif import *

class ResidualUnit(nn.Module):
    """
    Residual unit proposed in "Identity mappings in Deep Residual Networks"
    by He et al.
    """

    def __init__(self, l1, l2, w):
        super().__init__()
        self.normalize1 = nn.BatchNorm1d(l1)
        self.normalize2 = nn.BatchNorm1d(l2)
        self.act1 = self.act2 = nn.ReLU()

        padding = ((w - 1)) // 2

        self.conv1 = nn.Conv1d(l1, l2, w, dilation=1, padding=padding)
        self.conv2 = nn.Conv1d(l1, l2, w, dilation=1,  padding=padding)

    def forward(self, input_node):
        bn1 = self.normalize1(input_node)
        act1 = self.act1(bn1)
        conv1 = self.conv1(act1)
        assert conv1.shape == act1.shape
        bn2 = self.normalize2(conv1)
        act2 = self.act2(bn2)
        conv2 = self.conv2(act2)
        assert conv2.shape == act2.shape
        output_node = conv2 + input_node
        return output_node


class MotifModel(nn.Module):
    def __init__(self, hidden_size, window_size):
        super().__init__()
        self.conv1 = nn.Conv1d(4, hidden_size, kernel_size=3, padding=((3 - 1) // 2))
        self.norm1 = nn.BatchNorm1d(hidden_size)

        self.convstack = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(hidden_size, hidden_size, 3),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                )
                for i in range((window_size - 2) // 2)
            ]
        )
        # self.linear = nn.Linear(hidden_size, 2)
        self.conv_output = nn.Conv1d(hidden_size, 2, 1)
        self.act = nn.ReLU()
        self.window_size = window_size

    def forward(self, input, f=None, l=None, use_as_motif=False):
        if use_as_motif:
            pass
        else:
            input = input.transpose(1, 2) # BxLxC -> BxCxL

        x = self.conv1(input)
        x = self.norm1(x)
        x = self.act(x)
        # x = self.dropout(x)
        for i in range((self.window_size - 2) // 2):
            x = self.convstack[i](x)

        x = self.conv_output(x)
        if use_as_motif:
            pass
        else:
            x = x.sum(2)

        return x


class MotifModelOld(nn.Module):
    def __init__(self, hidden_size, window_size):
        super().__init__()
        self.conv1 = nn.Conv1d(4, hidden_size, kernel_size=5, padding=((5 - 1) // 2))
        self.norm1 = nn.BatchNorm1d(hidden_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size * 2, 1)
        self.norm2 = nn.BatchNorm1d(hidden_size * 2)
        self.conv3 = nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=window_size-4)
        self.norm3 = nn.BatchNorm1d(hidden_size)
        # self.linear = nn.Linear(hidden_size, 2)
        self.conv4 = nn.Conv1d(hidden_size, 2, 1)
        self.act = nn.ReLU()
    
    def forward(self, input, f=None, l=None, use_as_motif=False):
        if use_as_motif:
            pass
        else:
            input = input.transpose(1, 2) # BxLxC -> BxCxL

        x = self.conv1(input)
        x = self.norm1(x)
        x = self.act(x)
        # x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act(x)
        # x = self.dropout(x)

        # x = torch.squeeze(x)
        # x = self.linear(x)
        x = self.conv4(x)
        if use_as_motif:
            pass
        else:
            x = x.sum(2)
        x = softmax(x, dim=1)

        return x



class MotifModel_simplecnn(nn.Module):
    def __init__(self, num_proteins=2):
        super(MotifModel, self).__init__()
        self.linear1 = nn.Linear(4, 500)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(500, 2)

    def forward(self, input, mc, l):
        output = self.linear1(input)
        output = self.act(output)
        output = self.linear2(output)

        return output


class MotifModelDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, shuffle_by_group=False):
        self.path = path
        self.data = h5py.File(path, "r")
        self._l = self.dsize(self.data)
        self.max_l = self.dmax(self.data)
        self.shuffle_by_group = shuffle_by_group
    
    def __len__(self):
        return self._l
    
    @staticmethod
    def dmax(data):
        L = data["L"][:]
        return torch.max(torch.tensor(L))
    
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
    def shuffle_group_by_l(data, size=250):
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

        if self.shuffle_by_group or MotifModelDataset.dmax(data).data.item() > 20:
            i_s = MotifModelDataset.shuffle_group_by_l(data)
        else:
            i_s = list(range(MotifModelDataset.dsize(data)))
            shuffle(i_s)

        for i in i_s:
            X = data["X"][i]
            Y = data["Y"][i]
            F = data["F"][i]
            L = data["L"][i]

            # print(L)

            yield X[:L].astype(np.float32), Y.astype(np.long), F.astype(np.float32), L.astype(np.float32)
    
    def loader(self, batch_size):
        if self.max_l.data.item() > 20:
            batch_size = 250
        else:
            pass
        l = torch.utils.data.DataLoader(self, num_workers=0, batch_size=batch_size)
        return l


def equal_length(x):
    unique_x = torch.unique(x)
    res = unique_x.shape[0]
    if res == 1:
        return True
    else:
        return False


def evaluate_model(m, d, limit=float("inf"), bs=500, pbar=lambda x: x, num_proteins=2, quiet=True, evaluate_result=False):
    count = 0
    total_correct = 0
    total_seq = 0
    class_correct = list(0.0 for i in range(num_proteins))
    class_total = list(0.0 for i in range(num_proteins))
    dataset = d.loader(bs)

    if evaluate_result:
        permute_dict = permute(repeat=5)
        # each list include number of TP, FP, TN, FN
        tp_list, fp_list, tn_list, fn_list = list(), list(), list(), list()

    try:
        m.eval()
        # for x, y, f, l in pbar(DataLoader(d, batch_size=bs)):
        for i, (x, y, f, l) in pbar(enumerate(dataset)):

            x = x.cuda()
            y = y.cuda()
            f = f.cuda()
            l = l.cuda()

            # print(f"y:\n{y}")
            y = y.unsqueeze(1)
            with torch.no_grad():
                yp = m(x, f, l)
                # print(f"before softmax, {yp}")
                _, predicted = torch.max(yp, 1)
                # print(f"predicted, {predicted}")

                predicted_label = predicted.cpu().numpy().tolist()
                y = torch.squeeze(y)

                for idx, value in enumerate(predicted_label):
                    y_label = y[idx].data.item()
                    yp_label = predicted[idx].data.item()
                    if yp_label == y_label:
                        total_correct += 1
                        class_correct[yp_label] += 1
                    class_total[y_label] += 1
                    total_seq += 1
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

    print(f"class_correct: {class_correct}")
    print(f"class_total: {class_total}")
    print(f"total_corrent: {total_correct}")
    print(f"total_seq: {total_seq}")
    
    if not quiet:
        if min(class_total) == 0:
            acc = 0.0
            print(f"Accuracy: NOT FOUND")
            with open('binary_acc_prod.txt', 'a') as f:
                f.write('NOT FOUND' + '\n')
                f.close()
            return 0.0
    if quiet:
        if min(class_total) == 0:
            acc = -1.0
            return acc

    acc = total_correct * 1.0 / total_seq
    if not quiet:
        print('Accuracy', acc)
        class_acc = list()
        for i in range(num_proteins):
            if class_total[i] > 0:
                class_acc.append(class_correct[i] * 1.0 / class_total[i])
            else:
                class_acc.append(0.0)
    
    if evaluate_result:
        permute_dict = extract_dist(permute_dict, tp_list, fp_list, tn_list, fn_list)
        return acc, permute_dict

    return acc


def load_model(folder, name, step=None):
    if os.path.isfile(folder):
        return None, torch.load(folder)
    model_dir = os.path.join(folder, f"model_{name}")
    if not os.path.exists(model_dir):
        return None, None
    if step is None and os.listdir(model_dir):
        step = max(os.listdir(model_dir), key=int)
    path = os.path.join(model_dir, str(step))
    if not os.path.exists(path):
        return None, None
    return int(step), torch.load(path)


def save_model(model, folder, name, step):
    path = os.path.join(folder, f"model_{name}", str(step))
    try:
        os.makedirs(os.path.dirname(path))
    except FileExistsError:
        pass
    torch.save(model, path)



        

