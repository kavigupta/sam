import h5py
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
from utils_rbns import list2seq

from random import shuffle

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



class MotifModel_lstm(nn.Module):

    def __init__(self, num_proteins=104):
        super(MotifModel, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.act = nn.ReLU()
        self.residualunit1 = ResidualUnit(64, 64, 11)
        self.residualunit2 = ResidualUnit(64, 64, 11)
        self.residualunit3 = ResidualUnit(64, 64, 11)

        self.rnn = nn.LSTM(64, 64, num_layers=10)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.output = nn.Linear(64, 2)
        self.num_proteins = num_proteins

    def forward(self, input, mc, l): # X, concentration
        mc = mc.unsqueeze(1).expand(mc.shape[0], self.num_proteins)
        mc = mc.unsqueeze(1).transpose(1, 2)
        input = input.transpose(1, 2) # BxLxC -> BxCxL

        conv = self.conv1(input)
        conv = self.act(conv)
        conv = self.conv2(conv)
        conv = self.act(conv)
        conv = self.residualunit1(conv)
        conv = self.residualunit2(conv)
        conv = self.residualunit3(conv)
        conv = conv.permute(2, 0, 1) # -> LxBxC
        conv, (hn, cn) = self.rnn(conv)
        conv = conv[-1, :, :]
        conv = self.output(conv)
        conv = self.logsoftmax(conv) # conv.logsoftmax(dim=1)
        output = conv

        return output

class MotifModel_cnn(nn.Module):

    def __init__(self, num_proteins=104):
        super(MotifModel, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 2)
        self.residualunit1 = ResidualUnit(64, 64, 11)
        self.residualunit2 = ResidualUnit(64, 64, 11)
        self.conv3 = nn.Conv1d(64, 32, 10)
        self.conv4 = nn.Conv1d(32, 16, 10)
        self.conv5 = nn.Conv1d(16, 16, 1)
        self.conv6 = nn.Conv1d(16, 2, 1)
        self.act = nn.ReLU()
        self.num_proteins = num_proteins
        # self.motifs = read_motifs()

    def forward(self, input, mc, l): # X, concentration

        mc = mc.unsqueeze(1).expand(mc.shape[0], self.num_proteins)
        mc = mc.unsqueeze(1).transpose(1, 2)
        input = input.transpose(1, 2) # BxLxC -> BxCxL
        
        conv = self.conv1(input)
        conv = self.act(conv)

        conv = self.conv2(conv)
        conv = self.act(conv)

        conv = self.residualunit1(conv)
        conv = self.residualunit2(conv)
        
        conv = self.conv3(conv)
        conv = self.act(conv)

        conv = self.conv4(conv)
        conv = self.act(conv)

        conv = self.conv5(conv)
        conv = self.act(conv)

        conv = self.conv6(conv)
        # one option is ignoring mc for now
        prob = conv
        # the statement below does not work, so annotate it for now
        # prob = torch.sigmoid(torch.log(mc)-conv)

        output = torch.squeeze(prob)
        # print(f"output shape: {output.shape}")
        # exit(0)

        return output


class MotifModelDataset(torch.utils.data.IterableDataset):
    def __init__(self, map_index, path, protein, extract_large=True):
        self.path = path
        data = h5py.File(path, "r")
        self._l = self.dsize(data)
        self.protein = protein
        self.map_index = map_index
        self.extract_large = extract_large
        if extract_large is True:
            pass
            #  self.dataset_file = open(f"protein_{protein}.txt", 'w')
        # self.data_load_path = data_path
    
    def __len__(self):
        return self._l
    
    @staticmethod
    def dsize(data):
        # ys = [k for k in data.keys() if "Y" in k]
        Y = data["Y"][:]
        return Y.shape[0]
    
    def __iter__(self):
        if not self.extract_large:
            data = h5py.File(self.path, "r")
            i_s = list(range(MotifModelDataset.dsize(data)))
            shuffle(i_s)

            for i in i_s:
                X = data["X"][i]
                Y = data["Y"][i]
                F = data["F"][i]
                L = data["L"][i]

                yield X.astype(np.float32), Y.astype(np.float32), F.astype(np.float32), L.astype(np.float32)

        if self.extract_large:
            data = h5py.File(self.path, "r")
            # i_s = list(range(MotifModelDataset.dsize(data)))

            i_s = list(range(MotifModelDataset.dsize(data)))
            label_dict = {}
            # shuffle(i_s)

            count_control = 0
            count_protein = 0
            for i in i_s: # remove the padding
                if count_control == 100 and count_protein == 100:
                    # exit(0)
                    continue
                X = data["X"][i]
                Y = data["Y"][i]
                F = data["F"][i]
                L = data["L"][i]
            
                if int(Y) in self.map_index or int(Y)==0:
                    pass
                else:
                    continue

                if int(Y) == 0:
                    # print(f"control: {Y.astype(np.long)}")
                    if count_control >= 100:
                        continue
                    count_control += 1
                    Y = 0
                    # self.dataset_file.write(f"{list2seq(X[:20])}, {Y}\n")
                    yield X[:20].astype(np.float32), Y, F.astype(np.float32), L.astype(np.float32)
                elif self.protein in self.map_index[int(Y)]:
                    Y = 1 
                    if count_protein >= 100:
                        continue
                    count_protein += 1
                    # self.dataset_file.write(f"{list2seq(X[:20])}, {Y}\n")
                    # print(f"input: {self.protein}")
                    yield X[:20].astype(np.float32), Y, F.astype(np.float32), L.astype(np.float32)
                else:
                    continue
    
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


def evaluate_model(m, d, name, limit=float("inf"), bs=512, pbar=lambda x: x, num_proteins=2, quiet=True):
    count = 0
    total_correct = 0
    total_seq = 0
    class_correct = list(0.0 for i in range(num_proteins))
    class_total = list(0.0 for i in range(num_proteins))

    try:
        m.eval()
        for x, y, f, l in pbar(DataLoader(d, batch_size=bs)):
            
            if not equal_length(l):
                continue

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
            with open('raw_all_acc_prod.txt', 'a') as f:
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
        # class_acc = [if class_total[i]>0 class_correct[i] * 1.0 / class_total[i] else 0.0 for i in range(num_proteins)]
        # with open(f"raw_acc/raw_label_acc_{name}.txt", 'w') as f:
        #     for label_acc in class_acc:
        #         f.write(str(label_acc) + '\n')
        # f.close()


    return acc


def load_model(folder, name, step=None):
    # return None, torch.load(folder)
    
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



        

