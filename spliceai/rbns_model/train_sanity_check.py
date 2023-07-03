import numpy as np
import torch
import sklearn

from sklearn.ensemble import RandomForestClassifier

from train_utils import *

# Step 1: Read and preprocess data

# f = open('large_dataset/protein_19.txt')
f = open('protein_dataset/protein_19.txt')
vals = [line.strip().split(', ') for line in f]
f.close()

def convert(s):
    y = np.array([one_hot(c) for c in s]) # .ravel()
    return y

def one_hot(c):
    index = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
    x = np.zeros([4])
    if not c in index: # handle 'N'
        return x
    x[index[c]] = 1
    return x

xs = np.array([convert(val[0]) for val in vals])
ys = np.array([int(val[1]) for val in vals])


xs, ys = sklearn.utils.shuffle(xs, ys)

n_train = 100 # 8000
xs_train = xs[:n_train]
ys_train = ys[:n_train]
xs_test = xs[n_train:]
ys_test = ys[n_train:]

# split dataset

xs_train_torch = torch.FloatTensor(xs_train)
ys_train_torch = torch.LongTensor(ys_train)
xs_test_torch = torch.FloatTensor(xs_test)
ys_test_torch = torch.LongTensor(ys_test)


# Step 1: Train Simple CNN  # function equally to FCT in train_osb.py

n_steps = 1000
lr = 1e-1

m = MotifModel_simplecnn()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(m.parameters(), lr=lr)

for i in range(n_steps):
    optimizer.zero_grad()
    ps_hat = m(xs_train_torch)
    loss = criterion(ps_hat, ys_train_torch)
    loss.backward()
    optimizer.step()
    # print(loss)

    _, ys_hat = torch.max(ps_hat.data, 1)
    acc_train = np.mean(ys_hat.detach().numpy() == ys_train)
    ps_hat = m(xs_test_torch)
    _, ys_hat = torch.max(ps_hat.data, 1)
    acc_test = np.mean(ys_hat.detach().numpy() == ys_test)
    # print(i, acc_train, acc_test)
print(acc_train, acc_test)


# Step 2: Train CNN

n_steps = 1000
lr = 5e-2

m = MotifModel_cnn(hidden_size=16)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(m.parameters(), lr=lr)

for i in range(n_steps):
    optimizer.zero_grad()
    ps_hat = m(xs_train_torch)
    loss = criterion(ps_hat, ys_train_torch)
    loss.backward()
    optimizer.step()
    print(loss)

    _, ys_hat = torch.max(ps_hat.data, 1)
    acc_train = np.mean(ys_hat.detach().numpy() == ys_train)
    ps_hat = m(xs_test_torch)
    _, ys_hat = torch.max(ps_hat.data, 1)
    acc_test = np.mean(ys_hat.detach().numpy() == ys_test)
    print(i, acc_train, acc_test)
print(f"CNN: train: {acc_train}, test: {acc_test}")


# Step 3: Train CNN dropout

# n_steps = 100
# lr = 5e-2

# m = MotifModel_cnn_dropout()

# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(m.parameters(), lr=lr)

# for i in range(n_steps):
#     optimizer.zero_grad()
#     ps_hat = m(xs_train_torch)
#     loss = criterion(ps_hat, ys_train_torch)
#     loss.backward()
#     optimizer.step()
#     print(loss)

#     _, ys_hat = torch.max(ps_hat.data, 1)
#     acc_train = np.mean(ys_hat.detach().numpy() == ys_train)
#     ps_hat = m(xs_test_torch)
#     _, ys_hat = torch.max(ps_hat.data, 1)
#     acc_test = np.mean(ys_hat.detach().numpy() == ys_test)
#     print(i, acc_train, acc_test)
# print(f"CNN-dropout: train: {acc_train}, test: {acc_test}")


# Step 4: Train LSTM

# m = MotifModel_lstm()

# n_steps = 100
# lr = 1e-1

# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(m.parameters(), lr=lr)

# for i in range(n_steps):
#     optimizer.zero_grad()
#     ps_hat = m(xs_train_torch)
#     loss = criterion(ps_hat, ys_train_torch)
#     loss.backward()
#     optimizer.step()
#     print(loss)

#     _, ys_hat = torch.max(ps_hat.data, 1)
#     acc_train = np.mean(ys_hat.detach().numpy() == ys_train)
#     ps_hat = m(xs_test_torch)
#     _, ys_hat = torch.max(ps_hat.data, 1)
#     acc_test = np.mean(ys_hat.detach().numpy() == ys_test)
#     print(i, acc_train, acc_test)

