import numpy as np
import torch
import sklearn

from sklearn.ensemble import RandomForestClassifier

# Step 1: Read and preprocess data

f = open('protein_dataset/protein_19.txt')
vals = [line.strip().split(', ') for line in f]
f.close()

def convert(s):
    return np.array([one_hot(c) for c in s]).ravel()

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

n_train = 5000
xs_train = xs[:n_train]
ys_train = ys[:n_train]
xs_test = xs[n_train:]
ys_test = ys[n_train:]

# Step 2: Train random forest

rf = RandomForestClassifier(n_estimators=100)
rf.fit(xs_train, ys_train)

def acc(rf, xs, ys):
    ys_hat = rf.predict(xs)
    return np.mean(ys == ys_hat)

ys_hat = rf.predict(xs)
acc_train = acc(rf, xs_train, ys_train)
acc_test = acc(rf, xs_test, ys_test)
print('RF:', acc_train, acc_test)

# Step 3: Train FCN

input_dim = xs.shape[1]
hidden_dim = 500
output_dim = 2

model = torch.nn.Sequential(
    torch.nn.Linear(input_dim, hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dim, output_dim),
)

n_steps = 100
lr = 1e-1

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

xs_train_torch = torch.FloatTensor(xs_train)
ys_train_torch = torch.LongTensor(ys_train)
xs_test_torch = torch.FloatTensor(xs_test)
ys_test_torch = torch.LongTensor(ys_test)

for i in range(n_steps):
    optimizer.zero_grad()
    ps_hat = model(xs_train_torch)
    loss = criterion(ps_hat, ys_train_torch)
    loss.backward()
    optimizer.step()
    print(loss)

    _, ys_hat = torch.max(ps_hat.data, 1)
    acc_train = np.mean(ys_hat.detach().numpy() == ys_train)
    ps_hat = model(xs_test_torch)
    _, ys_hat = torch.max(ps_hat.data, 1)
    acc_test = np.mean(ys_hat.detach().numpy() == ys_test)
    print(i, acc_train, acc_test)
