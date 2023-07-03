import h5py

h5f = h5py.File('data_20/rbns_train.h5', "r")
# X = h5f['X'][:]
Y = h5f['Y'][:]
F = h5f['F'][:]
L = h5f['L'][:]
h5f.close()

y_set = set()
for i in range(Y.shape[0]):
    label = Y[i]
    if label in y_set:
        continue
    else:
        y_set.add(label)

print(y_set)


