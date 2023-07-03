import h5py

from utils_rbns import list2seq

path = 'binary_dataset/protein_RALYL/rbns_train.h5'

data = h5py.File(path, "r")
Y = data["Y"][:]
i_s = list(range(Y.shape[0]))

print(Y.shape[0])
exit(0)

for i in i_s:
    if i < 100000:
        continue
    X = data["X"][i]
    Y = data["Y"][i]
    F = data["F"][i]
    L = data["L"][i]
    
    if L > 20:
        print(list2seq(X[:]),Y,F,L)
