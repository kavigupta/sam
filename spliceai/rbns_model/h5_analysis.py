import h5py
from map_protein_name import get_map_protein
import random
import numpy as np

map_index = get_map_protein()

data = h5py.File("tiny_binary_dataset/protein_47/rbns_train.h5", "r")
# L = data["L"][:
# print(max(L))
X = data["X"][:]# .astype(int)
L = data["L"][:]# .astype(int)
Y = data["Y"][:]# .astype(int)
tmp_X = X[Y==1]
tmp_L = L[Y==1]
print(tmp_X.mean(0))
print(len(tmp_X.mean(0)))
# print(max(tmp_L), min(tmp_L))
# print(tmp_X[:, :tmp_L].mean(0))
# ddprint(X[:, :L][Y==1].mean(0))
# print(X[Y==1].mean(0))
exit(0)
# Y = data["Y"][:]
# dsize = Y.shape[0]

protein_idx = 6
protein_list = list()
count_protein = 0 
count_overlap = 0
count_control = 0

i_s = list(range(dsize))
random.shuffle(i_s)
for i in i_s:
    Y = data["Y"][i]
    X = data["X"][i]
    if int(Y) in map_index:
        pass
    else:
        continue
    if protein_idx in map_index[int(Y)]:
        protein_list.append(X.tolist())
        count_protein += 1
        if count_protein >= 5000:
            break
    else:
        continue

print(f"protein_list: {protein_list}")
print(f"{protein_list[0] in protein_list}")
for i in i_s:
    Y = data["Y"][i]
    X = data["X"][i]
    if int(Y) == 0:
        count_control += 1
        if count_control >= 5000:
            break
        if X.tolist() in protein_list:
            count_overlap += 1

print(f"count_protein: {count_protein}, count_overlap: {count_overlap}, count_control: {count_control}")





