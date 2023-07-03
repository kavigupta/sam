import re

path = f"binary_all_acc_tmp.txt"

psam_idx_acc_dict = dict()
with open(path, 'r') as f:
    i = 0
    for line in f:
        if i % 3 == 0:
            idx = int(line[:-1])
        if i % 3 == 2:
            content = float(line[:-1].split("; ")[0].split(": ")[1])
            psam_idx_acc_dict[idx] = content
        i += 1
f.close()

protein_psam_name_dict = {}
protein_rbns_name_dict = {}

with open('protein_psam_name.txt', 'r') as f:
    # f.readline()
    idx = 0
    for line in f:
        psam_name = re.split(',|\n', line)[:-1][0]
        # print(psam_name)
        protein_psam_name_dict[psam_name] = idx
        idx += 1
f.close()

with open('protein_rbns_name.txt', 'r') as f:
    # f.readline()
    idx = 1 # 0 rbns starts from 1
    for line in f:
        protein_rbns_name_dict[re.split(',|\n', line)[:-1][0]] = idx
        idx += 1
f.close()

rbns_name_acc_dict = dict()
for rbns_name in protein_rbns_name_dict:
    acc = ''
    for psam_name in protein_psam_name_dict:
        if psam_name == rbns_name:
            psam_idx = protein_psam_name_dict[psam_name]
            if psam_idx in psam_idx_acc_dict:
                acc = str(psam_idx_acc_dict[psam_idx])
                break
    print(f"{rbns_name}, {acc}")
    rbns_name_acc_dict[rbns_name] = acc

f = open("rbns_name_psam_acc.txt", 'w')
for rbns_name in rbns_name_acc_dict:
    f.write(f"{rbns_name}, {rbns_name_acc_dict[rbns_name]}\n")
f.close()





