import matplotlib.pyplot as plt
import re
import os
from scipy.spatial import distance
import numpy as np
import seaborn as sns

from os import listdir
from os.path import isfile, join

from utils import permute_l
import h5py

import pandas as pd
from constants import *

import logomaker

def read_roc(path):
    acc_dict = dict()
    threshold_list = list()
    non_bind_acc_list = list()
    bind_acc_list = list()
    with open(path, 'r') as f:
        i = 0
        for line in f:
            if i % 2 == 0:
                # print(line)
                threshold = float(line.split(": ")[1][:-1])
            elif i % 2 == 1:
                l = line.split(": ")[2][:-1]
                # print(f"l, {l}")
                acc_l = l.split(", ")
                non_bind_acc = 1- float(acc_l[0][1:]) 
                bind_acc = float(acc_l[1][:-2])
                acc_dict[threshold] = (non_bind_acc, bind_acc)
            i += 1
        f.close()

    for threshold, acc in sorted(acc_dict.items(), reverse=True):
        threshold_list.append(threshold)
        non_bind_acc_list.append(acc[0])
        bind_acc_list.append(acc[1])

    return threshold_list, bind_acc_list, non_bind_acc_list


def read_roc_2(path):
    acc_dict = dict()
    threshold_list = list()
    non_bind_acc_list = list()
    bind_acc_list = list()
    with open(path, 'r') as f:
        i = 0
        for line in f:
            if i % 2 == 0:
                # print(line)
                threshold = float(line.split(": ")[1][:-1])
            elif i % 2 == 1:
                l = line.split("; ")
                class_correct = l[1].split(": ")[1][1: -1].split(", ")
                class_total = l[2].split(": ")[1][1: -3].split(", ")
                c0 = float(class_correct[0])
                c1 = float(class_correct[1])
                t0 = float(class_total[0])
                t1 = float(class_total[1])
                print(c0, c1, t0, t1)
                TPR = c1 * 1.0 / t1
                FPR = 1 - c0 * 1.0 / t0
                
                acc_dict[threshold] = (FPR, TPR)
            i += 1
        f.close()

    for threshold, acc in sorted(acc_dict.items(), reverse=True):
        threshold_list.append(threshold)
        non_bind_acc_list.append(acc[0])
        bind_acc_list.append(acc[1])
    
    print(non_bind_acc_list, bind_acc_list)
    return threshold_list, non_bind_acc_list, bind_acc_list


def read_psam_idx_acc_threshold(path):
    protein_rbns_name_dict = dict()
    with open('intermediate_data/protein_rbns_name.txt', 'r') as f:
        # f.readline()
        idx = 1 # 0 rbns starts from 1
        for line in f:
            protein_rbns_name_dict[re.split(',|\n', line)[:-1][0]] = idx
            idx += 1
    f.close()
    
    protein_psam_idx_name_dict = dict()
    with open('intermediate_data/protein_psam_name.txt', 'r') as f:
        # f.readline()
        idx = 0
        for line in f:
            psam_name = re.split(',|\n', line)[:-1][0]
            # print(psam_name)
            protein_psam_idx_name_dict[idx] = psam_name
            idx += 1
    f.close()

    rbns_name_psam_acc_threshold_dict = dict()
    for rbns_name in protein_rbns_name_dict:
        rbns_name_psam_acc_threshold_dict[rbns_name] = ('', '')

    with open(path, 'r') as f:
        i = 0
        for line in f:
            if i % 29 == 0:
                idx = int(line[:-1])
                rbns_name = protein_psam_idx_name_dict[idx]
                acc, threshold = 0.0, 0.0
            elif i % 29 == 28:
                rbns_name_psam_acc_threshold_dict[rbns_name] = (acc, threshold)
            else:
                if 'threshold' in line:
                    tmp_threshold = float(line[:-1].split(": ")[1])
                if 'accuracy' in line:
                    tmp_acc = float(line.split("; ")[0].split(": ")[1])
                    if tmp_acc > acc:
                        acc = tmp_acc
                        threshold = tmp_threshold
            i += 1
    f.close()

    # f = open('intermediate_data/rbns_name_psam_acc_base_threshold.txt', 'w')
    f  = open('intermediate_data/psam_threshold.txt', 'w')
    for rbns_name, (acc, threshold) in sorted(rbns_name_psam_acc_threshold_dict.items()):
        if acc != '':
            # f.write(f"{rbns_name}, {acc}, {threshold}\n")
            f.write(f"{rbns_name}, {threshold}\n")
    f.close()

    return  


def read_rbns_acc(dir):
    rbns_acc_dict = dict()

    file_list = [f for f in listdir(dir) if isfile(join(dir, f))]
    for file_name in file_list:
        f = open(join(dir, file_name), 'r')
        protein = f.readline().split(", ")[0]
        content = f.readline()[:-1].split("; ")
        if 'trained' in content[0] or '' == content[0]:
            train_acc, test_acc = 0.0, 0.0
        else:
            train_acc, test_acc = content[1].split(": ")[1], content[2].split(": ")[1]
        rbns_acc_dict[protein] = (train_acc, test_acc)
    
    rbns_acc_f = open("intermediate_data/rbns_motif_model_acc.txt", 'w')
    for rbns_name, (train_acc, test_acc) in sorted(rbns_acc_dict.items()):
        rbns_acc_f.write(f"{rbns_name}, {train_acc}, {test_acc}\n")
    rbns_acc_f.close()

    return


def plot_line(x_list, y1_list, y2_list, title, x_label, y_label, label1, label2, fig_title, c1='C0', c2='C1', log=False):
    ax = plt.subplot(111)
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()

    plt.plot(y1_list, y2_list, c=c1)
    # plt.plot(x_list, y1_list, label=label1, c=c1)
    # plt.plot(x_list, y2_list, label=label2, c=c2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if log is True:
        plt.xscale("log")

    plt.title(title)
    plt.grid()
    plt.legend()
    plt.savefig(fig_title)
    plt.close()


def calculate_enrichment_dict(path):
    tp_list, fp_list, tn_list, fn_list = list(), list(), list(), list()
    tp_enrichment = list()
    fp_enrichment = list()
    tn_enrichment = list()
    fn_enrichment = list()
    protein_enrichment = list()
    print(path)

    f = open(path, 'r')
    first_line = f.readline()
    acc = float(first_line.split(',')[0])
    for line in f:
        content = line[:-1].split(', ')
        #! exchange fp, fn
        # tp_list.append(int(content[1]))
        # fp_list.append(int(content[2]))
        # tn_list.append(int(content[3]))
        # fn_list.append(int(content[4]))
        tp_list.append(int(content[1]))
        fn_list.append(int(content[2]))
        tn_list.append(int(content[3]))
        fp_list.append(int(content[4]))
    
    p_number = sum(tp_list) + sum(fn_list)
    n_number = sum(tn_list) + sum(fp_list)
    fn_number = sum(fn_list)

    print(p_number, n_number)
    for idx, v in enumerate(tp_list):
        # if idx % 100 == 0:
        print(idx, tp_list[idx] + fn_list[idx], tn_list[idx] + fp_list[idx], ((tp_list[idx] + fn_list[idx]) / p_number) / ((1.0 * (tn_list[idx] + fp_list[idx])) / n_number))
        # print(p_number, tn_list[idx] + fn_list[idx], n_number)
        tp_enrichment.append((v * 1.0 / p_number) / (1.0 * (tn_list[idx] + fp_list[idx]) / n_number))
        fp_enrichment.append((fp_list[idx] * 1.0 / p_number) / (1.0 * (tn_list[idx] + fp_list[idx]) / n_number))

        fn_enrichment.append((fn_list[idx] * 1.0 / p_number) / (1.0 * (tn_list[idx] + fp_list[idx]) / n_number))
        protein_enrichment.append(((tp_list[idx] + fn_list[idx]) / p_number) / ((1.0 * (tn_list[idx] + fp_list[idx])) / n_number))
        # protein_enrichment.append(((tp_list[idx] + fn_list[idx]) / 1.0) / ((1.0 * (tn_list[idx] + fp_list[idx])) / 1.0))


    enrichment_dict = {
        'tp_enrichment': tp_enrichment,
        'fp_enrichment': fp_enrichment,
        'fn_enrichment': fn_enrichment,
        'protein_enrichment': protein_enrichment, 
    }

    return enrichment_dict, acc


def read_protein_binding_enrichment(
    DIR="intermediate_data/protein_binding_distribution",
    read_accuracy=True,
):
    pbe_dict = dict()
    for file_name in listdir(DIR):
        if '46' not in file_name and '47' not in file_name:
            continue
        if 'nn' in file_name:
            if '20' in file_name:
                continue
            content = file_name.split('.')[0].split('_')
            protein_name = content[2]
            psam_idx = int(content[3])
            if psam_idx in [45, 17]: continue
            if psam_idx != -1:
                if psam_idx in pbe_dict:
                    protein_info_dict = pbe_dict[psam_idx]
                    protein_info_dict['protein_name'] = protein_name
                else:
                    protein_info_dict = {
                        'protein_name': protein_name,
                        'psam_idx': psam_idx
                    }
            else:
                continue
        else:
            content = file_name.split('.')[0].split('_')
            # print(content)
            psam_idx = int(content[1])
            if psam_idx in [45, 17]: continue # exceptions
            if psam_idx in pbe_dict:
                protein_info_dict = pbe_dict[psam_idx]
            else:
                protein_info_dict = {
                    'psam_idx': psam_idx,
                }

        path = join(DIR, file_name)
        enrichment_dict, acc = calculate_enrichment_dict(path)

        if 'nn' in file_name:
            protein_info_dict['nn_enrichment'] = enrichment_dict
            protein_info_dict['nn_acc'] = acc
        else:
            protein_info_dict['psam_enrichment'] = enrichment_dict
            protein_info_dict['psam_acc'] = acc

        
        pbe_dict[psam_idx] = protein_info_dict
    
    return pbe_dict    


def plot_enrichment_alignment_heatmap(pbe_dict):
    # each protein, the enrichment function of TP, FP
    # matrix to measure the distance
    tp_list, fp_list, fn_list = list(), list(), list()
    for psam_idx_1 in sorted(pbe_dict): # nn
        tp, fp, fn = list(), list(), list()
        for psam_idx_2 in sorted(pbe_dict): # psam
            nn_tp_enrichment = pbe_dict[psam_idx_1]['nn_enrichment']['tp_enrichment']
            nn_fp_enrichment = pbe_dict[psam_idx_1]['nn_enrichment']['fp_enrichment']
            psam_tp_enrichment = pbe_dict[psam_idx_2]['psam_enrichment']['tp_enrichment']
            psam_fp_enrichment = pbe_dict[psam_idx_2]['psam_enrichment']['fp_enrichment']
            nn_fn_enrichment = pbe_dict[psam_idx_1]['nn_enrichment']['fn_enrichment']
            psam_fn_enrichment = pbe_dict[psam_idx_2]['psam_enrichment']['fn_enrichment']

            tp_dist = distance.euclidean(nn_tp_enrichment, psam_tp_enrichment)
            fp_dist = distance.euclidean(nn_fp_enrichment, psam_fp_enrichment)
            fn_dist = distance.euclidean(nn_fn_enrichment, psam_fn_enrichment)
            
            tp.append(tp_dist)
            fp.append(fp_dist)
            fn.append(fn_dist)
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)
    
    plot_heatmap(np.array(tp_list), x_label='psam', y_label='nn', title='TP 5-mer Enrichment Distance', fig_title='tp_heatmap')
    plot_heatmap(np.array(fp_list), x_label='psam', y_label='nn', title='FP 5-mer Enrichment Distance', fig_title='fp_heatmap')
    plot_heatmap(np.array(fn_list), x_label='psam', y_label='nn', title='FN 5-mer Enrichment Distance', fig_title='fn_heatmap')
    
    return


def plot_heatmap(data, x_label, y_label, title, fig_title):
    # cmap = sns.cm.rocket_r

    ax = sns.heatmap(data)
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.set_title(title)

    plt.savefig(f"figures/{fig_title}.png")
    plt.close()


def mask_off(text_list, step=4):
    res_list = list()
    for i, text in enumerate(text_list):
        if i % step == 0:
            res_list.append(text)
        else:
            res_list.append(None)
    return res_list


def plot_enrichment_distribution(x_text_list, y1_list, y2_list, x_label, y_label, title, label1, label2, fig_title, chunk_size=205, path=f"figures/kmer_distribution"):
    x = np.arange(len(y1_list))
    nchunks = (len(y1_list) + chunk_size - 1) // chunk_size

    fig, axs = plt.subplots(ncols=1, nrows=nchunks, figsize=(20, 10))
    y1_l = np.array(y1_list)
    y2_l = np.array(y2_list)

    x_text_list = mask_off(x_text_list)

    x_text = np.array(x_text_list)
    print(x.shape, x_text.shape, y1_l.shape, y2_l.shape)

    mini, maxi = min(min(y1_list), min(y2_list)), max(max(y1_list), max(y2_list))

    for k in range(nchunks):
        chunk=slice(k * chunk_size, (k + 1) * chunk_size)
        axs[k].plot(x[chunk], y1_l[chunk], label=label1, c='C0')
        axs[k].plot(x[chunk], y2_l[chunk], label=label2, c='C1')
        axs[k].set_xticks(x[chunk])
        axs[k].set_xticklabels(x_text[chunk], rotation=90)
        axs[k].set_ylim(mini, maxi)
        # axs[k].xticks(rotation=90)
        axs[k].grid(axis="y")
    
    fig.tight_layout()
    plt.suptitle(title)
    plt.legend()
    plt.savefig(f"{path}/{fig_title}.png")
    plt.close()

    return 


def plot_kmer_distribution(pbe_dict):
    permute_5mer = permute_l(repeat=5)
    # print(permute_5mer)
    sorted(permute_5mer)
    # print(permute_5mer)

    for psam_idx in pbe_dict:
        protein_name = pbe_dict[psam_idx]["protein_name"]
        nn_tp_enrichment = pbe_dict[psam_idx]['nn_enrichment']['tp_enrichment']
        nn_fp_enrichment = pbe_dict[psam_idx]['nn_enrichment']['fp_enrichment']
        nn_fn_enrichment = pbe_dict[psam_idx]['nn_enrichment']['fn_enrichment']
        psam_tp_enrichment = pbe_dict[psam_idx]['psam_enrichment']['tp_enrichment']
        psam_fp_enrichment = pbe_dict[psam_idx]['psam_enrichment']['fp_enrichment']
        psam_fn_enrichment = pbe_dict[psam_idx]['psam_enrichment']['fn_enrichment']

        nn_acc = pbe_dict[psam_idx]['nn_acc']
        psam_acc = pbe_dict[psam_idx]['psam_acc']

        plot_enrichment_distribution(permute_5mer, nn_tp_enrichment, psam_tp_enrichment, 
            x_label='5-mer', y_label='enrichment', title=f"TP 5-mer enrichment (NN acc: {nn_acc}%, PSAM acc: {psam_acc}%)", 
            label1='nn', label2='psam', fig_title=f"{protein_name}_tp")
        plot_enrichment_distribution(permute_5mer, nn_fn_enrichment, psam_fn_enrichment, 
            x_label='5-mer', y_label='enrichment', title=f"FN 5-mer enrichment (NN acc: {nn_acc}%, PSAM acc: {psam_acc}%)", 
            label1='nn', label2='psam', fig_title=f"{protein_name}_fn")
        plot_enrichment_distribution(permute_5mer, nn_fp_enrichment, psam_fp_enrichment, 
            x_label='5-mer', y_label='enrichment', title=f"FP 5-mer enrichment (NN acc: {nn_acc}%, PSAM acc: {psam_acc}%)", 
            label1='nn', label2='psam', fig_title=f"{protein_name}_fp")
    
    return 


def plot_protein_enrichment(pbe_dict):
    permute_5mer = permute_l(repeat=5)
    # print(permute_5mer)
    sorted(permute_5mer)
    # print(permute_5mer)

    enrichment_dict = dict()
    for psam_idx in pbe_dict:
        protein_name = pbe_dict[psam_idx]["protein_name"]
        if protein_name in ['RBFOX2', 'RBFOX3']:
            nn_protein_enrichment = pbe_dict[psam_idx]['nn_enrichment']['protein_enrichment']
            psam_protein_enrichment = pbe_dict[psam_idx]['psam_enrichment']['protein_enrichment']
            enrichment_dict[protein_name] = {
                'nn': nn_protein_enrichment,
                'psam': psam_protein_enrichment,
            }
            
    plot_enrichment_distribution(permute_5mer, enrichment_dict['RBFOX2']['nn'], enrichment_dict['RBFOX3']['nn'],
        x_label='5-mer', y_label='enrichment', title=f"enrichment data 1", 
        label1="RBFOX2", label2="RBFOX3", fig_title=f"protein_enrichment_nn", path=f"figures/protein_enrichment")
    plot_enrichment_distribution(permute_5mer, enrichment_dict['RBFOX2']['psam'], enrichment_dict['RBFOX3']['psam'],
        x_label='5-mer', y_label='enrichment', title=f"enrichment data 1", 
        label1="RBFOX2", label2="RBFOX3", fig_title=f"protein_enrichment_psam", path=f"figures/protein_enrichment")
        


def read_single_protein_acc(dir):
    res_dict = dict()
    for file_name in os.listdir(dir):
        path = os.path.join(dir, file_name)
        f =  open(path, 'r')
        name_line = f.readline()
        rbns_n_name = name_line.split(', ')[0]
        acc_line = f.readline()
        if 'test' in acc_line:
            print(acc_line)
            test_acc = float(acc_line[:-1].split('; ')[-1].split(': ')[1])
        else:
            test_acc = 0.0
        res_dict[rbns_n_name] = test_acc
    for rbns_n_name in sorted(res_dict):
        print(f"{rbns_n_name}, {res_dict[rbns_n_name]}")
    

def render_psam(psam, normalize=True, **kwargs):
    if normalize:
        psam = psam / psam.sum(1)[:, None]
    assert psam.shape[1] == 4
    psam_df = pd.DataFrame(psam, columns=list("ACGT"))
    return logomaker.Logo(psam_df, **kwargs)


def generate_motif(motif_name):
    data = h5py.File(f"tiny_binary_dataset/protein_{motif_name}/rbns_train.h5", 'r')
    X = data["X"][:]
    L = data["L"][:]
    Y = data["Y"][:]
    tmpX = X[Y==1]
    tmpL = L[Y==1]

    return tmpX.mean(0)[:max(tmpL)]

def render_motif():
    args = get_args()
    print(args)
    motif_name = args.motif_name
    title = 'RBFOX3' if motif_name=="47" else 'RBFOX2'

    motif = generate_motif(motif_name)
    _, ax = plt.subplots(1, 1, figsize=(len(motif), 10))
    render_psam(motif, ax=ax)
    ax.set_title(title)
    
    plt.savefig(f"figures/{title}_motif.png")


if __name__ == "__main__":
    # x_list, y1_list, y2_list = read_roc_2(f"binary_all_acc_RBFOX3.txt")
    # plot_line(x_list, y1_list, y2_list, title='PSAM_RBFOX3_ROC', x_label=f"False Positive Rate", y_label=f"True Positive Rate", label1=f"non-binding", label2=f"binding", fig_title=f"threshold_psam_RBFOX3_roc.png", log=False)
    # read_psam_idx_acc_threshold("intermediate_data/binary_all_acc_tmp.txt")
    # read_rbns_acc("intermediate_data/single_proteins")

    # pbe_dict = read_protein_binding_enrichment()
    # plot_kmer_distribution(pbe_dict)
    # plot_enrichment_alignment_heatmap(pbe_dict)
    # path = "./rbns_model_binary_results/single_proteins_w_11"
    # read_single_protein_acc(path)

    # render_motif()

    # plot protein enrichment
    pbe_dict = read_protein_binding_enrichment()
    plot_protein_enrichment(pbe_dict)



    
                      
