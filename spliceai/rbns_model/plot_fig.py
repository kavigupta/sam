import matplotlib.pyplot as plt
import re
import os

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


if __name__ == "__main__":
    x_list, y1_list, y2_list = read_roc_2(f"binary_all_acc_RBFOX3.txt")
    plot_line(x_list, y1_list, y2_list, title='PSAM_RBFOX3_ROC', x_label=f"False Positive Rate", y_label=f"True Positive Rate", label1=f"non-binding", label2=f"binding", fig_title=f"threshold_psam_RBFOX3_roc.png", log=False)

                      
