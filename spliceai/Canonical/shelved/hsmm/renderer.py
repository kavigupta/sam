import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_transition_matrix(hsmm, ordered_states, size, path):
    ordering = [hsmm.states.index(i) for i in ordered_states]
    trans_matr = hsmm.transition_matrix[ordering][:, ordering]
    first, second = np.where(trans_matr)
    plt.figure(dpi=120, figsize=(size, size))
    plt.grid()
    plt.scatter(
        first,
        second,
        c=trans_matr[first, second],
        norm=mpl.colors.LogNorm(),
        cmap="jet",
    )
    plt.xticks(np.arange(len(ordered_states)), ordered_states, rotation=90)
    plt.yticks(np.arange(len(ordered_states)), ordered_states)
    plt.xlabel("From")
    plt.ylabel("To")
    plt.colorbar()
    plt.savefig(path)
    plt.show()
