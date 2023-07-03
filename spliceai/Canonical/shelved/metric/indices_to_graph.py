import numpy as np
import scipy


def generate_closeness_pairs(a, b, *, delta):
    result = []
    for i in range(a.shape[0]):
        b_index = np.searchsorted(b, a[i] - delta)
        for j in range(b_index, b.shape[0]):
            if b[j] - a[i] > delta:
                break
            assert not (a[i] - b[j] > delta)
            result.append([i, j])
    return np.array(result).T


def generate_closeness_matrix(a, b, *, delta):
    row_ind, col_ind = generate_closeness_pairs(a, b, delta=delta)
    return scipy.sparse.csr_matrix(
        ([True] * row_ind.shape[0], (row_ind, col_ind)),
        shape=(a.shape[0], b.shape[0]),
    )
