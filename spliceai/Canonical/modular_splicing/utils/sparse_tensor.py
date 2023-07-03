import scipy.sparse


def pad_sparse_motifs_with_cl(motif_ys):
    """
    Pad the given sparse motifs of shape (N, SL) in order to add a CL = SL * 2.

    Output is of shape (N, SL + CL) = (N, 3 * SL).

    Flows together, e.g., [[0, 1], [2, 3]] -> [[0, 0, 0, 1, 2, 3], [0, 1, 2, 3, 0, 0]]
    """
    pad = scipy.sparse.csr_matrix((1, motif_ys.shape[1]), dtype=motif_ys.dtype)
    motif_ys_left = scipy.sparse.vstack([pad, motif_ys[:-1]])
    motif_ys_right = scipy.sparse.vstack([motif_ys[1:], pad])
    motif_ys = scipy.sparse.hstack([motif_ys_left, motif_ys, motif_ys_right])

    return motif_ys
