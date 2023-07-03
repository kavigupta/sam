import math
import re

import numpy as np
import scipy.sparse

"""
One-hot encoding of the inputs: 0 is for padding, and 1, 2, 3, 4 correspond
to A, C, G, T respectively.
"""
IN_MAP = np.asarray(
    [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
)

"""
One-hot encoding of the outputs: 0 is for no splice, 1 is for acceptor,
2 is for donor and -1 is for padding.
"""
OUT_MAP = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])


def create_datapoints(
    seq, strand, tx_start, tx_end, jn_start, jn_end, *, SL, CL_max, mode="one-hot"
):
    """
    This function first converts the sequence into an integer array, where
    A, C, G, T, N are mapped to 1, 2, 3, 4, 0 respectively. If the strand is
    negative, then reverse complementing is done. The splice junctions
    are also converted into an array of integers, where 0, 1, 2, -1
    correspond to no splicing, acceptor, donor and missing information
    respectively. It then calls reformat_data and one_hot_encode
    and returns X, Y which can be used by Keras models.

    if you use external-indices, then the output is of the form
    -1 = no splicing, 0 = nothing, positive = the index into the annotation table + 1
    you will need to use the annotation table to get whether it's an acceptor/donor
    and other information
    """

    assert mode in ["one-hot", "sparse", "external-indices"]

    if seq is not None:
        seq = seq.decode("utf-8")
    strand = strand.decode("utf-8")
    tx_start = tx_start.decode("utf-8")
    tx_end = tx_end.decode("utf-8")
    jn_start = jn_to_list(jn_start)
    jn_end = jn_to_list(jn_end)

    assert len(jn_start) == len(jn_end)

    if mode == "external-indices":
        jn_start, jn_start_vals = jn_start
        jn_end, jn_end_vals = jn_end
        jn_start = [jn_start]
        jn_end = [jn_end]
        dtype_to_use_for_y = np.int64
        jn_start_vals = [x + 1 for x in jn_start_vals]
        jn_end_vals = [x + 1 for x in jn_end_vals]
    else:
        assert mode in ["one-hot", "sparse"]
        jn_start_vals, jn_end_vals = [None] * len(jn_start[0]), [None] * len(jn_end[0])
        dtype_to_use_for_y = np.int8

    assert len(jn_start) == len(jn_end) == 1
    assert len(jn_start[0]) == len(jn_start_vals)
    assert len(jn_end[0]) == len(jn_end_vals)

    if seq is not None:
        seq = (
            "N" * (CL_max // 2) + seq[CL_max // 2 : -CL_max // 2] + "N" * (CL_max // 2)
        )
        # Context being provided on the RNA and not the DNA

        seq = seq.upper().replace("A", "1").replace("C", "2")
        seq = seq.replace("G", "3").replace("T", "4").replace("N", "0")

    tx_start = int(tx_start)
    tx_end = int(tx_end)

    assert tx_start <= tx_end

    zero_value = -1
    Y0 = [
        scipy.sparse.dok_matrix((tx_end - tx_start + 1, 1), dtype=dtype_to_use_for_y)
        for t in range(1)
    ]

    if strand == "+":
        X0 = np.asarray(list(map(int, list(seq)))) if seq is not None else None

        for t in range(1):
            if len(jn_start[t]) > 0:
                zero_value = 0
                for c, v in zip(jn_start[t], jn_start_vals):
                    if tx_start <= c <= tx_end:
                        Y0[t][c - tx_start] = 2 if v is None else v
                for c, v in zip(jn_end[t], jn_end_vals):
                    if tx_start <= c <= tx_end:
                        Y0[t][c - tx_start] = 1 if v is None else v
                    # Ignoring junctions outside annotated tx start/end sites

    elif strand == "-":
        X0 = (
            (5 - np.asarray(list(map(int, list(seq[::-1]))))) % 5
            if seq is not None
            else None
        )  # Reverse complement

        for t in range(1):
            if len(jn_start[t]) > 0:
                zero_value = 0
                for c, v in zip(jn_end[t], jn_end_vals):
                    if tx_start <= c <= tx_end:
                        Y0[t][tx_end - c] = 2 if v is None else v
                for c, v in zip(jn_start[t], jn_start_vals):
                    if tx_start <= c <= tx_end:
                        Y0[t][tx_end - c] = 1 if v is None else v

    Xd, Yd, padding = reformat_data(X0, Y0, SL=SL, CL_max=CL_max)

    if mode == "one-hot":
        Yd[0] = Yd[0].toarray()
        Yd[0][Yd[0] == 0] = zero_value
        Yd[0][-1, -padding:] = -1

        X, Y = one_hot_encode(Xd, Yd)

        return X, Y
    elif mode == "sparse":
        return Xd, Yd
    elif mode == "external-indices":
        Yd[0] = Yd[0].toarray()
        Yd[0][Yd[0] == 0] = zero_value
        Yd[0][-1, -padding:] = -1

        Xd, _ = one_hot_encode(Xd, None)
        return Xd, Yd
    else:
        raise RuntimeError(f"Mode {mode!r} is not valid")


def jn_to_list(jn_start):
    """
    Convert the junction numbers from a string to a list of integers
    """
    if len(jn_start) == 1 and isinstance(jn_start[0], list):
        return jn_start
    jn_start = [x.decode("utf-8") for x in jn_start]
    jn_start = [list(map(int, [t for t in re.split(",", x) if t])) for x in jn_start]
    return jn_start


def reformat_data(X0, Y0, *, SL, CL_max):
    """
    This function converts X0, Y0 of the create_datapoints function into
    blocks such that the data is broken down into data points where the
    input is a sequence of length SL+CL_max corresponding to SL nucleotides
    of interest and CL_max context nucleotides, the output is a sequence of
    length SL corresponding to the splicing information of the nucleotides
    of interest. The CL_max context nucleotides are such that they are
    CL_max/2 on either side of the SL nucleotides of interest.
    """

    num_points = ceil_div(Y0[0].shape[0], SL)

    if X0 is not None:
        Xd = np.zeros((num_points, SL + CL_max))
        X0 = np.pad(X0, [0, SL], "constant", constant_values=0)
        for i in range(num_points):
            Xd[i] = X0[SL * i : CL_max + SL * (i + 1)]
    else:
        Xd = None

    padding_yd = num_points * SL - Y0[0].shape[0]
    Yd = [
        scipy.sparse.vstack(
            [Y0[0], scipy.sparse.dok_matrix((padding_yd, 1), dtype=Y0[0].dtype)]
        )
    ]
    Yd[0] = Yd[0].reshape((num_points, SL))

    return Xd, Yd, padding_yd


def ceil_div(x, y):
    return int(math.ceil(float(x) / y))


def one_hot_encode(Xd, Yd):
    if Xd is not None:
        Xd = IN_MAP[Xd.astype("int8")]
    if Yd is not None:
        Yd = [OUT_MAP[Yd[t].astype("int8")] for t in range(1)]
    return Xd, Yd
