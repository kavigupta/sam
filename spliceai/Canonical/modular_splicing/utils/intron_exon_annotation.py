"""
Module for handling the conversion of a normal dataset of splicepoints
to a dataset of intron/exon annotations.
"""

import numpy as np

IS_INTRON = {(1, 2): False, (2, 1): True}

ANNOTATION_INTRON = -2
ANNOTATION_EXON = -1


def to_intron_exon_annotation(ys):
    """
    Take the given annotation of outputs, and convert it to a binary
    annotation of introns and exons.

    Parameters
    ----------
    ys : np.ndarray
        (N, L, 3): The annotation of the outputs, in one-hot format.
            0 for null, 1 for acceptor, 2 for donor.

    Returns
    -------
    np.ndarray
        (N, L): The annotation of the outputs, in binary format.
            True for intron, False for exon.
    """
    out = np.zeros(ys.shape[:2], dtype=np.bool)
    last_element = 1  # start with an "acceptor", since we start in an exon
    for batch_idx in range(ys.shape[0]):
        y = ys[batch_idx].argmax(-1)
        [positions] = np.where(y)
        acc_don = list(y[positions])
        positions = [0] + list(positions) + [ys.shape[1]]

        acc_don = [last_element] + acc_don
        acc_don = acc_don + [3 - acc_don[-1]]

        for i in range(len(acc_don) - 1):
            key = acc_don[i], acc_don[i + 1]
            if key not in IS_INTRON:
                print("bad outcome", key)
            encoding = IS_INTRON.get(key, False)
            start, end = positions[i], positions[i + 1]
            out[batch_idx, start:end] = encoding

        last_element = acc_don[-2]
    return out


def to_onehot_intron_exon_annotation(ys):
    """
    Similar to to_intron_exon_annotation, but returns something that
        looks like a one-hot encoding of the original sequence, except
        with introns and exons encoded as ANNOTATION_INTRON and ANNOTATION_EXON,
        respectively, in the null channel.

    Parameters
    ----------
    ys : np.ndarray
        (N, L, 3): The annotation of the outputs, in one-hot format.
            0 for null, 1 for acceptor, 2 for donor.

    Returns
    -------
    np.ndarray
        (N, L, 3): The annotation of the outputs, in one-hot format.
            0 for null, 1 for acceptor, 2 for donor
            Uses ANNOTATION_INTRON and ANNOTATION_EXON for introns and
            exons in the null channel instead of 1
    """
    ys = ys[0].copy()
    annot = to_intron_exon_annotation(ys)
    annot = np.where(annot, ANNOTATION_INTRON, ANNOTATION_EXON)
    ys[ys[:, :, 0] != 0, 0] *= annot[ys[:, :, 0] != 0]
    return ys[None]
