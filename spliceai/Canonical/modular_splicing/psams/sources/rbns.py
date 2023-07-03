import functools

import numpy as np
import os

from ..psams import PSAM
from .data_directory import DATA_DIRECTORY

MOTIFS_DIRECTORY = os.path.join(DATA_DIRECTORY, "motifs")
SPLICE_MOTIFS_DIRECTORY = os.path.join(DATA_DIRECTORY, "splice_point_motifs")


@functools.lru_cache(None)
def read_rbns_motifs(
    directory=MOTIFS_DIRECTORY, splice_motifs_directory=SPLICE_MOTIFS_DIRECTORY
):
    """
    Read the RBNS motifs from the given directory.
    """
    results = {
        protein_path.split(".")[0]: read_rbns_motif_from_file(
            os.path.join(directory, protein_path)
        )
        for protein_path in os.listdir(directory)
    }
    path_fn = lambda name: read_splicepoint_matrix(
        os.path.join(splice_motifs_directory, f"{name}.txt")
    )
    results.update(
        {
            "3P": [path_fn("3HGC_human"), path_fn("3LGC_human")],
            "5P": [path_fn("5HGC_human"), path_fn("5LGC_human")],
        }
    )
    return results


@functools.lru_cache(None)
def read_rbns_motif_from_file(path, has_runinfo=True):
    """
    Read the RBNS motif from the given file.
    """
    with open(path) as f:
        contents = list(f)
    if has_runinfo:
        assert contents.pop(0).startswith("# runinfo")
    assert contents.pop(0).startswith("# ModelSetParams")
    matrices = []
    i = 0
    while contents:
        assert contents.pop(0) == f"# PSAM {i}\n"
        params = eval("dict({})".format(",".join(contents.pop(0).split()[1:])))
        assert contents.pop(0) == "#\tA\tC\tG\tU\tconsensus\n"
        rows = []
        while contents and contents[0].split():
            row = [float(x) for x in contents.pop(0).split()[:-1]]
            rows.append(row)
        if contents:
            contents.pop(0)
        rows = np.array(rows)
        matrices.append(PSAM(matrix=rows, **params))
        i += 1
    return matrices


@functools.lru_cache(None)
def read_splicepoint_matrix(path):
    """
    Read a splicepoint matrix. These should not be used.
    """
    with open(path) as f:
        motifs_file = f.read().split()
    matrix = []
    letters = list("ACGT")
    while motifs_file:
        first = motifs_file.pop(0)
        if first in letters:
            first_letter = letters.pop(0)
            assert first == first_letter, str((first, first_letter))
            matrix.append([])
        else:
            matrix[-1].append(float(first))
    matrix = np.array(matrix).T
    matrix = matrix / matrix.max(1)[:, None]
    return PSAM(
        A0=1,
        n=matrix.shape[0],
        acc_k=None,
        acc_shift=None,
        acc_scale=None,
        matrix=matrix,
        threshold=0.0001,
    )
