import functools

import attr
# import tqdm
import os
import numpy as np

import torch

from utils import batched

# from permacache import permacache, stable_hash

MOTIFS_DIRECTORY = os.path.join(
    os.path.dirname(__file__), "data_motifs", "motifs"
)
SPLICE_MOTIFS_DIRECTORY = os.path.join(
    os.path.dirname(__file__), "data_motifs", "splice_point_motifs"
)

@attr.s
class PSAM:
    A0 = attr.ib()
    n = attr.ib()
    acc_k = attr.ib()
    acc_shift = attr.ib()
    acc_scale = attr.ib()
    matrix = attr.ib()
    threshold = attr.ib(kw_only=True, default=0.2)


@functools.lru_cache(None)
def read_motif(path):
    with open(path) as f:
        contents = list(f)
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
        threshold= 0.0001,
    )


@functools.lru_cache(None)
def read_motifs(
    directory=MOTIFS_DIRECTORY,
    splice_motifs_directory=SPLICE_MOTIFS_DIRECTORY,
):  
    # protein_name = list()
    # protein_file = open('protein_name.txt', 'w')
    # for protein_path in os.listdir(directory):
    #     protein_file.write(protein_path.split('.')[0] + '\n')
        # protein_name.append(protein_path.split('.')[0])
    # protein_file = open('protein_name.txt', 'w')
    # protein_file.close()

    results = {
        protein_path.split(".")[0]: read_motif(os.path.join(directory, protein_path))
        for protein_path in os.listdir(directory)
    }
    path_fn = lambda name: read_splicepoint_matrix(
        os.path.join(splice_motifs_directory, f"{name}.txt"),
        )
    results.update(
        {
            "3P": [path_fn("3HGC_human"), path_fn("3LGC_human")],
            "5P": [path_fn("5HGC_human"), path_fn("5LGC_human")],
        }
    )
    # print(f"results: {results}")

    return results


def _motifs_list(motifs, x, model=None, silent=False):
    all_out = {}
    non_pad_values = x.sum(-1) > 0
    # X = torch.tensor(x.astype(np.float32)).cuda().transpose(2, 1)
    X = x.transpose(2, 1)
    if model is not None:
        splicepoint_motifs = batched(
            lambda inp: model(inp).softmax(-1), x.astype(np.float32), 100
        )
        clip_amount = (x.shape[1] - splicepoint_motifs.shape[1]) // 2
        splicepoint_motifs = np.pad(
            splicepoint_motifs, [(0, 0), (clip_amount, clip_amount), (0, 0)]
        )
        splicepoint_motifs = splicepoint_motifs[:, :, 1:] / 1e-4
        all_out["5P"], all_out["3P"] = splicepoint_motifs.transpose(2, 0, 1)

    names = sorted(motifs)
    # print(f"names: {names}")
    if not silent:
        names = tqdm.tqdm(names)

    for i, motif_name in enumerate(names):

        if motif_name in all_out:
            continue

        motif = motifs[motif_name]

        M = (
            torch.tensor([np.log(m.matrix + 1e-100).astype(np.float32) for m in motif])
            .cuda()
            .transpose(2, 1)
        )
        A0s = np.array([m.A0 for m in motif])
        A0s = A0s / A0s.max()
        A0s = torch.tensor(A0s[None, :, None]).cuda()
        out = torch.nn.functional.conv1d(X, M)
        out = out.exp() * A0s
        out = out.sum(1)
        out = out.cpu().numpy()
        out = np.pad(out, pad_width=[(0, 0), (0, X.shape[-1] - out.shape[-1])])

        out = out * non_pad_values.cpu().numpy()

        all_out[motif_name] = out
    # print(f"all out: {all_out}")
    # print(f"all out items: {all_out.items()}")
    # psam_name_f = open('protein_psam_name.txt', 'w')
    # for x, _ in sorted(all_out.items()):
    #     psam_name_f.write(f"{x}\n")
    # psam_name_f.close()
    # print(f"{[x for x, _ in sorted(all_out.items())]}")
    # print(f"{[v for _, v in sorted(all_out.items())]}")
    # exit(0)
    
    # do you need this line?
    all_out = [v for _, v in sorted(all_out.items())]
    return all_out


def motifs_for(motifs, x, model=None):
    res = np.array(_motifs_list(motifs, x, model, silent=True)).transpose(1, 2, 0)
    # res[res < 1] = 0
    return res



