import numpy as np
import tqdm.auto as tqdm
from permacache import permacache, stable_hash
from modular_splicing.dataset.datafile_object import SpliceAIDatafile
from modular_splicing.utils.arrays import Sparse
from .cassette_exons import locate_cassette_exon

from modular_splicing.legacy.hash_model import hash_model

from modular_splicing.utils.run_batched import run_batched


@permacache(
    "validation/knockdown_dataset/compute_knockdown/compute_with_and_without_knockouts_4",
    key_function=dict(
        experimental_setting=stable_hash,
        m=hash_model,
        table=lambda table: stable_hash(
            [
                np.array(table.index),
                {c: np.array(table[c]).tolist() for c in sorted(table.columns)},
            ]
        ),
    ),
    multiprocess_safe=True,
)
def compute_with_and_without_knockouts(
    experimental_setting, m, table, path, relevant_motif_id
):
    """
    Compute the model on the exons specified by the table, with and without knockouts.

    Args:
        experimental_setting: The experimental setting to use.
        m: The model to use.
        table: The table of exons to use.
        path: The path to the datafile to use. Only exons who appear within genes in this datafile will be used.
        relevant_motif_id: The motif to knock down.

    Returns a dictionary with keys:
        indices :: list of int:
            The indices of the exons in the table that were used.
        fs :: list of dictionaries from name to location:
            the feature indices for each exon.
        motifs :: Sparse array of dimensions (len(indices), L, M):
            The full motifs array.
        pred :: list of dictionaries from name to location:
            The model output for each exon on each feature.
        perturbed_pred :: list of dictionaries from name to location:
            The model output for each exon on each feature, with the relevant motif knocked down.
    """
    indices, xs, fs = collect_relevant_data(path, table, experimental_setting, m.cl)

    pred, perturbed_pred = compute_with_and_without_knockouts_on_data(
        m=m, xs=xs, relevant_motif_id=relevant_motif_id
    )

    motifs = Sparse.of(pred["motifs"])

    def select(yps):
        return [
            {k: yp[i - m.cl // 2, int(k[0] == "5'")] for k, i in f.items()}
            for yp, f in zip(yps, fs)
        ]

    pred = select(pred["y"])
    perturbed_pred = select(perturbed_pred["y"])
    return dict(
        fs=fs,
        indices=indices,
        motifs=motifs,
        pred=pred,
        perturbed_pred=perturbed_pred,
    )


def collect_relevant_data(datafile_path, table, experimental_setting, cl):
    """
    Collect the data relevant to the table from the given datafile.

    Args:
        datafile_path: The path to the datafile to use.
        table: The table of exons to use.
        experimental_setting: The experimental setting to use.
        cl: The context length

    Returns a tuple of:
        indices :: list of int:
            The indices of the exons in the table that were used.
        xs :: array of dimension (len(indices), L, 4):
            The text of the exons, with the context.
        fs :: list of dictionaries from name to location:
            the feature indices for each exon.
    """
    datafile = SpliceAIDatafile.load(datafile_path)
    indices = []
    xs = []
    fs = []
    for i, row in tqdm.tqdm(list(table.iterrows()), desc="Finding datapoints"):
        r = locate_cassette_exon(
            datafile, row, experimental_setting=experimental_setting, cl=cl
        )
        if r is None:
            continue
        indices.append(i)
        x, f = r
        xs.append(x)
        fs.append(f)

    xs = np.array(xs)

    return indices, xs, fs


def compute_with_and_without_knockouts_on_data(m, xs, relevant_motif_id):
    def run(**kwargs):
        def run_model(x):
            outputs = m(x, collect_intermediates=True, **kwargs)
            motifs = outputs["post_sparse_motifs_only"][:, :, relevant_motif_id]
            y = outputs["output"].softmax(-1)[:, :, 1:]
            return dict(motifs=motifs, y=y)

        return run_batched(run_model, xs, 64, pbar=tqdm.tqdm)

    def knock_down(x):
        x[:, :, relevant_motif_id] = 0
        return x

    pred = run()
    perturbed_pred = run(manipulate_post_sparse=knock_down)
    return pred, perturbed_pred
