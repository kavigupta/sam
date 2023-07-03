from permacache import permacache, stable_hash

import numpy as np
import pandas as pd
import tqdm.auto as tqdm

from modular_splicing.legacy.remapping_pickle import permacache_with_remapping_pickle
from modular_splicing.utils.arrays import Sparse
from modular_splicing.utils.run_batched import run_batched


@permacache_with_remapping_pickle(
    "modular_splicing/mrs/motifs/all_motifs_for_seqs_2",
    key_function=dict(model=stable_hash, seqs=stable_hash),
)
def all_motifs_for_seqs(model, seqs, mcl=20):
    """
    Produce all motifs for a set of sequences.

    Arguments:
        model: The model to use.
        seqs: The sequences to use (N, L, 4)
        mcl: The motif context length

    Returns:
        Sparse motif binding sites with (N, L - mcl, M)
    """
    result = run_batched(
        lambda x: model(x, only_motifs=True)["post_sparse_motifs_only"][
            :, mcl // 2 : -(mcl // 2)
        ],
        np.eye(4, dtype=np.float32)[seqs],
        1000,
        pbar=tqdm.tqdm,
    )

    result = result != 0

    return Sparse.of(result)


def subset_delta_results(delta, models, don_seqs, seeds, *, motif_idx):
    """
    Produce results for each subset of motif bindings for the given delta values.

    Arguments:
        delta: The delta values to use as the summary statistic
        models: The models to use
        don_seqs: The donor sequences to use
        seeds: The seeds to use. models should have keys "FM" and "AM #{seed}" for each seed.
        motif_idx: The motif index to analyze

    Returns:
        - the mean number of binding sites by position for AM #1
        - the matches mask for each model
        - a table with rows for each seed and columns for each subset.
    """
    results = {
        k: all_motifs_for_seqs(models[k], don_seqs).original[:, :, motif_idx]
        for k in models
    }

    matches = {k: (results[k] != 0).any(-1) for k in results}

    table = {
        i: {
            r"AM & FM": delta[matches[f"AM #{i}"] & matches["FM"]].mean(),
            r"AM \ FM": delta[matches[f"AM #{i}"] & (~matches["FM"])].mean(),
            r"FM \ AM": delta[matches["FM"] & (~matches[f"AM #{i}"])].mean(),
            r"~FM & ~AM": delta[(~matches["FM"]) & (~matches[f"AM #{i}"])].mean(),
            r"FM": delta[matches["FM"]].mean(),
            r"~FM": delta[~matches["FM"]].mean(),
            r"AM": delta[matches[f"AM #{i}"]].mean(),
            r"~AM": delta[~matches[f"AM #{i}"]].mean(),
        }
        for i in seeds
    }
    table = pd.DataFrame(table).T
    return (results["AM #1"] != 0).mean(0) * 100, matches, table


@permacache(
    "modular_splicing/mrs/motifs/subset_table_results",
    key_function=dict(delta=stable_hash, models=stable_hash, don_seqs=stable_hash),
)
def subset_table_results(delta, models, don_seqs, seeds, names):
    """
    See subset_delta_results, but for multiple motifs.
    """
    return {
        name: subset_delta_results(delta, models, don_seqs, seeds, motif_idx=i)
        for i, name in enumerate(tqdm.tqdm(names))
    }


def mean_subset_table_results(delta, models, don_seqs, seeds, names):
    """
    See subset_table_results, but with a mean taken across different seeds.
    """
    tables = subset_table_results(delta, models, don_seqs, seeds, names)
    mean = pd.DataFrame({k: tables[k][-1].mean() for k in tables}).T
    return mean
