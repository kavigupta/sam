import os
import pickle
import numpy as np
import h5py

import tqdm.auto as tqdm

from modular_splicing.dataset.data_rewriter import data_rewriter_types
from .gtex_dataset import GTExDataset
from modular_splicing.utils.construct import construct


def create_const_alt_data(
    *,
    path_whole,
    path_const_alt,
    gtex_index_rewriter_spec,
    bar_alt,
    bar_const,
    psi_calculation_width,
):
    path_probs = path_const_alt + ".psi.pkl"

    if all(os.path.exists(p) for p in [path_const_alt, path_probs]):
        with open(path_probs, "rb") as f:
            return pickle.load(f)

    dset = GTExDataset(
        path=path_whole,
        sl=5000,
        cl=400,
        cl_max=10_000,
        iterator_spec=dict(type="FastIter", shuffler_spec=dict(type="DoNotShuffle")),
        gtex_index_rewriter_spec=gtex_index_rewriter_spec,
        post_processor_spec=dict(type="IdentityPostProcessor"),
        psi_calculation_width=psi_calculation_width,
    )

    averaged_psi = construct(data_rewriter_types(), gtex_index_rewriter_spec)

    all_probs = []
    with h5py.File(path_whole, "r") as f_whole, h5py.File(
        path_const_alt, "w"
    ) as f_const_alt:
        for idx in tqdm.trange(len(f_whole) // 2):
            x, y = f_whole[f"X{idx}"][:], f_whole[f"Y{idx}"][0]
            y = np.array([averaged_psi.rewrite_indices(y, dset=dset) for y in y])
            y = y[..., 1:]
            all_probs.append(y[y != 0])
            both = y >= bar_alt
            const = y >= bar_const
            f_const_alt[f"X{idx}"] = x
            f_const_alt[f"Y{idx}"] = np.concatenate([both, const], axis=-1)
        f_const_alt["ordering"] = np.array([b"gtex_const_and_alt", b"gtex_const"])

    with open(path_probs, "wb") as f:
        pickle.dump(np.concatenate(all_probs), f)

    return np.concatenate(all_probs)


def create_const_alt_dataset(
    *,
    dir_whole,
    dir_const_alt,
    gtex_index_rewriter_spec,
    bar_alt,
    bar_const,
    data_chunks=[("train", "all"), ("test", "0")],
    psi_calculation_width=2000,
):
    results = []
    for data, chunk in data_chunks:
        path_whole = os.path.join(dir_whole, f"dataset_{data}_{chunk}.h5")
        try:
            os.makedirs(dir_const_alt)
        except FileExistsError:
            pass
        path_const_alt = os.path.join(dir_const_alt, f"dataset_{data}_{chunk}.h5")
        results.append(
            create_const_alt_data(
                path_whole=path_whole,
                path_const_alt=path_const_alt,
                gtex_index_rewriter_spec=gtex_index_rewriter_spec,
                bar_alt=bar_alt,
                bar_const=bar_const,
                psi_calculation_width=psi_calculation_width,
            )
        )
    return results
