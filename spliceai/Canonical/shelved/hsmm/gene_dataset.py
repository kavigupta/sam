import os
import h5py
import numpy as np
import torch

import tqdm.auto as tqdm
from permacache import permacache, stable_hash

from modular_splicing.utils.run_batched import run_batched
from modular_splicing.data_pipeline.reprocess_data import create_datapoints


def create_gene_dataset(original_datafile_path, *, create_datapoints_kwargs={}):
    modified_path = original_datafile_path.replace(".h5", "_genes.h5")
    if os.path.exists(modified_path):
        return modified_path
    with h5py.File(original_datafile_path, "r") as q:
        with h5py.File(modified_path, "w") as out:
            for idx in tqdm.trange(q["SEQ"].shape[0]):
                s = q["SEQ"][idx]

                x, y = create_datapoints(
                    s,
                    q["STRAND"][idx],
                    q["TX_START"][idx],
                    q["TX_END"][idx],
                    q["JN_START"][idx],
                    q["JN_END"][idx],
                    SL=5000,
                    CL_max=10_000,
                    **create_datapoints_kwargs,
                )
                x, y = x[:, 5000:-5000], y[0]
                x, y = np.concatenate(x), np.concatenate(y)
                [idxs] = np.where(x.sum(-1))
                x, y = x[: idxs[-1] + 1], y[: idxs[-1] + 1]
                assert x.shape[0] == len(s) - 10_000

                out[f"x{idx}"] = x
                out[f"y{idx}"] = y
    return modified_path


def run_on_entire_gene(model, x):
    chunk_size = 5000

    padded_length = (len(x) + chunk_size - 1) // chunk_size * chunk_size
    x_padded = np.pad(x, ([0, padded_length - x.shape[0]], [0, 0]))
    starts = [i * chunk_size for i in range(len(x_padded) // chunk_size)]
    pad = np.zeros((model.cl // 2, 4), dtype=np.uint8)
    x_padded = np.concatenate([pad, x_padded, pad])
    xs = np.array([x_padded[start : start + chunk_size + model.cl] for start in starts])
    yps = run_batched(lambda x: model(x).softmax(-1), xs.astype(np.float32), 16)
    yps = np.concatenate(yps)
    return yps[: x.shape[0]]


def run_on_all_genes(model, test_file_path):
    with h5py.File(create_gene_dataset(test_file_path), "r") as f:
        yps = []
        for idx in tqdm.trange(len(f) // 2):
            with torch.no_grad():
                yp = run_on_entire_gene(model, f[f"x{idx}"][:])
                yps.append(yp)
    return yps


def get_from_genes(test_file_path):
    with h5py.File(create_gene_dataset(test_file_path), "r") as f:
        xs, ys = [], []
        for idx in range(len(f) // 2):
            x = f[f"x{idx}"][:]
            y = f[f"y{idx}"][:]
            xs.append(x)
            ys.append(y)
    return xs, ys


def names(test_file_path):
    with h5py.File(test_file_path, "r") as f:
        return [x.decode("utf-8") for x in f["NAME"][:]]


def run_on_all_genes_binarized(model, test_file_path):
    return [yp > 1 for yp in run_on_all_genes_relative_to_score(model, test_file_path)]


def run_on_all_genes_with_threshold(model, test_file_path):
    _, ys = get_from_genes(test_file_path)
    yps = run_on_all_genes(model, test_file_path)
    ys_flat, yps_flat = np.concatenate(ys), np.concatenate(yps)
    quantiles = ys_flat[:, 1:].mean(0)
    thresholds = [
        np.quantile(yps_flat[:, i + 1], 1 - quantiles[i])
        for i in range(yps_flat.shape[1] - 1)
    ]
    return yps, thresholds


@permacache(
    "hsmm/gene_dataset/run_on_all_genes_relative_to_score",
    key_function=dict(model=stable_hash),
)
def run_on_all_genes_relative_to_score(model, test_file_path):
    yps, thresholds = run_on_all_genes_with_threshold(model, test_file_path)
    return [yp[:, 1:] / thresholds for yp in yps]


def plot_acc_by_length(
    stat, accuracies, name, ax, *, cut_off=0.01, chunks=10, style_fn=lambda k: {}
):
    low, high = np.quantile(stat, [cut_off, 1 - cut_off])
    boundaries = np.exp(np.linspace(np.log(low), np.log(high), chunks + 1))
    boundaries[0] = -float("inf")
    boundaries[-1] = float("inf")

    masks = [(lo < stat) & (stat < hi) for lo, hi in zip(boundaries, boundaries[1:])]

    stat_mean_vals = np.array([np.exp(np.mean(np.log(stat[mask]))) for mask in masks])
    acc_mean_vals = {
        k: np.array([accuracies[k][mask].mean() for mask in masks]) for k in accuracies
    }

    for k in acc_mean_vals:
        ax.plot(stat_mean_vals, 100 * acc_mean_vals[k], label=k, **style_fn(k))
    ax.set_xscale("log")
    ax.set_xlabel(name)
    ax.set_ylabel("accuracy [%]")
    ax.legend()
    ax.grid()
