from permacache import permacache, stable_hash

import tqdm.auto as tqdm
import numpy as np
import matplotlib.pyplot as plt

from modular_splicing.data_for_experiments.standardized_sample import (
    standardized_sample,
)
from modular_splicing.models.modules.lssi_in_model import LSSI_MODEL_THRESH
from modular_splicing.utils.plots.plot_psam import render_psam, information_psam

from modular_splicing.utils.run_batched import run_batched


def run_model(mod, cl):
    xs, _ = standardized_sample("dataset_test_0.h5", None, cl=cl)
    result = run_batched(mod, xs, 100, pbar=tqdm.tqdm)[:, cl // 2 : -cl // 2]
    overall = np.log(np.exp(result).sum(-1))
    mask = overall > LSSI_MODEL_THRESH + 0.5
    delta = result - overall[:, :, None]
    mean_vals = delta[mask].mean(0)
    delta = delta - mean_vals
    # TODO do not hardcode 3
    highlighted = np.eye(3, dtype=np.bool_)[delta.argmax(-1)] * mask[:, :, None]
    return xs, highlighted


def produce_filtered_and_psams(m, cl):
    mod = m.model.motif_model.parallel_models[1]
    filters = mod.get_filters()
    psams, means = produce_psams(mod, cl)
    # TODO fix this properly
    psams, means = psams[: len(filters)], means[: len(filters)]
    return m.seed, filters, psams, means


@permacache("notebooks/multiple-donors/run_model_6", key_function=dict(mod=stable_hash))
def produce_psams(mod, cl):
    xs, highlighted = run_model(mod, cl)
    psams = []
    for i in range(highlighted.shape[-1]):
        batch, seq = np.where(highlighted[:, :, i])
        seq, offset = np.meshgrid(seq, np.arange(-2, 6 + 1))
        seq = seq + offset + cl // 2
        psams.append(xs[batch, seq].mean(1))
    return psams, highlighted.mean((0, 1))


def plot(axs, seed, filters, psams, means):
    psams = [information_psam(psam) for psam in psams]
    mean_filt = np.mean(filters, 0)
    mean_psam = np.mean(psams, 0)
    for i, filt in enumerate(filters):
        render_psam(filt.T, ax=axs[i, 0], psam_mode="raw")
        axs[i, 0].axis("off")
        axs[i, 0].set_title(f"Filter {seed}.{i + 1}")

        render_psam(psams[i], ax=axs[i, 1], psam_mode="raw")
        axs[i, 1].axis("off")
        axs[i, 1].set_title(f"Logo {seed}.{i + 1} [{means[i]:.3%}]")

        render_psam((filt - mean_filt).T, ax=axs[i, 2], psam_mode="raw")
        axs[i, 2].axis("off")
        axs[i, 2].set_title(f"Filter {seed}.{i + 1} - mean")

        render_psam(psams[i] - mean_psam, ax=axs[i, 3], psam_mode="raw")
        axs[i, 3].axis("off")
        axs[i, 3].set_title(f"Logo {seed}.{i + 1} - mean")


def subplots(rows, columns, width=5, height=2):
    _, axs = plt.subplots(
        rows,
        columns,
        figsize=(columns * width, height * rows),
        tight_layout=True,
        facecolor="white",
    )
    return axs


def draw_psams(mod, cl):
    res_each = [
        produce_filtered_and_psams(m, cl=cl) for m in mod.non_binarized_models()
    ]
    axs = subplots(4 * len(res_each), len(res_each[0][1]))
    for res, axs in zip(res_each, axs.reshape(len(res_each), -1, *axs.shape[1:])):
        plot(axs.T, *res)
    plt.savefig(f"output-csvs/multiple-donor-models/{mod.name}.png")
