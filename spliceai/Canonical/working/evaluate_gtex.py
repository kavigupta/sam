from matplotlib import pyplot as plt
from more_itertools import chunked
import tqdm.auto as tqdm
from permacache import permacache, stable_hash
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import torch

from modular_splicing.dataset.generic_dataset import dataset_types
from modular_splicing.gtex_data.experiments.create_dataset import MarginalPsiV3
from modular_splicing.utils.construct import construct


@permacache(
    "working/evaluate_gtex/run_on_gtex_data",
    key_function=dict(mod=stable_hash, batch_size=None),
)
def run_on_gtex_data(
    mod, gtex_data_spec, *, num_tissues, limit=float("inf"), batch_size=32
):
    dset = construct(dataset_types(), gtex_data_spec)
    items = []
    limit = min(len(dset) // 2, limit)
    for i, x in enumerate(dset):
        if i >= limit:
            break
        items.append(x)

    all_settings = []
    all_ys = []
    all_yps = []
    for chunk in chunked(tqdm.tqdm(items), batch_size):
        ys = np.array([it["outputs"]["y"] for it in chunk])[:, :, 1:]
        chunk = [it["inputs"] for it in chunk]
        xs = np.array([it["x"] for it in chunk])
        settings = np.array([it["setting"] for it in chunk])
        s = settings[:, 0, 0]
        xs, settings = torch.tensor(xs).float().cuda(), torch.tensor(settings).cuda()
        with torch.no_grad():
            yps = mod(dict(x=xs, setting=settings)).softmax(-1)
            yps = yps.cpu().numpy()
            yps = yps[:, :, 1:]
        all_settings.append(s)
        all_ys.append(ys)
        all_yps.append(yps)
    all_settings = np.concatenate(all_settings)
    all_ys = np.concatenate(all_ys)
    all_yps = np.concatenate(all_yps)
    all_settings = all_settings.reshape(-1, num_tissues)
    assert (all_settings == np.arange(num_tissues)).all()
    all_ys = all_ys.reshape(-1, num_tissues, *all_ys.shape[1:]).transpose(0, 2, 1, 3)
    all_yps = all_yps.reshape(-1, num_tissues, *all_yps.shape[1:]).transpose(0, 2, 1, 3)

    multiplier = 20
    frequencies = multiplier * (all_ys > 0).mean((0, 1))
    thresholds = [
        [
            np.quantile(all_yps[:, :, t, c], 1 - frequencies[t, c])
            for c in range(all_yps.shape[-1])
        ]
        for t in range(all_yps.shape[-2])
    ]
    mask = (all_ys > 0).any((-1, -2)) | (all_yps > thresholds).any((-1, -2))

    return dict(
        mask=np.where(mask),
        ys=all_ys[mask],
        yps=all_yps[mask],
        thresholds=thresholds,
        frequencies=frequencies,
    )


def compute_constitu_mask(ys):
    return (ys == 0).all(-1) | (ys == 1).any(-1)


def topk(ys, yps, thresh_ys):
    """
    ys: float[N, C]
    yps: float[N, C]
    thresh_ys: float or float[C]

    Returns accs: float[C]
    """

    assert ys.shape == yps.shape
    assert len(ys.shape) == 2
    ys = ys > thresh_ys
    quantities = ys.mean(0)
    thresholds = [
        np.quantile(yps[:, c], 1 - quantities[c]) for c in range(yps.shape[-1])
    ]
    yps = yps > thresholds
    accs = (ys & yps).sum(0) / ys.sum(0)
    return accs


def topk_constitu(ys, yps):
    constitu_mask = compute_constitu_mask(ys)
    ys = ys[constitu_mask]
    yps = yps[constitu_mask]
    return topk(ys, yps, 0.5)


def rmse_alternative(ys, yps):
    constitu_mask = compute_constitu_mask(ys)
    ys = ys[~constitu_mask]
    yps = yps[~constitu_mask]
    return ((ys - yps) ** 2).mean(0) ** 0.5


def compute_all_stats(ys, yps, acc_fn):
    """
    Computes all stats of each tissue with respect to other tissues.

    Parameters
    ----------
    ys: float[N, T, C]
    yps: float[N, T, C]
    acc_fn: function that takes (ys, yps) and returns accs: float[C]

    Returns
    -------
    stats: float[T, T, C]
        stats[i, j, c] = acc_fn(ys[:, i, c], yps[:, j, c])
        that is the accuracy of predicting tissue i using a model trained on tissue j
    """
    assert ys.shape == yps.shape
    assert len(ys.shape) == 3
    T = ys.shape[1]
    stats = np.zeros((T, T, ys.shape[-1]))
    for i in range(T):
        for j in range(T):
            stats[i, j] = acc_fn(ys[:, i], yps[:, j])
    return stats


@permacache(
    "working/evaluate_gtex/evaluate_on_gtex_data",
    key_function=dict(mod=stable_hash, batch_size=None),
)
def evaluate_on_gtex_data(
    mod, gtex_data_spec, *, num_tissues, limit=float("inf"), batch_size=32
):
    res = run_on_gtex_data(
        mod,
        gtex_data_spec,
        num_tissues=num_tissues,
        limit=limit,
        batch_size=batch_size,
    )
    ys = res["ys"]
    yps = res["yps"]

    results = dict(
        topk={
            thresh: compute_all_stats(ys, yps, lambda ys, yps: topk(ys, yps, thresh))
            for thresh in np.arange(0, 1, 0.05)
        },
        topk_constitu=compute_all_stats(ys, yps, topk_constitu),
        rmse_alternative=compute_all_stats(ys, yps, rmse_alternative),
    )
    return results


def display_stats_matrix(stat_name, tissue_names, matr, fig, ax):
    im = ax.imshow(matr.mean(-1) * 100, cmap="gray_r")
    ax.set_xticks(range(len(tissue_names)))
    ax.set_yticks(range(len(tissue_names)))
    ax.set_xticklabels(tissue_names, rotation=90)
    ax.set_yticklabels(tissue_names)

    for i in range(len(tissue_names)):
        for j in range(len(tissue_names)):
            ax.text(
                j,
                i,
                f"{matr[i, j][0] * 100:.1f}\n{matr[i, j][1] * 100:.1f}",
                ha="center",
                va="center",
                color="red",
            )

    ax.set_xlabel("Train on")
    ax.set_ylabel("Test on")

    ax.set_title(stat_name)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im, cax=cax, orientation="vertical")


def sameness_advantage(matr):
    mask_same = np.eye(matr.shape[0], dtype=bool)
    mask_diff = ~mask_same
    return matr[mask_same].mean() - matr[mask_diff].mean()


def evaluate_on_tissue_groups(models, tissue_groups):
    dset_spec = dict(
        type="MultiTissueProbabilitiesH5Dataset",
        path=f"{MarginalPsiV3.data_path_folder}/dataset_test_0.h5",
        post_processor_spec=dict(type="IdentityPostProcessor"),
        datapoint_extractor_spec=dict(
            type="MultipleSettingDatapointExtractor", run_argmax=False
        ),
        iterator_spec=dict(type="FastIter", shuffler_spec=dict(type="DoNotShuffle")),
        tissue_groups=tissue_groups,
        sl=5000,
        cl=400,
        cl_max=10_000,
    )
    results = {}
    for mod in models:
        results[mod.name] = evaluate_on_gtex_data(
            mod.non_binarized_models()[0].model,
            dset_spec,
            num_tissues=len(tissue_groups),
        )
    return results


def flatten_results(results):
    flat_results = {}
    advantages = {}
    for k, res in results.items():
        flat_res = results[k].copy()
        del flat_res["topk"]
        for t in res["topk"]:
            flat_res[f"topk {t:.0%}"] = res["topk"][t]
        flat_res["1-rmse_alternative"] = 1 - flat_res.pop("rmse_alternative")
        flat_results[k] = flat_res
        adv = {k: sameness_advantage(v) for k, v in flat_res.items()}
        advantages[k] = adv
    return flat_results, advantages


def display_mean_results(flat_results):
    for m in flat_results:
        plt.scatter(
            np.arange(len(flat_results[m])),
            [
                100 * np.diag(flat_results[m][k].mean(-1)).mean()
                for k in flat_results[m]
            ],
            label=m,
        )
        plt.xticks(np.arange(len(flat_results[m])), list(flat_results[m]), rotation=90)
    plt.grid()
    plt.ylabel("Value [%]")
    plt.legend()
    plt.show()
