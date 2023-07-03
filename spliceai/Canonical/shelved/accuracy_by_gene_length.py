import pickle

import pandas as pd
import tqdm.auto as tqdm
import numpy as np
from modular_splicing.models.modules.lssi_in_model import BothLSSIModelsJustSplicepoints
from modular_splicing.models_for_testing.main_models import AM, FM
from shelved.hsmm.gene_dataset import run_on_all_genes_binarized

from modular_splicing.models_for_testing.load_model_for_testing import step_for_density
from modular_splicing.utils.io import load_model
from modular_splicing.gtex_data.experiments.create_dataset import MarginalPsiV3

spliceai_models = [
    (("spliceai-400", "full"), "model/standard-400-1"),
    (("spliceai-10k", "full"), "model/standard-10000-1"),
    # # WBL
    # (("spliceai-400", "wbl"), "model/msp-254a2_1"),
    # (("spliceai-10k", "wbl"), "model/msp-254b2_1"),
    # # TOS
    # (("spliceai-400", "tos"), "model/msp-257d4_1"),
    # (("spliceai-10k", "tos"), "model/msp-257e4_1"),
    # # TOSI
    # (("spliceai-400", "tosi"), "model/msp-257dq5_1"),
    # (("spliceai-10k", "tosi"), "model/msp-257eq5_1"),
    # # BAL
    # (("spliceai-400", "bal"), "model/msp-257dr5_1"),
    # (("spliceai-10k", "bal"), "model/msp-257er5_1"),
]

sparse_models = [
    (("AM", "full"), AM.non_binarized_model(1).path),
    (("FM", "full"), FM.non_binarized_model(1).path),
    # # WBL
    # (("AM", "wbl"), "model/msp-254d1_1"),
    # (("FM", "wbl"), "model/msp-254c1_1"),
    # # TOS
    # (("AM", "tos"), "model/msp-257b4_1"),
    # (("FM", "tos"), "model/msp-257a4_1"),
    # # TOSI
    # (("AM", "tosi"), "model/msp-257bq5_1"),
    # (("FM", "tosi"), "model/msp-257aq5_1"),
    # # BAL
    # (("AM", "bal"), "model/msp-257br5_1"),
    # (("FM", "bal"), "model/msp-257ar5_1"),
]


def load_models():
    model = {}
    model[("lssi", "full")] = (
        BothLSSIModelsJustSplicepoints.from_paths(
            "model/splicepoint-model-acceptor-1", "model/splicepoint-donor2-2.sh", 50
        )
        .cuda()
        .eval()
    )

    for name, path in spliceai_models:
        _, model[name] = load_model(path)

    for name, path in sparse_models:
        _, model[name] = load_model(path, step_for_density(path, 0.178e-2))

    model = {k: model[k].eval() for k in model}

    return model


def gene_length_statistics(ys):
    num_introns = []
    intron_length = []
    intron_length_mean = []
    gene_length = []
    is_first_intron = []

    for y in tqdm.tqdm(ys):
        pos, is_d = np.where(y[:, 1:])
        num_splicepoints = pos.size
        il = pos[1::2] - pos[::2]
        num_introns += [num_splicepoints // 2] * num_splicepoints
        intron_length += [x for l in il for x in [l, l]]
        intron_length_mean += [np.mean(il)] * num_splicepoints
        gene_length += [y.shape[0]] * num_splicepoints
        is_first_intron += [1] + [0] * (num_splicepoints - 1)
    return {
        "Gene Length": np.array(gene_length),
        "# introns": np.array(num_introns),
        "Mean Intron Length": np.array(intron_length_mean),
        "Intron Length": np.array(intron_length),
        "First Intron": np.array(is_first_intron),
    }


def compute_accuracies(model, ys):
    yps = {
        k: run_on_all_genes_binarized(model[k], "datafile_test_0.h5")
        for k in tqdm.tqdm(model)
    }
    is_acc = {}
    ys_flat = np.concatenate(ys)[:, 1:]
    for k in tqdm.tqdm(yps):
        yps_flat = np.concatenate(yps[k])
        is_acc[k] = yps_flat[ys_flat != 0]
    return yps, is_acc


def compute_by_category(statistics):
    categories = {
        "short": statistics["Gene Length"] < 10_000,
        "intermediate": (10_000 <= statistics["Gene Length"])
        & (statistics["Gene Length"] < 100_000),
        "long": 100_000 <= statistics["Gene Length"],
    }

    def collect_statistic(fn):
        result = {k: fn(categories[k]) for k in categories}
        result = {k: result[k] / sum(result.values()) for k in categories}
        return result

    statistics_by_category = pd.DataFrame(
        {
            "Gene %": collect_statistic(
                lambda x: (x * statistics["First Intron"]).sum()
            ),
            "Intron %": collect_statistic(lambda x: x.sum()),
            "Nucleotide %": collect_statistic(
                lambda x: (
                    x * statistics["First Intron"] * statistics["Gene Length"]
                ).sum()
            ),
        }
    )
    return categories, statistics_by_category


def plot_accs(ax, model_order, results, xl, yl, label=None):
    xs, ys = results[xl], results[yl]
    if label is not None:
        label = f"{label}: mean=({np.mean(xs):.2f}%, {np.mean(ys):.2f}%)"
    ax.scatter(xs, ys, marker=".", label=label)
    for i, k in enumerate(model_order):
        ax.text(xs[i], ys[i], s=" ".join(k), size=3)
    a, b = 10, 100
    ax.plot([a, b], [a, b], linestyle="--", color="black")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_xlim(a, b)
    ax.set_ylim(a, b)
    ax.grid("on")


def stats(vals):
    n = len(vals)
    k = np.sum(vals)
    return (
        k / n,
        (k - 1.96 * np.sqrt(k * (n - k) / n)) / n,
        (k + 1.96 * np.sqrt(k * (n - k) / n)) / n,
    )


def plot_by_category(model_order, is_acc, categories, *, ax):
    xs = [0]
    for u, v in zip(model_order, model_order[1:]):
        xs.append(xs[-1] + 1 + 0.5 * (u[0] != v[0]))
    xs = np.array(xs)
    slab = 0.5
    for off, category in zip(
        np.linspace(-slab / 2, slab / 2, len(categories)), categories
    ):
        means, lows, highs = zip(
            *[stats(is_acc[k][categories[category]]) for k in model_order]
        )
        lows = np.array(lows)
        highs = np.array(highs)
        ax.bar(
            off + xs,
            means,
            width=slab / (len(categories) - 1),
            label=category,
        )
        ax.errorbar(
            off + xs,
            means,
            yerr=np.array([means - lows, highs - means]),
            fmt="none",
            ecolor="black",
            capsize=2,
        )
    ax.set_xticks(xs, [" ".join(x) for x in model_order], rotation=90)
    ax.set_ylabel("Accuracy [%]")
    ax.grid(axis="y")
    ax.legend()


def gene_constiutive_map():
    with open(f"{MarginalPsiV3.data_path_folder}/psis.pkl", "rb") as f:
        psis = pickle.load(f)
    table = pd.read_csv(f"{MarginalPsiV3.data_path_folder}/splice_table.csv").fillna("")

    def mean(x):
        if not x.size:
            return 1
        return x.mean()

    constitu_by_site = (psis["psi_values"] == 1).all(-1)
    constitus_by_gene = (table.start_ids + "," + table.end_ids).apply(
        lambda x: constitu_by_site[[int(t) for t in x.split(",") if t]]
    )
    constitus_by_gene = dict(zip(table.name, constitus_by_gene))
    whole_gene_constitu = {k: mean(constitus_by_gene[k]) for k in constitus_by_gene}
    return whole_gene_constitu
