from collections import Counter
from functools import lru_cache, reduce
import itertools
import attr
from cached_property import cached_property
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from permacache import permacache, stable_hash
import torch
import tqdm.auto as tqdm
from sklearn.linear_model import LogisticRegression

from modular_splicing.data_for_experiments.standardized_sample import (
    standardized_sample_memcache,
    model_motifs_on_standardized_sample,
)
from modular_splicing.dataset.h5_dataset import H5Dataset
from modular_splicing.utils.arrays import Sparse
from modular_splicing.utils.plots.plot_psam import render_psam
from modular_splicing.utils.run_batched import run_batched
from modular_splicing.models.modules.lssi_in_model import LSSI_MODEL_THRESH
from modular_splicing.utils.sequence_utils import draw_bases


@permacache(
    "modular_splicing/other_experiments/motif_positioning/donorlike_motifs_3",
    key_function=dict(spm=stable_hash, model=stable_hash),
)
def donorlike_motifs(spm, model, *, path, amount, cl, donorlike_motif_idxs):
    mots = model_motifs_on_standardized_sample(
        model_for_motifs=model,
        indices=donorlike_motif_idxs,
        path=path,
        amount=amount,
        cl=cl,
    )
    xs, ys = standardized_sample_memcache(path, amount, cl=cl)
    spm_results = run_batched(
        lambda x: spm(x).log_softmax(-1)[:, :, 2],
        xs.astype(np.float32),
        128,
        pbar=tqdm.tqdm,
    )
    spm_results = spm_results - LSSI_MODEL_THRESH
    spm_results[spm_results <= 0] = 0
    spm_count = (spm_results != 0).mean()
    threshold = np.quantile(spm_results, 1 - spm_count)
    mots = mots - threshold
    mots[mots <= 0] = 0
    return xs, ys, mots, spm_results


def patches(xs, mots, cl, window):
    """
    Collect patches of size window from xs, using mots as a mask representing the locations of the centers.

    Do not include patches that are within cl of the start or end of the sequence.

    Parameters
        xs: (N, L)
        mots: (N, L)
        cl: context length of the original model
        window: size of the patch

    Returns
        (*, window)
    """
    assert window % 2 == 1
    mots = mots.copy()
    mots[:, : cl // 2] = 0
    mots[:, -cl // 2 :] = 0
    result = []
    for b, s in zip(*np.where(mots)):
        result.append(xs[b, s - window // 2 : s + 1 + window // 2])
    return np.array(result)


@attr.s
class DonorlikeResult:
    xs = attr.ib()
    ys = attr.ib()
    motif_values = attr.ib()
    splicepoint_values = attr.ib()
    cl = attr.ib()

    @classmethod
    def mean(cls, results):
        return cls(
            xs=results[0].xs,
            ys=results[0].ys,
            motif_values=np.mean([r.motif_values for r in results], 0),
            splicepoint_values=np.mean([r.splicepoint_values for r in results], 0),
            cl=results[0].cl,
        )

    def masks(self, patch_top_amount):
        mots, spmr = self.motif_values, self.splicepoint_values
        mots = mots[:, :, 0]
        mots_amount = (mots != 0).mean()
        spl_amount = (spmr != 0).mean()
        amount = mots_amount * patch_top_amount
        bar = np.quantile(spmr, 1 - amount)
        spmr = spmr > bar
        bar = np.quantile(mots, 1 - amount)
        mots = mots > bar
        amounts = dict(mots=mots_amount, spl=spl_amount, amount=amount)
        return amounts, mots, spmr

    def patches(self, patch_top_amount):
        amount, mots, spls = self.masks(patch_top_amount)
        mot = patches(self.xs, mots, self.cl, 21)
        spl = patches(self.xs, spls, self.cl, 21)
        return DonorlikeResultsPatches(amount, mot, spl)


@attr.s
class DonorlikeResultsPatches:
    amount = attr.ib()
    mot = attr.ib()
    spl = attr.ib()

    @property
    def patches(self):
        return self.mot, self.spl

    @cached_property
    def psams(self):
        return [x.mean(0) for x in self.patches]

    @cached_property
    def contrastive_logistic(self):
        a, b = self.patches
        train_accuracy, parameters = logregress(a, b)
        return parameters, train_accuracy

    def core_difference(self, first):
        a, b = self.patches
        a, b = core_counts(a, first), core_counts(b, first)
        return {k: a[k] - b[k] for k in a}


def core_counts(patches, start):
    patches = patches[:, [start, start + 1]]
    patches = patches.argmax(-1)
    patches = patches[:, 0] * 4 + patches[:, 1]
    bins = np.bincount(patches, minlength=16)
    drawn = draw_bases(np.array([np.arange(16) // 4, np.arange(16) % 4]).T)
    bins = bins / bins.sum()
    return dict(zip(drawn, bins))


@permacache(
    "working/adjusted_donor/logregress",
    key_function=dict(a=stable_hash, b=stable_hash),
)
def logregress(a, b):
    xs = np.concatenate([a, b])
    ys = np.concatenate(
        [np.ones(a.shape[0], dtype=bool), np.zeros(b.shape[0], dtype=bool)]
    )
    model = LogisticRegression(random_state=0, solver="liblinear").fit(
        xs.reshape(xs.shape[0], -1), ys
    )
    train_accuracy = (model.predict(xs.reshape(xs.shape[0], -1)) == ys).mean()
    parameters = model.coef_.reshape(xs.shape[1:])
    parameters = parameters - parameters.mean(-1)[:, None]
    return train_accuracy, parameters


@permacache(
    "working/adjusted_donor/compute_donor_adjustments_from_models_3",
    key_function=dict(models=stable_hash, spm=stable_hash),
)
def compute_donor_adjustments_from_models(
    seeds, models, spm, path, amount, cl, patch_top_amount
):
    results = []
    names = []
    for seed, mod in zip(seeds, models):
        res = DonorlikeResult(
            *donorlike_motifs(
                spm,
                mod,
                path=path,
                amount=amount,
                cl=cl,
                donorlike_motif_idxs=[0],
            ),
            cl,
        )
        results.append(res)
        names.append(f"seed={seed}")
    results += [DonorlikeResult.mean(results)]
    names.append("mean")
    results = [x.patches(patch_top_amount) for x in results]
    return results, names


def compute_donor_adjustments(series, spm, path, amount, cl, patch_top_amount):
    models = series.non_binarized_models()
    return compute_donor_adjustments_from_models(
        [x.seed for x in models],
        [x.model for x in models],
        spm,
        path,
        amount,
        cl,
        patch_top_amount,
    )


def render_donor_adjustment_psams(results, names):
    _, axs = plt.subplots(3, len(results), figsize=(15, 6), tight_layout=True)
    for ax in axs.flatten():
        ax.set_axis_off()

    for i, res in enumerate(results):
        m, s = res.psams
        c, t = res.contrastive_logistic
        render_psam(m, ax=axs[0, i], psam_mode="info")
        render_psam(s, ax=axs[1, i], psam_mode="info")
        render_psam(c, ax=axs[2, i], psam_mode="raw")
        axs[0, i].set_title(f"{names[i]}: Adj Don")
        axs[1, i].set_title("LSSI")
        axs[2, i].set_title(f"Classifier [{t:.2%}]")


def render_donor_adjustment_core_dimer(results, names):
    overall = pd.DataFrame(
        {n: res.core_difference(11) for n, res in zip(names, results)}
    )
    overall = overall[~(overall == 0).T.all()]
    for k in overall:
        plt.scatter(np.arange(len(overall[k])), 100 * overall[k], label=k)
    plt.xticks(np.arange(len(overall[k])), list(overall[k].index))
    plt.legend()
    plt.axhline(0, color="black")
    plt.ylabel("Adjusted AM % - Real AM %")
    plt.show()


@lru_cache(None)
def data():
    data = H5Dataset(
        path="dataset_test_0.h5",
        cl=400,
        cl_max=10_000,
        sl=5000,
        iterator_spec=dict(type="FastIter", shuffler_spec=dict(type="DoNotShuffle")),
        datapoint_extractor_spec=dict(type="BasicDatapointExtractor"),
        post_processor_spec=dict(type="IdentityPostProcessor"),
    )
    xs, ys = [], []
    for it in tqdm.tqdm(data, desc="Loading data"):
        x = it["inputs"]["x"]
        y = it["outputs"]["y"]
        xs.append(x)
        ys.append(y)
        if len(xs) > len(data) / 2:
            break
    return np.array(xs), np.array(ys)


@permacache(
    "working/adjusted_donor/run_tandem",
    key_function=dict(mod=stable_hash),
)
def run_tandem_on_setting(mod, setting):
    xs, _ = data()
    return run_batched(
        lambda x: mod(
            dict(x=x, setting=torch.zeros(x.shape, dtype=torch.long) + setting)
        ),
        xs,
        32,
        tqdm.tqdm,
    )


@permacache(
    "working/adjusted_donor/binary_yes_no",
    key_function=dict(mod=stable_hash),
)
def binary_yes_no(mod):
    _, y = data()
    yps_no = run_tandem_on_setting(mod, 0)
    yps_no = np.exp(yps_no)
    yps_no /= yps_no.sum(-1)[:, :, None]
    yps_yes = run_tandem_on_setting(mod, 1)
    yps_yes = np.exp(yps_yes)
    yps_yes /= yps_yes.sum(-1)[:, :, None]

    yps_no = binary_for_channel(yps_no, y, 2)
    yps_yes = binary_for_channel(yps_yes, y, 2)
    return Sparse.of(yps_no), Sparse.of(yps_yes)


def binary_for_channel(yps, ys, c):
    thresh = np.quantile(yps[:, :, c], 1 - (ys == c).mean())
    return yps[:, :, c] > thresh


def tuplize(mask):
    return set(zip(*mask.where))


@permacache(
    "working/adjusted_donor/correctness_overlaps",
    key_function=dict(mod=stable_hash),
)
def correctness_overlaps(mod):
    _, y = data()
    yps_no, yps_yes = binary_yes_no(mod)
    arrays = tuplize(Sparse.of(y == 2)), tuplize(yps_no), tuplize(yps_yes)
    universe = reduce(set.union, arrays)
    result = {}
    for code in itertools.product((0, 1), repeat=3):
        result[code] = {
            x for x in universe if all((x in arrays[i]) == code[i] for i in range(3))
        }

    result = {
        "".join(str(t) for t in x): len(result[x]) for x in result if x != (0, 0, 0)
    }
    return result
