import itertools
from collections import defaultdict
from types import SimpleNamespace
from matplotlib import pyplot as plt

import tqdm.auto as tqdm

import numpy as np
from modular_splicing.dataset.h5_dataset import H5Dataset
from modular_splicing.evaluation.predict_splicing import predict_splicepoints_cached
from shelved.plot_base_importance import plot_base_importance, Motif

from .evaluation import evaluation_data_spec

model_specs_no_target = {
    "just-spicepoints": "msp-238a3_1",
    "spliceai": "msp-245a1_1",
    "21x2+T100": "msp-233baz2_1",
    "13x4+T100": "msp-233bbz2_1",
    "output_21x2+T100": "msp-234baz2_1",
    "output_13x4+T100": "msp-234bbz2_1",
}

model_specs_target = {
    "just-spicepoints": "msp-238a3_1",
    "21x2": "msp-233baq2_1",
    "13x4": "msp-233bbq2_1",
    "output_21x2": "msp-234baq2_1",
    "output_13x4": "msp-234bbq2_1",
}


def cl_for_model(name):
    if name == "spliceai":
        return 10_000
    return 400


def branchpoint_datapoint_extractor_spec(
    *, modify_output, require_single_branch_site=False
):
    return evaluation_data_spec(
        provide_input=True,
        require_single_branch_site=require_single_branch_site,
        modify_output=modify_output,
    )["datapoint_extractor_spec"]


def load_data(models, *, cl, amount):
    model_names = list(models)

    dset = H5Dataset(
        path="dataset_train_all.h5",
        sl=1000,
        cl=cl,
        cl_max=10_000,
        iterator_spec=dict(
            type="FullyRandomIter", shuffler_spec=dict(type="SeededShuffler", seed=0)
        ),
        datapoint_extractor_spec=branchpoint_datapoint_extractor_spec(
            modify_output=False
        ),
        post_processor_spec=dict(type="IdentityPostProcessor"),
    )
    data = [
        datum
        for datum in list(tqdm.tqdm(itertools.islice(dset, amount), total=amount))
        if datum["outputs"]["mask"].all()
    ]

    xs = np.array([datum["inputs"]["x"] for datum in data])
    motifs = np.array([datum["inputs"]["motifs"] for datum in data])
    ys = np.array([datum["outputs"]["y"] for datum in data])

    out = {
        k: predict_splicepoints_cached(
            models[k], dict(x=xs, motifs=motifs), 16, pbar=tqdm.tqdm
        )
        for k in models
    }

    branch_site_pred = [out[k][:, :, 2:] for k in model_names]
    branch_site_pred = [
        (x > np.quantile(x, 1 - motifs.mean())).squeeze(-1) if x.size > 0 else None
        for x in branch_site_pred
    ]
    out = {k: out[k][:, :, :2] for k in models}

    thresholds = {
        k: np.percentile(out[k], 100 * (1 - ys.mean()), axis=(0, 1)) for k in out
    }
    yps_stack = np.array([out[k] > thresholds[k] for k in model_names])
    exons = exons_matching(np.eye(3)[ys][:, :, 1:])
    return SimpleNamespace(
        model_names=model_names,
        xs=xs,
        motifs=motifs,
        ys=ys,
        yps_stack=yps_stack,
        branch_site_pred=branch_site_pred,
        exons=exons,
    )


def get_features(ys_i, yps_i, motifs_i, branch_sites):
    feats = defaultdict(list)
    for position, feature in zip(*np.where(yps_i.any(0))):
        actual, pred = ys_i[position] == feature + 1, yps_i[:, position, feature]
        feats[position].append(create_motif(feature, actual, pred))

    branch_indices = sorted(
        {
            y
            for x in [motifs_i.squeeze(-1), *branch_sites]
            if x is not None
            for y in np.where(x)[0]
        }
    )
    for position in branch_indices:
        feats[position].append(
            create_motif(
                2,
                motifs_i[position, 0] != 0,
                [None if x is None else x[position] for x in branch_sites],
            )
        )
    return feats


def create_motif(feature, actual, pred):
    name = (
        {True: "T", False: "F"}[actual]
        + {0: "A", 1: "D", 2: "B"}[feature]
        + "".join({True: "1", False: "0", None: "_"}[p] for p in pred)
    )
    motif = Motif(
        name,
        {0: "blue", 1: "red", 2: "#080"}[feature],
        {True: {}, False: dict(linestyle="--")}[actual],
    )

    return motif


def draw_comparison_example(exon, data, *, cl, window=50, chunk_size=100):
    i, original_start, original_end = exon
    start, end = (
        original_start // chunk_size * chunk_size,
        (original_end + chunk_size - 1) // chunk_size * chunk_size,
    )
    if original_start - start < window:
        start -= chunk_size
    if end - original_end < window:
        end += chunk_size
    start = max(0, start)
    end = min(end, data.ys.shape[1])
    feats = get_features(
        data.ys[i, start:end],
        data.yps_stack[:, i, start:end],
        data.motifs[i, start + cl // 2 : end + cl // 2],
        branch_sites=[
            x[i, start:end] if x is not None else None for x in data.branch_site_pred
        ],
    )
    plot_base_importance(
        data.xs[i, start + cl // 2 : end + cl // 2],
        np.zeros(end - start),
        features=feats,
        chunk_size=chunk_size,
    )
    plt.suptitle(", ".join(data.model_names))


def exons_matching(mask):
    exons = []
    for i in range(mask.shape[0]):
        locs, acc_don = np.where(mask[i])
        if len(acc_don) < 2:
            continue
        if acc_don[:2].tolist() != [0, 1]:
            continue
        exons.append((i, *locs[:2]))
    return exons
