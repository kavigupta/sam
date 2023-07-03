import itertools
from functools import lru_cache

from permacache import permacache, drop_if_equal, stable_hash

import numpy as np
import tqdm.auto as tqdm
from modular_splicing.evaluation.predict_splicing import predict_splicepoints

from modular_splicing.legacy.hash_model import hash_model
from modular_splicing.dataset.h5_dataset import H5Dataset

from modular_splicing.utils.run_batched import run_batched


def standardized_sample(
    path,
    amount,
    cl=50,
    *,
    get_motifs=False,
    datapoint_extractor_spec=dict(type="BasicDatapointExtractor"),
    post_processor_spec=dict(type="IdentityPostProcessor"),
    **kwargs,
):
    """
    Returns a sample of the given size from the dataset at the given path.
    See _standardized_sample_dataset_object for more details on the order.

    If any elements have any positions masked out, they will be removed from
        the sample.

    Parameters:
        path: Path to the dataset.
        amount: Number of elements to sample.
        cl: Context length.
        get_motifs: Whether to return the motifs.
        datapoint_extractor_spec: Specification for the datapoint extractor.
        post_processor_spec: Specification for the post processor.
        kwargs: Additional arguments to pass to the dataset.

    Returns:
        xs: The inputs.
        ys: The outputs.
        motifs: The motifs, if get_motifs is True.
    """
    dset = _standardized_sample_dataset_object(
        path,
        cl,
        datapoint_extractor_spec=datapoint_extractor_spec,
        post_processor_spec=post_processor_spec,
        **kwargs,
    )
    xs = []
    ys = []
    rests = []
    masks = []
    for el in tqdm.tqdm(itertools.islice(dset, amount), total=amount):
        xs.append(el["inputs"]["x"])
        ys.append(el["outputs"]["y"])
        rests.append([el["inputs"].get("motifs", None)])
        masks.append(el["outputs"].get("mask", None))
    xs = np.array(xs)
    ys = np.array(ys)
    if any(mask is not None for mask in masks):
        assert not get_motifs
        masks = np.all(masks, 1)
        xs, ys = xs[masks], ys[masks]
    if get_motifs:
        return xs, ys, rests
    return xs, ys


@lru_cache(None)
def standardized_sample_memcache(*args, **kwargs):
    """
    Like standardized_sample, but caches the result in memory.
    """
    return standardized_sample(*args, **kwargs)


@permacache("eclip/run_eclip_motifs_on_data_2", key_function=dict(cl=drop_if_equal(50)))
def eclip_motifs_on_standardized_sample(
    *, path, amount, eclip_params, eclip_idxs, cl=50
):
    """
    Produce the eclip motifs for the given standardized sample.
    """
    dset = _standardized_sample_dataset_object(
        path,
        cl,
        datapoint_extractor_spec=dict(
            type="BasicDatapointExtractor",
            rewriters=[
                dict(
                    type="AdditionalChannelDataRewriter",
                    out_channel=["inputs", "motifs"],
                    data_provider_spec=dict(type="eclip", params=eclip_params),
                )
            ],
        ),
        post_processor_spec=dict(
            type="FlattenerPostProcessor", indices=[("inputs", "motifs")]
        ),
    )
    all_eclip_motifs = []
    for (m,) in tqdm.tqdm(itertools.islice(dset, amount), total=amount):
        eclip_motifs = m[cl // 2 : -cl // 2, eclip_idxs]
        all_eclip_motifs.append(eclip_motifs)
    return np.array(all_eclip_motifs)


def model_motifs_on_standardized_sample(*, model_for_motifs, indices, path, amount, cl):
    """
    Produce the given model's motifs on the standardied sample.
    """

    def run(x):
        motifs = model_for_motifs(x, only_motifs=True)
        motifs = motifs["post_sparse_motifs_only"]
        motifs = motifs[:, :, indices]
        return motifs

    cl = model_for_motifs.cl if cl is None else cl

    xs, _ = standardized_sample(path, amount, cl)
    return run_batched(run, xs, bs=16, pbar=tqdm.tqdm)


def _standardized_sample_dataset_object(path, cl, sl=None, cl_max=10_000, **kwargs):
    """
    Load the given data using the standardized ordering.

    seeded, with seed=0 in the fully random iteration mode.

    The order will be stable so long as the dataset has the same number of
        chunks and the same length in each chunk (i.e. it will be stable
        across different "interpretations" of the same dataset, such as
        additional inputs (eclip etc) or intron/exon annotations).
    """
    dset = H5Dataset(
        path=path,
        cl=cl,
        cl_max=cl_max,
        sl=sl,
        iterator_spec=dict(
            type="FullyRandomIter", shuffler_spec=dict(type="SeededShuffler", seed=0)
        ),
        **kwargs,
    )

    return dset


@permacache(
    "modular_splicing/data_for_experiments/standardized_sample/calibrate_thresholds_on_standardized_sample",
    key_function=dict(m=stable_hash, bs=None),
)
def calibrate_thresholds_on_standardized_sample(m, *, path, amount, bs=32, **kwargs):
    xs, ys = standardized_sample(path=path, amount=amount, cl=m.cl, **kwargs)
    yps = predict_splicepoints(m, xs, batch_size=bs, pbar=tqdm.tqdm)

    true_fractions = [(ys == c + 1).mean() for c in range(yps.shape[-1])]
    thresholds = [
        np.quantile(yps[:, :, c], 1 - f) for c, f in enumerate(true_fractions)
    ]
    return thresholds
