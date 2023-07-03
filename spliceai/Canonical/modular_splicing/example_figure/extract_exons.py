"""
Methods for extracting exons from the test data.
"""
import tqdm.auto as tqdm
import numpy as np

from modular_splicing.data_for_experiments.standardized_sample import (
    standardized_sample,
)
from modular_splicing.evaluation.predict_splicing import predict_splicepoints_cached

from .data import dataset_for_species
from .run_models import compute_thresholds


def get_exons(
    *, flank, cl, lssi_model, species, use_prediction, lssi_filter_multiplier
):
    """
    Get exons from the standardized sample. This is a generator.

    Each exon produced is a dictionary with keys:
        - "x": the sequence
        - "y": the labels
        - "y_to_use": the labels to use in deciding whether to keep
            this exon or not. These are the true labels if use_prediction
            is False, and the predicted labels if use_prediction is True.
        - "s": the start
        - "e": the end
        - "dataset_idx": the index of the dataset that the sequence this exon is contained in came from
        - "startpos": the start position of the exon in the sequence
        - "spl": the LSSI scores for this exon.

    We also return `flank` bases on either side of the exon in `y`, and
    `cl` bases on either side of the exon in `x`.

    The `lssi_filter_multiplier` is applied to the LSSI scores to produce
        less stringent predictions. This enables us to pick up on more false exons.
    """
    _, path, kwargs = dataset_for_species(species)
    xs, ys, idxs = get_data(path=path, cl=cl, **kwargs)
    yps, lssi = predict_lssi(
        lssi_model,
        xs=xs,
        species=species,
        lssi_filter_multiplier=lssi_filter_multiplier,
    )
    assert yps.shape == ys.shape, str((yps.shape, ys.shape))

    for x, y, y_to_use, lss, i in zip(
        xs, ys, (yps if use_prediction else ys), lssi, idxs
    ):
        [splice_idxs] = np.where(y_to_use)
        for start, end in zip(splice_idxs, splice_idxs[1:]):
            if y_to_use[[start, end]].tolist() != [1, 2]:
                continue
            if not ((0 < start - flank) and (end + flank < y.shape[0])):
                continue
            startpos, endpos = start - flank, end + flank
            x_e, y_e, y_to_use_e, lss_e = (
                x[startpos : endpos + cl],
                y[startpos:endpos],
                y_to_use[startpos:endpos],
                lss[startpos:endpos],
            )
            start, end = start - startpos, end - startpos
            yield dict(
                x=x_e,
                y=y_e,
                y_to_use=y_to_use_e,
                s=start,
                e=end,
                startpos=startpos,
                dataset_idx=i,
                spl=lss_e,
            )


def get_data(*, path, cl, **kwargs):
    """
    Get the data from the standardized sample.

    Returns
    -------
    xs : (N, L + cl, 4)
        The sequences
    ys : (N, L)
        The labels
    idxs : (N,)
        The indices of the datasets that the sequences came from
    """
    # None means read all the data
    xs, ys, idxs = standardized_sample(
        path,
        None,
        cl=cl,
        sl=5000,
        get_motifs=True,
        datapoint_extractor_spec=dict(
            type="BasicDatapointExtractor",
            rewriters=[
                dict(
                    type="AdditionalChannelDataRewriter",
                    out_channel=["inputs", "motifs"],
                    data_provider_spec=dict(type="index_tracking"),
                ),
            ],
        ),
        **kwargs,
    )
    assert all(len(idx) == 1 for idx in idxs)
    idxs = [idx[0] for idx in idxs]
    assert all((idx == idx[0]).all() for idx in idxs)
    idxs = [idx[0] for idx in idxs]
    return xs, ys, idxs


def predict_lssi(lssi_model, *, xs, species, lssi_filter_multiplier):
    """
    Predict the LSSI scores for the standardized sample.

    Also produce a binarized version of the LSSI scores, where the
    threshold is calibrated on the standardized sample, and then multiplied
    by `lssi_filter_multiplier`. This multiplier is performed in the logit
    space.
    """
    lssi_thresh = compute_thresholds(lssi_model, species)

    lssi = np.log(
        predict_splicepoints_cached(lssi_model, xs, batch_size=128, pbar=tqdm.tqdm)
    )

    yps = lssi > np.log(lssi_thresh) * lssi_filter_multiplier
    yps = yps.any(-1) * (1 + yps.argmax(-1))
    return yps, lssi
