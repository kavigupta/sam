from permacache import permacache, stable_hash

import tqdm.auto as tqdm

from modular_splicing.data_for_experiments.standardized_sample import (
    standardized_sample,
)
from modular_splicing.module_substitution.evaluate_combinations import combine_models
from modular_splicing.models_for_testing.list import FM

from .motif_statistics import motif_statistics
from .compute_perturbations import motif_perturbations_individual


@permacache(
    "modular_splicing/motif_perturbations/perturbations_on_standardized_sample/motif_perturbations_individual_on_standardized_sample",
    key_function=dict(
        m=stable_hash,
        bs=None,
    ),
    multiprocess_safe=True,
)
def motif_perturbations_individual_on_standardized_sample(
    m,
    bs=32,
    value_range=10_000,
    is_binary=False,
    amount=5000,
    *,
    path,
    sl,
    datapoint_extractor_spec=dict(type="BasicDatapointExtractor"),
    num_output_channels=3,
):
    """
    Run several motif perturbations on a standardized sample.

    Parameters
    ----------
    m : torch.nn.Module
        The model to use.
    bs : int
        The batch size to use. Should not affect the results.
    value_range : int
        The value range to use. Pick a large value to cover the range.
    is_binary : bool
        Whether to use thresholds or not. Set to True to treat the motif model as binary.
    amount : int
        The number of samples to draw.
    path : str
        The path to use for the data.
    sl : int
        The sequence length to use.
    datapoint_extractor_spec : dict
        The datapoint extractor to use.
    num_output_channels : int
        The number of output channels to use.

    Returns
    -------
    a list of outputs as in `motif_perturbations_individual`.
    """
    if not is_binary:
        threshold_info = motif_statistics(
            m,
            count=2000,
            bs=bs // 4,
            pbar=tqdm.tqdm,
            min_value=-value_range,
            max_value=value_range,
        )
    else:
        threshold_info = None
    xs, ys = standardized_sample(
        path,
        cl=m.cl,
        amount=amount,
        sl=sl,
        datapoint_extractor_spec=datapoint_extractor_spec,
    )
    return [
        motif_perturbations_individual(
            m,
            x,
            y,
            threshold_info,
            pbar=lambda x: x,
            include_threshold=0.001,
            bs=bs,
            num_output_channels=num_output_channels,
        )
        for x, y in tqdm.tqdm(list(zip(xs, ys)), desc="samples")
    ]


def all_mpi_on_standardized_sample(models, *, always_use_fm, is_binary, **kwargs):
    """
    Run motif_perturbations_individual_on_standardized_sample on all models.

    If `always_use_fm` is True, then use the FM model instead of the original model
        for the downstream.
    """

    def full_model(m):
        if always_use_fm:
            return combine_models(m.model, FM.binarized_model(1).model)
        return m.model

    perturbations = {
        m.name: motif_perturbations_individual_on_standardized_sample(
            full_model(m),
            path="dataset_train_all.h5",
            is_binary=is_binary,
            value_range=1000,
            **kwargs,
        )
        for m in tqdm.tqdm(models, desc="models")
    }

    return perturbations
