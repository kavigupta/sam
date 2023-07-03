import numpy as np

from permacache import permacache, stable_hash, drop_if_equal

from modular_splicing.evaluation.predict_motifs import predict_motifs
from modular_splicing.eclip.test_motifs.calibrate_sparsity import (
    compute_sparsity_thresholds_for_model,
)
from modular_splicing.fit_rbns.rbns_data import RBNSData


class RBNSEvaluator:
    """
    Represents the data, names, thresholds, models, and whether it is a test condition or not,
    for evaluating RBNS models.

    Parameters
    ----------
    fm_model_non_binarized : torch.nn.Module
        The non-binarized model to use as the baseline
    am_models : dict[str, torch.nn.Module]
        The binarized models to use as the adjusted models
    genome_calibration_amount : int
        The amount of data to use for calibrating the sparsity thresholds
    rbnsp_names : dict[str, list[str]]
        The names of the RBNS motifs to use for each AM model
    is_test : bool
        Whether to use the test set or not
    """

    def __init__(
        self,
        fm_model_non_binarized,
        am_models,
        genome_calibration_amount,
        rbnsp_names,
        *,
        is_test
    ):
        self.data = RBNSData()
        self.rbnsp_names = rbnsp_names
        self.names = {
            k: sorted(set(self.rbnsp_names[k]) & set(self.data.names))
            for k in self.rbnsp_names
        }
        self.is_test = is_test

        self.fm_model_non_binarized = fm_model_non_binarized
        self.am_models = am_models

        self.thresholds = {}
        self.sparsities = {}
        for name in self.am_models:
            (
                self.sparsities[name],
                self.thresholds[name],
            ) = compute_sparsity_thresholds_for_model(
                binarized_to_calibrate_to_model=self.am_models[name],
                non_binarized_model=self.fm_model_non_binarized[name],
                motif_indices=list(range(len(self.rbnsp_names[name]))),
                amount=genome_calibration_amount,
                path="dataset_train_all.h5",
                cl=None,
            )

    def calculate_statistics_for_all_motifs(self, model_name):
        return calculate_statistics_for_all_motifs(
            data=self.data,
            motif_names=self.names[model_name],
            motif_names_for_index=self.rbnsp_names[model_name],
            fm_model_non_binarized=self.fm_model_non_binarized[model_name],
            am_model=self.am_models[model_name],
            thresholds=self.thresholds[model_name],
            sparsities=self.sparsities[model_name],
            is_test=self.is_test,
        )

    def rbnsp_index(self, model_name, name):
        return self.rbnsp_names[model_name].index(name)

    def stable_hash(self):
        d = dict(**self.__dict__)
        d["data"] = stable_hash([d["data"].__dict__, type(d["data"]).__name__])
        return stable_hash(d)


@permacache(
    "validation/calculate_statistics_for_all_motifs",
    key_function=dict(
        data=lambda x: stable_hash([x.__dict__, type(x).__name__]),
        motif_names=stable_hash,
        motif_names_for_index=stable_hash,
        kwargs=stable_hash,
        is_test=drop_if_equal(False),
    ),
)
def calculate_statistics_for_all_motifs(
    data, motif_names, motif_names_for_index, *, is_test, **kwargs
):
    """
    Calculate the statistics for all motifs in the RBNS dataset.

    See `calculate_statistics` for the parameters and return type, this
    function just iterates over all motifs.
    """
    return {
        motif_name: calculate_statistics_faster_cache(
            data=data,
            motif_name=motif_name,
            **kwargs,
            is_test=is_test,
            motif_index=motif_names_for_index.index(motif_name),
        )
        for motif_name in motif_names
    }


@permacache(
    "validation/test_on_rbns.cache",
    key_function=dict(
        data=lambda x: stable_hash([x.__dict__, type(x).__name__]),
        motif_name=stable_hash,
        kwargs=stable_hash,
        is_test=drop_if_equal(False),
    ),
)
def calculate_statistics_faster_cache(data, motif_name, is_test, **kwargs):
    return calculate_statistics(datum=data.data(motif_name, is_test=is_test), **kwargs)


def calculate_statistics(
    *, datum, fm_model_non_binarized, motif_index, am_model, thresholds, sparsities
):
    """
    Calculate the statistics for a single motif in the RBNS dataset.

    Parameters
    ----------
    datum : SimpleNamespace
        The datum to compute results on. See `RBNSData.data` for the fields.
    fm_model_non_binarized : torch.nn.Module
        The non-binarized model to use as the baseline
    motif_index : int
        The index of the motif in the model
    am_model : torch.nn.Module
        The binarized model to use as the adjusted model
    thresholds : list[float]
        The thresholds to use for the adjusted model (calibrated on the genome)
    sparsities : list[float]
        The sparsities to use for the adjusted model (calibrated on the genome).
        Just used to return.
    """
    fm_out = predict_motifs(
        fm_model_non_binarized,
        datum.x.astype(np.float32),
        bs=512,
    )[:, :, motif_index].max(-1)

    am_out = (
        predict_motifs(
            am_model,
            datum.x.astype(np.float32),
            bs=512,
        )[:, :, motif_index]
        > 0
    ).any(-1)

    threshold_genome = thresholds[motif_index]
    fm_to_genome = fm_out > threshold_genome

    threshold_overall = np.quantile(fm_out, 1 - (am_out).mean())
    fm_to_rbns = fm_out >= threshold_overall

    return dict(
        sparsity=sparsities[motif_index],
        threshold_genome=threshold_genome,
        threshold_overall=threshold_overall,
        acc_fm_to_genome=np.mean(fm_to_genome == datum.y),
        acc_fm_to_rbns=np.mean(fm_to_rbns == datum.y),
        acc_am=np.mean(am_out == datum.y),
    )
