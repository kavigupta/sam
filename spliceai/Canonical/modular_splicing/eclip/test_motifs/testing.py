import tqdm.auto as tqdm
import numpy as np

import attr
from permacache import permacache, stable_hash, drop_if_equal
from modular_splicing.data_for_experiments.standardized_sample import (
    model_motifs_on_standardized_sample,
)

from .baseline import fm_baseline_nonbinarized
from .calibrate_sparsity import calibrate_sparsities
from ..data.eclips_for_testing import EclipDataForTesting, extract_actual_range
from .names import get_testing_names


@attr.s
class CoverageResult:
    """
    A class to hold coverage results. Consists of three arrays, each keyed by motif.

    covered_fractions: the fraction of the eclip data covered by the motif.
    sparsities: the sparsity of the motif.
    eclip_sparsities: the sparsity of the eclip motif.
    """

    covered_fractions = attr.ib()
    sparsities = attr.ib()
    eclip_sparsities = attr.ib()

    @property
    def mean_coverage(self):
        return {k: v.mean() for k, v in self.covered_fractions.items()}

    @property
    def reals_and_controls(self):
        reals = {k[0]: v for k, v in self.covered_fractions.items() if k[1] == "real"}
        controls = {
            k[0]: v for k, v in self.covered_fractions.items() if k[1] == "control"
        }
        assert set(reals.keys()) == set(controls.keys())
        return reals, controls

    @property
    def mean_enrichment(self):
        reals, controls = self.reals_and_controls
        return {k: reals[k].mean() / controls[k].mean() for k in reals}

    @property
    def enrichment_by_motif(self):
        reals, controls = self.reals_and_controls
        return {k: reals[k] / controls[k] for k in reals}


@attr.s
class SparsityMatchedCoverageResult:
    """
    Represents the results of a sparsity-matched coverage test.

    Attributes:
        signal, control: A dictionary that maps model names to CoverageResult objects.
    """

    signal = attr.ib()
    control = attr.ib()

    @property
    def relative_mean_enrichment(self):
        me_signal = self.signal.mean_enrichment
        me_control = self.control.mean_enrichment
        assert set(me_signal.keys()) == set(me_control.keys())
        return {k: me_signal[k] / me_control[k] - 1 for k in me_signal}

    @property
    def relative_enrichment_by_motif(self):
        me_signal = self.signal.enrichment_by_motif
        me_control = self.control.enrichment_by_motif
        assert set(me_signal.keys()) == set(me_control.keys())
        return {k: me_signal[k] / me_control[k] - 1 for k in me_signal}

    @property
    def intron_exon_controlled_mean_enrichment(self):
        rme = self.relative_mean_enrichment
        return (rme["introns"] + rme["exons"]) / 2


def sparsity_matched_coverage_results(
    *, motif_names_source, test_models, data_amount, cl=None
):
    """
    Collects the sparsity-corrected coverage results for the given models.

    Arguments:
        motif_names_source: A tag representing the motifs to be used. See get_motif_names.
        test_models: A dictionary from model names to model objects.
        data_amount: The amount of data to be used for testing.

    Returns dictionary with keys
        result: A dictionary that maps model names to SparsityMatchedCoverageResult objects.
        names: The names of the motifs used.
    """
    indices_rbns = get_testing_names(motif_names_source=motif_names_source)
    am_entire_rbns = sparsity_matched_coverage_results_for_data(
        control_model=fm_baseline_nonbinarized(motif_names_source),
        test_models=test_models,
        amount=data_amount,
        path="dataset_test_0.h5",
        annotation_path="dataset_intron_exon_annotations_test_0.h5",
        eclip_idxs=indices_rbns.eclip_idxs,
        motif_idxs=indices_rbns.motif_idxs,
        cl=cl,
    )
    return dict(
        out=am_entire_rbns,
        common_names=indices_rbns.common_names,
    )


@permacache(
    "modular_splicing/eclip/test_motifs/testing/sparsity_matched_coverage_results_for_data_5",
    key_function=dict(
        control_model=stable_hash,
        test_models=stable_hash,
        cl=drop_if_equal(None),
    ),
)
def sparsity_matched_coverage_results_for_data(
    control_model,
    test_models,
    amount,
    path,
    annotation_path,
    eclip_idxs,
    motif_idxs,
    cl,
):
    """
    Produce sparsity-corrected coverage results for the given models on the given data

    Arugments:
        control_model: The model to be used as the control/baseline.
        test_models: A dictionary from model names to model objects of the models to be tested.
        amount: The amount of data to be used for testing.
        path: The path to the dataset.
        annotation_path: The path to the intron/exon annotation dataset.
        eclip_idxs: the indices into the eclip motif order to be used.
        motif_idxs: the indices into the model's motif order to be used.
        common_names: the names of the motifs to be used.
        cl: the context length of the models.

    Returns:
        dictionary from model names to SparsityMatchedCoverageResult objects.
    """
    print("Testing sparsity matched coverage results", test_models.keys())
    mode = ("from_5'", 50)

    eclip_d = EclipDataForTesting(
        amount,
        eclip_idxs,
        mode=mode,
        path=path,
        annotation_path=annotation_path,
    )
    control_model_unbinarized = model_motifs_on_standardized_sample(
        model_for_motifs=control_model,
        indices=motif_idxs,
        path=path,
        amount=amount,
        cl=cl,
    )
    test_models_binarized = {
        am_model_name: model_motifs_on_standardized_sample(
            model_for_motifs=test_models[am_model_name],
            indices=motif_idxs,
            path=path,
            amount=amount,
            cl=cl,
        )
        for am_model_name in tqdm.tqdm(test_models)
    }
    am_vs_control = {
        am_model_name: compare_am_to_control(
            eclip_d.filtered_eclips,
            test_models_binarized[am_model_name],
            control_model_unbinarized,
            eclip_d.all_eclip_motifs,
            mode=mode,
        )
        for am_model_name in tqdm.tqdm(test_models_binarized)
    }
    return am_vs_control


def compare_am_to_control(
    filtered_eclips,
    all_am_binarized,
    all_fm_less_sparse,
    all_eclip_motifs,
    *,
    mode,
):
    stats_am = all_coverage_stats(
        filtered_eclips,
        all_am_binarized,
        all_eclip_motifs,
        mode=mode,
    )
    stats_fm_calibrated_to_am = all_coverage_stats(
        filtered_eclips,
        calibrate_sparsities(all_am_binarized, all_fm_less_sparse),
        all_eclip_motifs,
        mode=mode,
    )
    return SparsityMatchedCoverageResult(
        signal=stats_am, control=stats_fm_calibrated_to_am
    )


def all_coverage_stats(
    filtered_eclips,
    motifs_binarized,
    all_eclip_motifs,
    *,
    mode,
):
    """
    Produce all coverage statistics for the given eclip results.

    Arguments:
        filtered_eclips: The eclip results to be used.
        motifs_binarized: The binarized motifs to be used.
        all_eclip_motifs: The eclip motifs to be used (as an array).
        mode: The mode to be used for the coverage statistics.

    Returns:
        a CoverageResult object.
    """
    motifs_binarized = motifs_binarized != 0
    covered_counts = {
        k: eclip_covered_counts(
            filtered_eclips[k],
            motifs_binarized,
            cl=motifs_binarized.shape[1] - all_eclip_motifs.shape[1],
            mode=mode,
        )
        for k in filtered_eclips
    }
    covered_fractions = {
        k: covered_counts[k]["covered_count"] / covered_counts[k]["total_count"]
        for k in covered_counts
    }
    sparsities = motifs_binarized.mean((0, 1))
    eclip_sparsities = all_eclip_motifs.mean((0, 1))[:, 0].sum(-1)
    return CoverageResult(
        covered_fractions=covered_fractions,
        sparsities=sparsities,
        eclip_sparsities=eclip_sparsities,
    )


def eclip_covered_counts(eclips, motif_binary, cl, mode=("range",)):
    """
    Count the number of eclips covered by the given motifs. Done on a per-motif basis.

    Arguments:
        eclips: The eclip results to be used
        motif_binary: The binarized motifs to be used
        cl: The context length of the motifs (effectively their padding)
        mode: The mode to be used for the coverage statistics.

    Returns:
        total_count: the total number of eclips for each motif
        covered_count: the number of eclips covered by each motif
    """
    total_count = np.zeros(motif_binary.shape[2], dtype=np.int)
    covered_count = np.zeros(motif_binary.shape[2], dtype=np.int)
    for eclip in eclips:
        total_count[eclip["motif_idx"]] += 1
        start, end = extract_actual_range(eclip, mode)
        if motif_binary[
            eclip["batch_idx"],
            cl // 2 + start : cl // 2 + end,
            eclip["motif_idx"],
        ].any():
            covered_count[eclip["motif_idx"]] += 1
    return dict(total_count=total_count, covered_count=covered_count)
