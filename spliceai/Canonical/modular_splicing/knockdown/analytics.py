# TODO rename this file to something more descriptive

import tqdm.auto as tqdm
import attr

from permacache import permacache, stable_hash

import numpy as np
from modular_splicing.utils.construct import construct

from modular_splicing.knockdown.compute_knockdown import (
    compute_with_and_without_knockouts,
)
from modular_splicing.knockdown.experimental_setting import experimental_settings
from modular_splicing.knockdown.compute_in_silico_delta_psi import annotator_types

from modular_splicing.knockdown.utils import render_cell_line, render_cell_lines
from modular_splicing.knockdown.pipeline import dataset_for_cell_line

from modular_splicing.motif_names import get_motif_names

from .accuracy import ResultOfAnalyses


def knockdown_analysis(models, cell_lines, annotation_spec, analyses, setting="SE"):
    """
    Run knockdown analysis for all models given.

    Results are summed across cell lines.

    Args:
        models: dict[model name -> ([model (ensemble)], model names source)]
        cell_lines: All cell lines to include as data sources.
        annotation_spec: The specification for how to annotate exons
        analyses: list of Analysis objects to runs
        setting: The experimental setting to use. One of "SE", "A3SS", "A5SS".

    Returns:
        dict[model name -> dict[analysis name -> ResultOfAnalyses]]
    """
    return {
        name: knockdown_analysis_for_model(
            ms=models[name][0],
            setting=setting,
            motif_names_source=models[name][1],
            cell_lines=list(cell_lines),
            annotation_spec=annotation_spec,
            analyses=analyses,
        )
        for name in tqdm.tqdm(models, desc=f"model: {render_cell_lines(cell_lines)}")
    }


def knockdown_analysis_for_model(
    *,
    ms,
    setting,
    motif_names_source,
    cell_lines,
    annotation_spec,
    analyses,
):
    """
    Run the analyses for the given model (ensemble).

    See knockdown_analysis for argument descriptions. Only difference is that
        this function operates on a single model (ensemble).

    Returns:
        dict[analysis name -> ResultOfAnalyses]
    """
    by_cell_line = [
        knockdown_analysis_for_model_and_cell_line(
            ms=ms,
            setting=setting,
            motif_names_source=motif_names_source,
            cell_line=cell_line,
            analyses=analyses,
            annotation_spec=annotation_spec,
        )
        for cell_line in cell_lines
    ]

    all_motifs = sorted({motif for motifs in by_cell_line for motif in motifs})

    return {
        analysis: ResultOfAnalyses(
            {
                motif: sum(by[motif][analysis] for by in by_cell_line if motif in by)
                for motif in all_motifs
            }
        )
        for analysis in analyses
    }


@permacache(
    "modular_splicing/knockdown/analytics/knockdown_analysis_for_model_and_cell_line",
    key_function=dict(ms=stable_hash),
)
def knockdown_analysis_for_model_and_cell_line(
    *, ms, setting, motif_names_source, cell_line, analyses, annotation_spec
):
    """
    Run the analyses for the given model (ensemble) and cell line.

    Like knockdown_analysis_for_model, but only for a single cell line.

    # Cached at this point so that the dataset does not need to be loaded when
        accessing these results in future
    """
    dset = dataset_for_cell_line(cell_line=cell_line)
    motifs = sorted(
        set(dset.index) & set(get_motif_names(motif_names_source=motif_names_source))
    )
    return {
        motif: run_analyses_for_model_cell_line_and_motif(
            ms=ms,
            cell_line=cell_line,
            setting=setting,
            motif=motif,
            motif_names_source=motif_names_source,
            analyses=analyses,
            annotation_spec=annotation_spec,
        )
        for motif in tqdm.tqdm(
            motifs,
            desc="knockdown_analysis_for_model_and_cell_line: "
            + f"{render_cell_line(cell_line)} {setting}",
        )
    }


def run_analyses_for_model_cell_line_and_motif(
    *, ms, cell_line, setting, motif, motif_names_source, analyses, annotation_spec
):
    """
    Run the analyses for the given model (ensemble), cell line, and motif.

    Like knockdown_analysis_for_model_and_cell_line, but only for a single motif.
    """
    at = annotate_table_with_results(
        ms=ms,
        cell_line=cell_line,
        setting=setting,
        motif_names_source=motif_names_source,
        motif=motif,
        annotation_spec=annotation_spec,
    )
    analyses = {analysis: analysis.analyze(at) for analysis in analyses}
    return analyses


def annotate_table_with_results(
    *, ms, cell_line, setting, motif_names_source, motif, annotation_spec
):
    """
    Annotate the given table with the in silico knockdown statistic, along with
        the minimum count column. The in silico knockdown statistic will be labeled
        `in_siilco_knockdown_stat`, and the minimum count column will be labeled
        `min_count`.
    """
    dset = dataset_for_cell_line(cell_line=cell_line)
    table = dset.loc[motif][setting]

    motif_idx = get_motif_names(motif_names_source=motif_names_source).index(motif)
    results = [
        compute_with_and_without_knockouts(
            experimental_settings[setting],
            m,
            table,
            "datafile_train_all.h5",
            motif_idx,
        )
        for m in ms
    ]
    indices = [result["indices"] for result in results]
    assert all(indices[0] == i for i in indices)
    indices = indices[0]

    annotator = construct(
        annotator_types(),
        annotation_spec,
    )

    res = annotator.annotate(experimental_settings[setting], results)

    assert res["mask"].dtype == np.bool

    annotated_table = table.loc[indices].copy()
    annotated_table["in_silico_knockdown_stat"] = res["stat"]
    annotated_table = annotated_table[res["mask"]]
    add_min_count_column(annotated_table)
    return annotated_table


def add_min_count_column(at):
    """
    Take the minimum count across the two conditions for each row in the table.

    Specifically, we sum across the two replicates, and then take the minimum
        of the *_SAMPLE_1 and *_SAMPLE_2 sums.
    """
    if "min_count" in at:
        return
    counts = at[[x for x in at if "JC_SAMPLE_" in x]].applymap(
        lambda x: sum(int(t) for t in x.split(","))
    )
    min_count = np.minimum(
        (counts.IJC_SAMPLE_1 + counts.SJC_SAMPLE_1),
        (counts.IJC_SAMPLE_2 + counts.SJC_SAMPLE_2),
    )
    at["min_count"] = min_count


def counts_matrix(x, y):
    """
    Produce a matrix of counts for the given binary vectors.

    The matrix is of shape (2, 2), where the first index is the value of x,
        and the second index is the value of y. The value at the matrix is
        the number of times that x and y were both 0 or both 1.
    """
    assert x.dtype == y.dtype == np.bool
    return np.array([[((x == i) & (y == j)).sum() for j in range(2)] for i in range(2)])


def directional_accuracy(annotated_table):
    """
    Compute the directional accuracy of the in silico knockdown statistic.
    """
    x, y = np.array(annotated_table.in_silico_knockdown_stat), np.array(
        annotated_table.IncLevelDifference
    )
    return counts_matrix(x > 0, y > 0)


def non_directional_accuracy(annotated_table):
    """
    Compute the non-directional accuracy of the in silico knockdown statistic.
    """
    x, y = np.abs(np.array(annotated_table.in_silico_knockdown_stat)), np.abs(
        np.array(annotated_table.IncLevelDifference)
    )

    return counts_matrix(
        (np.abs(x) > np.median(np.abs(x))), (np.abs(y) > np.median(np.abs(y)))
    )


@attr.s(hash=True)
class Analysis:
    """
    Class representing an analysis to be run on the in silico knockdown statistic.

    The analysis is run by calling the `analyze` method, which takes a table
        annotated with the in silico knockdown statistic, and returns a counts
        matrix.

    Parameters:
        is_directional: Whether the analysis uses directional accuracy or not.
        filter: the filter to use in performing the analysis. See `analyze` for
            a list of valid filters.
    """

    is_directional = attr.ib()
    filter = attr.ib()

    @property
    def name(self):
        return f"{'directional' if self.is_directional else 'non_directional'} [{self.filter}]"

    def analyze(self, at):
        metric = {
            True: directional_accuracy,
            False: non_directional_accuracy,
        }[self.is_directional]
        filter = {
            "FDR < 0.05": lambda x: x.FDR < 0.05,
            "FDR < 0.25": lambda x: x.FDR < 0.25,
            "min_count > 50": lambda x: x.min_count > 50,
            "|delta_psi| > 0.1": lambda x: np.abs(x.IncLevelDifference) > 0.1,
            "|delta_psi| > 0.25": lambda x: np.abs(x.IncLevelDifference) > 0.25,
        }[self.filter](at)
        return metric(at[filter])
