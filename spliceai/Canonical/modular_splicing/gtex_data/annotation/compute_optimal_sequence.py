import pulp

from permacache import permacache, stable_hash
import tqdm.auto as tqdm

from modular_splicing.gtex_data.pipeline.gtex_data_table import (
    exon_junction_data,
    gene_expr_data,
)

from .annotations import annotation_types
from .extract_sequence_info import extract_sequence_info


def choose_optimal_annotation_sequence(
    annotations, junctions, junction_tpms, site_tpms, *, num_sites, cost_params
):
    """
    Choose the optimal annotation sequence.

    Parameters
    ----------
    annotations : list of Annotations
        The annotations to choose from.
    junctions : list of lists of ints
        Each element is a two element list (start, end),
            representing the start and end of a junction,
            in the index of the sites list.
    junction_tpms : list of floats
        The tpm of each junction in the gene, averaged across all tissues.
    site_tpms : list of floats
        The tpm of each site in the gene, averaged across all tissues.
    num_sites : int
        The number of sites in the gene.
    cost_params : dict
        The parameters for the additional cost function. Contains keys
            "annot_cost" and "other_cost".
    """
    prob = pulp.LpProblem("problem", pulp.LpMinimize)
    var_annotations = pulp.LpVariable.dicts(
        "annotations", (range(len(annotations)),), cat="Binary"
    )
    var_junctions = pulp.LpVariable.dicts(
        "junctions", (range(len(junctions)),), cat="Binary"
    )
    for i in range(len(annotations)):
        for j in range(i + 1, len(annotations)):
            if not annotations[i].compatible_with_annotation(annotations[j]):
                prob += pulp.lpSum([var_annotations[i], var_annotations[j]]) <= 1

    for i in range(len(annotations)):
        for j, junc in enumerate(junctions):
            if not annotations[i].compatible_with_junction(junc):
                prob += pulp.lpSum([var_annotations[i], var_junctions[j]]) <= 1

    for i in range(num_sites):
        prob += (
            pulp.lpSum(
                [var_annotations[v] for v, a in enumerate(annotations) if i in a.sites]
            )
            >= 1
        )
    junction_reward = pulp.lpSum(
        [var_junctions[j] * junction_tpms[j] for j in range(len(junction_tpms))]
    )
    annotation_cost = pulp.lpSum(
        [
            var_annotations[i]
            * annotations[i].cost(cost_params)
            * site_tpms[list(annotations[i].sites)].mean()
            for i in range(len(annotations))
        ]
    )
    prob.objective = -junction_reward + annotation_cost
    pulp.PULP_CBC_CMD(msg=0).solve(prob)

    assert prob.status == 1

    return sorted(
        [
            annotations[i]
            for i in range(len(annotations))
            if pulp.value(var_annotations[i])
        ],
        key=lambda x: min(x.sites),
    )


def all_annotations(sites, usable, skippable):
    """
    Find all possible annotations for the given sites.
    """
    for annotation_type in annotation_types:
        yield from annotation_type.find_patterns(sites, usable, skippable)


def choose_optimal_sequence_for_gene(genes, juncs, gene_ensg, cost_params):
    """
    Compute the optimal sequence for the given gene.

    Parameters
    ----------
    genes : FeatureDataBySample
        The genes to extract information from.
    juncs : FeatureDataBySample
        The junctions to extract information from.
    gene_ensg : str
        The ensembl gene id to extract information for.
    cost_params : dict
        The parameters for the additional cost function. Contains keys
            "annot_cost" and "other_cost".

    Returns
    -------
    intermediates: dict
        Intermediate information for the gene, useful for graphing
        the results.
    psi_calculation_info: dict
        Information about the psi calculation. Contains keys
        - sites_orig : list of tuple of (int, str)
            The original sites in the gene, in the form int(index, site).
        - tpm_junc_each : array of floats
            The tpm of each junction in the gene, averaged across all tissues.
        - index_juncs : list of tuples of (int, int)
            The index of the junctions in the gene.
    annotations_chosen : list of Annotations
        The annotations chosen for the gene.
    """
    (
        sites_orig,
        tpm_junc_each,
        sites,
        psis,
        tpm_density,
        index_juncs,
        tpm_juncs,
    ) = extract_sequence_info(genes, juncs, gene_ensg)
    annotations = list(all_annotations(sites, psis > 0, psis <= 0.5))
    annotations_chosen = choose_optimal_annotation_sequence(
        annotations,
        index_juncs,
        tpm_juncs,
        tpm_density,
        num_sites=len(sites),
        cost_params=cost_params,
    )
    return (
        dict(sites=sites, psis=psis, index_juncs=index_juncs, tpm_juncs=tpm_juncs),
        dict(sites=sites_orig, tpm_junc_each=tpm_junc_each, index_juncs=index_juncs),
        annotations_chosen,
    )


@permacache("modular_splicing/gtex_data/annotation/compute_optimal_sequence_for_gene")
def compute_optimal_sequence_for_gene(gene_ensg, cost_params):
    genes = gene_expr_data()
    junction_data = exon_junction_data()
    return choose_optimal_sequence_for_gene(
        genes, junction_data, gene_ensg, cost_params
    )


@permacache(
    "modular_splicing/gtex_data/annotation/compute_optimal_sequences_all",
    key_function=dict(gene_ensgs=stable_hash),
)
def compute_optimal_sequences_all(gene_ensgs, *, cost_params):
    result = {}
    for gene_ensg in tqdm.tqdm(gene_ensgs):
        result[gene_ensg] = compute_optimal_sequence_for_gene(gene_ensg, cost_params)
    return result
