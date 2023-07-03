import numpy as np
import tqdm.auto as tqdm
from permacache import permacache, stable_hash

from .gtex_data_table import exon_junction_data


@permacache(
    "modular_splicing/gtex_data/pipeline/marginal_psis/compute_psis_from_junction_data_2",
    key_function=dict(gene_ensgs=stable_hash),
)
def compute_psis_from_junction_data(
    gene_ensgs, *, tissue_idxs, ensembl_version, denominator_technique="one_step"
):
    """
    Compute the marginal PSIs for each gene in gene_ensgs. Cached

    Parameters
    ----------
    gene_ensgs : list of str
        List of gene ensembl IDs
    tissue_idxs : list of int
        List of tissue indices to use, as returned by `gtex_data_table.FeatureDataBySample.group_by_tissue`
    ensembl_version : int
        Ensembl version to use

    Returns
    -------
    dict
        Dictionary mapping gene ensembl IDs to a dict with the following keys:
        - sites: list of (int, str) tuples, each representing a splice site.
            See `modular_splicing.gtex_data.pipeline.splice_table.splice_sites_junc`
        - scores_by_tissue: np.ndarray of shape (len(sites), len(tissue_idxs))
            Each row is the marginal PSI for the corresponding site in `sites`
        - gene_obj: pyensembl.Gene
            The gene object for the gene with ID `gene_ensg`
    """
    from pyensembl import EnsemblRelease

    data = EnsemblRelease(ensembl_version)
    exons = exon_junction_data()

    result = {}
    for gene_ensg in tqdm.tqdm(gene_ensgs):
        (sites, scores_by_tissue, _) = compute_splice_site_psis(
            exons,
            tissue_idxs,
            gene_ensg,
            smoothing=0.1,
            denominator_technique=denominator_technique,
        )
        gene_obj = data.gene_by_id(gene_ensg.split(".")[0])
        result[gene_ensg] = dict(
            sites=sites,
            scores_by_tissue=scores_by_tissue,
            gene_obj=gene_obj,
        )

    return result


def compute_splice_site_psis(
    exons, idxs_by_tissue, gene_ensg, *, smoothing, denominator_technique
):
    """
    Compute the marginal PSIs for each splice site in a given gene.

    Parameters
    ----------
    exons : FeatureDataBySample
        Exon junction data
    idxs_by_tissue : list of list of int
        List of tissue indices to use, as returned by `gtex_data_table.FeatureDataBySample.group_by_tissue`
    gene_ensg : str
        Gene ensembl ID
    smoothing : float
        Amount of smoothing to apply to the marginal PSIs. Added to the denominator

    Returns
    -------
    splice_sites_all: list of (int, str) tuples
        List of splice sites in the gene
    by_tissue: np.ndarray of shape (len(splice_sites_all), len(idxs_by_tissue))
        Each row is the marginal PSI for the corresponding site in `splice_sites_all`
    denominators: np.ndarray of shape (len(splice_sites_all), len(idxs_by_tissue))
        Each row is the denominator for the corresponding site in `splice_sites_all`
    """
    [junc_idxs] = np.where(np.array(exons.meta_1) == gene_ensg)
    junc_arr = np.array([exons[i] for i in junc_idxs])
    if junc_arr.shape[0] == 0:
        junc_arr = np.zeros((0, len(idxs_by_tissue)))
    junc_arr = np.array([junc_arr[:, idxs].sum(1) for idxs in idxs_by_tissue]).T

    junc_ids = [parse_junc(exons.meta_0[i]) for i in junc_idxs]
    junc_sites = [set(splice_sites_junc(t)) for t in junc_ids]
    splice_sites_all = sorted({s for ss in junc_sites for s in ss})

    by_tissue = []
    denominators = []
    for s in splice_sites_all:
        relevant_junctions = [s in ss for ss in junc_sites]
        current_related_cluster = related_sites(
            s, junc_sites, relatedness_technique=denominator_technique
        )
        related_junctions = [ss & current_related_cluster != set() for ss in junc_sites]
        numerator = junc_arr[relevant_junctions].sum(0)
        denominator = junc_arr[related_junctions].sum(0)
        by_tissue.append(numerator / (denominator + smoothing))
        denominators.append(denominator)
    by_tissue = np.array(by_tissue).T
    denominators = np.array(denominators).T

    return (splice_sites_all, by_tissue, denominators)


def related_sites(site, junc_sites, *, relatedness_technique):
    """
    Compute the set of related sites to a given site, using the given
        technique for determining relatedness

    Parameters
    ----------
    site : X
        The current site to compute related sites for
    junc_sites : list of set of X
        List of sets of sites, each representing a junction
    relatedness_technique : str
        Either
        - "one_step": Related sites are the current site and those that
            share a junction with the current site
        - "closure": Related sites are the current site and those that
            share a junction with the current site or any of the other
            related sites

    Returns
    -------
    set of X
        Related sites
    """
    if relatedness_technique == "one_step":
        return {site} | set.union(*[ss for ss in junc_sites if site in ss])
    elif relatedness_technique == "closure":
        related_sites = {site}
        while True:
            new_related_sites = related_sites | set.union(
                *[ss for ss in junc_sites if ss & related_sites != set()]
            )
            if new_related_sites == related_sites:
                return related_sites
            related_sites = new_related_sites
    else:
        raise ValueError("Unknown relatedness_technique")


def parse_junc(junc_id):
    """
    Parse a junction of the form chr1_100_200 into (100, 200).
        Chromosome is dropped as it is constant for all junctions in a gene

    Parameters
    ----------
    junc_id : str
        Junction ID, as returned by `gtex_data_table.FeatureDataBySample.meta_0`

    Returns
    -------
    (int, int)
        Start and end of the junction
    """
    _, start, end = junc_id.split("_")
    start, end = int(start), int(end)
    # convert to the splice sites by expanding out by 1
    start -= 1
    end += 1
    return start, end


def splice_sites_junc(junc):
    """
    Get the splice sites for a junction

    Parameters
    ----------
    junc : (int, int)
        Start and end of the junction

    Returns
    -------
    generator of (int, str)
        Each element is a splice site, represented as a tuple of (position, which)
        where `which` is either "A" or "D" for the end of an intron and the start
        of an intron, respectively. (note! not acceptor and donor, as the names
        might imply, that only applies to the positive strand)
    """
    start, end = junc
    yield start, "D"
    yield end, "A"
