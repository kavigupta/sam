import numpy as np

from modular_splicing.gtex_data.pipeline.marginal_psis import (
    compute_splice_site_psis,
    parse_junc,
)


def extract_sequence_info(genes, juncs, gene_ensg, denominator_technique="one_step"):
    """
    Extract information about the given gene.

    Parameters
    ----------
    genes : FeatureDataBySample
        The genes to extract information from.
    juncs : FeatureDataBySample
        The junctions to extract information from.
    gene_ensg : str
        The ensembl gene id to extract information for.

    Returns:
    --------
    sites_orig: list of (int, string) tuples
        The sites in the gene, either "A" or "D", in order,
            and their original position in the chromosome order.
    tpm_juncs: array of (J, T)
        The tpm of each junction in the gene, per tissue.
    sites : list of strings
        The sites in the gene, either "A" or "D", in order.
    psis : list of floats
        The psi values for each site in the gene, averaged across all tissues.
    tpm_sites : list of floats
        How much junction incidence occurs at the given site,
            normalized so that the maximum is 1.
    index_juncs: list of lists of ints
        Each element is a two element list (start, end),
            representing the start and end of a junction,
            in the index of the sites list.
    tpm_juncs : list of floats
        The tpm of each junction in the gene, averaged across all tissues.

    """
    sites, psis, tpm_sites = compute_splice_site_psis(
        juncs,
        [list(range(len(genes.tissue_ids)))],
        gene_ensg,
        smoothing=0.1,
        denominator_technique=denominator_technique,
    )
    if psis.size != 0:
        [psis] = psis
    if tpm_sites.size != 0:
        [tpm_sites] = tpm_sites
    [junc_idxs] = np.where(np.array(juncs.meta_1) == gene_ensg)
    span_juncs = [parse_junc(juncs.meta_0[i]) for i in junc_idxs]
    tpm_juncs = np.array([juncs[i] for i in junc_idxs])
    if tpm_juncs.size == 0:
        tpm_juncs = np.zeros((0, len(genes.tissue_ids)))
    tpm_juncs_mean = tpm_juncs.mean(-1)
    if tpm_sites.size and tpm_sites.max() > 0:
        tpm_sites = tpm_sites / tpm_sites.max()
    if tpm_juncs_mean.size and tpm_juncs_mean.max() > 0:
        tpm_juncs_mean /= tpm_juncs_mean.max()
    pos_l = [x for x, _ in sites]
    index_juncs = [[pos_l.index(s), pos_l.index(e)] for s, e in span_juncs]

    assert sites == sorted(sites)
    sites_just_ad = [x for _, x in sites]

    return sites, tpm_juncs, sites_just_ad, psis, tpm_sites, index_juncs, tpm_juncs_mean
