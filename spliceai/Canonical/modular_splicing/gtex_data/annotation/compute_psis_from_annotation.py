import numpy as np
import tqdm.auto as tqdm

from permacache import permacache, stable_hash

from modular_splicing.gtex_data.annotation.compute_optimal_sequence import (
    compute_optimal_sequences_all,
)


def compute_psis_from_annotation(sites, index_juncs, tpm_junc_each, annotations_chosen):
    psis = np.zeros((len(sites), tpm_junc_each.shape[1]))
    for annot in annotations_chosen:
        psis[list(annot.sites)] = annot.compute_psis(index_juncs, tpm_junc_each)
    return psis


@permacache(
    "modular_splicing/gtex_data/annotation/statistics/compute_psis_all_3",
    key_function=dict(gene_ensgs=stable_hash),
)
def compute_psis_all(gene_ensgs, cost_params, tissue_idxs, *, ensembl_version):
    from pyensembl import EnsemblRelease

    data = EnsemblRelease(ensembl_version)
    seq_all = compute_optimal_sequences_all(gene_ensgs, cost_params=cost_params)
    psis = {}
    for k in tqdm.tqdm(seq_all):
        intermediates, psi_calculation_info, annotations_chosen = seq_all[k]
        tpm_junc_each = np.array(
            [psi_calculation_info["tpm_junc_each"][:, i].sum(1) for i in tissue_idxs]
        ).T
        psi_for_gene = compute_psis_from_annotation(
            intermediates["sites"],
            psi_calculation_info["index_juncs"],
            tpm_junc_each,
            annotations_chosen,
        )
        psis[k] = dict(
            sites=psi_calculation_info["sites"],
            scores_by_tissue=psi_for_gene.T,
            gene_obj=data.gene_by_id(k.split(".")[0]),
        )

    return psis
