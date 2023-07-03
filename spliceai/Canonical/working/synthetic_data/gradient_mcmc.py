import numpy as np


def select_independent_mutations(
    saliency_map,
    mech,
    codon_table,
    rna,
    motifs_score,
    *,
    rng,
    temperature,
    negativity_bias
):
    muts, map_back = saliency_map.sample_nonconflicting_mutations(
        rna, codon_table, motif_width=mech.motif_width(), rng=rng
    )
    rna_new = rna.copy()
    for mut in muts:
        mut.perform(rna_new)
    motifs_new_score = mech.predict_motifs(rna_new, cut_off=False)
    dmotifs = motifs_new_score - motifs_score
    dreward_dmotifs = dmotifs * saliency_map.saliency
    dreward_dmotifs[:, :2] = 0
    dreward_dmut = np.zeros(len(muts) + 1, dtype=np.float64)
    np.add.at(dreward_dmut, map_back, dreward_dmotifs.sum(-1))
    dreward_dmut -= negativity_bias * dreward_dmut.max()
    # remove the last element, which represents all non-codons
    dreward_dmut = dreward_dmut[:-1]
    prob_ratios = np.exp(dreward_dmut * temperature)
    bad_muts_mask = rng.uniform(size=len(muts)) > prob_ratios
    muts = [m for i, m in enumerate(muts) if not bad_muts_mask[i]]
    return muts
