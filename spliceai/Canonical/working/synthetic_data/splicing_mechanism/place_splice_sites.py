import tqdm.auto as tqdm
from permacache import permacache, stable_hash

import numpy as np


def modified_codon(codon_idxs, rna, updates, start_location, end_location):
    original = rna[codon_idxs]
    modified = np.repeat(original.copy()[None], updates.shape[0], 0)
    mask_to_modify = in_range(codon_idxs, start_location, end_location)
    modified[:, mask_to_modify] = updates[
        :, codon_idxs[mask_to_modify] - start_location
    ]
    return original, modified


def consistent_codon_mask(
    codon_table, codon_idxs, rna, updates, start_location, end_location, mode
):
    original, modified = modified_codon(
        codon_idxs, rna, updates, start_location, end_location
    )
    original, modified = codon_table.to_id(original), codon_table.to_id(modified)
    mask = [
        codon_table.ids_correspond_to_same_amino(original, m, mode) for m in modified
    ]
    return mask


def in_range(codon_idxs, start_location, end_location):
    return (codon_idxs >= start_location) & (codon_idxs < end_location)


def filter_for_consistency_with_indices(
    codon_table, codon_idxs, rna, updates, start_location, end_location, mode
):
    [relevant_codons] = np.where(
        (in_range(codon_idxs, start_location, end_location)).any(-1)
    )
    for codon in relevant_codons:
        mask = consistent_codon_mask(
            codon_table,
            codon_idxs[codon],
            rna,
            updates,
            start_location,
            end_location,
            mode=mode,
        )
        updates = updates[mask]
    return updates


def place_consistent(
    codon_table, rna, exon_indices, intron_indices, motif, *, site, rng
):
    left_rad, right_rad = motif.radii_each()
    width = left_rad + right_rad + 1
    start_location, end_location = motif.start_end(site)
    if start_location < 0 or end_location > rna.shape[0]:
        return False

    updated = rng.choice(4, size=(100_000, width))
    mot = motif.score(updated, pad=False)
    assert mot.shape[1] == 1
    mot = mot[:, 0]
    updated = updated[mot != 0]

    for codon_idxs, mode in [
        (exon_indices, "same_amino"),
        (intron_indices, "any_amino"),
    ]:
        updated = filter_for_consistency_with_indices(
            codon_table,
            codon_idxs,
            rna,
            updated,
            start_location,
            end_location,
            mode,
        )
    if updated.shape[0] == 0:
        return False
    selected = rng.choice(updated.shape[0])
    updated = updated[selected]
    assert motif.score(updated, pad=False)[0] != 0
    rna[start_location:end_location] = updated
    assert motif.score(rna)[site] != 0
    return True


def place_on_indices(
    codon_table,
    rna,
    exon_indices,
    intron_indices,
    motif,
    *,
    sites,
    frac_place,
    rng,
    update_pbar,
):
    # sample a binomial number of sites
    num_sites = rng.binomial(sites.size, frac_place)
    if num_sites == 0:
        update_pbar(1)
        return []
    chosen_sites = set()
    extra_sites = set()
    while True:
        site = rng.choice(sites.flatten())
        if site in extra_sites:
            continue
        if place_consistent(
            codon_table, rna, exon_indices, intron_indices, motif, site=site, rng=rng
        ):
            chosen_sites.add(site)
            update_pbar(1 / num_sites)
            extra_sites.update(range(*motif.start_end(site)))
            if len(chosen_sites) == num_sites:
                return sorted(extra_sites)


def add_fake_splice_sites(
    codon_table,
    rna,
    exon_indices,
    intron_indices,
    motif,
    *,
    frac_place,
    rng,
    update_pbar,
):
    fake_sites = []
    for indices in [exon_indices, intron_indices]:
        fake_sites += place_on_indices(
            codon_table,
            rna,
            exon_indices,
            intron_indices,
            motif,
            sites=indices,
            rng=rng,
            frac_place=frac_place,
            update_pbar=update_pbar,
        )
    return fake_sites


def add_real_splice_sites(
    codon_table,
    rna,
    exon_indices,
    intron_indices,
    motif,
    real_sites,
    *,
    rng,
    update_pbar,
):
    protected = []
    for site in real_sites:
        if not place_consistent(
            codon_table, rna, exon_indices, intron_indices, motif, site=site, rng=rng
        ):
            raise ValueError("Real site could not be placed")
        update_pbar(1 / real_sites.size)
        assert motif.score(rna)[site] != 0
        protected.extend(range(*motif.start_end(site)))
    return protected


@permacache(
    "synthetic_data/splicing_mechanism/place_splice_sites/add_splice_sites_9",
    key_function=dict(
        codon_table=stable_hash,
        rna=stable_hash,
        splicing_mechanism=stable_hash,
        exon_indices=stable_hash,
        intron_indices=stable_hash,
        splice_sites=stable_hash,
    ),
    multiprocess_safe=True,
)
def add_splice_sites(
    codon_table,
    *,
    rna,
    splicing_mechanism,
    exon_indices,
    intron_indices,
    splice_sites,
    seed,
    frac_place,
):
    rna = rna.copy()
    rng = np.random.RandomState(seed)
    protected_sites = []
    pbar = tqdm.tqdm(total=6)
    for acc_don in 0, 1:
        motif = splicing_mechanism.splice_site_motifs[acc_don]
        protected_sites += add_fake_splice_sites(
            codon_table,
            rna,
            exon_indices,
            intron_indices,
            motif,
            rng=rng,
            frac_place=frac_place,
            update_pbar=pbar.update,
        )
    for acc_don in 0, 1:
        motif = splicing_mechanism.splice_site_motifs[acc_don]
        protected_sites += add_real_splice_sites(
            codon_table,
            rna,
            exon_indices,
            intron_indices,
            motif,
            splice_sites[1 - acc_don],
            rng=rng,
            update_pbar=pbar.update,
        )
    pbar.close()
    return rna, sorted(set(protected_sites))
