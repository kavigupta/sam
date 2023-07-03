import attr

import numpy as np


@attr.s
class SaliencyMap:
    codon_address = attr.ib()
    saliency = attr.ib()
    absolute_saliency = attr.ib()
    bar = attr.ib()
    salient_locations = attr.ib()
    salient_motifs = attr.ib()
    possibly_salient_codons = attr.ib()
    possibly_non_salient_codons = attr.ib()
    all_codons = attr.ib()

    @classmethod
    def of(cls, unchangable_motifs, codon_address, salience, frac_keep):
        salience = salience * (~unchangable_motifs)
        absolute_salience = np.abs(salience)
        bar = np.quantile(absolute_salience, 1 - frac_keep)
        salient_locations, salient_motifs = np.where(absolute_salience > bar)
        valid_codons = set(codon_address.index_map[~unchangable_motifs.any(-1)]) - {-1}
        salient_codons = set(codon_address.index_map[salient_locations]) - {-1}
        non_salient_codons = valid_codons - salient_codons
        salient_codons = np.array(sorted(salient_codons))
        non_salient_codons = np.array(sorted(non_salient_codons))
        all_codons = np.array(sorted(valid_codons))

        return cls(
            codon_address,
            salience,
            absolute_salience,
            bar,
            salient_locations,
            salient_motifs,
            salient_codons,
            non_salient_codons,
            all_codons,
        )

    def frac_possibly_salient_codons(self):
        return self.possibly_salient_codons.shape[0] / self.codon_address.num_codons()

    def _sample_codon(self, rng, *, codon_type):
        if codon_type == "salient":
            codons = self.possibly_salient_codons
        elif codon_type == "non_salient":
            codons = self.possibly_non_salient_codons
        elif codon_type == "any":
            codons = self.all_codons
        else:
            raise ValueError(f"Unknown codon_type: {codon_type}")
        return rng.choice(codons)

    def sample_mutation(self, codon_table, rna, rng, *, codon_type):
        while True:
            codon_idx = self._sample_codon(rng, codon_type=codon_type)
            mut = self.codon_address.sample_mutation(codon_table, rna, codon_idx, rng)
            if mut is not None:
                return mut

    def sample_mutation_series(self, codon_table, mech, rna, motifs, rng):
        from working.synthetic_data.realization import apply_local, compute_ranges

        radius = mech.motif_width() // 2

        mutations = []
        while True:
            for _ in range(rng.geometric(self.frac_possibly_salient_codons()) - 1):
                mutations.append(
                    self.sample_mutation(
                        codon_table, rna, rng, codon_type="non_salient"
                    )
                )

            mut = self.sample_mutation(codon_table, rna, rng, codon_type="salient")
            mutations.append(mut)
            rna_footprint = mut.footprint()
            ranges = compute_ranges(rna_footprint)

            mut.perform(rna)

            motif_undo, _, mot_ranges = apply_local(
                mech.predict_motifs,
                rna,
                motifs,
                radius,
                ranges,
            )
            delta_poss, delta_ids = [], []
            for (start, end), ol in zip(mot_ranges, motif_undo.old):
                delta_pos, delta_id = np.where(motifs[start:end] != ol)
                delta_poss.extend(delta_pos + start)
                delta_ids.extend(delta_id)
            real_change = (
                np.array(self.saliency[delta_poss, delta_ids]) > self.bar
            ).any()
            motif_undo()
            if real_change:
                break
        return mutations

    def sample_nonconflicting_mutations(self, rna, codon_table, *, motif_width, rng):
        assert motif_width % 2 == 1
        motif_radius = motif_width // 2
        map_back = np.zeros(rna.shape[0], dtype=np.int64) - 1
        selected_codons = []
        for _ in range(rna.shape[0] // 3):
            mut = self.sample_mutation(codon_table, rna, rng, codon_type="any")
            lo, hi = mut.codon_idxs.min(), 1 + mut.codon_idxs.max()
            if (map_back[lo - motif_radius : hi + motif_radius + 1] >= 0).any():
                continue
            selected_codons.append(mut)
            codon_identifier = len(selected_codons)
            for i in mut.codon_idxs:
                map_back[i - motif_radius : i + motif_radius + 1] = codon_identifier
        return selected_codons, map_back
