import attr

import numpy as np

from working.synthetic_data.mutation.codon_mutation import CodonMutation


@attr.s
class CodonAddress:
    unchangable_positions_mask = attr.ib()
    all_codon_indices = attr.ib()
    is_exon = attr.ib()
    index_map = attr.ib()

    @classmethod
    def of(cls, eci, ici, unchangable_positions_mask):
        aci = np.concatenate([eci, ici])
        is_exon = np.concatenate(
            [
                np.ones(eci.shape[0], dtype=np.bool_),
                np.zeros(ici.shape[0], dtype=np.bool_),
            ]
        )
        changeable = ~unchangable_positions_mask[aci].any(-1)
        aci, is_exon = aci[changeable], is_exon[changeable]
        index_map = np.zeros(unchangable_positions_mask.shape[0], dtype=np.int32) - 1
        index_map[aci] = np.arange(aci.shape[0])[:, None]
        return cls(unchangable_positions_mask, aci, is_exon, index_map)

    def num_codons(self):
        return self.all_codon_indices.shape[0]

    def sample_mutation(self, codon_table, rna, codon_address, rng):
        loc = self.all_codon_indices[codon_address]
        new_seq = codon_table.sample_mutation(
            rna[loc],
            mode="same_amino" if self.is_exon[codon_address] else "any_amino",
            rng=rng,
        )
        if new_seq is None:
            return None
        return CodonMutation(
            loc,
            new_seq,
        )
