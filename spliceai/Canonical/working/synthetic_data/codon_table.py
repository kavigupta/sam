from functools import cached_property
import numpy as np
import attr

from modular_splicing.utils.sequence_utils import all_seqs, draw_bases, parse_bases

CODON_TO_ID = np.array([1, 4, 16])

_STANDARD_CODON_TABLE = [
    ["UUU", "UUC"],  # Phe/F
    ["CUU", "CUC", "CUA", "CUG", "UUA", "UUG"],  # Leu/L
    ["AUU", "AUC", "AUA"],  # Ile/I
    ["AUG"],  # Met/M
    ["GUU", "GUC", "GUA", "GUG"],  # Val/V
    ["UCU", "UCC", "UCA", "UCG", "AGU", "AGC"],  # Ser/S
    ["CCU", "CCC", "CCA", "CCG"],  # Pro/P
    ["ACU", "ACC", "ACA", "ACG"],  # Thr/T
    ["GCU", "GCC", "GCA", "GCG"],  # Ala/A
    ["UAU", "UAC"],  # Tyr/Y
    ["CAU", "CAC"],  # His/H
    ["CAA", "CAG"],  # Gln/Q
    ["AAU", "AAC"],  # Asn/N
    ["AAA", "AAG"],  # Lys/K
    ["GAU", "GAC"],  # Asp/D
    ["GAA", "GAG"],  # Glu/E
    ["UGU", "UGC"],  # Cys/C
    ["UGG"],  # Trp/W
    ["CGU", "CGC", "CGA", "CGG", "AGA", "AGG"],  # Arg/R
    ["GGU", "GGC", "GGA", "GGG"],  # Gly/G
    ["UAA", "UGA", "UAG"],  # Stop
]


@attr.s
class CodonTable:
    per_codon = attr.ib()

    @classmethod
    def sample(cls, rng):
        """
        Sample codons entirely randomly. There are 22 codons, and a stop codon.

        We represent the table as a list of 23 arrays, each of shape (k_i, 3, 4), where
            all elements are unique 3-element codons and sum_i k_i = 64.

        The first 22 arrays are the codons for each amino acid, and the last array is the
            stop codon.

        We guarantee at least one codon for each amino acid, but otherwise the codons are
            sampled uniformly at random.
        """
        one_each = rng.choice(64, size=23, replace=False)
        remainder = np.setdiff1d(np.arange(64), one_each)
        remainder_assignment = rng.choice(23, size=len(remainder), replace=True)
        result_each = [
            np.array([one_each[i], *remainder[remainder_assignment == i]])
            for i in range(23)
        ]
        codons = np.array(list(all_seqs(3))).argmax(-1)
        result = [codons[i] for i in result_each]
        return cls(result)

    @classmethod
    def standard(cls):
        """
        The standard codon table. See https://en.wikipedia.org/wiki/DNA_and_RNA_codon_tables
        """

        # original is strings, convert to arrays.

        return cls(
            [
                np.array([parse_bases(x.replace("U", "T")) for x in codons]).argmax(-1)
                for codons in _STANDARD_CODON_TABLE
            ]
        )

    def codons_for(self, amino):
        return self.per_codon[amino]

    def num_aminos(self):
        return len(self.per_codon)

    def sample_protein(self, rng, *, num_aminos, pseudo):
        """
        Sample a protein sequence of length `num_aminos` from the codon table.
        Do not include the stop codon, in anything but the end codon.
        """
        if pseudo:
            weights = np.array([len(x) for x in self.per_codon])
            weights = weights / weights.sum()
        else:
            weights = np.ones(self.num_aminos())
            weights[-1] = 0
            weights = weights / weights.sum()
        aminos = rng.choice(len(self.per_codon), size=num_aminos, p=weights)
        if not pseudo:
            aminos[-1] = len(self.per_codon) - 1
        return aminos

    def draw(self):
        return [draw_bases(x) for x in self.per_codon]

    @cached_property
    def _codon_id_to_other_codons_with_same_amino(self):
        result = {}
        for amino in range(self.num_aminos()):
            codons = self.codons_for(amino)
            for i, codon in enumerate(codons):
                codon_id = codon @ CODON_TO_ID
                result[codon_id] = np.concatenate([codons[:i], codons[i + 1 :]])
        return result

    @cached_property
    def _codon_id_to_other_codons_with_any_amino(self):
        result = {}
        sequences = np.array(list(all_seqs(3))).argmax(-1)
        for seq in sequences:
            codon_id = seq @ CODON_TO_ID
            result[codon_id] = sequences[(sequences != seq).all(-1)]

        return result

    @cached_property
    def _all_same_codon_pairs_set(self):
        return {
            (x @ CODON_TO_ID, y @ CODON_TO_ID)
            for xs in self.per_codon
            for x in xs
            for y in xs
        }

    def sample_mutation(self, codon, *, rng, mode):
        if mode == "same_amino":
            other_codons = self._codon_id_to_other_codons_with_same_amino[
                codon @ CODON_TO_ID
            ]
        elif mode == "any_amino":
            other_codons = self._codon_id_to_other_codons_with_any_amino[
                codon @ CODON_TO_ID
            ]
        else:
            raise ValueError(mode)
        if len(other_codons) == 0:
            return None
        return other_codons[rng.choice(other_codons.shape[0])]

    def to_id(self, codon):
        return codon @ CODON_TO_ID

    def ids_correspond_to_same_amino(self, a, b, mode):
        if mode == "same_amino":
            return (a, b) in self._all_same_codon_pairs_set
        elif mode == "any_amino":
            return True
        else:
            raise ValueError(mode)
