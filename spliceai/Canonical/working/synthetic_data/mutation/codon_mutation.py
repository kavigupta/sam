import attr

from .mutation import Mutation

import numpy as np


@attr.s
class CodonMutation(Mutation):
    """
    Represents a mutation of a codon.
    """

    codon_idxs = attr.ib()
    new_codon = attr.ib()

    def perform(self, rna):
        rna[self.codon_idxs] = self.new_codon

    def footprint(self):
        return self.codon_idxs

    @classmethod
    def combine(cls, mutations):
        assert all(isinstance(m, cls) for m in mutations)
        return cls(
            np.concatenate([m.codon_idxs for m in mutations]),
            np.concatenate([m.new_codon for m in mutations]),
        )
