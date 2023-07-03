import numpy as np


from modular_splicing.dataset.additional_data import (
    AdditionalData,
    DatafileReferencingAdditionalData,
)
from modular_splicing.utils.genome_statistics import gene_at_richness


class GeneATRichnessAdditionalData(DatafileReferencingAdditionalData):
    def __init__(self, *args, cl_max, **kwargs):
        super().__init__(*args, **kwargs)
        self.cl_max = cl_max

    def compute_additional_input(self, original_input, path, i, j):
        aligner = self.aligners[path]

        # the mean of the values across all batches should be 1
        # (correction * values_each * gene_sizes).sum() / (gene_sizes.sum()) = 1
        # num_genes * correction / gene_sizes.sum() = 1
        # correction = gene_sizes.mean()

        gene_idx, _ = aligner.get_gene_idx(i, j)
        gene_at = gene_at_richness(self.datafiles[path], gene_idx, self.cl_max)
        res = np.zeros((*original_input.shape[:1], 1)) + gene_at
        return res


class ATRichAdditionalData(AdditionalData):
    def __init__(self, chunk_width, select_bases="AT"):
        self.chunk_width = chunk_width
        self.select_bases = select_bases

    def compute_additional_input(self, original_input, path, i, j):
        assert len(original_input.shape) == 2 and original_input.shape[-1] == 4

        assert original_input.shape[0] % self.chunk_width == 0
        original_input = original_input.reshape(-1, self.chunk_width, 4)

        channels = ["ACGT".index(x) for x in self.select_bases]
        is_at = original_input[:, :, channels].sum(-1)
        is_anything = original_input.sum(-1)

        at_frac = is_at.sum(-1) / (is_anything.sum(-1) + 1e-10)

        at_frac = at_frac[:, None].repeat(self.chunk_width, axis=1)
        at_frac = at_frac.reshape(-1, 1)

        return at_frac
