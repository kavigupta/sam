import numpy as np

from modular_splicing.dataset.additional_data import DatafileReferencingAdditionalData


class GeneLengthWeightingAdditionalData(DatafileReferencingAdditionalData):
    def compute_additional_input(self, original_input, path, i, j):
        raise RuntimeError("Not implemented")

    def compute_additional_output(self, original_output, path, i, j, cl_max):
        aligner = self.aligners[path]

        # the mean of the values across all batches should be 1
        # (correction * values_each * gene_sizes).sum() / (gene_sizes.sum()) = 1
        # num_genes * correction / gene_sizes.sum() = 1
        # correction = gene_sizes.mean()

        gene_idx, _ = aligner.get_gene_idx(i, j)
        val = aligner.gene_sizes.mean() / aligner.gene_sizes[gene_idx]

        res = np.zeros((*original_output.shape[:1], 1)) + val
        return res
