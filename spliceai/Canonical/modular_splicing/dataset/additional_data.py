from abc import ABC, abstractmethod
import os

import numpy as np


class AdditionalData(ABC):
    """
    Handles additional data to be loaded.

    Can be called with arguments
    - path: path to the dataset
    - i: index of the chunk
    - j: index of the sequence in the chunk
    - cl_max: context length of the underlying dataset
    - target: either "inputs" or "outputs", depending on whether the
        additional data is an input or an output. This impacts the
        padding.
    """

    @abstractmethod
    def compute_additional_input(self, original_input, path, i, j):
        """
        Required, compute the additional data as if it were an input.
        """

    def compute_additional_output(self, original_input, path, i, j, *, cl_max):
        """
        Compute the additional data as if it were an output. By default
        this is the same as compute_additional_input, except that it
        trims the padding on either side.
        """
        extra = self.compute_additional_input(original_input, path, i, j)
        return extra[cl_max // 2 : extra.shape[0] - cl_max // 2]

    def __call__(self, el, path, i, j, *, cl_max, target):
        if target == "inputs":
            return self.compute_additional_input(el["inputs"]["x"], path, i, j)
        elif target == "outputs":
            return self.compute_additional_output(
                el["outputs"]["y"], path, i, j, cl_max=cl_max
            )
        raise RuntimeError(f"Unknown target {target}")

    def classify_path(self, path):
        """
        Convert a path to a dataset to a path to whether it
        is a training, validation or test set.

        This is done by looking at the name and only works for
        datasets named `dataset_train_all.h5` or `dataset_test_0.h5`.

        Arguments:
        path: path to the dataset

        Returns:
        bool, whether it is a training set
        """
        if os.path.basename(path) == "dataset_train_all.h5":
            is_train = True
        elif os.path.basename(path) == "dataset_test_0.h5":
            is_train = False
        else:
            raise RuntimeError(f"Cannot figure out which dataset to load for {path}")

        return is_train


class IndexTrackingAdditionalData(AdditionalData):
    """
    Tracks the indices of the original input.

    Useful for referencing the original coordinates.

    Produces an output of shape (n, 2) where n is the number of datapoints.

    The first column is the index of the original chunk and the second column
    is the index of the datapoint within the chunk.
    """

    def compute_additional_input(self, original_input, path, i, j):
        indices = np.zeros((*original_input.shape[:1], 2), dtype=np.int64)
        indices[:, 0] += i
        indices[:, 1] += j
        return indices


class DatafileReferencingAdditionalData(AdditionalData):
    """
    Subclass of AdditionalData that sets up aligners
    for you based on the datafiles you provide.

    This is useful in case you need to access the
    original datafile to compute additional data.
    """

    def __init__(self, datafiles, sl):
        self.datafiles = datafiles
        from modular_splicing.dataset.dataset_aligner import DatasetAligner

        self.aligners = {
            path: DatasetAligner(
                dataset_path=path,
                datafile_path=self.datafiles[path],
                sl=sl,
            )
            for path in self.datafiles
        }

        self._sl = sl


def additional_data_types():
    from modular_splicing.eclip.data.dataset import EClipAdditionalData
    from shelved.branch_sites.pipeline import (
        BranchSiteAdditionalData,
        BranchSiteMaskAdditionalData,
    )
    from shelved.full_gtex_dataset.dataset.unified_alternative_dataset import (
        MaskOutOtherSplicepointsDatasetRewriter,
        SelectChannels,
        ExpandUnifiedAlternativeDatasetRewriter,
    )
    from modular_splicing.dataset.secondary_structure.rna_fold import (
        SubstructureProbabilityInformationAdditionalData,
    )
    from shelved.gene_length.length_additional_data import (
        GeneLengthWeightingAdditionalData,
    )
    from shelved.gene_length.at_rich_additional_data import (
        ATRichAdditionalData,
        GeneATRichnessAdditionalData,
    )

    from modular_splicing.evolutionary_conservation.phylo_p import PhyloPAdditionalData

    from modular_splicing.dataset.filter_by_lssi import FilterByIncorrectLSSI

    return dict(
        eclip=EClipAdditionalData,
        branch_site=BranchSiteAdditionalData,
        branch_site_mask=BranchSiteMaskAdditionalData,
        substructure_probabilities=SubstructureProbabilityInformationAdditionalData,
        index_tracking=IndexTrackingAdditionalData,
        gene_length_weighting=GeneLengthWeightingAdditionalData,
        expand_unified_alternative_dataset=ExpandUnifiedAlternativeDatasetRewriter,
        mask_out_other_splicepoints_dataset=MaskOutOtherSplicepointsDatasetRewriter,
        select_channels=SelectChannels,
        at_rich_additional_input=ATRichAdditionalData,
        gene_level_at_rich_additional_input=GeneATRichnessAdditionalData,
        phylo_p=PhyloPAdditionalData,
        FilterByIncorrectLSSI=FilterByIncorrectLSSI,
    )
