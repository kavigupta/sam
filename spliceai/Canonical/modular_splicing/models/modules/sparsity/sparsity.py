from abc import ABC, abstractmethod


class SparseLayer(ABC):
    """
    Base class for sparse layers.

    This class uses the term `sparsity` to refer to the fraction of outputs
        that are zero, i.e., correctly. In other places, the term `sparsity`
        is used to refer to the fraction of inputs that are nonzero, which is
        the opposite. This is confusing, but we are stuck with it.

    In training mode, can update the thresholds to be used for sparsity.
    In eval mode, freezes these thresholds.
    """

    @abstractmethod
    def update_sparsity(self, update_by):
        """
        Update the sparsity of the layer by the given amount.

        Effectively, multiply the density by `update_by`.

        E.g., 25% sparse, update_by = 0.5 -> 62.5% sparse
        """

    @abstractmethod
    def set_sparsity(self, sparsity):
        """
        Set the sparsity of the layer to the given value.

        E.g., 25% sparse, sparsity = 0.7 -> 70% sparse
        """

    @abstractmethod
    def get_sparsity(self):
        """
        Get the sparsity of the layer.

        This is the fraction of outputs that are zero.
        """

    @abstractmethod
    def forward(self, x):
        """
        Run on the given input.

        NOTE: this is in (N, C, L) format, not (N, L, C) format.

        Parameters
        ----------
        x : torch.Tensor (N, C, L)
            Input to the layer.

        Returns
        -------
        torch.Tensor (N, C, L)
            Output of the layer. Should be sparse.
        """

    @abstractmethod
    def thresholds_numpy(self, num_motifs):
        """
        Produce the thresholds for each motif.

        Parameters
        ----------
        num_motifs : int
            Number of motifs in the layer.

        Returns
        -------
        numpy.ndarray (num_motifs,)
            Thresholds for each motif.
        """

    @abstractmethod
    def motif_index(self, num_channels):
        """
        Maps each valid index in the output array to a true motif index. Do not provide
            an index for indices in the output array that do not correspond to a real motif

        Basically used for situations where the sparse layer has dropped motifs.

        E.g., if there were originally 10 motifs from 0 to 9, but we've dropped
            motifs 2 and 7, then the output array will still have 10 channels,
            and we return {
                0: 0,
                1: 1,
                3: 2,
                4: 3,
                5: 4,
                6: 5,
                8: 6,
                9: 7
            }.

        Returns None if no motifs have been dropped.
        """


def sparse_layer_types():
    from .no_sparsity import NoSparsity
    from .spatially_sparse_by_channel import (
        SpatiallySparseByChannel,
    )
    from .parallel_spatially_sparse import (
        ParallelSpatiallySparse,
    )
    from .spatially_sparse_across_channels import (
        SpatiallySparseAcrossChannels,
    )
    from .spatially_sparse_across_channels_drop_motifs import (
        SpatiallySparseAcrossChannelsDropMotifs,
    )

    return dict(
        NoSparsity=NoSparsity,
        SpatiallySparseByChannel=SpatiallySparseByChannel,
        SpatiallySparseAcrossChannels=SpatiallySparseAcrossChannels,
        SpatiallySparseAcrossChannelsDropMotifs=SpatiallySparseAcrossChannelsDropMotifs,
        ParallelSpatiallySparse=ParallelSpatiallySparse,
    )
