from abc import ABC, abstractmethod

import numpy as np
from modular_splicing.dataset.additional_data import additional_data_types

from modular_splicing.utils.construct import construct


class DataRewriter(ABC):
    """
    Represents a generic method for rewriting a datapoint.
    """

    @abstractmethod
    def rewrite_datapoint(self, *, el, i, j, dset):
        """
        Rewrite the given datapoint in some way, given the contextual information
        """


class AdditionalChannelDataRewriter(DataRewriter):
    """
    Runs the underlying data provider and combines the result with the existing data
        in the given channel using the given combinator.
    """

    def __init__(
        self,
        out_channel,
        data_provider_spec,
        combinator_spec=dict(type="NovelCombinator"),
    ):
        self.out_channel = out_channel
        self.data_provider = construct(
            additional_data_types(),
            data_provider_spec,
        )
        self.combinator = construct(combinator_types(), combinator_spec)

    def rewrite_datapoint(self, *, el, i, j, dset):
        k1, k2 = self.out_channel
        additional_data = self.data_provider(
            el=el,
            path=dset.path,
            i=i,
            j=j,
            cl_max=dset.cl_max,
            target=k1,
        )
        el[k1][k2] = self.combinator.combine(
            current=el[k1].get(k2, None), novel=additional_data
        )
        return el


def data_rewriter_types():
    from shelved.full_gtex_dataset.gtex_dataset import gtex_index_rewriter_types
    from modular_splicing.models.entire_model.reconstruct_sequence import (
        ReconstructSequenceDataRewriter,
    )

    return dict(
        AdditionalChannelDataRewriter=AdditionalChannelDataRewriter,
        **gtex_index_rewriter_types(),
        ReconstructSequenceDataRewriter=ReconstructSequenceDataRewriter,
    )


class Combinator(ABC):
    """
    Represents a generic way to combine two datapoints.
    """

    @abstractmethod
    def combine(self, *, current, novel):
        """
        Combine the current value (None if it does not exist) with the novel value
        """


class NovelCombinator(Combinator):
    """
    Check that the current value does not exist, and then take the novel value.
    """

    def combine(self, *, current, novel):
        assert current is None
        return novel


class ReplacingCombinator(Combinator):
    """
    Check that the current value exists, then replace it
    """

    def combine(self, *, current, novel):
        assert current is not None
        return novel


class OneHotConcatenatingCombinator(Combinator):
    """
    Add the novel value to the end of the current value, as a one-hot vector

    E.g., [[0, 1, 0], [1, 0, 0]] `comb` [[0], [1]] --> [[0, 1, 0, 0], [0, 0, 0, 1]]
    """

    def combine(self, *, current, novel):
        current[novel.any(-1), 0] = 0
        current = np.concatenate([current, novel], axis=1)
        return current


def combinator_types():
    return dict(
        NovelCombinator=NovelCombinator,
        ReplacingCombinator=ReplacingCombinator,
        OneHotConcatenatingCombinator=OneHotConcatenatingCombinator,
    )
