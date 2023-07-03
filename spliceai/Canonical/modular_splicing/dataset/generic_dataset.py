from abc import ABC, abstractmethod
import torch

from .datapoint_extractor import datapoint_extractor_types
from .iterator import iterator_types
from .post_processor import post_processor_types
from modular_splicing.utils.construct import construct


class GenericDataset(ABC, torch.utils.data.IterableDataset):
    """
    Dataset base class. Contains some methods for iterating through the dataset.
    """

    def __init__(self, *, iterator_spec, datapoint_extractor_spec, post_processor_spec):
        self.iterator_spec = iterator_spec
        self.datapoint_extractor = construct(
            datapoint_extractor_types(), datapoint_extractor_spec, dset=self
        )
        self.post_processor = construct(post_processor_types(), post_processor_spec)

    def get_datapoint(self, *args, **kwargs):
        return self.datapoint_extractor.extract_datapoint(*args, **kwargs)

    def post_process(self, *args, **kwargs):
        return self.post_processor.post_process(*args, **kwargs)

    @abstractmethod
    def run_with_data_generator(self, fn_accepting_data, *, generator):
        """
        Run the generator function with the data for this dataset.
        """
        pass

    @abstractmethod
    def length_each(self, data):
        """
        Give the length of each data chunk, as a list.
        """
        pass

    @abstractmethod
    def data_for_chunk(self, data, chunk_idx):
        """
        Return the data for a given chunk. Generally a dictionary.
        """
        pass

    @abstractmethod
    def cached_data(self, data):
        """
        Cache the given data
        """
        pass

    @abstractmethod
    def clip(self, *, value, is_output):
        """
        Clip the given value, dependent on whether it is an output. Returns a sequence of clipped values
        """
        pass

    def iterate(self, data):
        """
        Iterate through the dataset. Handles clipping and post-processing.

        Yields a sequnece of post-processed objects.
        """
        iterator = construct(iterator_types(), self.iterator_spec)
        for el in iterator.iterate(self, data):
            el = {
                input_or_output: {
                    k: self.clip(value=value, is_output=input_or_output == "outputs")
                    for k, value in res.items()
                }
                for input_or_output, res in el.items()
            }
            for el2 in zipper_dictionaries(el):
                yield self.post_process(el2)

    def __iter__(self):
        return self.run_with_data_generator(self.iterate)


def dataset_types():
    from .h5_dataset import H5Dataset
    from shelved.full_gtex_dataset.dataset.unified_alternative_dataset import (
        UnifiedAlternativeDataset,
        NonConflictingAlternativeDataset,
    )
    from modular_splicing.dataset.multi_tissue_h5_dataset import (
        MultiTissueProbabilitiesH5Dataset,
    )
    from modular_splicing.dataset.duplicated_setting_h5_file import (
        DuplicatedSettingH5Dataset,
    )
    from modular_splicing.dataset.concatenated_dataset import ConcatenateDatasets

    return dict(
        H5Dataset=H5Dataset,
        MultiTissueProbabilitiesH5Dataset=MultiTissueProbabilitiesH5Dataset,
        UnifiedAlternativeDataset=UnifiedAlternativeDataset,
        NonConflictingAlternativeDataset=NonConflictingAlternativeDataset,
        DuplicatedSettingH5Dataset=DuplicatedSettingH5Dataset,
        ConcatenateDatasets=ConcatenateDatasets,
    )


def zipper_dictionaries(elements):
    [length] = {len(x) for xs in elements.values() for x in xs.values()}
    for i in range(length):
        yield {k: {k2: x[i] for k2, x in v.items()} for k, v in elements.items()}
