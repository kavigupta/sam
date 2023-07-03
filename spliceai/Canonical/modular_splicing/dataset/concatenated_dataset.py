import re
from modular_splicing.dataset.datapoint_extractor import DatapointExtractor
from modular_splicing.dataset.generic_dataset import GenericDataset, dataset_types
from modular_splicing.utils.construct import construct
from modular_splicing.utils.mulitple_contexts import MultipleContexts
from modular_splicing.utils.multi_h5_file import (
    compute_index,
    compute_prefix,
    construct_key_map,
)


class ConcatenatedDatapointExtractor(DatapointExtractor):
    def __init__(self, dset):
        self.concatenated_dataset = dset

    def extract_datapoint(self, data, i, j):
        which_dset, _ = self.concatenated_dataset.indices_in_original(i)
        return self.concatenated_dataset.datasets[
            which_dset
        ].datapoint_extractor.extract_datapoint(data, i, j)

    def shape_offset(self):
        [amount] = {
            dset.datapoint_extractor.shape_offset()
            for dset in self.concatenated_dataset.datasets
        }
        return amount


class ConcatenateDatasets(GenericDataset):
    """
    Represents a concatenation of multiple datasets.
    """

    def __init__(
        self,
        *,
        path,
        specs,
        path_replacement_specs,
        **kwargs,
    ):
        assert len(specs) == len(path_replacement_specs)
        assert all("iterator_spec" not in spec for spec in specs)
        assert all("post_processor_spec" not in spec for spec in specs)
        super().__init__(
            iterator_spec=kwargs["iterator_spec"],
            datapoint_extractor_spec=dict(type="ConcatenatedDatapointExtractor"),
            post_processor_spec=kwargs["post_processor_spec"],
        )
        paths = [
            construct(
                dict(
                    regex_replace=lambda regex, replacement, path: re.sub(
                        regex, replacement, path
                    ),
                    identity=lambda path: path,
                ),
                path_replacement_spec,
                path=path,
            )
            for path_replacement_spec in path_replacement_specs
        ]
        self.datasets = [
            construct(dataset_types(), spec, path=path, **kwargs)
            for spec, path in zip(specs, paths)
        ]
        with self._data() as data:
            self.key_map = construct_key_map([x.keys() for x in data])

    def _data(self):
        return MultipleContexts([dataset._data() for dataset in self.datasets])

    def __len__(self):
        return sum(len(x) for x in self.datasets)

    def length_each(self, data):
        assert len(data) == len(self.datasets)
        each = [x.length_each(y) for x, y in zip(self.datasets, data)]
        return [x for xs in each for x in xs]

    def data_for_chunk(self, data, out_idx):
        i, j = self.indices_in_original(out_idx)
        result = self.datasets[i].data_for_chunk(data[i], j)
        result = {compute_prefix(k) + str(out_idx): v for k, v in result.items()}
        return result

    def indices_in_original(self, out_idx):
        relevant_data_keys = [k for k in self.key_map if compute_index(k) == out_idx]
        i_s, j_s = zip(*[self.key_map[k] for k in relevant_data_keys])
        [i] = set(i_s)
        [j] = {compute_index(j) for j in j_s}
        return i, j

    def cached_data(self, data):
        caches_each = [x.cached_data(y) for x, y in zip(self.datasets, data)]
        return {k: caches_each[i1][i2] for k, (i1, i2) in self.key_map.items()}

    def run_with_data_generator(self, fn_accepting_data):
        with self._data() as data:
            yield from fn_accepting_data(data)

    def clip(self, *, value, is_output):
        # assumes all datasets have the same clip method
        return self.datasets[0].clip(value=value, is_output=is_output)
