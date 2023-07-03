from abc import ABC, abstractmethod

import attr
import h5py
import numpy as np

from modular_splicing.dataset.datapoint_extractor import BasicDatapointExtractor


class MultiSettingH5File(ABC):
    """
    Multiple settings in one h5 file, which expands the data with duplicates
    of the same data, but with different settings.

    Provides extra key "S0", "S1", etc. which are the settings for each
    datapoint, to be used as an input.
    """

    def __init__(self, path):
        """
        path: path to h5 file to open
        metadata_name: name of metadata file, in same directory as h5 file
        datapoint_extractor: only used to get batch/sequence axis
        """
        self.file = h5py.File(path, "r")
        self.valid_keys = {k for k in self.file.keys() if self._valid_key(k)}
        assert not any(k.startswith("S") for k in self.valid_keys)
        self.valid_keys |= {f"S{k[1:]}" for k in self.valid_keys}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def keys(self):
        return self.valid_keys

    def __getitem__(self, key):
        assert key in self.keys(), key
        if key.startswith("S"):
            return np.arange(self._batch_length_for(key)) % self._expansion_factor(key)
        return MultiSettingH5FileDataWrapper(self, key)

    def __contains__(self, key):
        return key in self.valid_keys

    def __len__(self):
        return len(self.valid_keys)

    def items(self):
        return ((k, self[k]) for k in self.keys())

    def true_shape(self, key):
        assert key in self.keys(), key
        original_shape = list(self.file[key].shape)
        original_shape[self._shape_offset(key)] *= self._expansion_factor(key)
        return tuple(original_shape)

    @abstractmethod
    def _valid_key(self, key):
        pass

    @abstractmethod
    def _shape_offset(self, key):
        pass

    @abstractmethod
    def _expansion_factor(self, key):
        pass

    @abstractmethod
    def _batch_length_for(self, key):
        pass

    @abstractmethod
    def actually_find(self, key):
        pass


@attr.s
class MultiSettingH5FileDataWrapper:
    """
    Allows for indexing into a subset of indices of an object.
    """

    dset = attr.ib()
    key = attr.ib()

    @property
    def shape(self):
        return self.dset.true_shape(self.key)

    def __getitem__(self, sli):
        assert sli == slice(None)
        return self.dset.actually_find(self.key)


class MultipleSettingDatapointExtractor(BasicDatapointExtractor):
    """
    Like BasicDatapointExtractor, but allows for multiple settings, which are
        passed as an integer in the ["inputs"]["setting"] field.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def extract_datapoint(self, data, i, j):
        el = super().extract_datapoint(data, i, j)
        setting_value = data[f"S{i}"][j]
        # just to be compatible with the slicing code
        setting_array = np.zeros_like(el["inputs"]["x"], dtype=np.int64) + setting_value
        el["inputs"]["setting"] = setting_array
        return el

    def shape_offset(self):
        return 1 if self.clip_y_zeros else 0
