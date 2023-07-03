from .h5_dataset import H5Dataset
from .multi_setting_h5_file import MultiSettingH5File


class DuplicatedSettingH5File(MultiSettingH5File):
    """
    Just duplicates the data identically for each setting.
    """

    def __init__(self, path, datapoint_extractor, num_settings):
        super(DuplicatedSettingH5File, self).__init__(path)
        self.datapoint_extractor = datapoint_extractor
        self.num_settings = num_settings

    def _valid_key(self, key):
        return key[0] in "XY"

    def _shape_offset(self, key):
        return self.datapoint_extractor.shape_offset() if key[0] == "Y" else 0

    def _expansion_factor(self, key):
        return self.num_settings

    def _batch_length_for(self, key):
        assert key[0] == "S"
        return self[key.replace("S", "X")].shape[0]

    def actually_find(self, key):
        obj = self.file[key][:]
        assert self._shape_offset(key) in {0, 1}
        first_dims = [-1]
        if self._shape_offset(key) == 1:
            first_dims = [1, -1]
            obj = obj.reshape(-1, *obj.shape[2:])
        obj = obj[:, None].repeat(self._expansion_factor(key), axis=1)
        obj = obj.reshape(-1, *obj.shape[2:])
        obj = obj.reshape(*first_dims, *obj.shape[1:])
        return obj


class DuplicatedSettingH5Dataset(H5Dataset):
    """
    Represent a multi-tissue probabilities dataset.
    """

    def __init__(self, *, num_settings, **kwargs):
        self.num_settings = num_settings
        super().__init__(**kwargs)

    def _data(self):
        return DuplicatedSettingH5File(
            self.path, self.datapoint_extractor, num_settings=self.num_settings
        )
