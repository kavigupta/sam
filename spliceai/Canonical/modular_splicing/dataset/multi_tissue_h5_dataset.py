import json
import os
import pickle


import numpy as np
from modular_splicing.dataset.h5_dataset import H5Dataset
from modular_splicing.dataset.multi_setting_h5_file import MultiSettingH5File


class MultiTissueProbabilitiesH5File(MultiSettingH5File):
    """
    Represents a multi-tissue probabilities dataset.

    Parameters
    ----------
    path : str
        The path to the h5 file.
    tissue_groups : list of list of str
        The tissue groups. Each tissue group is a list of tissues.
    """

    def __init__(self, path, datapoint_extractor, *, tissue_groups):
        super().__init__(path)
        with open(os.path.join(os.path.dirname(path), "psis.pkl"), "rb") as f:
            metadata = pickle.load(f)
        with open(os.path.join(os.path.dirname(path), "tissue_names.json"), "r") as f:
            tissues = json.load(f)

        self.tissue_groups = tissue_groups

        self.which_splicepoint = np.array(
            [
                {
                    # + strand is 5' to 3'
                    ("+", 0): 2,
                    ("+", 1): 1,
                    # - strand is 3' to 5'
                    ("-", 0): 1,
                    ("-", 1): 2,
                }[(strand, is_end)]
                for _, strand, _, is_end in metadata["all_keys"]
            ]
        )
        psi = metadata["psi_values"]
        psi_selected = []
        for tissue_group in tissue_groups.values():
            assert isinstance(tissue_group, (list, tuple))
            tissue_idxs = [tissues.index(t) for t in tissue_group]
            psi_selected.append(psi[:, tissue_idxs].mean(axis=1))
        self.psi = np.stack(psi_selected, axis=1)
        self.datapoint_extractor = datapoint_extractor

    def _valid_key(self, key):
        return key[0] in "XY"

    def _shape_offset(self, key):
        return self.datapoint_extractor.shape_offset() if key[0] == "Y" else 0

    def _expansion_factor(self, key):
        return self.psi.shape[1]

    def _batch_length_for(self, key):
        assert key[0] == "S"
        return self[key.replace("S", "X")].shape[0]

    def actually_find(self, key):
        """
        Returns the actual data for the given key, expanded to the correct shape.
        """
        obj = self.file[key][:]
        assert self._shape_offset(key) in {0, 1}
        first_dims = [-1]
        if self._shape_offset(key) == 1:
            first_dims = [1, -1]
            obj = obj.reshape(-1, *obj.shape[2:])
        obj = obj[:, None].repeat(self._expansion_factor(key), axis=1)
        if key[0] == "Y":
            obj = self.expand_y(obj)
        obj = obj.reshape(-1, *obj.shape[2:])
        obj = obj.reshape(*first_dims, *obj.shape[1:])
        return obj

    def expand_y(self, obj):
        """
        Expand the Y data to the correct shape.
        """
        batch_idx, tissue_idx, seq_idx = np.where(obj > 0)
        psi_idx = obj[batch_idx, tissue_idx, seq_idx]
        splice_idxs = self.which_splicepoint[psi_idx]
        psi_values = self.psi[psi_idx, tissue_idx]
        psi_nan = np.isnan(psi_values)
        psi_values[psi_nan] = 0
        out = np.zeros((*obj.shape, 3))
        out[:, :, :, 0] = 1
        out[batch_idx, tissue_idx, seq_idx, splice_idxs] = psi_values
        out[batch_idx, tissue_idx, seq_idx, 0] = 1 - psi_values
        out[batch_idx[psi_nan], tissue_idx[psi_nan], seq_idx[psi_nan]] = 0
        return out


class MultiTissueProbabilitiesH5Dataset(H5Dataset):
    """
    Represent a multi-tissue probabilities dataset.
    """

    def __init__(self, *, tissue_groups, **kwargs):
        self.tissue_groups = tissue_groups
        super().__init__(**kwargs)

    def _data(self):
        return MultiTissueProbabilitiesH5File(
            self.path, self.datapoint_extractor, tissue_groups=self.tissue_groups
        )
