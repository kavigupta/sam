import h5py
import os

from modular_splicing.dataset.generic_dataset import GenericDataset
from modular_splicing.utils.multi_h5_file import MultiH5File


class H5Dataset(GenericDataset):
    """
    Represents a dataset that is stored in an HDF5 file.

    Data in this file must be stored under keys C# where C is a character and # is a number.

    Each chunk in the data file must have the same number of datapoints, but may have different
    lengths of the sequences.

    A chunk is a set of datapoints that all have the same #, and are identifiable by character.

    Extra arguments:
    path: path to the HDF5 file. If a list is provided, the files are concatenated as per multi_h5_dataset.
    symbol_for_length: symbol that corresponds to data whose lengths correspond to the sequence
        length of the datapoints.
    sl: sequence length to use for the datapoints
    cl: context length to use for the datapoints
    cl_max: context length of the underlying data. Will be verified.
    """

    def __init__(
        self,
        *,
        path,
        symbol_for_length="Y",
        sl,
        cl,
        cl_max,
        equalize_sizes_by_subsampling=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.path = path
        self.symbol_for_length = symbol_for_length
        self.sl = sl
        self.cl = cl
        # cl_max = 400 # cx: add for now
        self.cl_max = cl_max
        self.equalize_sizes_by_subsampling = equalize_sizes_by_subsampling
        for check_path in self.path if isinstance(self.path, list) else [self.path]:
            assert os.path.exists(check_path), f"{check_path} does not exist"
        with self._data() as q:
            cl_max = self.datapoint_extractor.compute_cl(q["X0"].shape, q["Y0"].shape)
        assert cl_max == self.cl_max, f"{cl_max} != {self.cl_max}"

    def _data(self):
        return multi_h5_dataset(
            self.path,
            equalize_sizes_by_subsampling=self.equalize_sizes_by_subsampling,
            batch_indices_by_prefix={
                self.symbol_for_length: self.datapoint_extractor.shape_offset()
            },
        )

    def __len__(self):
        length = 0
        with self._data() as data:
            for i in range(len(self.length_each(data))):
                Y = data[f"{self.symbol_for_length}{i}"]
                sl_batch = (
                    self.datapoint_extractor.sl_axis(Y.shape) // self.sl
                    if self.sl is not None
                    else 1
                )
                length += self.datapoint_extractor.batch_axis(Y.shape) * sl_batch
        return length

    def length_each(self, data):
        lends = len([k for k in data.keys() if self.symbol_for_length in k])
        return [
            self.datapoint_extractor.batch_axis(
                data[self.symbol_for_length + str(i)].shape
            )
            for i in range(lends)
        ]

    def data_for_chunk(self, data, i):
        return {k: v[:] for k, v in data.items() if k[1:] == str(i)}

    def cached_data(self, data):
        return {k: v[:] for k, v in data.items()}

    def run_with_data_generator(self, fn_accepting_data):
        with self._data() as data:
            yield from fn_accepting_data(data)

    def clip(self, *, value, is_output):
        if not is_output:
            value = clip_datapoint(value, CL=self.cl, CL_max=self.cl_max)
        return modify_sl(value, self.sl, self.cl, is_output=is_output)


def clip_datapoint(x, *, CL, CL_max):
    """
    Remove excess context from each side of the datapoint.

    Args:
        x: datapoint (L + CL_max, C)
        CL: context length to use
        CL_max: context length of the underlying data
    Returns:
        out: datapoint (L + CL, C)
    """
    clip = (CL_max - CL) // 2
    if clip == 0:
        return x
    assert clip > 0
    return x[clip:-clip]


def modify_sl(value, SL, CL, *, is_output):
    """
    Modify the sequence length of the given datapoint.

    If it is an output, then we ensure we keep context on both sides of the sequence.

    Args:
        value: datapoint (SL_original + CL, C) if it is an input else (SL_original, C)
        SL: sequence length to target
        CL: context length to use
        is_output: whether the datapoint is an output

    Returns:
        out: list of datapoints (SL + CL, C) if it is an input else (SL, C)
    """
    actual_sl = value.shape[0] if is_output else value.shape[0] - CL

    if SL is None or actual_sl < SL:
        return [value]
    assert actual_sl % SL == 0, "the SL must be a factor of the SL from the data"
    chunks = actual_sl // SL
    values = []
    for c in range(chunks):
        start, end = c * SL, (c + 1) * SL
        values.append(value[start : (end if is_output else end + CL)])
    return values


def multi_h5_dataset(path, **kwargs):
    """
    If path is a string, then it just returns an h5py.File object.
    If path is a list, then it returns a MultiH5File object. See
        the documentation of MultiH5File for more information.
    """
    if isinstance(path, list):
        return MultiH5File(path, **kwargs)
    assert isinstance(path, str)
    return h5py.File(path, "r")
