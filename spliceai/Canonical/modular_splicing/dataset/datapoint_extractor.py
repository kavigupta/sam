from abc import ABC, abstractmethod

import numpy as np
from .data_rewriter import data_rewriter_types

from modular_splicing.utils.construct import construct


class DatapointExtractor(ABC):
    @abstractmethod
    def extract_datapoint(self, data, i, j):
        """
        Get the given data point
        """
        pass

    @abstractmethod
    def shape_offset(self):
        """
        how many extra axes are in the output's shape.
        E.g., this is often 1 for spliceai data because there's
        just an extra axis at the front of the output data.
        """
        pass

    def compute_cl(self, x_shape, y_shape):
        """
        Compute the context length for the given input and output shapes.
        """
        return x_shape[1] - self.sl_axis(y_shape)

    def batch_axis(self, y_shape):
        """
        Compute the value of the batch axis on the given shape.
        """
        return y_shape[self.shape_offset()]

    def sl_axis(self, y_shape):
        """
        Compute the value of the sl axis on the given shape.
        """
        return y_shape[1 + self.shape_offset()]


class BasicDatapointExtractor(DatapointExtractor):
    """
    Extract datapoints and put them into a dictionary, with X values going into
        ["inputs"]["x"] and Y values going into ["outputs"]["y"]

    Args:
        dset: the dataset context
        run_argmax: whether to run argmax on the Y values
        clip_y_zeros: whether to remove the first axis of the Y values. This is
            useful for spliceai data, which has an extra axis at the front.
        rewriters: a list of data rewriters to apply to the datapoints. Will be
            applied in the order given.
    """

    def __init__(self, dset, run_argmax=True, clip_y_zeros=True, rewriters=()):
        self.dset = dset
        self.run_argmax = run_argmax
        self.clip_y_zeros = clip_y_zeros
        self.rewriters = [construct(data_rewriter_types(), rw) for rw in rewriters]

    def extract_datapoint(self, data, i, j):
        x = data[f"X{i}"][j]
        y = data[f"Y{i}"]
        if self.clip_y_zeros:
            y = y[0]
        y = y[j]
        el = dict(inputs=dict(x=x), outputs=dict(y=y))
        for rew in self.rewriters:
            el = rew.rewrite_datapoint(el=el, i=i, j=j, dset=self.dset)
        el["inputs"]["x"] = el["inputs"]["x"].astype(np.float32)
        if self.run_argmax:
            el["outputs"]["y"] = el["outputs"]["y"].argmax(-1)
        return el

    def shape_offset(self):
        return 1 if self.clip_y_zeros else 0


def datapoint_extractor_types():
    from modular_splicing.dataset.multi_setting_h5_file import (
        MultipleSettingDatapointExtractor,
    )
    from modular_splicing.dataset.concatenated_dataset import (
        ConcatenatedDatapointExtractor,
    )

    return dict(
        BasicDatapointExtractor=BasicDatapointExtractor,
        MultipleSettingDatapointExtractor=MultipleSettingDatapointExtractor,
        ConcatenatedDatapointExtractor=ConcatenatedDatapointExtractor,
    )
