import numpy as np
import torch.nn as nn

from modular_splicing.utils.construct import construct

from .residual_unit import ResidualUnit


def no_preprocess(x):
    return x, {}


class SpliceAI(nn.Module):
    """
    Original spliceai code; translated from the original tensorflow
        code.
    """

    def __init__(
        self,
        l,
        w,
        ar,
        preprocess=no_preprocess,
        starting_channels=4,
        output_size=3,
        *,
        module_spec=dict(type="ResidualUnit"),
    ):
        super().__init__()
        assert len(w) == len(ar)
        self.w = w
        self.cl = 2 * sum(ar * (w - 1))
        self.conv1 = nn.Conv1d(starting_channels, l, 1)
        self.conv2 = nn.Conv1d(l, l, 1)

        def get_mod(i):
            res = construct(
                dict(ResidualUnit=ResidualUnit), module_spec, l=l, w=w[i], ar=ar[i]
            )
            return res

        self.convstack = nn.ModuleList([get_mod(i) for i in range(len(self.w))])
        self.skipconv = nn.ModuleList(
            [
                nn.Conv1d(l, l, 1) if self._skip_connection(i) else None
                for i in range(len(self.w))
            ]
        )
        self.output = nn.Conv1d(l, output_size, 1)

        self.preprocess = preprocess

        self.hooks_handles = []

    def forward(self, x, collect_intermediates=False, collect_gradients=False):
        if isinstance(x, dict):
            x = x["x"]
        collect = lambda x: x if collect_intermediates else None
        x = x.transpose(1, 2)
        intermediates = {}
        if hasattr(self, "preprocess"):
            x, extras = self.preprocess(x)
            intermediates.update(extras.items())

        conv = self.conv1(x)
        skip = self.conv2(conv)

        intermediates["skips"] = [collect(skip)]

        for i in range(len(self.w)):
            conv = self.convstack[i](conv)

            if self._skip_connection(i):
                # Skip connections to the output after every 4 residual units
                skip = skip + self.skipconv[i](conv)
                intermediates["skips"].append(collect(skip))

        skip = skip[:, :, self.cl // 2 : -self.cl // 2]

        y = self.output(skip)

        y = y.transpose(1, 2)
        if collect_gradients:
            [s.retain_grad() for s in intermediates["skips"]]

        if collect_intermediates:
            intermediates["output"] = y
            return intermediates
        return y

    def _skip_connection(self, i):
        return ((i + 1) % 4 == 0) or ((i + 1) == len(self.w))


class SpliceAIModule(nn.Module):
    """
    Wrapper around the original spliceai module to make it
        compatible with the rest of the codebase.

    Only accepts windows of 80, 400, 2000, or 10000.
    """

    def __init__(
        self,
        *,
        L=32,
        window,
        CL_max=10_000,
        input_size=4,
        output_size=3,
        spliceai_spec=dict(type="SpliceAI"),
    ):
        super().__init__()
        W, AR, _, _ = get_hparams(window, CL_max=CL_max)
        self.spliceai = construct(
            dict(SpliceAI=SpliceAI),
            spliceai_spec,
            l=L,
            w=W,
            ar=AR,
            starting_channels=input_size,
            output_size=output_size,
        )

    def forward(self, x):
        return self.spliceai(x)


def get_hparams(window, CL_max):
    """
    Get hyperparameters, based on the window size.

    This is based on the original SpliceAI code.
    """
    if window == 80:
        W = np.asarray([11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1])
        BATCH_SIZE = 18
    elif window == 400:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
        BATCH_SIZE = 18
    elif window == 2000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11, 21, 21, 21, 21])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10])
        BATCH_SIZE = 12
    elif window == 10000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11, 21, 21, 21, 21, 41, 41, 41, 41])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10, 25, 25, 25, 25])
        BATCH_SIZE = 6
    else:
        raise AssertionError("Invalid window: {}".format(window))

    # Hyper-parameters:
    # L: Number of convolution kernels
    # W: Convolution standard_args.window size in each residual unit
    # AR: Atrous rate in each residual unit

    CL = 2 * np.sum(AR * (W - 1))
    assert CL <= CL_max and CL == window

    return W, AR, BATCH_SIZE, CL
