import attr

import numpy as np
import torch


@attr.s
class PSAM:
    """
    Represent a position-specific matrix, using a numpy array.
    """

    A0 = attr.ib()
    n = attr.ib()
    acc_k = attr.ib()
    acc_shift = attr.ib()
    acc_scale = attr.ib()
    matrix = attr.ib()
    threshold = attr.ib(kw_only=True, default=0.2)


@attr.s
class TorchPSAM:
    """
    Represent a position-specific matrix, in torch.

    Not trainable, just uses tensors.

    Shapes are
        A0s: (1, num_psams, 1)
        M: (num_psams, 4, psam_width)
        threshold: number

    M is stored in log-space but A0s are not.
    """

    A0s = attr.ib()
    M = attr.ib()
    threshold = attr.ib()

    @staticmethod
    def from_psams(motif):
        """
        Convert a PSAM to a TorchPSAM.

        Also does normalization of the A0s (by dividing by the maximum A0 value).
        """
        assert len({m.threshold for m in motif}) == 1
        matrices = [m.matrix for m in motif]
        maximal_length = max([m.shape[0] for m in matrices])
        matrices = [pad_with_ones(m, maximal_length) for m in matrices]
        M = torch.tensor(
            np.array([np.log(m + 1e-100).astype(np.float32) for m in matrices])
        ).transpose(2, 1)
        A0s = np.array([m.A0 for m in motif])
        A0s = A0s / A0s.max()
        A0s = torch.tensor(A0s[None, :, None])
        return TorchPSAM(A0s, M, motif[0].threshold)

    def process(self, x):
        """
        Process the given sequence. The sequence is in the form N x 4 x L.

        Produces an output of shape N x L.

        Each output corresponds to a centered psam at that point. We pad the
            output with 0s on both sides. We also ensure that the output is
            0 on all positions where the center of the psam is an N.
        """
        # x : N x 4 x L
        out = torch.nn.functional.conv1d(x, self.M.to(x.device))
        # out : N x num_psams x (L - w + 1)
        out = out.exp() * self.A0s.to(x.device)
        # out : N x num_psams x (L - w + 1)
        out = out.sum(1)
        # out : N x (L - w + 1)
        out = out / self.threshold
        padding_amount = x.shape[-1] - out.shape[-1]
        pad = [padding_amount // 2, padding_amount - padding_amount // 2]
        out = torch.nn.functional.pad(out, pad=pad)
        # out : N x L
        out = out * (x.sum(1) > 0)
        return out


def pad_with_ones(x, to_size):
    """
    Pad on either side of the psam with uniform 1 values, which are irrelevant to the PSAM.
    """
    pad_amount = to_size - x.shape[0]
    assert pad_amount >= 0
    pad_left = pad_amount // 2
    pad_right = pad_amount - pad_left
    result = np.concatenate([np.ones((pad_left, 4)), x, np.ones((pad_right, 4))])
    return result
