import unittest
import numpy as np

import torch

from modular_splicing.models.lssi import AsymmetricConv, SplicePointIdentifier
from parameterized import parameterized


class AsymmetricConvWindowTest(unittest.TestCase):
    @parameterized.expand(
        [(left, right) for left in (2, 4, 6, 8) for right in (2, 4, 6, 8)]
    )
    def test_asymmetric_conv_window(self, l, r):
        m = AsymmetricConv(4, 10, 50, left=l, right=r)
        m.clipping = "none"
        x = torch.randn(1, 4, 1000, requires_grad=True)
        m(x)[0, :, 500].sum().backward()
        self.assertEqual(
            np.where(x.grad.numpy()[0].any(0))[0].tolist(),
            list(range(500 - l, 500 + r + 1)),
        )


class LSSIConvWindowTest(unittest.TestCase):
    @parameterized.expand(
        [(left, right) for left in (2, 4, 6, 8) for right in (2, 4, 6, 8)]
    )
    def test_asymmetric_conv_window(self, l, r):
        m = SplicePointIdentifier(50, (l, r), 100)
        m.conv_layers[0].clipping = "none"
        x = torch.randn(1, 1000, 4, requires_grad=True)
        m(x)[0, 500, 0].sum().backward()
        self.assertEqual(
            np.where(x.grad.numpy()[0].any(1))[0].tolist(),
            list(range(500 - l, 500 + r + 1)),
        )
