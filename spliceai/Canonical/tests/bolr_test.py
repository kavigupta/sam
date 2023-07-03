import unittest

import numpy as np
import torch

from parameterized import parameterized

from modular_splicing.models.modules.activations.backward_only_leaky_relu import (
    BackwardsOnlyLeakyReLU,
)


class TestDensityForSameEntropy(unittest.TestCase):
    @parameterized.expand([(seed,) for seed in range(100)])
    def test_bolr_forward(self, seed):
        np.random.seed(seed)
        torch.manual_seed(np.random.randint(2**32))
        x = torch.randn(3, 7, 29)
        y = BackwardsOnlyLeakyReLU(np.random.randn())(x)
        self.assertTrue(torch.all(y == torch.nn.functional.relu(x)))

    @parameterized.expand([(seed,) for seed in range(100)])
    def test_bolr_backward(self, seed):
        np.random.seed(seed)
        torch.manual_seed(np.random.randint(2**32))
        x = torch.randn(3, 7, 29, requires_grad=True)
        k = np.random.randn()
        y = BackwardsOnlyLeakyReLU(k)(x)
        y.sum().backward()
        self.assertTrue(torch.all(x.grad[x > 0] == 1))
        self.assertTrue(torch.all(x.grad[x <= 0] == k))
