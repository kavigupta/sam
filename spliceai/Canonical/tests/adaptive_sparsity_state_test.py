import unittest

import numpy as np
from torch import nn

from modular_splicing.train.adaptive_sparsity_threshold_manager import (
    AdaptiveSparsityThresholdManager,
)


class AdaptiveSparsityStateTest(unittest.TestCase):
    def test_idemnipotent_setup(self):
        m = nn.Module()
        x = AdaptiveSparsityThresholdManager.setup(
            m, maximal_threshold=90, minimal_threshold=80, decrease_per_epoch=1
        )
        y = AdaptiveSparsityThresholdManager.setup(
            m, maximal_threshold=90, minimal_threshold=80, decrease_per_epoch=1
        )
        self.assertIs(x, y)

    def test_no_max_threshold(self):
        m = nn.Module()
        mgr = AdaptiveSparsityThresholdManager.setup(
            m, maximal_threshold=None, minimal_threshold=80, decrease_per_epoch=1
        )
        epochs = 0
        accuracies = np.random.rand(1000) * 10 + 75
        for acc in accuracies:
            epochs += 0.1
            self.assertEqual(mgr.passes_accuracy_threshold(acc, epochs), acc >= 80)

    def test_decreasing_threshold(self):
        m = nn.Module()
        mgr = AdaptiveSparsityThresholdManager.setup(
            m, maximal_threshold=90, minimal_threshold=80, decrease_per_epoch=1
        )
        self.assertTrue(mgr.passes_accuracy_threshold(89, 1))
        self.assertFalse(mgr.passes_accuracy_threshold(88, 1.1))
        self.assertTrue(mgr.passes_accuracy_threshold(88, 2))

    def test_increasing_threshold(self):
        m = nn.Module()
        mgr = AdaptiveSparsityThresholdManager.setup(
            m, maximal_threshold=90, minimal_threshold=80, decrease_per_epoch=1
        )
        self.assertTrue(mgr.passes_accuracy_threshold(89, 1))
        self.assertFalse(mgr.passes_accuracy_threshold(88, 1.1))
        self.assertTrue(mgr.passes_accuracy_threshold(90, 2))
        self.assertFalse(mgr.passes_accuracy_threshold(88, 2.1))
        self.assertTrue(mgr.passes_accuracy_threshold(99, 3))
        self.assertTrue(mgr.passes_accuracy_threshold(90, 3.1))
