import unittest
from parameterized import parameterized

import numpy as np
from experiments.msp import starting_point_for_different_number_motifs

from modular_splicing.utils.entropy_calculations import hbern, density_for_same_entropy


class TestDensityForSameEntropy(unittest.TestCase):
    def verify_inverse(self, num_motifs_original, density_original, num_motifs_new):
        try:
            density_new = density_for_same_entropy(
                num_motifs_original, density_original, num_motifs_new
            )
        except TypeError as e:
            self.assertEqual(
                str(e),
                "Domain error: num_motifs_original * hbern(density_original) > num_motifs_new",
            )
            return

        entropy_original = num_motifs_original * hbern(density_original)
        entropy_new = num_motifs_new * hbern(density_new)
        self.assertAlmostEqual(entropy_original, entropy_new, places=5)

    @parameterized.expand([(seed,) for seed in range(1000)])
    def test_density_for_same_entropy(self, seed):
        rng = np.random.RandomState(seed)
        num_motifs_original = rng.randint(1, 100)
        density_original = rng.uniform(0.01, 0.5)
        num_motifs_new = rng.randint(1, 100)
        self.verify_inverse(num_motifs_original, density_original, num_motifs_new)


class TestStartingPoint(unittest.TestCase):
    @parameterized.expand([(seed,) for seed in range(1000)])
    def test_density_for_same_entropy(self, seed):
        rng = np.random.RandomState(seed)
        num_motifs_original = rng.randint(1, 100)
        density_original = rng.uniform(0.01, 0.5)
        num_motifs_new = rng.randint(1, 100)
        sparsity_update = rng.uniform(0.01, 0.9)
        try:
            starting_point = starting_point_for_different_number_motifs(
                num_motifs_original, density_original, num_motifs_new, sparsity_update
            )
        except TypeError as e:
            self.assertEqual(
                str(e),
                "Domain error: num_motifs_original * hbern(density_original) > num_motifs_new",
            )
            return
        densities = starting_point * sparsity_update ** np.arange(100)

        print(densities)

        entropies = num_motifs_new * hbern(densities)

        print(entropies)

        target_entropy = num_motifs_original * hbern(density_original)
        idx = np.abs(entropies - target_entropy).argmin()

        print(entropies[idx], target_entropy)

        self.assertTrue(abs(entropies[idx] - target_entropy) / target_entropy < 1e-5)
