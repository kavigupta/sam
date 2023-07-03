import unittest

from modular_splicing.models.modules.sparsity.spatially_sparse_across_channels_drop_motifs import (
    SpatiallySparseAcrossChannelsDropMotifs,
)


class SparseDropMotifsTest(unittest.TestCase):
    def testInitialDoesntDrop(self):
        sparse = SpatiallySparseAcrossChannelsDropMotifs(
            0.75, num_channels=20, sparse_drop_motif_frequency=1
        )
        self.assertFalse(sparse.enough_dropped())
        sparse = SpatiallySparseAcrossChannelsDropMotifs(
            0.75, num_channels=20, sparse_drop_motif_frequency=1 - 1e-10
        )
        self.assertTrue(sparse.enough_dropped())

    def testDropsInTandem(self):
        sparse = SpatiallySparseAcrossChannelsDropMotifs(
            0.75, num_channels=20, sparse_drop_motif_frequency=0.5
        )
        self.assertTrue(sparse.enough_dropped())
        sparse.update_sparsity(0.5)
        self.assertTrue(sparse.enough_dropped())
        self.assertEqual(len(sparse.dropped), 1)
        sparse.update_sparsity(0.5)
        self.assertTrue(sparse.enough_dropped())
        self.assertEqual(len(sparse.dropped), 2)

    def testDropsFaster(self):
        sparse = SpatiallySparseAcrossChannelsDropMotifs(
            0.75, num_channels=20, sparse_drop_motif_frequency=0.5
        )
        self.assertTrue(sparse.enough_dropped())
        sparse.update_sparsity(0.25)
        self.assertTrue(sparse.enough_dropped())
        self.assertEqual(len(sparse.dropped), 2)
        sparse.update_sparsity(0.25)
        self.assertTrue(sparse.enough_dropped())
        self.assertEqual(len(sparse.dropped), 4)

    def testDropsSlower(self):
        sparse = SpatiallySparseAcrossChannelsDropMotifs(
            0.75, num_channels=20, sparse_drop_motif_frequency=0.25
        )
        self.assertTrue(sparse.enough_dropped())
        sparse.update_sparsity(0.5)
        self.assertTrue(sparse.enough_dropped())
        self.assertEqual(len(sparse.dropped), 0)
        sparse.update_sparsity(0.5)
        self.assertTrue(sparse.enough_dropped())
        self.assertEqual(len(sparse.dropped), 1)
        sparse.update_sparsity(0.5)
        self.assertTrue(sparse.enough_dropped())
        self.assertEqual(len(sparse.dropped), 1)
        sparse.update_sparsity(0.5)
        self.assertTrue(sparse.enough_dropped())
        self.assertEqual(len(sparse.dropped), 2)
