import unittest

from modular_splicing.data_pipeline.chunk_iteration_order import (
    dataset_indices_generator,
)


class TestChunkIterationOrder(unittest.TestCase):
    def test_even_chunks(self):
        self.assertEqual(
            list(dataset_indices_generator(9, 3)),
            [
                (0, [0, 1, 2]),
                (1, [3, 4, 5]),
                (2, [6, 7, 8]),
            ],
        )
        self.assertEqual(
            list(dataset_indices_generator(10, 5)),
            [
                (0, [0, 1, 2, 3, 4]),
                (1, [5, 6, 7, 8, 9]),
            ],
        )
        self.assertEqual(
            list(dataset_indices_generator(10, 10)),
            [
                (0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            ],
        )
        self.assertEqual(
            list(dataset_indices_generator(10, 1)),
            [
                (0, [0]),
                (1, [1]),
                (2, [2]),
                (3, [3]),
                (4, [4]),
                (5, [5]),
                (6, [6]),
                (7, [7]),
                (8, [8]),
                (9, [9]),
            ],
        )

    def test_extra_at_end(self):
        self.assertEqual(
            list(dataset_indices_generator(9, 4)),
            [
                (0, [0, 1, 2, 3]),
                (1, [4, 5, 6, 7, 8]),
            ],
        )

        self.assertEqual(
            list(dataset_indices_generator(10, 4)),
            [
                (0, [0, 1, 2, 3]),
                (1, [4, 5, 6, 7, 8, 9]),
            ],
        )

    def test_smaller_than_chunk(self):
        self.assertEqual(
            list(dataset_indices_generator(3, 4)),
            [
                (0, [0, 1, 2]),
            ],
        )

        self.assertEqual(
            list(dataset_indices_generator(0, 5)),
            [(0, [])],
        )
