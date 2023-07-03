import unittest

from modular_splicing.gtex_data.pipeline.marginal_psis import related_sites


class RelatednessTest(unittest.TestCase):
    def test_one_step_relatedness(self):
        self.assertEqual(
            related_sites(
                1,
                [{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}],
                relatedness_technique="one_step",
            ),
            {1, 2},
        )
        self.assertEqual(
            related_sites(
                2,
                [{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}],
                relatedness_technique="one_step",
            ),
            {1, 2, 3},
        )
        self.assertEqual(
            related_sites(
                1, [{1, 2}, {1, 3}, {1, 4}, {2, 6}], relatedness_technique="one_step"
            ),
            {1, 2, 3, 4},
        )

    def test_one_step_relatedness_with_tuples(self):
        self.assertEqual(
            related_sites(
                (1, 2),
                [
                    {(1, 2), (2, 3)},
                    {(2, 3), (3, 4)},
                    {(3, 4), (4, 5)},
                    {(4, 5), (5, 6)},
                    {(5, 6), (6, 7)},
                ],
                relatedness_technique="one_step",
            ),
            {(1, 2), (2, 3)},
        )
        self.assertEqual(
            related_sites(
                ("A", 3974598),
                [
                    {("A", 3974598), ("A", 3974599)},
                    {("A", 3974599), ("A", 3974600)},
                    {("A", 3974600), ("A", 3974601)},
                    {("A", 3974601), ("A", 3974602)},
                    {("A", 3974602), ("A", 3974603)},
                ],
                relatedness_technique="one_step",
            ),
            {("A", 3974598), ("A", 3974599)},
        )

    def test_closure_relatedness(self):
        self.assertEqual(
            related_sites(
                1,
                [{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}],
                relatedness_technique="closure",
            ),
            {1, 2, 3, 4, 5, 6},
        )
        self.assertEqual(
            related_sites(
                1,
                [{1, 2}, {1, 3}, {1, 4}, {5, 6}],
                relatedness_technique="closure",
            ),
            {1, 2, 3, 4},
        )
        self.assertEqual(
            related_sites(
                1,
                [{1, 2}, {1, 3}, {1, 4}, {2, 6}],
                relatedness_technique="closure",
            ),
            {1, 2, 3, 4, 6},
        )
