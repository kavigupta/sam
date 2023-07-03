import unittest

import torch

from modular_splicing.models_for_testing.main_models import AM


class AMTest(unittest.TestCase):
    def test_am(self):
        model = AM.non_binarized_model(1, density_override=0.75).model

        variable = torch.nn.Parameter(torch.rand(1, 100, 4).cuda())
        result = model(variable, collect_intermediates=True)
        result["nonsparse_motifs"][0, 50].sum(-1).backward()
        self.assertEqual(
            torch.where(variable.grad[0].sum(-1) != 0)[0].tolist(),
            list(range(40, 60 + 1)),  # 21 wide
        )

        variable = torch.nn.Parameter(torch.rand(1, 100, 4).cuda())
        result = model(variable, collect_intermediates=True)
        result["pre_adjustment_motifs"][0, 50].sum(-1).backward()
        self.assertEqual(
            torch.where(variable.grad[0].sum(-1) != 0)[0].tolist(),
            list(range(45, 55 + 1)),  # 11 wide
        )
