"""
This class is just a regression test to ensure that the models can be loaded.

It does nothing if you haven't trained any models yet.
"""

import unittest
import os

import torch

from parameterized import parameterized

from modular_splicing.utils.io import load_model

if os.path.exists("model"):
    models = [f"model/{x}" for x in os.listdir("model")]
else:
    models = []

models = sorted(models)
models = [x for x in models if os.path.isdir(x)]


class EveryModelLoadable(unittest.TestCase):
    @parameterized.expand([(x,) for x in models], skip_on_empty=True)
    def test_every_model_loadable(self, name):
        load_model(name, map_location=torch.device("cpu"))
