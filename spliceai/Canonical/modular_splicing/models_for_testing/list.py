"""
Contains a list of models for testing.
"""

from .spliceai_models import spliceai_400, spliceai_10k
from .main_models import FM, AM, FM_sai, AM_sai
from .lssi_models import LSSI, LSSI_EXTRAS
from .eclip_models import FM_eclip_18

MAIN_MODELS = [
    FM,
    AM,
    FM_sai,
    AM_sai,
]

# just here to suppress import warnings
models_spliceai = [spliceai_400, spliceai_10k]
models_lssi = [LSSI, LSSI_EXTRAS]
models_eclip = [FM_eclip_18]


def binarized_models():
    return [model for models in MAIN_MODELS for model in models.binarized_models()]


def non_binarized_models(override_density=None):
    return [
        model
        for models in MAIN_MODELS
        for model in models.non_binarized_models(override_density)
    ]
