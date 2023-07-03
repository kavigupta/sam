from .models_for_testing import ModelForTestingLastStep

LSSI = [
    ModelForTestingLastStep(
        name_prefix="acceptor",
        path_prefix="model/splicepoint-model-acceptor",
        seed=1,
        before_prefix="-",
    ),
    ModelForTestingLastStep(
        name_prefix="donor",
        path_prefix="model/splicepoint-model-donor",
        seed=1,
        before_prefix="-",
    ),
]

LSSI_EXTRAS = [
    [
        ModelForTestingLastStep(
            name_prefix="acceptor",
            path_prefix="model/msp-262a6",
            seed=i,
            before_prefix="_",
        ),
        ModelForTestingLastStep(
            name_prefix="donor",
            path_prefix="model/msp-262da5",
            seed=i,
            before_prefix="_",
        ),
    ]
    for i in (2, 3, 4, 5)
]
