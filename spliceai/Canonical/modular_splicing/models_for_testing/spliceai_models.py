from .models_for_testing import ModelForTestingLastStep


spliceai_400 = [
    ModelForTestingLastStep(
        name_prefix="SpliceAI 400",
        path_prefix="model/standard-400",
        seed=seed,
        before_prefix="-",
    )
    for seed in range(5)
]

spliceai_10k = [
    ModelForTestingLastStep(
        name_prefix="SpliceAI 10k",
        path_prefix="model/standard-10000",
        seed=seed,
        before_prefix="-",
    )
    for seed in range(5)
]
