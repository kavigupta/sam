from modular_splicing.models_for_testing.models_for_testing import (
    EndToEndModelsForTesting,
    STANDARD_DENSITY,
)

gtex_multi_aggregator = EndToEndModelsForTesting(
    name="GTEx multi-aggregator",
    path_prefix="model/msp-307a3",
    seeds=(1, 2, 3),
    density=STANDARD_DENSITY,
)

gtex_pre_sparse = EndToEndModelsForTesting(
    name="GTEx pre-sparse",
    path_prefix="model/msp-308a3",
    seeds=(1, 2, 3),
    density=STANDARD_DENSITY,
)

gtex_post_sparse = EndToEndModelsForTesting(
    name="GTEx post-sparse",
    path_prefix="model/msp-309a3",
    seeds=(1, 2, 3),
    density=STANDARD_DENSITY,
)

gtex_just_whole_blood = EndToEndModelsForTesting(
    name="GTEx just whole blood",
    path_prefix="model/msp-317a1",
    seeds=(2, 3),
    density=STANDARD_DENSITY,
)

gtex_models = [gtex_pre_sparse, gtex_post_sparse, gtex_just_whole_blood]
