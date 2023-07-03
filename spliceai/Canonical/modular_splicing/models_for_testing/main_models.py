from .models_for_testing import EndToEndModelsForTesting, STANDARD_DENSITY, TEN_EPOCHS


FM = EndToEndModelsForTesting(
    name="FM",
    path_prefix="model/msp-273.665a3",
    seeds=(1,),
    density=STANDARD_DENSITY,
    binarized_seeds=(1,),
    binarized_step=TEN_EPOCHS[10],
)

# extra FMs just for the e2e tests
# we don't want to use these for anything related
# to motif analysis because they will all have
# near identical motifs.
FM_full = EndToEndModelsForTesting(
    name="FM",
    path_prefix="model/msp-273.665a3",
    seeds=(1, 2, 3, 4, 5),
    density=STANDARD_DENSITY,
    binarized_seeds=(1, 2, 3, 4, 5),
    binarized_step=TEN_EPOCHS[10],
)

AM = EndToEndModelsForTesting(
    name="AM",
    path_prefix="model/msp-274.790a3",
    seeds=(1, 2, 3, 4, 5),
    density=STANDARD_DENSITY,
    binarized_seeds=(1, 2, 3, 4, 5),
    binarized_step=TEN_EPOCHS[10],
)

FM_sai = EndToEndModelsForTesting(
    name="FM/sai",
    path_prefix="model/msp-276.710a1",
    seeds=(1,),
    density=STANDARD_DENSITY,
    binarized_seeds=(1,),
    binarized_step=TEN_EPOCHS[10],
)

# see FM_full for details
FM_sai_full = EndToEndModelsForTesting(
    name="FM/sai",
    path_prefix="model/msp-276.710a1",
    seeds=(1, 2, 3, 5, 10),
    density=STANDARD_DENSITY,
    binarized_seeds=(1,),
    binarized_step=TEN_EPOCHS[10],
)


AM_sai = EndToEndModelsForTesting(
    name="AM/sai",
    path_prefix="model/msp-275.835a1",
    seeds=(1, 2, 3, 5),
    density=STANDARD_DENSITY,
    binarized_seeds=(1, 2, 3, 5, 7),
    binarized_step=TEN_EPOCHS[10],
)
