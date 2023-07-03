from modular_splicing.utils.entropy_calculations import density_for_same_entropy
from .models_for_testing import EndToEndModelsForTesting, STANDARD_DENSITY

# all of these use aat
AM_21_rbns = EndToEndModelsForTesting(
    name="AM 21 [rbns]",
    path_prefix=f"model/msp-265a1",
    seeds=(1, 2, 3),
    density=STANDARD_DENSITY,
)

AM_21_rna_compete = EndToEndModelsForTesting(
    name="AM 21 [rnacompete]",
    path_prefix=f"model/msp-279aa1",
    seeds=(1, 2, 3),
    density=density_for_same_entropy(79, STANDARD_DENSITY, 146),
)

AM_21_hybrid = EndToEndModelsForTesting(
    name="AM 21 [hybrid]",
    path_prefix=f"model/msp-279ba1",
    seeds=(1, 2, 3),
    density=density_for_same_entropy(79, STANDARD_DENSITY, 100),
)
