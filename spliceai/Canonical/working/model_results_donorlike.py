from modular_splicing.models_for_testing.models_for_testing import (
    EndToEndModelsForTesting,
    STANDARD_DENSITY,
)

am_ad_sparse = EndToEndModelsForTesting(
    name="AM + adjusted donor",
    path_prefix="model/msp-311a1",
    seeds=(1, 2, 3),
    density=STANDARD_DENSITY,
)

am_ad_nonsparse = EndToEndModelsForTesting(
    name="AM + adjusted donor + no-sparse",
    path_prefix="model/msp-311b2",
    seeds=(1, 2, 3),
    density=STANDARD_DENSITY,
)

am_aa_sparse = EndToEndModelsForTesting(
    name="AM + adjusted acceptor",
    path_prefix="model/msp-311c1",
    seeds=(1, 2, 3),
    density=STANDARD_DENSITY,
)

am_aa_nonsparse = EndToEndModelsForTesting(
    name="AM + adjusted acceptor + no-sparse",
    path_prefix="model/msp-311d1",
    seeds=(1, 2, 3),
    density=STANDARD_DENSITY,
)

am_ad_sparse_tandem = EndToEndModelsForTesting(
    name="AM + adjusted donor + tandem",
    path_prefix="model/msp-311aa1",
    seeds=(1, 2, 3),
    density=STANDARD_DENSITY,
)

am_ad_nonsparse_tandem = EndToEndModelsForTesting(
    name="AM + adjusted donor + no-sparse + tandem",
    path_prefix="model/msp-311ba1",
    seeds=(1, 2, 3),
    density=STANDARD_DENSITY,
)

am_aa_sparse_tandem = EndToEndModelsForTesting(
    name="AM + adjusted acceptor + tandem",
    path_prefix="model/msp-311ca1",
    seeds=(1, 2, 3),
    density=STANDARD_DENSITY,
)

am_aa_nonsparse_tandem = EndToEndModelsForTesting(
    name="AM + adjusted acceptor + no-sparse + tandem",
    path_prefix="model/msp-311da1",
    seeds=(1, 2, 3),
    density=STANDARD_DENSITY,
)

am_adjusted_donor = [am_ad_sparse, am_ad_nonsparse, am_aa_sparse, am_aa_nonsparse]
am_adjusted_donor_tandem = [
    am_ad_sparse_tandem,
    am_ad_nonsparse_tandem,
    am_aa_sparse_tandem,
    am_aa_nonsparse_tandem,
]
