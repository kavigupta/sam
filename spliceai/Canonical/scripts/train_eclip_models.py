from modular_splicing.eclip.trained_on_eclip.train import (
    eclip_model_from_name,
    model_specs,
)
from modular_splicing.eclip.test_motifs.names import get_testing_names


for seed in (1, 2, 3):
    for motif in get_testing_names(motif_names_source="eclip_18").common_names:
        for name in model_specs:
            print(seed, motif, name)
            eclip_model_from_name(name, motif, seed=seed)
