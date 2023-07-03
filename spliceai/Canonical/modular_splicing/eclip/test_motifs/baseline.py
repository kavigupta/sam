from modular_splicing.models_for_testing.list import FM, FM_eclip_18
from modular_splicing.utils.io import load_model


fm_baselines_nonbinarized = {
    "rbns": FM,
    "eclip_18": FM_eclip_18,
}


def fm_baseline_nonbinarized(motif_names_source):
    fm_model = fm_baselines_nonbinarized[motif_names_source]
    fm_baseline = fm_model.non_binarized_model(seed=1, density_override=0.75).model
    return fm_baseline
