from modular_splicing.models_for_testing.list import non_binarized_models


def short_donor_models(at_sparsity=0.178e-2):
    """
    Produce the models for the knockdown analysis.

    Output is a dictionary mapping from model name to ([list of models], motif_names_source).
    """

    return {x.name: ([x.model], "rbns") for x in non_binarized_models(at_sparsity)}
