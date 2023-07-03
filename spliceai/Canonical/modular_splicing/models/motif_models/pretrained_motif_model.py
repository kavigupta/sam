from modular_splicing.utils.io import load_model


def PretrainedMotifModel(
    model_path, model_step, finetunable, num_motifs, input_size, channels
):
    """
    Load a motif model from a saved e2e model.
    """
    del input_size, channels, num_motifs
    assert model_step is not None
    _, entire_model = load_model(model_path, model_step)
    motif_model = entire_model.motif_model
    if not finetunable:
        for param in motif_model.parameters():
            param.requires_grad = False
    return motif_model
