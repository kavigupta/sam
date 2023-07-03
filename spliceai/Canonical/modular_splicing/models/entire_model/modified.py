from modular_splicing.models.modules.sparsity.drop_motifs_in import (
    DropMotifsIn,
)
from modular_splicing.models.modules.sparsity.discretize_motifs_in import (
    DiscretizeMotifsIn,
)
from modular_splicing.utils.io import load_model

from .entire_model_types import entire_model_types

from modular_splicing.utils.construct import construct


def DroppedMotifFromModel(
    *,
    original_model_path,
    original_model_step,
    dropped_motifs,
    input_size=4,
    sparsity=None,
    finetune_motifs=False,
    cl,
):
    """
    Represents a model where we have dropped some motifs from the original model.
    """
    del sparsity  # unused
    return manipulate_sparse_layer(
        input_size,
        lambda layer: DropMotifsIn(layer, dropped_motifs),
        original_model_path,
        original_model_step,
        disable_finetune_motif_model=not finetune_motifs,
    )


def DiscretizeMotifModel(
    *,
    original_model_path,
    original_model_step,
    input_size=4,
    sparsity=None,
    cl,
    reset_adaptive_sparsity_threshold_manager=False,
):
    """
    Represents a model where we have discretized the motifs in the original model.
    """
    del sparsity  # unused
    return manipulate_sparse_layer(
        input_size,
        lambda layer: DiscretizeMotifsIn(layer),
        original_model_path,
        original_model_step,
        disable_finetune_motif_model=True,
        reset_adaptive_sparsity_threshold_manager=reset_adaptive_sparsity_threshold_manager,
    )


def manipulate_sparse_layer(
    input_size,
    manipulator,
    original_model_path,
    original_model_step,
    disable_finetune_motif_model=False,
    reset_adaptive_sparsity_threshold_manager=False,
):
    """
    Manipulate the sparse layer of a model, and return the new model.

    Parameters
    ----------
    input_size : int
        The input size of the model.
    manipulator : function
        A function that takes a sparse layer and returns a new sparse layer.
    original_model_path : str
        The path to the original model. Can also be a dictionary, in which case we
            just use it as a model spec.
    original_model_step : int
        The step of the original model. Must be None if original_model_path is a
            dictionary.
    disable_finetune_motif_model : bool
        If True, we disable finetuning of the motif model.
    reset_adaptive_sparsity_threshold_manager : bool
        If True, we reset the adaptive sparsity threshold manager.
    """
    assert input_size == 4
    if isinstance(original_model_path, str):
        step, underlying_model = load_model(original_model_path, original_model_step)
        assert step == original_model_step
    else:
        assert original_model_step == None
        underlying_model = construct(
            entire_model_types(), original_model_path, input_size=4
        )

    if type(underlying_model).__name__ in {
        "MotifIncrementalDropper",
        "MotifIncrementalAdder",
    }:
        underlying_model = underlying_model.model
    if disable_finetune_motif_model:
        for x in underlying_model.motif_model.parameters():
            x.requires_grad = False
    if hasattr(underlying_model, "_adaptive_sparsity_threshold_manager"):
        underlying_model._adaptive_sparsity_threshold_manager.previous_epoch = 0
        if reset_adaptive_sparsity_threshold_manager:
            del underlying_model._adaptive_sparsity_threshold_manager
    enforcer = underlying_model.sparsity_enforcer
    enforcer.sparse_layer = manipulator(enforcer.sparse_layer)
    return underlying_model
