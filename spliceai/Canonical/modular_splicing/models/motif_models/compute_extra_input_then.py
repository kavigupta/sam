import torch.nn as nn

from modular_splicing.utils.construct import construct

from modular_splicing.models.motif_models.types import motif_model_types


class ComputeExtraInputThen(nn.Module):
    """
    Computes a new input channel, then passes it to the next module.

    Effectively, runs the `compute` model on the input (using the input channel if specified, else "x"),
    then passes the result to the `then` model, as the `output_channel` channel. The result
    of the `then` model is returned.

    Parameters
    ----------
    input_channel : str, optional
        The input channel to use. If None, uses "x".
    output_channel : str
        The channel to pass the output of `comput` to `then` via.
    compute_spec : dict
        The specification for the `compute` model.
    then_spec : dict
        The specification for the `then` model.
    **kwargs
        Additional arguments to pass when constructing the `then` model.
    """

    _input_is_dictionary = True

    def __init__(
        self, *, input_channel=None, output_channel, compute_spec, then_spec, **kwargs
    ):
        super().__init__()
        self.compute = construct(motif_model_types(), compute_spec)
        self.then = construct(motif_model_types(), then_spec, **kwargs)
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.motif_model_dict = self.then.motif_model_dict
        assert self.compute._input_is_dictionary
        assert self.then._input_is_dictionary

    def forward(self, x, **kwargs):
        for_compute = x.copy()
        if self.input_channel is not None:
            for_compute["x"] = x[self.input_channel]
        computed = self.compute(for_compute, **kwargs)
        if self.compute.motif_model_dict:
            computed = computed["motifs"]
        x[self.output_channel] = computed
        return self.then(x, **kwargs)

    def notify_sparsity(self, sparsity):
        self.then.notify_sparsity(sparsity)
