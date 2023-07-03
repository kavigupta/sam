import torch
import torch.nn as nn

from modular_splicing.utils.construct import construct

from modular_splicing.models.motif_models.types import motif_model_types


class UseInputsIn(nn.Module):
    """
    Use the given channels as inputs in the given underlying motif model.

    This is useful when we want to concatenate several channels together
    and then use them as inputs to a model.

    Parameters
    ----------
    input_channels : list of str
        The channels to use as inputs. Will be concatenated together, in order.
    underlying_model_spec : dict
        The specification for the underlying motif model.
    input_size : int
        The size of the input ("x") to this model.
    new_input_size : int
        The size of the input (the new "x") to this model.
    """

    _input_is_dictionary = True

    def __init__(
        self,
        input_channels,
        underlying_model_spec,
        input_size,
        new_input_size,
        **kwargs
    ):
        del input_size

        super().__init__()
        self.input_channels = input_channels
        self.underlying_model = construct(
            motif_model_types(),
            underlying_model_spec,
            input_size=new_input_size,
            **kwargs
        )
        self.new_input_size = new_input_size
        self.motif_model_dict = self.underlying_model.motif_model_dict
        assert self.underlying_model._input_is_dictionary

    def forward(self, x, **kwargs):
        for_channels = [x[k] for k in self.input_channels]
        for_channels = torch.cat(for_channels, axis=-1)
        assert for_channels.shape[-1] == self.new_input_size
        for_underlying = x.copy()
        for_underlying["x"] = for_channels
        return self.underlying_model(for_underlying, **kwargs)

    def notify_sparsity(self, sparsity):
        self.underlying_model.notify_sparsity(sparsity)
