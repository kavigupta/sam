import torch
import torch.nn as nn
from modular_splicing.models.lssi import AsymmetricConv

from modular_splicing.models.modules.lssi_in_model import (
    LSSI_MODEL_THRESH,
    load_individual_lssi_model,
)

from modular_splicing.utils.construct import construct


class MultipleSplicepointModels(nn.Module):
    """
    Represents a set of motifs that are all alternate interpretations
    of the splicepoint.

    Specifically, if interpreted as log-probabilities, the probabilities
    of each subcomponent here should add up to the LSSI probability for the
    requested splicepoint.

    Sometimes this will be below -10 in which case the motif value will
    probably be clamped to the equivalent of -10 by a later sparsity layer.
    This is fine, as that indicates that the particular split is not useful.
    This model can also be used without sparsity.
    """

    def __init__(
        self,
        *,
        input_size,
        channels,
        num_motifs,
        splicepoint_model_path,
        splicepoint_model_channel,
        splicepoint_splitter_spec
    ):
        super().__init__()
        assert input_size == 4
        self.num_motifs = num_motifs
        self.splicepoint_model = load_individual_lssi_model(
            splicepoint_model_path, trainable=False
        )
        self.splicepoint_model_channel = splicepoint_model_channel
        asym_layer = self.splicepoint_model.conv_layers[0]
        self.splicepoint_splitter = construct(
            dict(
                AsymmetricLinearSplicepointSplitter=AsymmetricLinearSplicepointSplitter
            ),
            splicepoint_splitter_spec,
            input_size=input_size,
            num_motifs=num_motifs,
            channels=channels,
            left=asym_layer.left,
            right=asym_layer.right,
        )

    def forward(self, x):
        if isinstance(x, dict):
            x = x["x"]

        # keep it as (N, L, 1) so that it can be multiplied and added to
        # the other motifs
        splicepoint = self.splicepoint_model(x).log_softmax(-1)[
            :, :, [self.splicepoint_model_channel]
        ]
        splicepoint = torch.maximum(
            splicepoint,
            torch.tensor(LSSI_MODEL_THRESH).to(splicepoint.device),
        )
        to_split = splicepoint > LSSI_MODEL_THRESH

        split = self.splicepoint_splitter(x)
        # when splicepoint is not present, suppress the split
        # value to avoid leakage of information
        split = split * to_split.float()

        split = split.log_softmax(-1)
        result = split + splicepoint

        # assert (torch.abs(result.exp().sum(-1) - splicepoint.exp().sum(-1)) < 1e-5).all()
        return result

    def notify_sparsity(self, sparsity):
        pass

    def get_filters(self):
        return self.splicepoint_splitter.get_filters()


class AsymmetricLinearSplicepointSplitter(nn.Module):
    """
    Represents an asymmetric linear function applied to split up the splicepoints
    """

    def __init__(self, *, input_size, num_motifs, channels, left, right):
        super().__init__()
        del channels
        self.asymm_conv = AsymmetricConv(
            in_channels=input_size,
            out_channels=num_motifs,
            cl=max(left, right) * 2,
            left=left,
            right=right,
        )
        self.asymm_conv.clipping = "none"

    def forward(self, x):
        # x and return is (batch, seq, channels)
        return self.asymm_conv(x.transpose(1, 2)).transpose(1, 2)

    def get_filters(self):
        return self.asymm_conv.conv.weight.detach().cpu().numpy()
