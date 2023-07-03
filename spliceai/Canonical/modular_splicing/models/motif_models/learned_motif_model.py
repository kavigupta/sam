import torch.nn as nn

from modular_splicing.utils.construct import construct

from modular_splicing.models.modules.residual_unit_stack import ResidualStack
from modular_splicing.models.modules.fc_motif_feature_extractor import (
    FCMotifFeatureExtractor,
    get_fc_block,
)


class LearnedMotifModel(nn.Module):
    """
    Represents a purely neural model of motifs. This can be used
        standalone or as part of an adjusted motif model.

    Shape: (N, L, input_size) -> (N, L, num_motifs)

    Parameters
    ----------
    input_size : int
        The number of input channels.
    channels : int
        The number of channels in the motif feature extractor.
    motif_width : int
        The width of the motif feature extractor.
    motif_feature_extractor_spec : dict
        The specification of the motif feature extractor.
    motif_fc_layers : int
        The number of fully connected layers in the motif feature reprocessor.
    num_motifs : int
        The number of motifs (size of the output).
    """

    motif_model_dict = False
    _input_is_dictionary = True

    def __init__(
        self,
        *,
        input_size,
        channels,
        motif_width,
        motif_feature_extractor_spec,
        motif_fc_layers,
        num_motifs,
    ):
        super().__init__()
        self.motif_feature_extractor = construct(
            dict(
                ResidualStack=ResidualStack,
                FCMotifFeatureExtractor=FCMotifFeatureExtractor,
            ),
            motif_feature_extractor_spec,
            input_channels=input_size,
            hidden_channels=channels,
            width=motif_width,
        )
        self.motif_feature_reprocessor = get_fc_block(motif_fc_layers, channels)
        self.reorient = nn.Linear(channels, num_motifs)
        # just for backwards compatibility
        self.residual_motifs = NoResidualMotifs()

    def forward(self, sequence):
        # just for checking backwards compatibility
        assert isinstance(self.residual_motifs, NoResidualMotifs)
        if isinstance(sequence, dict):
            sequence = sequence["x"]
        x = sequence
        x = self.motif_feature_extractor(x)
        x = x.transpose(1, 2)

        x = self.motif_feature_reprocessor(x)

        x = x.permute(2, 0, 1)
        x = self.reorient(x)
        x = x.transpose(0, 1)
        return x

    def notify_sparsity(self, sparsity):
        pass


class NoResidualMotifs(nn.Module):
    pass
