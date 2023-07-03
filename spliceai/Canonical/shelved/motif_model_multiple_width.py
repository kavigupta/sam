import numpy as np
import torch
import torch.nn as nn
from modular_splicing.models.motif_models.types import motif_model_types

from modular_splicing.utils.construct import construct


class MotifModelMultipleWidths(nn.Module):
    motif_model_dict = True

    def __init__(
        self,
        input_size,
        channels,
        num_motifs,
        widths,
        width_to_architecture_spec,
        motif_fc_layers,
    ):
        super().__init__()
        if len(widths) < num_motifs:
            # just pad it out
            widths = widths + [widths[-1]]
        assert len(widths) == num_motifs
        self.num_motifs = num_motifs
        self.widths = widths
        by_width_idx = {width: 0 for width in sorted(set(widths))}
        self.motif_idx_to_within_width_idx = []
        for idx in range(num_motifs):
            self.motif_idx_to_within_width_idx.append(by_width_idx[widths[idx]])
            by_width_idx[widths[idx]] += 1

        width_to_spec = {
            width: construct(
                dict(force_conv_size=force_conv_size),
                width_to_architecture_spec,
                total_width=width,
                motif_fc_layers=motif_fc_layers,
            )
            for width in sorted(set(widths))
        }
        self.motif_models = nn.ModuleDict(
            {
                str(width): construct(
                    motif_model_types(),
                    width_to_spec[width],
                    input_size=input_size,
                    channels=channels,
                    num_motifs=self.widths.count(width),
                )
                for width in sorted(set(widths))
            }
        )

    def forward(self, x):
        per_width_results = {
            width: self.motif_models[str(width)](x) for width in self.widths
        }

        out = torch.zeros(
            (*x["x"].shape[:-1], self.num_motifs),
            dtype=x["x"].dtype,
            device=x["x"].device,
        )
        for width in set(self.widths):
            [idxs] = np.where(np.array(self.widths) == width)
            out[:, :, idxs] = per_width_results[width]
        return dict(motifs=out)

    def notify_sparsity(self, sparsity):
        for model in self.models_to_choose_from:
            model.notify_sparsity(sparsity)


def force_conv_size(conv_size, total_width, motif_fc_layers):
    assert (total_width - 1) % (conv_size - 1) == 0, str((conv_size, total_width))
    num_convs = (total_width - 1) // (conv_size - 1)
    if num_convs % 2 == 0:
        num_convs //= 2
    else:
        num_convs = num_convs / 2
    return dict(
        type="LearnedMotifModel",
        motif_width=total_width,
        motif_fc_layers=motif_fc_layers,
        motif_feature_extractor_spec=dict(type="ResidualStack", depth=num_convs),
    )
