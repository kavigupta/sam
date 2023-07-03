import torch
import torch.nn as nn
from modular_splicing.models.modules.final_layer.final_processor import FinalProcessor

from modular_splicing.utils.construct import construct
from modular_splicing.models.modules.lssi_in_model import BothLSSIModels


class SplicepointBasedBranchSitePredictor(nn.Module):
    def __init__(
        self,
        *,
        input_size,
        sparsity,
        cl,
        splicepoint_model_spec,
        channels,
        final_processor_spec,
        masker_spec=dict(type="OnlyAllowA", factor=10),
    ):
        super().__init__()

        assert input_size == 4
        del sparsity
        self.splicepoint_model = construct(
            dict(BothLSSIModels=BothLSSIModels), splicepoint_model_spec
        )

        self.upscale = nn.Linear(2, channels)

        self.final_processor = construct(
            dict(FinalProcessor=FinalProcessor),
            final_processor_spec,
            num_motifs=channels,
            channels=channels,
            output_size=4,
        )

        self.masker = construct(dict(OnlyAllowA=OnlyAllowA), masker_spec)
        self.cl = cl

    def forward(self, x, collect_losses=False):
        x = x["x"]
        out, residue = self.splicepoint_model(x)
        out = self.upscale(out)
        out = self.final_processor(out)
        out_nad = out[:, :, :-1]
        out_nad = out_nad + residue
        out_branch = out[:, :, [-1]]
        out_branch = self.masker(x, out_branch)
        out = torch.cat([out_nad, out_branch], dim=-1)
        out = out[:, self.cl // 2 : out.shape[1] - self.cl // 2]
        if collect_losses:
            return {"output": out}
        else:
            return out


class OnlyAllowA(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x, out):
        is_a = x[:, :, [0]]
        is_not_a = 1 - is_a
        correction = is_not_a * -self.factor
        out = out + correction
        return out
