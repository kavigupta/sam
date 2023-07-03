import torch
import torch.nn as nn
from modular_splicing.models.modules.lssi_in_model import both_lssi_model_types

from modular_splicing.models.modules.spliceai import SpliceAIModule
from modular_splicing.models.motif_models.types import motif_model_types

from modular_splicing.utils.construct import construct


class PreprocessedSpliceAI(nn.Module):
    """
    Used to represent the SpliceAI model as an end-to-end module.

    Includes a preprocessor specification. The preprocessor specification
    can be used to specify a motif model, or to specify that no preprocessor
    by using the Identity preprocessor.

    Can also include a splicepoint model specification, which will be
    concatenated to the output of the preprocessor.
    """

    def __init__(
        self,
        *,
        preprocessor_spec,
        spliceai_spec,
        splicepoint_model_spec=None,
        input_size=4,
        post_processor=4,
        sparsity=None,
        cl,
    ):
        super().__init__()
        assert input_size == 4
        del sparsity  # unused
        self.preprocessor = construct(
            dict(
                Identity=nn.Identity,
                Motifs=lambda motifs_spec: construct(
                    motif_model_types(), motifs_spec, input_size=input_size
                ),
            ),
            preprocessor_spec,
        )
        if splicepoint_model_spec is not None:
            self.splicepoint_model = construct(
                both_lssi_model_types(), splicepoint_model_spec
            )
        else:
            self.splicepoint_model = None
        self.spliceai = construct(
            dict(SpliceAIModule=SpliceAIModule),
            spliceai_spec,
            window=cl,
            input_size=post_processor + (splicepoint_model_spec is not None) * 2,
        )

    @property
    def cl(self):
        return self.spliceai.spliceai.cl

    def forward(self, x, collect_intermediates=False, collect_losses=False):
        masked = self.preprocessor(x)
        if getattr(self, "splicepoint_model", None) is not None:
            splicepoints, _ = self.splicepoint_model(x)
            x = torch.cat([masked, splicepoints], dim=2)
        else:
            x = masked
        if isinstance(x, dict):
            x = x["x"]
        x = self.spliceai(x)
        if collect_intermediates or collect_losses:
            return dict(output=x)
        else:
            return x
