import torch
import torch.nn as nn

from modular_splicing.utils.io import load_model


LSSI_MODEL_THRESH = -10.0


class BothLSSIModels(nn.Module):
    """
    Represents both an acceptor and a donor model. Core LSSI as used
    in the main model.

    Also performs multiple projections, to represent the splicepoint as
        a motif as well as the splicepoint as a final mask.

    Both are computed from a common origin: running each of the models
        on the input separately, concatenating the results, and then
        clipping the results to a minimum value (LSSI_MODEL_THRESH).

    You can set `no_linear_layers` to True to disable the linear layers
        which means you can't run `forward` on this model, but you can
        run `forward_just_splicepoints` on it. This option is to use
        this model standalone since it leads to it having a stable
        hash value.

    Input: (N, L, 4)
    Output: (motif_results, residual_results)
        motif_results: (N, L, 2): the splicepoint as a motif
        residual_results: (N, L, 3): the splicepoint as a mask

    Arguments
    ---------
    acceptor: str
        Path to the acceptor model
    donor: str
        Path to the donor model
    trainable: bool
        Whether to make the model trainable
    no_linear_layers: bool
        Whether to disable the linear layers
    """

    def __init__(
        self,
        acceptor,
        donor,
        *,
        trainable=False,
        no_linear_layers=False,
    ):
        super().__init__()
        self.use_splicepoint = None if no_linear_layers else nn.Linear(2, 3)
        self.splicepoint_transform_motif = None if no_linear_layers else nn.Linear(2, 2)
        self.models = nn.ModuleList(
            [
                load_individual_lssi_model(acceptor, trainable=trainable),
                load_individual_lssi_model(donor, trainable=trainable),
            ]
        )

    def forward(self, input, manipulate_splicepoint_motif=None):
        assert not hasattr(self, "sparsity_technique")
        spl = self.forward_just_splicepoints(input)
        spl = torch.maximum(
            spl,
            torch.tensor(LSSI_MODEL_THRESH).to(spl.device),
        )
        spl_motif = spl_residual = spl
        if manipulate_splicepoint_motif is not None:
            spl_motif = manipulate_splicepoint_motif(spl_motif)
        spl_motif = self.splicepoint_transform_motif(spl_motif)
        spl_residual = self.use_splicepoint(spl_residual)
        return spl_motif, spl_residual

    def forward_just_splicepoints(self, input):
        """
        Compute the splicepoint scores. Computes the splicepoint scores for
        both acceptor and donor sites, and returns them concatenated.
        """
        splicepoint_results = torch.stack(
            [self.models[c](input).log_softmax(-1)[:, :, c + 1] for c in range(2)],
            dim=2,
        )
        return splicepoint_results


class BothLSSIModelsDummy(nn.Module):
    """
    Identical to BothLSSIModels in shapes, but just returns 0.

    Used as a dummy model for the splicepoint model when we don't want
    to use it.
    """

    def forward(self, input):
        return torch.zeros_like(input[:, :, :2]), torch.zeros_like(input[:, :, :3])


def both_lssi_model_types():
    return dict(
        BothLSSIModels=BothLSSIModels,
        BothLSSIModelsDummy=BothLSSIModelsDummy,
    )


def load_individual_lssi_model(path, *, trainable):
    """
    Load the individual LSSI model from the given path

    Parameters
    ----------
    path : str
        The path to the model
    trainable : bool
        Whether the model should be trainable
    """
    _, m = load_model(path)
    m = m.cpu()
    assert trainable in {True, False}

    m.conv_layers[0].clipping = "none"
    if not trainable:
        for p in m.parameters():
            p.requires_grad = False
    return m


class BothLSSIModelsJustSplicepoints(nn.Module):
    """
    Splicepoint model that just returns the splicepoint scores.

    Wrapper around BothLSSIModels that just returns the splicepoint scores,
        and clips them so that the model acts as if it has the given cl.

    Places 0s in the null value, just to ensure that the output is of shape
        (N, L, 3) for consistency with the other models. This means the
        numbers aren't log-normalized, but that's fine for relative
        comparisons.
    """

    @classmethod
    def from_paths(cls, acc, don, cl):
        return cls(BothLSSIModels(acc, don, no_linear_layers=True), cl)

    def __init__(self, splicepoint_model, cl):
        super().__init__()
        self.splicepoint_model = splicepoint_model
        self.cl = cl
        self.version = 2

    def forward(self, inp):
        x = self.splicepoint_model.forward_just_splicepoints(inp)
        clip = self.cl // 2
        x = x[:, clip : x.shape[1] - clip, :]
        x = torch.cat([torch.zeros_like(x)[:, :, :1], x], dim=2)
        return x
