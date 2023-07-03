import torch.nn as nn

from modular_splicing.utils.construct import construct


class ProductSparsityPropagation(nn.Module):
    """
    Propagate the influences via multiplication, which mantains sparisty.
    """

    def forward(self, influence, to_influence):
        return influence * to_influence


class ClearInfluenceOnChannels(nn.Module):
    def __init__(self, channel_idxs, and_then_spec):
        super().__init__()
        self.channel_idxs = channel_idxs
        self.and_then = construct(sparsity_propagation_types(), and_then_spec)

    def forward(self, influence, to_influence):
        influence[:, :, self.channel_idxs] = 0
        return self.and_then(influence, to_influence)


class AddOnlyToNonZero(nn.Module):
    """
    Propagate the influences via addition, but only allow the influence to be
    applied to locations that are not zero.
    """

    def forward(self, influence, to_influence):
        original_values = to_influence
        zero_values = (original_values == 0).all(-1)
        influence = influence.clone()
        influence[zero_values] = 0
        return influence + to_influence


class NoSparsityPropagation(nn.Module):
    """
    Propagate the influences via addition, which does not mantain sparisty.
    """

    def forward(self, influence, to_influence):
        return influence + to_influence


class OnlyAllowInfluenceOnSplicesites(nn.Module):
    """
    Propagate the influences via addition, but only allow the influence to be
    applied to real splicesites.

    Note: not using the -10 bar since it has been passed through a linear
        layer at this point. Instead we use the most common value as the
        zero value that we do not allow to be influenced.
    """

    def forward(self, influence, to_influence):
        original_splice_site_values = to_influence[:, :, :2]
        zero_values, _ = original_splice_site_values.reshape(
            -1, original_splice_site_values.shape[-1]
        ).mode(0)
        valid_locations = (original_splice_site_values != zero_values).any(-1)
        influence = influence.clone()
        influence[~valid_locations] = 0
        influence[:, :, 2:] = 0
        return influence + to_influence


class JustUseInfluence(nn.Module):
    """
    Just use influence values rather than propagating the original value.
    """

    def forward(self, influence, to_influence):
        return influence


def sparsity_propagation_types():
    return dict(
        ProductSparsityPropagation=ProductSparsityPropagation,
        AddOnlyToNonZero=AddOnlyToNonZero,
        ClearInfluenceOnChannels=ClearInfluenceOnChannels,
        NoSparsityPropagation=NoSparsityPropagation,
        OnlyAllowInfluenceOnSplicesites=OnlyAllowInfluenceOnSplicesites,
        JustUseInfluence=JustUseInfluence,
    )
