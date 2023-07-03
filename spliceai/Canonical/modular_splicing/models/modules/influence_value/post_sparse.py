import torch.nn as nn
from modular_splicing.models.modules.influence_value.influence_value_calculator import (
    post_sparse_types,
)
from modular_splicing.models.modules.sparsity_propagation import (
    sparsity_propagation_types,
)

from modular_splicing.utils.construct import construct


class ReducedDimensionalityPostSparse(nn.Module):
    def __init__(self, underlying_spec, num_dims, **kwargs):
        super().__init__()
        self.underlying = construct(post_sparse_types(), underlying_spec, **kwargs)
        self.num_dims = num_dims

    def forward(self, x):
        x = self.underlying(x)
        x[:, :, self.num_dims :] = 0
        return x


class SparsityPropagationPostSparse(nn.Module):
    def __init__(self, underlying_spec, sparsity_propagation_spec, **kwargs):
        super().__init__()
        self.underlying = construct(post_sparse_types(), underlying_spec, **kwargs)
        self.sparsity_propagation = construct(
            sparsity_propagation_types(),
            sparsity_propagation_spec,
        )

    def forward(self, x):
        inf = self.underlying(x)
        x = self.sparsity_propagation(inf, x)
        return x


class LinearConvPostSparse(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels, width):
        super().__init__()
        del hidden_channels
        assert width % 2 == 1
        self.conv = nn.Conv1d(
            input_channels, output_channels, width, padding=width // 2
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x
