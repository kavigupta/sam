import torch.nn as nn

from .residual_unit import HalfResidualUnit, ResidualUnit


class ResidualStack(nn.Module):
    """
    Represents a stack of ResidualUnits. The number of ResidualUnits is
        determined by the `depth` parameter, which can be an integer or a
        half integer. If it is a half integer, the last ResidualUnit is
        replaced by a HalfResidualUnit. The width of the ResidualUnits is
        determined such that the total width of the stack is `width`.

    Shapes: (N, L, C) -> (N, L, C)

    Parameters
    ----------
    input_channels : int, optional
        The number of input channels. If None, no initial convolution is
        performed.
    hidden_channels : int
        The number of channels in the hidden layers of the ResidualUnits.
    output_channels : int, optional
        The number of output channels. If None, no final convolution is
        performed.
    width : int
        The total width of the stack.
    depth : float
        The number of ResidualUnits in the stack. Can be a half integer.
    extra_stack_1x1_layers : int, optional
        The number of 1x1 convolutional layers to add after each
        ResidualUnit.
    """

    def __init__(
        self,
        *,
        input_channels=None,
        hidden_channels,
        output_channels=None,
        width,
        depth,
        extra_stack_1x1_layers=0,
    ):
        super().__init__()
        if input_channels is not None:
            self.initial_conv = nn.Conv1d(input_channels, hidden_channels, 1)
        else:
            self.initial_conv = nn.Identity()
        number_resblocks = depth
        use_half = False
        if depth != int(depth):
            number_resblocks = int(depth)
            use_half = True
        assert depth * 2 == int(depth * 2)
        assert (width - 1) % (depth * 2) == 0
        kernel_size = (width - 1) // (depth * 2) + 1
        kernel_size = int(kernel_size)
        assert kernel_size % 2 == 1
        stack = []
        for _ in range(number_resblocks):
            stack.append(ResidualUnit(l=hidden_channels, w=kernel_size, ar=1))
            stack.extend(
                [
                    ResidualUnit(l=hidden_channels, w=1, ar=1)
                    for _ in range(extra_stack_1x1_layers)
                ]
            )
        if use_half:
            stack += [HalfResidualUnit(l=hidden_channels, w=kernel_size, ar=1)]
            stack += [
                HalfResidualUnit(l=hidden_channels, w=1, ar=1)
                for _ in range(extra_stack_1x1_layers)
            ]
        self.stack = nn.Sequential(*stack)
        if output_channels is not None:
            self.final_conv = nn.Conv1d(hidden_channels, output_channels, 1)
        else:
            self.final_conv = nn.Identity()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.initial_conv(x)
        x = self.stack(x)
        x = self.final_conv(x)
        x = x.transpose(1, 2)
        return x
