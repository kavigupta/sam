import torch.nn as nn
import torch.nn.functional as F


class BackwardsOnlyLeakyReLU(nn.Module):
    """
    Acts like a ReLU in the forwards direction, but in the backwards
    direction, it acts like a leaky ReLU with a negative slope of
    `slope_on_negatives`.
    """

    def __init__(self, slope_on_negatives):
        super().__init__()
        self.slope_on_negatives = slope_on_negatives

    def forward(self, x):
        x_without_grad = x.detach()

        # value that is zero in real terms but has a gradient of 1 on the backwards pass for negative values
        conditional_zero = F.relu(-x_without_grad) - F.relu(-x)

        # add to the real relu to get the backwards-only relu
        return F.relu(x) + conditional_zero * self.slope_on_negatives
