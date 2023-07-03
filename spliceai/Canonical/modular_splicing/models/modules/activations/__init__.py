import torch.nn as nn

from .backward_only_leaky_relu import BackwardsOnlyLeakyReLU


def activation_types():
    return dict(ReLU=nn.ReLU, BackwardsOnlyLeakyReLU=BackwardsOnlyLeakyReLU)
