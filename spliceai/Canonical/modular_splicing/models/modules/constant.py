import torch
import torch.nn as nn


class Constant(nn.Module):
    """
    Returns a constant tensor of the specified size, regardless of the input.

    The tensor is a learnable parameter.

    Parameters:
        size: int
            the size of the tensor to return


    Input: any tensor of shape (N1, N2, ..., Nk)
    Output: a tensor of shape (B, size)
    """

    def __init__(self, size):
        super().__init__()
        self.size = size
        self.param = nn.Parameter(torch.zeros(size))

    def __call__(self, x):
        return self.param.expand(*x.shape[:-1], self.size)
