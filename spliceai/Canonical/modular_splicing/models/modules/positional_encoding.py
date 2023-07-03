import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    from the tutorial https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Input: (L, N, C)
    Output: (L, N, C)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe : (L_max, C)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        cos_vals = torch.cos(position * div_term)
        if d_model % 2 == 1:
            cos_vals = cos_vals[:, :-1]
        pe[:, 1::2] = cos_vals
        # pe : (L_max, C)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe : (L_max, 1, C)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # self.pe[: x.size(0), :] : (L, 1, C)
        # x : (L, N, C)
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
