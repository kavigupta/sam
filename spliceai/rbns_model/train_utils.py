import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualUnit(nn.Module):
    """
    Residual unit proposed in "Identity mappings in Deep Residual Networks"
    by He et al.
    """

    def __init__(self, l1, l2, w):
        super().__init__()
        self.normalize1 = nn.BatchNorm1d(l1)
        self.normalize2 = nn.BatchNorm1d(l2)
        self.act1 = self.act2 = nn.ReLU()

        padding = ((w - 1)) // 2

        self.conv1 = nn.Conv1d(l1, l2, w, dilation=1, padding=padding)
        self.conv2 = nn.Conv1d(l1, l2, w, dilation=1, padding=padding)

    def forward(self, input_node):
        bn1 = self.normalize1(input_node)
        act1 = self.act1(bn1)
        conv1 = self.conv1(act1)
        assert conv1.shape == act1.shape
        bn2 = self.normalize2(conv1)
        act2 = self.act2(bn2)
        conv2 = self.conv2(act2)
        assert conv2.shape == act2.shape
        output_node = conv2 + input_node
        return output_node


class MotifModel_simplecnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 500, 20)
        self.act = nn.ReLU()
        self.linear = nn.Linear(500, 2)
    
    def forward(self, input):
        input = input.transpose(1, 2) # BxLxC -> BxCxL

        conv = self.conv1(input)
        conv = torch.squeeze(conv)
        conv = self.act(conv)
        output = self.linear(conv)

        return output


class MotifModel_cnn(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.conv1 = nn.Conv1d(4, hidden_size, kernel_size=5, padding=((5 - 1) // 2))
        self.norm1 = nn.BatchNorm1d(hidden_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 20)
        self.norm2 = nn.BatchNorm1d(hidden_size)
        self.linear = nn.Linear(hidden_size, 2)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, input):
        print(f"input: {input.shape}")
        input = input.transpose(1, 2) # BxLxC -> BxCxL

        x = self.conv1(input)
        x = self.norm1(x)
        x = self.act(x)
        # x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        # x = self.dropout(x)

        print(x.shape)
        x = torch.squeeze(x)
        x = self.linear(x)

        return x


class MotifModel_cnn_dropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 500, kernel_size=5, padding=((5 - 1) // 2))
        self.conv2 = nn.Conv1d(500, 500, 20)
        self.linear = nn.Linear(500, 2)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, input):
        input = input.transpose(1, 2) # BxLxC -> BxCxL

        x = self.conv1(input)
        x = self.act(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.dropout(x)

        x = torch.squeeze(x)
        x = self.linear(x)

        return x


class MotifModel_lstm(nn.Module):

    def __init__(self):
        super(MotifModel_lstm, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=5, padding=((5-1)//2))
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=((5-1)//2))
        self.act = nn.ReLU()
        self.residualunit1 = ResidualUnit(64, 64, 11)
        self.residualunit2 = ResidualUnit(64, 64, 11)
        self.residualunit3 = ResidualUnit(64, 64, 11)

        self.rnn = nn.LSTM(64, 64, num_layers=10)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.output = nn.Linear(64, 2)

    def forward(self, input): # X, concentration
        input = input.transpose(1, 2) # BxLxC -> BxCxL

        conv = self.conv1(input)
        conv = self.act(conv)
        conv = self.conv2(conv)
        conv = self.act(conv)
        conv = self.residualunit1(conv)
        conv = self.residualunit2(conv)
        conv = self.residualunit3(conv)
        conv = conv.permute(2, 0, 1) # -> LxBxC
        conv, (hn, cn) = self.rnn(conv)
        conv = conv[-1, :, :]
        conv = self.output(conv)
        conv = self.logsoftmax(conv) # conv.logsoftmax(dim=1)
        output = conv

        return output

