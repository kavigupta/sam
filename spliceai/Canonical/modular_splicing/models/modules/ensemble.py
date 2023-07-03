import torch
import torch.nn as nn


class ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, *args, **kwargs):
        outputs = [m(*args, **kwargs) for m in self.models]
        return torch.stack(outputs).mean(0)

    @property
    def _input_dictionary(self):
        return self.models[0]._input_dictionary
