import numpy as np
import torch.nn as nn

from modular_splicing.utils.io import load_model


class MotifModelForRobustDownstream(nn.Module):
    motif_model_dict = True

    def __init__(
        self, input_size, channels, num_motifs, models_for_motifs, trainable=False
    ):
        super().__init__()
        models = []
        sparsity_enforcers = []
        for path, step in models_for_motifs:
            _, model = load_model(path, step)
            models.append(model.motif_model)
            sparsity_enforcers.append(model.sparsity_enforcer)
        self.motif_models = nn.ModuleList(models)
        self.sparsity_enforcers = nn.ModuleList(sparsity_enforcers)
        for param in self.parameters():
            param.requires_grad = trainable

    def forward(self, x):
        # always run this module in eval mode
        self.eval()
        index = np.random.choice(len(self.motif_models))
        model = self.motif_models[index]
        sparsity_enforcer = self.sparsity_enforcers[index]
        y = model(x)
        if not getattr(model, "motif_model_dict", False):
            y = dict(motifs=y)
        _, y["motifs"] = sparsity_enforcer(
            y["motifs"],
            splicepoint_results=None,
            manipulate_post_sparse=None,
            collect_intermediates=False,
        )
        return y

    def notify_sparsity(self, sparsity):
        for model in self.models_to_choose_from:
            model.notify_sparsity(sparsity)
