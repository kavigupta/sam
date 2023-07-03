import torch
import torch.nn as nn

from modular_splicing.motif_names import get_motif_names
from modular_splicing.utils.io import load_model


class NeuralFixedMotif(nn.Module):
    """
    Uses a folder full of Neural fixed motifs as a fixed motif model.

    The folder should contain a file for each motif, named by the
        format `path_format`, which should be of the form, e.g.

        path/to/folder/{motif}-rbns

    Each folder will have its last saved model loaded, and the motif
        model will be extracted from it.

    We do not finetune the motifs, but we do add a final reweighting layer
        since they were trained independently and thus don't share the same
        output scale.
    """

    def __init__(
        self, *, num_motifs, path_format, removed_motifs, input_size=4, channels=None
    ):
        super().__init__()

        assert "3P" in removed_motifs and "5P" in removed_motifs
        del channels
        assert input_size == 4
        self.motifs = sorted(set(get_motif_names("rbns")) - set(removed_motifs))
        self.num_motifs = num_motifs
        print(len(self.motifs), self.num_motifs)
        assert self.num_motifs % 2 == 0
        assert self.num_motifs in {len(self.motifs), 1 + len(self.motifs)}
        self.motif_models = nn.ModuleDict({})
        for mot in self.motifs:
            mot_model_path = path_format.format(motif=mot)
            step, model = load_model(mot_model_path)
            assert step is not None, mot
            self.motif_models[mot] = model.motif_model.cpu()
        for param in self.motif_models.parameters():
            param.requires_grad = False
        self.reweighter = nn.BatchNorm1d(self.num_motifs)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["x"]
        out = torch.zeros(x.shape[0], x.shape[1], self.num_motifs, device=x.device)
        for i, k in enumerate(self.motifs):
            out[:, :, i] = self.motif_models[k](x)["motifs"][:, :, 0]
        out = self.reweighter(out.transpose(1, 2)).transpose(1, 2)
        return out

    def notify_sparsity(self, sparsity):
        pass
