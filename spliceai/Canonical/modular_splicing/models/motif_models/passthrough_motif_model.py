import torch.nn as nn


class PassthroughMotifsFromData(nn.Module):
    """
    Pass through motifs from the input.
    """

    _input_is_dictionary = True
    motif_model_dict = False

    def __init__(self, input_size, num_motifs, channels):
        super().__init__()
        del channels
        self.input_size = input_size
        self.num_motifs = num_motifs
        self.processor = nn.Identity()

    def forward(self, input_dict):
        motifs = input_dict["motifs"].float()
        if hasattr(self, "processor") and not isinstance(self.processor, nn.Identity):
            assert motifs.shape[-1] == self.input_size
            motifs = self.processor(motifs)
        assert motifs.shape[-1] == self.num_motifs
        return motifs

    def notify_sparsity(self, sparsity):
        pass
