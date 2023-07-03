from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from modular_splicing.psams.motif_types import motifs_types
from modular_splicing.psams.psams import TorchPSAM

from modular_splicing.utils.construct import construct


class PSAMMotifModel(nn.Module):
    """
    Motif model using PSAMs.

    Parameters
    ----------
    motif_spec : dict
        Specification of the motifs to use.
    exclude_names : list of str
        Names of motifs to exclude.
    include_names : list of str
        Names of motifs to include. If provided, `exclude_names` must be empty.
    input_size : int
        Size of the input sequence (4).
    channels : int
        Unused
    num_motifs : int
        Number of motifs. This must be at least the number of motifs in
        `motif_spec`, and the extra motifs will be filled with padding.
    rbns_psam_spec : dict
        Specification of the RbnsPsamMotifs model to use.
    """

    def __init__(
        self,
        motif_spec,
        exclude_names=(),
        *,
        include_names=None,
        input_size,
        channels,
        num_motifs,
        rbns_psam_spec=dict(type="RbnsPsamMotifs"),
    ):
        super().__init__()
        assert input_size == 4
        del channels
        exclude_names = set(exclude_names)
        motifs = construct(motifs_types(), motif_spec)
        if include_names is not None:
            assert exclude_names == set()
            exclude_names = set(motifs) - set(include_names)
        motifs = {k: v for k, v in motifs.items() if k not in exclude_names}
        assert num_motifs >= len(motifs), str((num_motifs, len(motifs)))
        self.padding_motifs = num_motifs - len(motifs)
        self.model = construct(
            dict(RbnsPsamMotifs=RbnsPsamMotifs), rbns_psam_spec, motifs=motifs
        )

    def forward(self, sequence):
        motifs = self.model.get_motifs(sequence)
        motifs = F.pad(motifs, (0, self.padding_motifs))
        return motifs

    def notify_sparsity(self, sparsity):
        pass


class FixedMotifs(ABC, nn.Module):
    """
    Here for legacy related reasons. Only the base
    class for the RbnsPsamMotifs model, no other use.

    The normalization layer created in this class is unused.
    """

    def __init__(self, num_fixed_motifs):
        super().__init__()
        self.norm = nn.BatchNorm1d(num_fixed_motifs)

    @abstractmethod
    def get_motifs(self, sequence):
        pass


class RbnsPsamMotifs(FixedMotifs):
    """
    Represents an RBNS PSAM model. The PSAMs are not trainable.

    Runs them in alphabetical order, in parallel, and then concatenates the
    results.

    Parameters
    ----------
    motifs : dict
        Dictionary of PSAMs to use
    thresholding_mode : str
        How to threshold the PSAMs. One of "none" or "subtractive".
    """

    thresholding_mode = "none"
    _input_is_dictionary = True

    def __init__(self, *, motifs, thresholding_mode="none"):
        names = sorted(motifs)
        super().__init__(len(names))
        self.psams = [TorchPSAM.from_psams(motifs[k]) for k in names]
        self.thresholding_mode = thresholding_mode

    def get_motifs(self, sequence):
        if isinstance(sequence, dict):
            sequence = sequence["x"]
        procd_seq = torch.zeros((*sequence.shape[:-1], len(self.psams))).to(
            sequence.device
        )
        for i, p in enumerate(self.psams):
            procd_seq[:, :, i] = p.process(sequence.transpose(1, 2))
        if self.thresholding_mode == "none":
            pass
        elif self.thresholding_mode == "subtractive":
            procd_seq = F.relu(procd_seq - 1)
        else:
            raise RuntimeError(f"Thresholding mode {self.thresholding_mode} is invalid")
        return procd_seq
