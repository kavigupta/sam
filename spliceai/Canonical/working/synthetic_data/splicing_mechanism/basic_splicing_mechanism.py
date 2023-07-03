import attr

import numpy as np
import torch

from .splicing_mechanism import SplicingMechanism, log


@attr.s
class BasicEffect:
    """
    Represents the effect of a motif on a splice site, which is constant
        for short distances and then zero for long distances.
    """

    motif_idx = attr.ib()
    site_idx = attr.ib()
    effect = attr.ib()
    start_disp = attr.ib()
    end_disp = attr.ib()
    highpoint = attr.ib()  # only choices are left and right

    def process(self, motifs):
        input_idx = self.motif_idx + 2
        filt = (
            torch.linspace(
                0, 1, self.end_disp - self.start_disp + 1, dtype=motifs.dtype
            )
            * self.effect
        )
        if self.highpoint == "left":
            filt = filt.flip(0)
        x = motifs[:, input_idx]
        x = torch.nn.functional.conv1d(x[None, None], filt[None, None])[0, 0]
        x = torch.cat([torch.zeros(-self.start_disp), x, torch.zeros(self.end_disp)])
        return x

    def effect_applies_to_disp(self, disp):
        # linear ramp
        if self.highpoint == "left":
            numerator = disp - self.start_disp
        else:
            assert self.highpoint == "right"
            numerator = self.end_disp - disp
        denominator = self.end_disp - self.start_disp
        frac = numerator / denominator
        return frac * (frac > 0) * (frac < 1) * self.effect


@attr.s
class BasicSplicingMechanism(SplicingMechanism):
    """
    Represents a splicing mechanism with basic motifs that influence splice sites.
    """

    splice_site_motifs = attr.ib()
    other_motifs = attr.ib()
    effects = attr.ib()
    splice_site_cutoff = attr.ib()

    def predict_motifs(self, rna, **kwargs):
        return np.array(
            [
                mot.score(rna, **kwargs)
                for mot in self.splice_site_motifs + self.other_motifs
            ]
        ).T

    def processed_motifs(self, motifs):
        numpy = isinstance(motifs, np.ndarray)
        if numpy:
            motifs = torch.from_numpy(motifs)

        sums = [0, 0]
        for eff in self.effects:
            sums[eff.site_idx] = sums[eff.site_idx] + eff.process(motifs)
        result = log(1e-5 + motifs[:, :2]) + torch.stack(sums, dim=1)
        if numpy:
            return result.numpy()
        return result

    def predict_splicing_pattern_from_motifs(self, motifs):
        motifs = motifs > self.splice_site_cutoff
        return np.where(motifs[:, 1])[0], np.where(motifs[:, 0])[0]

    def motif_width(self):
        return max(mot.width() for mot in self.splice_site_motifs)

    def motif_processing_width(self):
        return (
            max(max(abs(eff.end_disp), abs(eff.start_disp)) for eff in self.effects) * 2
            + 1
        )
