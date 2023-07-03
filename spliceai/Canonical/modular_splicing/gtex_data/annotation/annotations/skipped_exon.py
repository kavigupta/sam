import numpy as np
from .annotation import Annotation


class SkippedExon(Annotation):
    """
    Represents an exon that can be skipped.
    """

    @classmethod
    def patterns(self):
        return ["AD"]

    def compatible_with_overlapping_junction(self, start, end):
        # anything that overlaps this and doesn't land at a skip is fine
        return True

    def cost(self, cost_params):
        return cost_params["annot_cost"]

    def short_name(self):
        return "S"

    def color(self):
        return "red"

    def compute_psis(self, junctions, tpm_junc):
        a, d = self.sites
        tpm_by_site = self.compute_junction_totals(junctions, tpm_junc)
        tpm_over_exon = tpm_junc[
            [start <= a and end >= d for start, end in junctions]
        ].sum(0)
        tpm_by_site = tpm_by_site.mean(0)
        psi_by_site = tpm_by_site / (tpm_by_site + tpm_over_exon + 1e-8)
        return np.array([psi_by_site, psi_by_site])
