import numpy as np
from .annotation import Annotation


class ExclusiveExons(Annotation):
    """
    Represents a pair of exons, where exactly one
        of the two must be included in a transcript.
    """

    @classmethod
    def patterns(self):
        return ["ADAD"]

    def compatible_with_overlapping_junction(self, start, end):
        [a1, d1, a2, d2] = self.sites
        # can't skip both exons
        return not (start < a1 and end < d2)

    def cost(self, cost_params):
        return cost_params["annot_cost"]

    def short_name(self):
        return "X"

    def color(self):
        return "green"

    def compute_psis(self, junctions, tpm_junc):
        tpm_by_site = self.compute_junction_totals(junctions, tpm_junc)
        tpm_by_site = np.array(
            [tpm_by_site[0] + tpm_by_site[1], tpm_by_site[2] + tpm_by_site[3]]
        )
        psi_by_site = tpm_by_site / (tpm_by_site.sum(0) + 1e-8)
        psi_by_site = np.array(
            [psi_by_site[0], psi_by_site[0], psi_by_site[1], psi_by_site[1]]
        )
        return psi_by_site
