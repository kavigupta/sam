import numpy as np
from .annotation import Annotation


class AlternateSites(Annotation):
    """
    Represents a pair of alternate sites, that is,
        a pair of sites where one should be included and the other should not.
    """

    @classmethod
    def patterns(self):
        return ["AA", "DD", "AAA", "DDD"]

    def compatible_with_overlapping_junction(self, start, end):
        if {start, end} & set(self.sites) != set():
            # does not end at one of the alternates, must be skip or internal
            return True
        return False

    def cost(self, cost_params):
        return cost_params["annot_cost"]

    def short_name(self):
        return "A"

    def color(self):
        return "blue"

    def compute_psis(self, junctions, tpm_junc):
        tpm_by_site = self.compute_junction_totals(junctions, tpm_junc)
        psi_by_site = tpm_by_site / (tpm_by_site.sum(0) + 1e-8)
        return psi_by_site
