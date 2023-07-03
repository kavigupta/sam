import numpy as np
from .annotation import Annotation


class Constituitive(Annotation):
    """
    Represent a constituitive annotation cluster, which is a site
        that should always included in a transcript.
    """

    @classmethod
    def patterns(self):
        return ["A", "D"]

    def compatible_with_overlapping_junction(self, start, end):
        [site] = self.sites
        return site in {start, end}

    def cost(self, cost_params):
        return 0

    def short_name(self):
        return "C"

    def color(self):
        return "black"

    def compute_psis(self, junctions, tpm_junc):
        return np.ones((1, len(self.sites)))
