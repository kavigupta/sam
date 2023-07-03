import numpy as np

from .annotation import Annotation


class Null(Annotation):
    """
    Represents a site that should never be included in a transcript.
    """

    @classmethod
    def find_patterns(cls, sites, usable, skippable):
        return super().find_patterns(sites, np.ones_like(usable), skippable)

    @classmethod
    def patterns(self):
        return ["A", "D"]

    def compatible_with_overlapping_junction(self, start, end):
        [site] = self.sites
        return site not in {start, end}

    def cost(self, cost_params):
        return 0

    def short_name(self):
        return "N"

    def color(self):
        return "gray"

    def compute_psis(self, junctions, tpm_junc):
        return np.zeros((1, len(self.sites)))
