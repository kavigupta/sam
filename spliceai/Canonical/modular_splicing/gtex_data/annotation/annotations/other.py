import numpy as np
from .annotation import Annotation


class Other(Annotation):
    """
    Represents an annotation cluster that is not one of the other types.

    Compatible with all junctions.
    """

    @classmethod
    def patterns(self):
        return ["A", "D"]

    def compatible_with_overlapping_junction(self, start, end):
        return True

    def cost(self, cost_params):
        return cost_params["other_cost"]

    def short_name(self):
        return "O"

    def color(self):
        return "magenta"

    def compute_psis(self, junctions, tpm_junc):
        return np.nan + np.zeros((1, len(self.sites)))
