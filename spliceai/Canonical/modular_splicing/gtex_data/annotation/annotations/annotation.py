from abc import ABC, abstractmethod
import attr
from matplotlib import patches
import numpy as np

from ..pattern import find_pattern


@attr.s
class Annotation(ABC):
    """
    Represents an annotation cluster, a set of sites along with a certain
        notion of what transcripts are compatible with it.
    """

    # list of sites that are included in the annotation cluster, in order
    sites = attr.ib()

    def compatible_with_junction(self, junc):
        """
        Check if the junction is compatible with this annotation cluster.

        This function handles the specific cases of
            - junctions that do not overlap the annotation cluster at all (return True)
            - junctions that land on a site that is not in the annotation cluster (return False)

        All other cases are delegated to the `compatible_with_overlapping_junction`
            method, which is implemented by subclasses.

        Parameters
        ----------
        junc : tuple of int
            The junction to check, as a tuple of (start, end).

        Returns
        -------
        bool
            True if the junction is compatible with this annotation cluster.
        """
        start, end = junc
        if end < self.sites[0] or start > self.sites[-1]:
            # does not overlap us at all
            return True
        sites = set(self.sites)
        if any(self.sites[0] <= x <= self.sites[-1] and x not in sites for x in junc):
            # lands on a skip
            return False
        return self.compatible_with_overlapping_junction(start, end)

    @classmethod
    def find_patterns(cls, sites, usable, skippable):
        """
        Find all instances of the patterns that this annotation cluster.

        Parameters
        ----------
        sites : list of str
            List of sites, in order, where ecach site is "A" or "D".
        usable : list of bool
            List of booleans indicating which sites are usable.
        skippable : list of bool
            List of booleans indicating which sites are skippable.

        Returns
        -------
        list of Annotation
            List of annotations that match the patterns.
        """
        for pat in cls.patterns():
            for s in find_pattern(sites, usable, skippable, pat):
                yield cls(s)

    def compatible_with_annotation(self, other):
        """
        Check if this annotation cluster is compatible with another annotation cluster.

        This is true iff the two clusters do not overlap at all.

        Parameters
        ----------
        other : Annotation
            The other annotation cluster.

        Returns
        -------
        bool
            True if the two annotation clusters are compatible.
        """
        a, b = self.sites[0], self.sites[-1]
        c, d = other.sites[0], other.sites[-1]
        return a <= b < c <= d or c <= d < a <= b

    @abstractmethod
    def compatible_with_overlapping_junction(self, start, end):
        """
        Handles the case where the junction overlaps the annotation cluster,
            but does not land on a site that is not in the annotation cluster.

        Parameters
        ----------
        start : int
            The start of the junction.
        end : int
            The end of the junction.

        Returns
        -------
        bool
            True if the junction is compatible with this annotation cluster.
        """
        pass

    @classmethod
    @abstractmethod
    def patterns(cls):
        """
        The patterns that this annotation cluster matches.

        Returns
        -------
        list of str
            List of patterns. Each is something like "AADD" or "ADAD".
        """
        pass

    @abstractmethod
    def cost(self, cost_params):
        """
        Compute the additional cost of this annotation cluster.
        """
        pass

    @abstractmethod
    def short_name(self):
        """
        A short name for this annotation cluster.
        """
        pass

    @abstractmethod
    def color(self):
        """
        A color for this annotation cluster, in the form of a matplotlib color.
        """
        pass

    @abstractmethod
    def compute_psis(self, junctions, tpm_junc):
        """
        Compute the psi values for this annotation cluster.

        Parameters
        ----------
        junctions : list of tuple of int of length J
            The junctions to compute psi values for.
        tpm_junc : array of float of shape (J, T)
            The TPM values for each junction in each sample.

        Returns
        -------
        array of float of shape (I, T)
            The psi values for each splice site in each sample.
        """

    def compute_junction_totals(self, junctions, tpm_junc):
        tpm_by_site = []
        for site in self.sites:
            mask = [site in junction for junction in junctions]
            tpm_by_site.append(tpm_junc[mask].sum(0))
        tpm_by_site = np.array(tpm_by_site)
        return tpm_by_site

    def render(self, ax, *, y_center, y_rad):
        """
        Plot the given annotation cluster.
        """
        start, end = self.sites[0] - 0.4, self.sites[-1] + 0.4
        rect = patches.Rectangle(
            (start, y_center - y_rad),
            end - start,
            y_rad * 2,
            linewidth=1,
            edgecolor=self.color(),
            facecolor="none",
        )
        ax.text(start, y_center + y_rad * 1.2, self.short_name(), color=self.color())
        ax.add_patch(rect)
        ax.scatter(self.sites, [y_center] * len(self.sites), color=self.color())
