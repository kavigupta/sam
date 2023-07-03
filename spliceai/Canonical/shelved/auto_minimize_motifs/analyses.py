from abc import ABC, abstractmethod

import attr
import tqdm.auto as tqdm
from permacache import permacache
import numpy as np

TOL = 1e-5


class Analysis(ABC):
    def mean_number_analyses_under_null_hypothesis(self, num_runs, num_motifs, count):
        return np.mean(
            [
                len(
                    self.run_analyses(
                        num_motifs,
                        np.argsort(
                            np.random.RandomState(seed).randn(count, num_motifs)
                        ),
                    )
                )
                for seed in tqdm.trange(num_runs)
            ]
        )

    def run_analyses(self, num_motifs, importance_outcomes):
        return [
            parameter
            for parameter in self.parameters(num_motifs)
            if self.analyze_outcomes(parameter, importance_outcomes)
        ]

    @abstractmethod
    def parameters(self, num_motifs):
        pass

    @abstractmethod
    def analyze_outcomes(self, parameter, importance_outcomes):
        """
        index_outcomes: an array of the form NxM that contains the importance outcomes for each of the N
            runs and M motifs

            index_outcomes[n, m] = k <-> in the nth run, the mth motif had an importance of k (smaller is more important)
        """


@permacache("auto_minimize_motifs/analyses/mean_number_analyses_under_null_hypothesis")
def mean_number_analyses_under_null_hypothesis(analyzer, num_runs, num_motifs, count):
    return analyzer.mean_number_analyses_under_null_hypothesis(
        num_runs, num_motifs, count
    )


@attr.s
class MotifInTopK(Analysis):
    k = attr.ib()
    bar = attr.ib()
    version = attr.ib(default=2, kw_only=True)

    def parameters(self, num_motifs):
        return list(range(num_motifs))

    def analyze_outcomes(self, motif_idx, importance_outcomes):
        return (importance_outcomes[:, motif_idx] < self.k).mean() + TOL > self.bar


@attr.s
class MotifInBottomK(Analysis):
    k = attr.ib()
    bar = attr.ib()
    version = attr.ib(default=2, kw_only=True)

    def parameters(self, num_motifs):
        return list(range(num_motifs))

    def analyze_outcomes(self, motif_idx, importance_outcomes):
        return (
            importance_outcomes.shape[-1] - 1 - importance_outcomes[:, motif_idx]
            < self.k
        ).mean() + TOL > self.bar


@attr.s
class MotifMoreImportantThanOther(Analysis):
    k = attr.ib()
    bar = attr.ib()

    def parameters(self, num_motifs):
        return [(i, j) for i in range(num_motifs) for j in range(num_motifs)]

    def analyze_outcomes(self, motifs, importance_outcomes):
        motif_idx_1, motif_idx_2 = motifs
        return (
            importance_outcomes[:, motif_idx_2]
            >= self.k + importance_outcomes[:, motif_idx_1]
        ).mean() + TOL > self.bar


@attr.s
class MotifsSubstitutable(Analysis):
    delta = attr.ib()
    bar_delta = attr.ib()
    bar_directionality = attr.ib()

    def parameters(self, num_motifs):
        return [(i, j) for i in range(num_motifs) for j in range(i + 1, num_motifs)]

    def analyze_outcomes(self, motifs, importance_outcomes):
        motif_idx_1, motif_idx_2 = motifs
        deltas = np.abs(
            importance_outcomes[:, motif_idx_1] - importance_outcomes[:, motif_idx_2]
        )
        if not (deltas >= self.delta).mean() + TOL > self.bar_delta:
            return False

        directionality = np.mean(
            importance_outcomes[:, motif_idx_1] > importance_outcomes[:, motif_idx_2]
        )
        directionality = max(directionality, 1 - directionality)
        return directionality <= self.bar_directionality + TOL


@attr.s
class MotifsCorrelationInRange(Analysis):
    min_corr = attr.ib()
    max_corr = attr.ib()

    def parameters(self, num_motifs):
        return [(i, j) for i in range(num_motifs) for j in range(i + 1, num_motifs)]

    def analyze_outcomes(self, motifs, importance_outcomes):
        a, b = importance_outcomes.T[list(motifs)]
        return self.min_corr <= np.corrcoef(a, b)[0, 1] <= self.max_corr


@attr.s
class ConflictAmongTopK(Analysis):
    k = attr.ib()
    dependence_bar = attr.ib()

    def parameters(self, num_motifs):
        return [(i, j) for i in range(num_motifs) for j in range(i + 1, num_motifs)]

    def analyze_outcomes(self, motifs, importance_outcomes):
        a, b = importance_outcomes.T[list(motifs)] <= self.k
        if a.sum() == 0 or b.sum() == 0:
            return False
        return (a & b).mean() / (a.mean() * b.mean()) < self.dependence_bar
