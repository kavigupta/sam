import attr

import scipy.stats


@attr.s
class Accuracy:
    """
    Represents the accuracy of a prediction.

    The accuracy is represented as a mean value, and a confidence interval (95% by default).
    """

    mean = attr.ib()
    low = attr.ib()
    hi = attr.ib()

    @staticmethod
    def of(matr, confidence=0.95):
        """
        Compute the accuracy of a prediction as well as a confidence interval,
        given a confusion matrix. Only supports binary classification.
        """
        assert matr.shape == (2, 2)
        correct = matr[0, 0] + matr[1, 1]
        total = matr.sum()
        low, hi = scipy.stats.binom.interval(confidence, total, correct / total)
        return Accuracy(correct / total, low / total, hi / total)

    def asarray(self):
        return [self.mean, self.low, self.hi]


@attr.s
class ResultOfAnalyses:
    """
    Represents the results of a set of analyses.

    The results are represented as a confusion matrix for each motif,
        stored inside a dictionary from motif name to confusion matrix.
    """

    results_each = attr.ib()

    def accuracy_by_motif(self):
        return {
            motif: Accuracy.of(self.results_each[motif]) for motif in self.results_each
        }

    def accuracy_overall(self):
        return Accuracy.of(sum(self.results_each.values()))
