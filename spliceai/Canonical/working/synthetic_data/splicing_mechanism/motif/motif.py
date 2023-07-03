from abc import ABC, abstractmethod

import numpy as np


class Motif(ABC):
    @abstractmethod
    def score(self, rna, pad=True):
        """
        Score the given RNA sequence. If pad is True, then the score is padded
        with zeros on the left and right.
        """
        pass

    def width(self):
        """
        Returns an upper bound on the width of the motif.
        """
        radius_left, radius_right = self.radii_each()
        return max(radius_left, radius_right) * 2 + 1

    def start_end(self, idx):
        """
        Returns the start and end for the given motif and binding index idx
        """
        radius_left, radius_right = self.radii_each()
        return (
            idx - self.activation_point(),
            idx - self.activation_point() + radius_right + radius_left + 1,
        )

    @abstractmethod
    def radii_each(self):
        """
        Returns the left and right radii of the motif.
        """
        pass

    @abstractmethod
    def activation_point(self):
        """
        Returns the activation point of the motif.
        """
        pass

    def _scores_on_random_sequences(self, num_samples):
        """
        Returns the scores on random sequences of the same length as the motif.

        Returns:
            sequences: (num_samples, width)
            scores: (num_samples,)
        """
        left, right = self.radii_each()
        true_width = right + left + 1
        sequences = np.random.RandomState(0).choice(4, size=(num_samples, true_width))
        scores = self.score(sequences, pad=False)
        assert scores.shape[1] == 1
        scores = scores[:, 0]
        return sequences, scores

    def empirical_psam(self, num_samples=10_000):
        """
        Returns the empirical PSAM for the motif, based on looking at
            what random sequences it matches.
        """
        sequences, scores = self._scores_on_random_sequences(num_samples)
        sequences = sequences[scores != 0]
        return np.eye(4)[sequences].mean(0)

    def empirical_density(self, num_samples=10_000):
        """
        Returns the empirical density for the motif, based on looking at
            what random sequences it matches.
        """
        _, scores = self._scores_on_random_sequences(num_samples)
        return (scores != 0).mean()

    def footprint_entirely_contained_within(self, mask):
        """
        Check if the footprint of the motif is entirely contained within the
        given mask.
        """
        left, right = self.radii_each()
        return np.all([shift(mask, k) for k in range(-left, right + 1)], axis=0)


def shift(arr, k):
    """
    Shift the given array by k positions, so that e.g., [1, 2, 3, 4, 5] shifted by 2 becomes [3, 4, 5, 0, 0].
    Pad with 0s of the same dtype so that the result has the same length as the input.
    """
    if k == 0:
        return arr.copy()
    if k > 0:
        return np.concatenate([arr[k:], np.zeros(k, dtype=arr.dtype)])
    else:
        return np.concatenate([np.zeros(-k, dtype=arr.dtype), arr[:k]])
