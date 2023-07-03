from abc import ABC, abstractmethod

import numpy as np
import torch


class SplicingMechanism(ABC):
    """
    Represents a splicing mechanism.
    """

    @abstractmethod
    def predict_motifs(self, rna):
        """
        Predict the motifs on the given RNA sequence.

        Input: RNA sequence (N,), elements are integers in [0, 3]
        Output: motif binding affinities (N, M). First 2/M channels represent
            3' and 5' motifs, respectively.
        """

    @abstractmethod
    def processed_motifs(self, motifs):
        """
        Process motifs locally to get a more accurate prediction. Should work on
            both numpy arrays and torch tensors. The outputs should be logits.

        Input: motif affinities (N, M)
        Output: motif affinities (N, M2)
        """

    @abstractmethod
    def predict_splicing_pattern_from_motifs(self, motifs):
        """
        Predict the splicing pattern on the given RNA sequence, given the motif
            binding affinities.
        """

    @abstractmethod
    def motif_width(self):
        """
        Returns an upper bound on the width of the motifs.
        """

    @abstractmethod
    def motif_processing_width(self):
        """
        Returns an upper bound on the width of the motif processing.
        """

    def predict_splicing_pattern(self, rna):
        """
        Predict the splicing pattern on the given RNA sequence.

        Specifically, return two lists, (5'_sites, 3'_sites), where
            5'_sites[i] is the 5' site of the i-th intron
            3'_sites[i] is the 3' site of the i-th intron
        """
        mots = self.predict_motifs(rna)
        mots = self.processed_motifs(mots)
        return self.predict_splicing_pattern_from_motifs(mots)

    def score_from_motifs(self, motifs, true_splice):
        """
        Compute a cross-entropy loss between the scores we have for the
            splicing pattern and the true splicing pattern.
        """
        motifs = self.processed_motifs(motifs)

        log_p = log_sigmoid(motifs)
        log_1_p = log_sigmoid(-motifs)
        five_prime, three_prime = true_splice
        mask_five_prime = np.zeros(motifs.shape[0], dtype=np.bool)
        mask_five_prime[five_prime] = 1
        for_five = log_p[mask_five_prime, 1].sum() + log_1_p[~mask_five_prime, 1].sum()

        mask_three_prime = np.zeros(motifs.shape[0], dtype=np.bool)
        mask_three_prime[three_prime] = 1
        for_three = (
            log_p[mask_three_prime, 0].sum() + log_1_p[~mask_three_prime, 0].sum()
        )

        return for_five + for_three

    def motif_saliency_map(self, motifs, true_splice):
        """
        Returns a saliency map for the given RNA sequence.

        Uses the loss function score(true) - score(pred)
        """
        motifs_t = torch.tensor(motifs, requires_grad=True)
        score = self.score_from_motifs(motifs_t, true_splice)
        score.backward()

        return motifs_t.grad.numpy()

    def unchangable_motifs(self, protected_mask):
        """
        Convert a mask of protected RNA positions to a mask of unchangable motifs.

        An unchangable motif is a motif whose footprint is entirely contained within the
            protected region. Non-unchangable motifs are not necessarily changable, but
            unchanged motifs are always unchangable.

        Input: protected_mask (N,), elements are 0 or 1
        Output: unchangable_motifs (N, M), elements are 0 or 1
        """
        return np.array(
            [
                motif.footprint_entirely_contained_within(protected_mask)
                for motif in self.splice_site_motifs + self.other_motifs
            ]
        ).T


def copy(tensor):
    """
    Copies the given tensor or numpy array
    """
    if isinstance(tensor, np.ndarray):
        return tensor.copy()
    return tensor.clone()


def concatenate(t1, t2):
    """
    Concatenates the given tensors or numpy arrays
    """
    if isinstance(t1, np.ndarray):
        return np.concatenate((t1, t2), axis=0)
    return torch.cat((t1, t2), dim=0)


def log(x):
    """
    Applies log to the given tensor or numpy array
    """
    if isinstance(x, np.ndarray):
        return np.log(x)
    return torch.log(x)


def log_sigmoid(x):
    """
    Applies log(sigmoid(x)) to the given tensor or numpy array
    """
    if isinstance(x, np.ndarray):
        return np.log(1 / (1 + np.exp(-x)))
    return torch.nn.functional.logsigmoid(x)


def except_true(pred_splice, true_splice):
    """
    Returns a splicing pattern that is the same as pred_splice except that the
        introns that are spliced in true_splice are not spliced in pred_splice.
    """
    pred_don, pred_acc = pred_splice
    true_don, true_acc = true_splice
    return (
        np.array(
            [don for don in pred_don if don not in true_don], dtype=pred_don.dtype
        ),
        np.array(
            [acc for acc in pred_acc if acc not in true_acc], dtype=pred_acc.dtype
        ),
    )
