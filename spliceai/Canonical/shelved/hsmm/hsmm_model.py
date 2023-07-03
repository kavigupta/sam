import copy
import attr
import numpy as np
import torch
import torch.nn as nn

from modular_splicing.utils.arrays import run_length_encoding


@attr.s
class HSMM:
    initial = attr.ib()
    distance_distributions = attr.ib()
    transition_distributions = attr.ib()

    @property
    def states(self):
        return sorted(self.transition_distributions, key=str)

    def clip_distances(self):
        to_split = {s for s in self.states if self.distance_distributions[s][0] == 0}

        split_lengths = {}
        initial = {}
        for s in self.initial:
            if s in to_split:
                initial["prefix", s] = self.initial[s]
            else:
                initial[s] = self.initial[s]
        dists = {}
        for s in self.states:
            dist = self.distance_distributions[s]
            dist = clip_distances(dist)
            if s in to_split:
                dists["prefix", s], dists["suffix", s] = split_distances(dist)
                split_lengths[s] = len(dists["prefix", s])
            else:
                dists[s] = dist
        transitions = {}
        for s in self.states:
            transition = copy.deepcopy(self.transition_distributions[s])
            for sprime in list(transition):
                if sprime in to_split:
                    transition["prefix", sprime] = transition.pop(sprime)
            if s in to_split:
                transitions["prefix", s] = {("suffix", s): 1}
                transitions["suffix", s] = transition
            else:
                transitions[s] = transition
        return HSMM(initial, dists, transitions), split_lengths

    def to_torch(self, **kwargs):
        return HSMMTorch.from_probs(
            pi=torch.tensor(self.initial_array),
            p=padded_stack(
                [
                    np.concatenate([[0], self.distance_distributions[s]])
                    for s in self.states
                ]
            ),
            a=torch.tensor(self.transition_matrix),
            **kwargs,
        )

    @property
    def transition_matrix(self):
        # M[s, s']
        return np.array(
            [
                [
                    self.transition_distributions[s_in].get(s_out, 0)
                    for s_out in self.states
                ]
                for s_in in self.states
            ]
        )

    @property
    def initial_array(self):
        return np.array([self.initial.get(s, 0) for s in self.states])

    def score_distance(self, state, dist):
        result = self.distance_distributions[state]
        if dist <= len(result):
            return result[dist - 1]
        else:
            return 0

    def score_sequence(
        self, annotated_sequence, states, preprocess_states=lambda x, _: x
    ):
        lengths, _, values = run_length_encoding(annotated_sequence)
        values = [states[v] for v in values]
        values, lengths = preprocess_states(values, lengths)

        return self.score_rle_sequence(lengths, values)

    def score_rle_sequence(self, lengths, values):
        probs = []

        probs += [self.initial[values[0]]]
        probs += [self.score_distance(values[0], lengths[0])]

        for i in range(1, len(lengths)):
            probs += [self.score_distance(values[i], lengths[i])]
            probs += [self.transition_distributions[values[i - 1]].get(values[i], 0)]
        return np.log(np.array(probs) + 1e-1000).sum()

    @property
    def max_distances(self):
        return {k: len(v) for k, v in self.distance_distributions.items()}


def padded_stack(tensors):
    """
    Stack tensors with padding.
    """
    tensors = [torch.tensor(t) for t in tensors]
    max_length = max(t.shape[0] for t in tensors)
    return torch.stack(
        [torch.nn.functional.pad(t, (0, max_length - t.shape[0])) for t in tensors],
        dim=0,
    )


def clip_distances(dist):
    """
    Clip the distance distribution to the maximum nonzero index.
    """
    (nonzero_indices,) = np.nonzero(dist)
    return dist[: max(nonzero_indices) + 1]


def split_distances(dist):
    """
    Split the distance distribution into two parts.
    """
    (nonzero_indices,) = np.nonzero(dist)
    dist_1 = np.zeros(nonzero_indices[0])
    dist_1[-1] = 1
    dist_2 = dist[nonzero_indices[0] :]
    return dist_1, dist_2


class HSMMTorch(nn.Module):
    @classmethod
    def from_probs(cls, pi, p, a, *, tol=1e-100, **kwargs):
        return cls(
            S=p.shape[0],
            D=p.shape[1] - 1,
            log_pi=torch.log(pi + tol),
            log_p=torch.log(p + tol),
            log_a=torch.log(a + tol),
            **kwargs,
        )

    def __init__(self, *, S, D, log_pi=None, log_a=None, log_p=None, trainable):
        super().__init__()
        if log_pi is None:
            log_pi = torch.randn(S)
        if log_a is None:
            log_a = torch.randn(S, S)
        if log_p is None:
            log_p = torch.randn(S, D + 1)
        self.log_pi = nn.Parameter(log_pi)
        self.log_a = nn.Parameter(log_a)
        self.log_p = nn.Parameter(log_p)
        self.trainable = trainable
        for param in self.parameters():
            param.requires_grad = self.trainable

    def infer_states(self, log_o):
        from .hsmm_forward_backward import HSMMForwardBackward

        return HSMMForwardBackward(
            log_o=log_o,
            log_pi=self.log_pi,
            log_p=self.log_p,
            log_a=self.log_a,
        ).log_Lqt
