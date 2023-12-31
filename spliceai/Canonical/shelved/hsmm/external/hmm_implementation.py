"""
External implementation of HMM algorithm in torch.

Sourced from https://github.com/TreB1eN/HiddenMarkovModel_Pytorch/blob/master/HiddenMarkovModel.py
"""

import torch
import numpy as np
import pdb


class HiddenMarkovModel(object):
    """
    Hidden Markov self Class

    Parameters:
    -----------

    - S: Number of states.
    - T: numpy.array Transition matrix of size S by S
         stores probability from state i to state j.
    - E: numpy.array Emission matrix of size S by N (number of observations)
         stores the probability of observing  O_j  from state  S_i.
    - T0: numpy.array Initial state probabilities of size S.
    """

    def __init__(self, T, T0, epsilon=0.001, maxStep=10):
        # Max number of iteration
        self.maxStep = maxStep
        # convergence criteria
        self.epsilon = epsilon
        # Number of possible states
        self.S = T.shape[0]
        # Number of possible observations
        self.prob_state_1 = []
        # Transition matrix
        self.T = torch.tensor(T)
        # Initial state vector
        self.T0 = torch.tensor(T0)

    def initialize_viterbi_variables(self, shape):
        pathStates = torch.zeros(shape, dtype=torch.float64)
        pathScores = torch.zeros_like(pathStates)
        states_seq = torch.zeros([shape[0]], dtype=torch.int64)
        return pathStates, pathScores, states_seq

    def belief_propagation(self, scores):
        return scores.view(-1, 1) + torch.log(self.T)

    def viterbi_inference(self, obs_prob_full):
        """
        obs_prob_full: the log of the emissions probabilities
        """
        self.N = len(obs_prob_full)
        shape = [self.N, self.S]
        # Init_viterbi_variables
        pathStates, pathScores, states_seq = self.initialize_viterbi_variables(shape)
        # initialize with state starting log-priors
        pathScores[0] = torch.log(self.T0) + obs_prob_full[0]
        for step, obs_prob in enumerate(obs_prob_full[1:]):
            # propagate state belief
            belief = self.belief_propagation(pathScores[step, :])
            # the inferred state by maximizing global function
            pathStates[step + 1] = torch.argmax(belief, 0)
            # and update state and score matrices
            pathScores[step + 1] = torch.max(belief, 0)[0] + obs_prob
        # infer most likely last state
        states_seq[self.N - 1] = torch.argmax(pathScores[self.N - 1, :], 0)
        for step in range(self.N - 1, 0, -1):
            # for every timestep retrieve inferred state
            state = states_seq[step]
            state_prob = pathStates[step][state]
            states_seq[step - 1] = state_prob
        return states_seq, torch.exp(pathScores)  # turn scores back to probabilities

    def initialize_forw_back_variables(self, shape):
        self.forward = torch.zeros(shape, dtype=torch.float64)
        self.backward = torch.zeros_like(self.forward)
        self.posterior = torch.zeros_like(self.forward)

    def _forward(model, obs_prob_seq):
        model.scale = torch.zeros([model.N], dtype=torch.float64)  # scale factors
        # initialize with state starting priors
        init_prob = model.T0 * obs_prob_seq[0]
        # scaling factor at t=0
        model.scale[0] = 1.0 / init_prob.sum()
        # scaled belief at t=0
        model.forward[0] = model.scale[0] * init_prob
        # propagate belief
        for step, obs_prob in enumerate(obs_prob_seq[1:]):
            # previous state probability
            prev_prob = model.forward[step].unsqueeze(0)
            # transition prior
            prior_prob = torch.matmul(prev_prob, model.T)
            # forward belief propagation
            forward_score = prior_prob * obs_prob
            forward_prob = torch.squeeze(forward_score)
            # scaling factor
            model.scale[step + 1] = 1 / forward_prob.sum()
            # Update forward matrix
            model.forward[step + 1] = model.scale[step + 1] * forward_prob

    def _backward(self, obs_prob_seq_rev):
        # initialize with state ending priors
        self.backward[0] = self.scale[self.N - 1] * torch.ones(
            [self.S], dtype=torch.float64
        )
        # propagate belief
        for step, obs_prob in enumerate(obs_prob_seq_rev[:-1]):
            # next state probability
            next_prob = self.backward[step, :].unsqueeze(1)
            # observation emission probabilities
            obs_prob_d = torch.diag(obs_prob)
            # transition prior
            prior_prob = torch.matmul(self.T, obs_prob_d)
            # backward belief propagation
            backward_prob = torch.matmul(prior_prob, next_prob).squeeze()
            # Update backward matrix
            self.backward[step + 1] = self.scale[self.N - 2 - step] * backward_prob
        self.backward = torch.flip(self.backward, [0, 1])

    def forward_backward(self, obs_prob_seq):
        """
        runs forward backward algorithm on observation sequence

        Arguments
        ---------
        - obs_prob_seq : matrix of size N by S, where N is number of timesteps and
            S is the number of states

        Returns
        -------
        - forward : matrix of size N by S representing
            the forward probability of each state at each time step
        - backward : matrix of size N by S representing
            the backward probability of each state at each time step
        - posterior : matrix of size N by S representing
            the posterior probability of each state at each time step
        """
        self._forward(obs_prob_seq)
        obs_prob_seq_rev = torch.flip(obs_prob_seq, [0, 1])
        self._backward(obs_prob_seq_rev)
