import attr
from cached_property import cached_property

import torch
import torch.nn as nn
import numpy as np


@attr.s
class HSMMForwardBackward:
    log_o = attr.ib(type="NxSxT")
    log_pi = attr.ib(type="S")
    log_p = attr.ib(type="Sx(D+1)")
    log_a = attr.ib(type="SxS")

    @classmethod
    def from_logits(cls, logit_o, logit_pi, logit_p, logit_a):
        return cls(
            torch.nn.functional.logsigmoid(logit_o),
            logit_pi.log_softmax(0),
            logit_p.log_softmax(1),
            logit_a.log_softmax(1),
        )

    @property
    def N(self):
        return self.log_o.shape[0]

    @property
    def S(self):
        return self.log_o.shape[1]

    @property
    def T(self):
        return self.log_o.shape[2]

    @property
    def D(self):
        return self.log_p.shape[1] - 1

    @property
    def device(self):
        return self.log_o.device

    @log_o.validator
    def check(self, *_, **__):
        assert self.log_o.shape == (self.N, self.S, self.T)
        assert self.log_pi.shape == (self.S,)
        assert self.log_p.shape == (self.S, self.D + 1)
        assert self.log_a.shape == (self.S, self.S)

    @cached_property
    def log_Y(self):
        # N x S x (T + 1)
        return torch.cumsum(torch.nn.functional.pad(self.log_o, (1, 0)), axis=-1)

    def log_Z(self, t_s, t_e):
        return self.log_Y[:, :, t_e] - self.log_Y[:, :, t_s - 1]

    @cached_property
    def alpha_quantities(self):
        # alpha : N x S x T. Is 1-indexed as well
        log_alpha = torch.zeros(self.N, self.S, self.T + 1, device=self.device)
        log_alpha_star = torch.zeros(self.N, self.S, self.T + 1, device=self.device)

        log_alpha_star[:, :, 0] = self.log_pi
        for t in range(1, 1 + self.T):
            d = torch.arange(1, 1 + min(self.D, t), device=self.device)
            # N x S x D
            lz = self.log_Z(t - d + 1, [t])
            lp = self.log_p[None, :, d]
            log_alpha[:, :, t] = (lz + lp + log_alpha_star[:, :, t - d]).logsumexp(-1)

            # log_a[None] : 1x i x j, log_alpha[:,:, [t]] : n x i x 1
            # common: n x i x j
            log_alpha_star[:, :, t] = (
                self.log_a[None] + log_alpha[:, :, [t]]
            ).logsumexp(1)
        return dict(log_alpha=log_alpha, log_alpha_star=log_alpha_star)

    @cached_property
    def beta_quantities(self):
        # beta : N x S x T. Is 1-indexed
        log_beta = torch.zeros(self.N, self.S, self.T + self.D + 1, device=self.device)
        log_beta_star = torch.zeros(
            self.N, self.S, self.T + self.D + 1, device=self.device
        )

        log_beta[:, :, self.T + 1 :] = log_beta_star[:, :, self.T + 1 :] = -float("inf")

        for t in reversed(range(0, self.T)):
            d = torch.arange(1, 1 + self.D, device=self.device)

            # N x S x D
            lz = self.log_Z(np.array([t + 1]), torch.clip(t + d, 0, self.T))
            lp = self.log_p[None, :, d]
            log_beta_star[:, :, t] = (lz + lp + log_beta[:, :, t + d]).logsumexp(-1)

            # log_beta_star[:, :, [t]]: n x j x 1, log_a.T[None] : 1 x j x i
            # output: n x i
            log_beta[:, :, t] = (
                log_beta_star[:, :, [t]] + self.log_a.T[None]
            ).logsumexp(1)
        return dict(
            log_beta=log_beta[:, :, : self.T + 1],
            log_beta_star=log_beta_star[:, :, : self.T + 1],
        )

    @cached_property
    def log_Lqt(self):
        # F[n,q,t,d]

        ts, ds = torch.meshgrid(
            torch.arange(0, 1 + self.T, device=self.device),
            torch.arange(1, 1 + self.D, device=self.device),
        )

        beta = self.beta_quantities["log_beta"][:, :, ts]
        z = self.log_Z(ts - ds + 1, ts)
        p = self.log_p[None, :, ds]
        alpha = self.alpha_quantities["log_alpha_star"][:, :, ts - ds]

        # return beta + z + p + alpha - correction

        mask = torch.where(ts >= ds, 0.0, -float("inf"))

        F = (beta + z + p + alpha + mask)[:, :, 1:].logsumexp(-1)

        correction = self.alpha_quantities["log_alpha"][:, :, -1].logsumexp(1)[
            :, None, None
        ]

        return F - correction
