import copy
import attr
import numpy as np
from permacache.dict_function import drop_if_equal

import torch
import torch.nn as nn
import torch.fft
import tqdm.auto as tqdm

from permacache import permacache, stable_hash
from .hsmm_model import HSMM

from modular_splicing.utils.construct import construct


class SumOfPDFs(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K
        self.log_epsilon = nn.Parameter(torch.tensor(-5, dtype=torch.float64))
        self.logit_pdf_a = nn.Parameter(torch.randn(K, dtype=torch.float64))

    def forward(self, amount):
        epsilon = torch.exp(self.log_epsilon)
        pdf_b = -torch.expm1(-epsilon) * torch.exp(
            -epsilon * torch.arange(amount, dtype=torch.float64)
        )
        wholepdf = addpdf(pdf_b, self.logit_pdf_a.softmax(0))
        wholepdf = wholepdf[: -self.K]
        return wholepdf

    def pdf(self, amount):
        return self(amount)

    def loss(self, pdf_d):
        pdf_d = torch.tensor(pdf_d + 1e-8)
        predicted = self.forward(pdf_d.shape[0])
        return (pdf_d * (pdf_d.log() - (predicted + 1e-8).log())).sum()


def addpdf(a, b):
    al, bl = a.shape[0], b.shape[0]

    a = nn.functional.pad(a, [0, bl])
    b = nn.functional.pad(b, [0, al])

    return torch.fft.ifft(torch.fft.fft(a) * torch.fft.fft(b)).real


class MixtureOfGeometrics(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K
        self.logit_alphas = nn.Parameter(torch.randn(K, dtype=torch.float64))
        self.log_epsilons = nn.Parameter(torch.randn(K, dtype=torch.float64))

    @property
    def epsilons(self):
        return torch.exp(self.log_epsilons)

    @property
    def cbeta(self):
        return -torch.expm1(-self.epsilons)

    def forward(self, amount):
        # beta = exp(- epsilon)
        # 1 - beta = 1 - exp(-epsilon) = -(exp(-epsilon) - 1)
        epsilons = self.epsilons[:, None]
        log_cbeta = self.cbeta[:, None].log()
        log_probs = log_cbeta - epsilons * torch.arange(amount, dtype=torch.float64)
        log_probs = self.logit_alphas.log_softmax(-1)[:, None] + log_probs
        log_probs = log_probs.logsumexp(0)
        return log_probs

    def pdf(self, amount):
        return self(amount).exp()

    def loss(self, pdf_d):
        pdf_d = pdf_d.copy()
        pdf_d = torch.tensor(pdf_d)
        predicted = self.forward(pdf_d.shape[0])
        return (pdf_d * ((pdf_d + 1e-10).log() - predicted)).sum()


@permacache(
    "hsmm/shorten_distance_distributions/train", key_function=dict(dpdf=stable_hash)
)
def train(dpdf, model_spec, num_itr, seed, lr=1e-1):
    torch.manual_seed(seed)
    s = construct(
        dict(SumOfPDFs=SumOfPDFs, MixtureOfGeometrics=MixtureOfGeometrics), model_spec
    )
    opt = torch.optim.Adam(s.parameters(), lr=lr)
    losses = []
    for itr in range(num_itr):
        opt.zero_grad()
        loss = s.loss(dpdf)
        losses.append(loss.detach().numpy())
        if itr % 200 == 0:
            print(itr, losses[-1])
        loss.backward()
        opt.step()
    return losses, s


def replace_with_mixture_of_geometrics(m, g, s):
    alpha = g.logit_alphas.softmax(0).detach().cpu().numpy()
    cbeta = g.cbeta.detach().cpu().numpy()

    def distribute_entrances(d):
        if s in d:
            v = d.pop(s)
            for i in range(alpha.shape[0]):
                d[("geom", i, s)] = alpha[i] * v

    # initial distribution should be unchanged since the state is still equally likely to happen
    initial = copy.deepcopy(m.initial)
    distribute_entrances(initial)
    # transitions into the state should be distributed among the various representatives
    transition = copy.deepcopy(m.transition_distributions)
    for v in transition.values():
        distribute_entrances(v)
    # transitions out of the state should have the self-loop added in
    exit_distro = transition.pop(s)
    for i in range(alpha.shape[0]):
        sprime = ("geom", i, s)
        new_exit_distro = {k: exit_distro[k] * cbeta[i] for k in exit_distro}
        new_exit_distro[sprime] = new_exit_distro.get(sprime, 0) + 1 - cbeta[i]
        transition[sprime] = new_exit_distro
    distance_distribution = copy.deepcopy(m.distance_distributions)
    del distance_distribution[s]
    for i in range(alpha.shape[0]):
        distance_distribution["geom", i, s] = [1]
    return (
        HSMM(
            initial=initial,
            transition_distributions=transition,
            distance_distributions=distance_distribution,
        ),
        BestMatchingState(alpha, cbeta),
    )


def replace_with_best_mixture_of_geometrics(m, s, n_seeds, K):
    z = m.distance_distributions[s]
    g = min(
        [
            train(
                z,
                model_spec=dict(type="MixtureOfGeometrics", K=K),
                num_itr=1000,
                seed=seed,
            )[1]
            for seed in tqdm.trange(n_seeds)
        ],
        key=lambda g: ((g.pdf(z.size).detach().numpy() - z) ** 2).sum(),
    )
    return replace_with_mixture_of_geometrics(m, g, s)


@permacache(
    "hsmm/shorten_distance_distributions/shorten_states",
    key_function=dict(m=stable_hash, ignore=stable_hash, K=drop_if_equal(10)),
)
def shorten_states(m, max_length, n_seeds, ignore=(), K=10):
    bms = {}
    for s in m.states:
        if s in ignore:
            continue
        print(s)
        print(len(m.distance_distributions[s]))
        if len(m.distance_distributions[s]) > max_length:
            m, bms[s] = replace_with_best_mixture_of_geometrics(m, s, n_seeds, K=K)
    return m, bms


@attr.s
class BestMatchingState:
    alpha = attr.ib()
    cbeta = attr.ib()

    def __call__(self, n):
        return np.argmax(
            np.log(self.alpha) + np.log(self.cbeta) + n * np.log(1 - self.cbeta)
        )
