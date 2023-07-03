import copy
from functools import lru_cache
import numpy as np
from permacache import permacache, stable_hash
from sklearn.decomposition import PCA
import sklearn.linear_model
import tqdm.auto as tqdm

import torch
import torch.nn as nn
from modular_splicing.data_for_experiments.standardized_sample import (
    standardized_sample,
)
from modular_splicing.dataset.h5_dataset import H5Dataset
from modular_splicing.example_figure.data import dataset_for_species
from modular_splicing.lssi.analyze import topk
from modular_splicing.utils.run_batched import run_batched

from modular_splicing.utils.construct import construct


def train_surrogate(inp, out, arch_spec, seed, epochs, batch_size=128, lr=1e-3):
    """
    Train a surrogate model with the given architecture
    """
    torch.manual_seed(seed)
    mod = construct(
        dict(Conv1d=nn.Conv1d),
        arch_spec,
        in_channels=inp.shape[1],
        out_channels=out.shape[1],
    )
    losses = []
    for epoch in tqdm.trange(epochs, desc="Epoch"):
        mod = copy.deepcopy(mod)
        mod, ls = train_surrogate_single_epoch(
            mod, inp, out, seed, batch_size, lr, epoch
        )
        losses.append(np.mean(ls))
    return mod, losses


@permacache(
    "working/what_does_psc_do/train_surrogate_single_epoch_2",
    key_function=dict(
        mod=stable_hash,
        inp=stable_hash,
        out=stable_hash,
    ),
)
def train_surrogate_single_epoch(mod, inp, out, seed, batch_size, lr, epoch):
    """
    Train a single epoch of the surrogate model
    """
    mod = mod.cuda()
    inp = torch.tensor(inp).cuda()
    out = torch.tensor(out).cuda()
    torch.manual_seed(seed)
    mod.train()
    optimizer = torch.optim.Adam(mod.parameters(), lr=lr)
    loss = nn.MSELoss()
    losses = []
    pbar = tqdm.tqdm(
        torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(inp, out), batch_size=batch_size
        ),
        desc=f"Epoch {epoch}",
    )
    for batch in pbar:
        optimizer.zero_grad()
        pred = mod(batch[0])
        l = loss(pred, batch[1])
        losses.append(l.item())
        pbar.set_postfix(loss=l.item())
        l.backward()
        optimizer.step()
    return mod.cpu(), losses


@permacache(
    "working/what_does_psc_do/fit_linear_convolution",
    key_function=dict(inp=stable_hash, out=stable_hash),
)
def fit_linear_convolution(inp, out, width, sample_n=None):
    """
    Fit a linear convolution with a bias from inp to out

    Parameters
        inp: (N, L, C1)
        out: (N, L, C2)
        width: int

    Returns
        filter: (C2, C1, width)
        bias: (C2,)
    """
    N, L, C1 = inp.shape
    _, _, C2 = out.shape
    assert (N, L, C2) == out.shape
    assert width % 2 == 1
    L_out = L - width + 1

    batch_idxs = np.arange(N)
    seq_idxs = np.arange(L_out) + width // 2
    offset_idxs = np.arange(-(width // 2), width // 2 + 1)

    # output indices
    bo, so = np.meshgrid(batch_idxs, seq_idxs, indexing="ij")
    # input indices
    bi, si, oi = np.meshgrid(batch_idxs, seq_idxs, offset_idxs, indexing="ij")

    linear_input = inp[bi, si + oi, :]
    assert linear_input.shape == (N, L_out, width, C1)
    linear_input = linear_input.reshape(N * L_out, width * C1)
    linear_output = out[bo, so, :]
    assert linear_output.shape == (N, L_out, C2)
    linear_output = linear_output.reshape(N * L_out, C2)
    if sample_n is not None:
        idxs = np.random.choice(N * L_out, sample_n, replace=False)
        linear_input = linear_input[idxs]
        linear_output = linear_output[idxs]
    model = sklearn.linear_model.LinearRegression(fit_intercept=True)
    model.fit(linear_input, linear_output)
    r2 = model.score(linear_input, linear_output)
    pred = model.predict(linear_input)
    var_out = np.var(linear_output)
    result = np.mean((linear_output - pred) ** 2)
    print(result, var_out)
    r2_direct = 1 - result / var_out
    print(r2, r2_direct)
    assert model.intercept_.shape == (C2,)
    assert model.coef_.shape == (C2, width * C1)
    coef = model.coef_.reshape(C2, width, C1)
    coef = np.transpose(coef, (0, 2, 1))
    return model, coef, model.intercept_, r2, pred


def fit_pca(reprocessed, num_components, sampling):
    vectors = reprocessed.reshape(-1, reprocessed.shape[-1])
    sampling = np.random.RandomState(0).choice(len(vectors), size=sampling)
    p = PCA(num_components, whiten=True)
    p.fit(vectors[sampling])
    return p


@lru_cache(None)
def data():
    _, data_path, dataset_kwargs = dataset_for_species("fly")
    data = H5Dataset(
        path=data_path,
        cl=400,
        **dataset_kwargs,
        sl=5000,
        iterator_spec=dict(type="FastIter", shuffler_spec=dict(type="DoNotShuffle")),
        datapoint_extractor_spec=dict(type="BasicDatapointExtractor"),
        post_processor_spec=dict(type="IdentityPostProcessor"),
    )
    xs, ys = [], []
    for it in tqdm.tqdm(data, desc="Loading data"):
        x = it["inputs"]["x"]
        y = it["outputs"]["y"]
        xs.append(x)
        ys.append(y)
        if len(xs) > len(data) / 2:
            break
    return np.array(xs), np.array(ys)


def run_mod(mod):
    xs, _ = data()
    return run_batched(lambda x: mod(x).softmax(-1), xs, 32, tqdm.tqdm)


def accs(mod):
    _, ys = data()
    yps = run_mod(mod)
    return [topk(yps[:, :, c], ys, c) for c in (1, 2)]


class AndThenReduceDim(nn.Module):
    def __init__(self, psc, p):
        super().__init__()
        self.psc = psc
        self.p = p

    def forward(self, motifs):
        motifs = self.psc(motifs)
        motifs_n = motifs.cpu().numpy()
        motifs_n_shape = motifs_n.shape
        motifs_n = motifs_n.reshape(-1, motifs_n.shape[-1])
        motifs_n = self.p.inverse_transform(self.p.transform(motifs_n))
        motifs_n = motifs_n.reshape(motifs_n_shape)
        motifs = torch.tensor(motifs_n.astype(np.float32), device=motifs.device)
        return motifs

    def __permacache_hash__(self):
        print("hashing")
        return stable_hash(self.p.components_)


@permacache(
    "working/what_does_psc_do/accuracy_with_dimensionality_reduction_3",
    key_function=dict(model=stable_hash, reprocessed=stable_hash),
)
def accuracy_with_dimensionality_reduction(model, reprocessed, dims):
    p10 = fit_pca(reprocessed, dims, 100_000)
    mod_2 = model.model
    mod_2.influence_calculator.sparse_reprocessor = AndThenReduceDim(
        mod_2.influence_calculator.sparse_reprocessor, p10
    )
    return accs(mod_2)
