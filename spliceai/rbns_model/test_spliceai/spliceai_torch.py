import h5py

import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os

# from motifs import read_motifs, motifs_for
from utils import clip_datapoint, modify_sl, print_topl_statistics


class ResidualUnit(nn.Module):
    """
    Residual unit proposed in "Identity mappings in Deep Residual Networks"
    by He et al.
    """

    def __init__(self, l, w, ar):
        super().__init__()
        self.normalize1 = nn.BatchNorm1d(l)
        self.normalize2 = nn.BatchNorm1d(l)
        self.act1 = self.act2 = nn.ReLU()

        padding = (ar * (w - 1)) // 2

        self.conv1 = nn.Conv1d(l, l, w, dilation=ar, padding=padding)
        self.conv2 = nn.Conv1d(l, l, w, dilation=ar, padding=padding)

    def forward(self, input_node):
        bn1 = self.normalize1(input_node)
        act1 = self.act1(bn1)
        conv1 = self.conv1(act1)
        assert conv1.shape == act1.shape
        bn2 = self.normalize2(conv1)
        act2 = self.act2(bn2)
        conv2 = self.conv2(act2)
        assert conv2.shape == act2.shape
        output_node = conv2 + input_node
        return output_node


class SpliceAI(nn.Module):
    def __init__(
        self,
        l,
        w,
        ar,
        starting_channels=4, # rbns_motifs: 102, psam_motifs: 81
    ):
        super().__init__()
        assert len(w) == len(ar)
        self.w = w
        self.cl = 2 * sum(ar * (w - 1))

        self.conv1 = nn.Conv1d(starting_channels, l, 1)
        self.conv2 = nn.Conv1d(l, l, 1)

        def get_mod(i):
            res = ResidualUnit(l, w[i], ar[i])
            return res

        self.convstack = nn.ModuleList([get_mod(i) for i in range(len(self.w))])
        self.skipconv = nn.ModuleList(
            [
                nn.Conv1d(l, l, 1) if self._skip_connection(i) else None
                for i in range(len(self.w))
            ]
        )
        self.output = nn.Conv1d(l, 3, 1)

        self.presparse_norm = nn.BatchNorm1d(starting_channels-2, affine=False)
        self.hooks_handles = []

    def forward(self, x):
        # print('cl', self.cl)
        x = x.transpose(1, 2)
    
        conv = self.conv1(x)
        skip = self.conv2(conv)

        for i in range(len(self.w)):
            conv = self.convstack[i](conv)

            if self._skip_connection(i):
                # Skip connections to the output after every 4 residual units
                skip = skip + self.skipconv[i](conv)
        
        skip = skip[:, :, self.cl // 2 : -self.cl // 2]

        y = self.output(skip)

        y = y.transpose(1, 2)

        return y


def batchdot(a, b):
    assert a.shape == b.shape
    *NN, C = a.shape
    a = a.reshape(-1, 1, C)
    b = b.reshape(-1, C, 1)
    ab = torch.bmm(a, b)
    return ab.reshape(*NN)


def getx(data, i, j=None, *, motifs_mode):
    """
    gets the x values from the given dataset. If Xi is present in the data,
        returns that, otherwise unpacks and uses the Mi value
    """
    if motifs_mode == "none" or motifs_mode == "learned":
        X = data[f"X{i}"]
        if j is None:
            return X
        else:
            return X[j]

    assert motifs_mode == "given"

    M = data[f"M{i}"]
    sh, M = M[:, 0], M[:, 1:]
    if j is None:
        X = np.zeros(tuple(sh))
        X[tuple(M)] = data[f"V{i}"]
        return X
    else:
        correct_idxs = M[0] == j
        m = M[1:, correct_idxs]
        x = np.zeros(tuple(sh)[1:])
        x[tuple(m)] = data[f"V{i}"][correct_idxs]
        return x


class SpliceAIDataset(torch.utils.data.IterableDataset):
    @staticmethod
    def of(path, cl, cl_max, *, stretch=1, **kwargs):
        assert cl % stretch == 0
        cl //= stretch
        data = SpliceAIDataset(path, cl, cl_max, **kwargs)
        if stretch != 1:
            data = StretchData(data, stretch)
        return data

    def __init__(
        self,
        path,
        cl,
        cl_max,
        sl=None,
        shuffled=False,
        seed=None,
        separate_motifs=False,
        iterator_strategy="fast",
        motifs_mode="none",
    ):
        self.path = path
        self._shuffled = shuffled
        self._seed = seed
        self.cl = cl
        self.cl_max = cl_max
        self.sl = sl
        self._iterator = dict(
            fast=self._fast_iter, fully_random=self._fully_random_iter
        )[iterator_strategy]
        self.motifs_mode = motifs_mode
        self.separate_motifs = separate_motifs
        if self.separate_motifs:
            assert self.motifs_mode == "learned"

    def __len__(self):
        if hasattr(self, "_l"):
            return self._l

        self._l = 0
        with h5py.File(self.path, "r") as data:
            for i in range(self.dsize(data)):
                Y = data[f"Y{i}"]
                self._l += Y.shape[1] * Y.shape[2] // self.sl
        return self._l

    @staticmethod
    def dsize(data):
        ys = [k for k in data.keys() if "Y" in k]
        return len(ys)

    @staticmethod
    def _length_each(data):
        return [data["Y" + str(i)].shape[1] for i in range(SpliceAIDataset.dsize(data))]

    def _fully_random_iter(self, data, shuffle):
        ijs = [(i, j) for i, l in enumerate(self._length_each(data)) for j in range(l)]
        shuffle(ijs)
        data = {k: v[:] for k, v in data.items()}
        for i, j in ijs:
            x, y = getx(data, i, j, motifs_mode=self.motifs_mode), data[f"Y{i}"][0][j]
            if self.separate_motifs:
                m = getx(data, i, j, motifs_mode="given")
                yield x, y, m
            else:
                yield x, y

    def _fast_iter(self, data, shuffle):
        i_s = list(range(SpliceAIDataset.dsize(data)))
        shuffle(i_s)
        for i in i_s:
            data_for_i = {k: v[:] for k, v in data.items() if k[1:] == str(i)}
            j_s = list(range(data[f"Y{i}"].shape[1]))
            shuffle(j_s)
            for j in j_s:
                x, y = (
                    getx(data_for_i, i, j, motifs_mode=self.motifs_mode),
                    data_for_i[f"Y{i}"][0][j],
                )
                if self.separate_motifs:
                    m = getx(data_for_i, i, j, motifs_mode="given")
                    yield x, y, m
                else:
                    yield x, y

    def clip(self, x, y):
        x = clip_datapoint(x, CL=self.cl, CL_max=self.cl_max)
        if self.sl is None:
            return [x], [y]
        return modify_sl(x, y, self.sl)

    def __iter__(self):
        if not self._shuffled:
            shuffle = lambda x: None
        elif self._seed is not None:
            shuffle = np.random.RandomState(self._seed).shuffle
        else:
            shuffle = np.random.shuffle

        with h5py.File(self.path, "r") as data:
            for x, y, *perhaps_m in self._iterator(data, shuffle):
                xs, ys = self.clip(x, y)
                perhaps_ms = [self.clip(m, y)[0] for m in perhaps_m]
                for i in range(len(xs)):
                    perhaps_m = [ms[i] for ms in perhaps_ms]
                    yield (xs[i].astype(np.float32), ys[i].argmax(-1), *perhaps_m)


def evaluate_model(
    m,
    d,
    limit=float("inf"),
    bs=32,
    separately_classified=False,
    pbar=lambda x: x,
    model_kwargs={},
    **kwargs,
):
    ytrues = None
    ypreds = None
    count = 0
    try:
        m.eval()
        for x, y in pbar(DataLoader(d, batch_size=bs)):
            x = x.cuda()
            y = y.cuda()
            with torch.no_grad():
                yp = m(x, **model_kwargs).softmax(-1)

            if ytrues is None:
                ytrues, ypreds = [
                    [
                        []
                        for _ in range(
                            yp.shape[-1] if separately_classified else yp.shape[-1] - 1
                        )
                    ]
                    for _ in range(2)
                ]

            for c in range(0 if separately_classified else 1, yp.shape[-1]):
                ytrues[c if separately_classified else c - 1].append(
                    (y[:, :, c] if separately_classified else y == c)
                    .flatten()
                    .cpu()
                    .numpy()
                )
                ypreds[c if separately_classified else c - 1].append(
                    yp[:, :, c].flatten().detach().cpu().numpy()
                )
            count += bs
            if count >= limit:
                break
    finally:
        m.train()

    by_c = []
    for c in range(1, len(ytrues) + 1):
        yt = np.concatenate(ytrues[c - 1])
        yp = np.concatenate(ypreds[c - 1])
        by_c.append(print_topl_statistics(yt, yp, **kwargs))
    return by_c


def predict(m, d, bs=32, pbar=lambda x: x):
    results = []
    try:
        m.eval()
        for x, _ in pbar(DataLoader(d, batch_size=bs)):
            x = x.cuda()
            with torch.no_grad():
                results.append(m(x).detach().cpu().numpy())
    finally:
        m.train()
    return np.concatenate(results)


def load_model(folder, step=None):
    def hook(m):
        if hasattr(m, "_load_hook"):
            m._load_hook()
        return m

    kwargs = {}
    if not torch.cuda.is_available():
        kwargs = dict(map_location=torch.device("cpu"))

    if os.path.isfile(folder):
        return None, hook(torch.load(folder, **kwargs))

    model_dir = os.path.join(folder, f"model_")
    if not os.path.exists(model_dir):
        return None, None

    if step is None and os.listdir(model_dir):
        step = max(os.listdir(model_dir), key=int)

    path = os.path.join(model_dir, str(step))
    if not os.path.exists(path):
        return None, None

    return int(step), hook(torch.load(path, **kwargs))


def save_model(model, folder, attr, step):
    path = os.path.join(folder, f"model_{attr}", str(step))
    try:
        os.makedirs(os.path.dirname(path))
    except FileExistsError:
        pass
    torch.save(model, path)


def model_steps(folder):
    return sorted([int(x) for x in os.listdir(os.path.join(folder, "model"))])


class ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, *args, **kwargs):
        outputs = [m(*args, **kwargs) for m in self.models]
        return torch.stack(outputs).mean(0)
