from abc import ABC, abstractmethod

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


class SpatiallySparse(nn.Module):
    def __init__(
        self,
        sparsity,
        num_channels,
        momentum=0.1,
        by_magnitude=True,
        discontinuity_mode="none",
    ):
        super().__init__()
        self.sparsity = sparsity
        self.thresholds = torch.nn.parameter.Parameter(
            torch.zeros(num_channels), requires_grad=False
        )
        self.by_magnitude = by_magnitude
        self.momentum = momentum
        self.discontinuity_mode = discontinuity_mode

    def update_sparsity(self, update_by):
        print(f"Originally dropping {self.sparsity}")
        self.sparsity = 1 - (1 - self.sparsity) * update_by
        print(f"Now dropping {self.sparsity}")

    def get_sparsity(self):
        return self.sparsity

    def forward(self, x):
        N, C, L = x.shape

        if self.by_magnitude:
            magnitudes = x.abs()
        else:
            magnitudes = x

        if self.training:
            to_drop = max(1, int(N * L * self.sparsity))
            thresholds, _ = torch.kthvalue(
                magnitudes.transpose(1, 2).reshape(N * L, C), k=to_drop, dim=0
            )

            self.thresholds.data = (
                self.thresholds.data * (1 - self.momentum) + thresholds * self.momentum
            )

        discontinuity_mode = getattr(self, "discontinuity_mode", "none")

        if discontinuity_mode == "none":
            mask = magnitudes > self.thresholds[None, :, None]
            return x * mask
        elif discontinuity_mode == "subtraction":
            assert not self.by_magnitude
            x = x - self.thresholds[None, :, None]
            return torch.nn.functional.relu(x)
        else:
            raise RuntimeError(f"Incorrect discontinuity mode: {discontinuity_mode}")


class SpatiallySparseAcrossChannels(nn.Module):
    def __init__(self, sparsity, num_channels, **kwargs):
        super().__init__()
        del num_channels  # unused
        self.sparse = SpatiallySparse(sparsity, 1, **kwargs)

    def forward(self, x):
        N, C, L = x.shape
        return self.sparse(x.reshape(N * C, 1, L)).reshape(N, C, L)

    def update_sparsity(self, update_by):
        return self.sparse.update_sparsity(update_by)

    def get_sparsity(self):
        return self.sparse.get_sparsity()


class SpatiallySparseAcrossChannelsDropMotifs(nn.Module):
    def __init__(
        self,
        sparsity,
        *,
        num_channels,
        sparse_drop_motif_frequency,
        zero_count_momentum=0.1,
        **kwargs,
    ):
        super().__init__()
        self.sparse = SpatiallySparseAcrossChannels(sparsity, 1, **kwargs)
        self.initial_sparsity = sparsity
        self.sparse_decrease_frequency = sparse_drop_motif_frequency
        self.dropped = []
        self.zero_counts = torch.nn.parameter.Parameter(
            torch.zeros(num_channels), requires_grad=False
        )
        self.zero_count_momentum = zero_count_momentum

    def forward(self, x):
        x[:, self.dropped, :] = 0
        x = self.sparse(x)
        if self.training:
            zero_count_estimate = (x == 0).float().mean(0).mean(1)
            self.zero_counts.data = (
                self.zero_counts.data * (1 - self.zero_count_momentum)
                + self.zero_count_momentum * zero_count_estimate
            )
        return x

    def enough_dropped(self):
        init_nonzero = 1 - self.initial_sparsity
        next_nonzero_bar = init_nonzero * self.sparse_decrease_frequency ** (
            len(self.dropped) + 1
        )
        return (1 - self.get_sparsity()) > next_nonzero_bar

    def update_sparsity(self, update_by):
        self.sparse.update_sparsity(update_by)
        _, order = self.zero_counts.sort()
        for motif in reversed(order):
            if self.enough_dropped():
                break
            motif = motif.item()
            if motif not in self.dropped:
                print("dropping motif", motif)
                self.dropped.append(motif)

    def get_sparsity(self):
        return self.sparse.get_sparsity()


sparse_layer_types = dict(
    SpatiallySparse=SpatiallySparse,
    SpatiallySparseAcrossChannels=SpatiallySparseAcrossChannels,
    SpatiallySparseAcrossChannelsDropMotifs=SpatiallySparseAcrossChannelsDropMotifs,
)


class GaussianNoiseLayer(nn.Module):
    def __init__(self, sparsity, k):
        """
        See ``Noise Based Information Bottleneck'' in https://www.overleaf.com/read/xxwdpmhyjrcm.

        We use sparsity == delta / k to match the definition of sparsity we use in
            the other models.
        """
        super().__init__()
        self.norm = nn.BatchNorm1d(k, affine=False)
        self.sparsity = sparsity

    @property
    def sigma(self):
        return (4 ** (2 * self.sparsity) - 1) ** -0.5

    def forward(self, x):
        x = self.norm(x)
        if not self.training:
            return x
        n = self.sigma * torch.randn_like(x)
        return x + n


def get_sparse_layer(sparsity_technique, sparsity, num_motifs):
    if sparsity_technique == "percentile-by-channel":
        return SpatiallySparse(1 - sparsity, num_motifs, by_magnitude=False)
    elif sparsity_technique == "percentile-across-channels":
        return SpatiallySparseAcrossChannels(1 - sparsity, by_magnitude=False)
    elif sparsity_technique == "gaussian-noise":
        return GaussianNoiseLayer(sparsity, num_motifs)
    else:
        assert sparsity_technique in {"l1"}
        return torch.nn.ReLU()


class LearnedMotifsModel(torch.nn.Module):
    def __init__(
        self,
        *,
        num_motifs,
        psams_per_motif,
        motif_length,
        sparsity,
        sparsity_technique,
        fixed_motifs,
    ):
        super().__init__()

        if fixed_motifs != None:
            num_motifs -= fixed_motifs.count
            self.fixed_motifs = fixed_motifs

        assert motif_length % 2 == 1, (
            "motifs must have an odd length for padding to work properly."
            " Round up to the nearest odd number if necessary"
        )
        self.num_motifs = num_motifs
        self.psams_per_motif = psams_per_motif
        self.log_psams = torch.nn.Conv1d(
            4,
            num_motifs * psams_per_motif,
            motif_length,
            padding=(motif_length - 1) // 2,
        )
        self.norm = torch.nn.BatchNorm1d(num_motifs)

        self.sparse = get_sparse_layer(sparsity_technique, sparsity, num_motifs)

    def forward(self, x):
        N, _, L = x.shape
        original_x = x

        x = self.log_psams(x).reshape(N, self.num_motifs, self.psams_per_motif, L)
        x = x.exp().sum(2).log()
        pre_sparse = x
        x = self.sparse(x)

        if hasattr(self, "fixed_motifs"):
            fixed_motifs = self.fixed_motifs(original_x)
            x = torch.cat([fixed_motifs, x], axis=1)

        return x, dict(motifs_seq=pre_sparse)

    def extract_motifs(self, relative=True):
        psam_params = self.log_psams._parameters
        M = psam_params["weight"]
        M = M.reshape(self.num_motifs, self.psams_per_motif, *M.shape[1:])
        A0 = psam_params["bias"]
        A0 = A0.reshape(self.num_motifs, self.psams_per_motif, *A0.shape[1:])

        M = M.transpose(2, 3)

        M, A0 = M.detach().cpu().numpy(), A0.detach().cpu().numpy()

        print(M.shape)

        M_mod = M - M.max(3)[:, :, :, None]
        A0_mod = A0 + M.max(3).sum(2)
        if relative:
            A0_mod -= A0_mod.max()

        M_mod = np.exp(M_mod)
        A0_mod = np.exp(A0_mod)
        return M_mod, A0_mod


class SimpleResidualModel(torch.nn.Module):
    def __init__(
        self, l, w, *, sparsity, sparsity_technique, architecture, ar=1, count=4
    ):
        super().__init__()
        self.embed = torch.nn.Conv1d(4, l, 1)

        def get_layer():
            if architecture == "residual":
                return ResidualUnit(l, w, ar)
            elif architecture == "convolutional":
                return torch.nn.Sequential(
                    torch.nn.Conv1d(l, l, w, dilation=ar, padding=(ar * (w - 1)) // 2),
                    torch.nn.ReLU(),
                )
            else:
                assert False, "unreachable"

        self.net = torch.nn.Sequential(*[get_layer() for _ in range(count)])
        self.sparse = get_sparse_layer(sparsity_technique, sparsity, l)

    def forward(self, x):
        x = self.embed(x)
        x = self.net(x)
        pre_sparse = x
        x = self.sparse(x)
        return x, dict(motifs_seq=pre_sparse)


class FixedMotifs:
    def __init__(self, splicepoint_model):
        self.splicepoint_model = splicepoint_model
        self.motifs = read_motifs()

    @property
    def count(self):
        return len(self.motifs)

    def __call__(self, x):
        device, dtype = x.device, x.dtype
        x = x.detach().cpu().numpy()
        x = motifs_for(
            self.motifs, x.transpose(0, 2, 1), self.splicepoint_model
        ).transpose(0, 2, 1)
        return torch.tensor(x).type(dtype).to(device)


class SpliceAI(nn.Module):
    def __init__(
        self,
        l,
        w,
        ar,
        motifs=None,
        preprocess=None,
        sparsity=None,
        spatial_sparse=None,
        starting_channels=4, # rbns_motifs: 102, psam_motifs: 81
        use_splice_site=False,
    ):
        super().__init__()
        assert len(w) == len(ar)
        self.w = w
        self.cl = 2 * sum(ar * (w - 1))

        self.conv1 = nn.Conv1d(starting_channels, l, 1)
        self.conv2 = nn.Conv1d(l, l, 1)

        def get_mod(i):
            res = ResidualUnit(l, w[i], ar[i])
            if spatial_sparse is not None:
                res = nn.Sequential(res, SpatiallySparse(spatial_sparse, l))
            return res

        self.convstack = nn.ModuleList([get_mod(i) for i in range(len(self.w))])
        self.skipconv = nn.ModuleList(
            [
                nn.Conv1d(l, l, 1) if self._skip_connection(i) else None
                for i in range(len(self.w))
            ]
        )
        self.output = nn.Conv1d(l, 3, 1)

        self.sparsity = sparsity
        self.motifs = motifs
        if preprocess is not None:
            self.preprocess = preprocess

        self.presparse_norm = nn.BatchNorm1d(starting_channels-2, affine=False)
        self.use_splice_site = use_splice_site
        self.hooks_handles = []

    def forward(self, x, collect_intermediates=False, collect_gradients=False):
        # print('cl', self.cl)
        collect = lambda x: x if collect_intermediates else None
        x = x.transpose(1, 2)
        intermediates = {}
        if hasattr(self, "preprocess"):
            non_splice_site_motifs, splice_site_motifs, extras = self.preprocess(motifs=self.motifs, x=x, use_splice_site=self.use_splice_site)
            intermediates.update(extras.items())
        
        if self.sparsity is not None:
            non_splice_site_motifs = self.presparse_norm(non_splice_site_motifs)

            x = non_splice_site_motifs

            B, C, L = x.shape
            # to_drop = int(B * L * C * (1 - self.sparsity))
            # threshold, _ = torch.kthvalue(torch.flatten(non_splice_site_motifs), k=to_drop)
            # non_splice_site_motifs[non_splice_site_motifs < threshold] = 0 
            to_drop = max(1, int(B * L * (1 - self.sparsity)))
            thresholds, _ = torch.kthvalue(
                x.transpose(1, 2).reshape(B * L, C), k=to_drop, dim=0
            )
            mask = x > thresholds[None, :, None]
            x = x * mask

            splice_site_motifs[splice_site_motifs < -10.0] = 0
        
            x = torch.cat((x, splice_site_motifs), dim=1)
        
        if self.use_splice_site:
            x = torch.cat((x, splice_site_motifs), dim=1)

        conv = self.conv1(x)
        skip = self.conv2(conv)

        intermediates["skips"] = [collect(skip)]

        for i in range(len(self.w)):
            conv = self.convstack[i](conv)

            if self._skip_connection(i):
                # Skip connections to the output after every 4 residual units
                skip = skip + self.skipconv[i](conv)
                intermediates["skips"].append(collect(skip))
        
        skip = skip[:, :, self.cl // 2 : -self.cl // 2]

        y = self.output(skip)

        y = y.transpose(1, 2)
        if collect_gradients:
            [s.retain_grad() for s in intermediates["skips"]]

        if collect_intermediates:
            intermediates["output"] = y
            return intermediates

        return y

    def _skip_connection(self, i):
        return ((i + 1) % 4 == 0) or ((i + 1) == len(self.w))

    def enable_guided_backprop(self):
        if not hasattr(self, "hooks_handles"):
            self.hooks_handles = []
        assert not self.hooks_handles, "already in guided backprop mode"
        for submodule in self.modules():
            if isinstance(submodule, nn.ReLU):
                handle = submodule.register_backward_hook(self._relu_hook)
                self.hooks_handles.append(handle)

    def disable_guided_backprop(self):
        if not hasattr(self, "hooks_handles"):
            self.hooks_handles = []
        for handle in self.hooks_handles:
            handle.remove()
        self.hooks_handles = []

    @staticmethod
    def _relu_hook(module, grad_in, grad_out):
        if isinstance(module, nn.ReLU):
            return (torch.clamp(grad_in[0], min=0),)


class SpatialSoftmax(nn.Module):
    def __init__(self, width, activation="exp"):
        super().__init__()
        self.width = width
        self.activation = activation
        assert self.width % 2 == 1

    def forward(self, z):
        _, _, C = z.shape
        original = z
        if self.activation == "exp":
            z = z.exp()
        elif self.activation == "abs":
            z = z.abs() + 1e-9
        else:
            raise RuntimeError(f"Invalid activation: {self.activation}")
        after_activation = z
        z = z.transpose(1, 2)
        z = torch.nn.functional.conv1d(
            z, torch.ones(C, C, self.width).to(z.device), padding=self.width // 2
        )
        z = z.transpose(1, 2)
        return original * after_activation / z


class SpatialExponentialDepression(nn.Module):
    def __init__(self, width, num_features):
        print(width, num_features)
        super().__init__()
        self.width = width
        assert self.width % 2 == 1
        self.norm = nn.BatchNorm1d(num_features)
        self.max_pool = nn.MaxPool1d(self.width, padding=self.width // 2, stride=1)

    def forward(self, z):
        z = z.transpose(1, 2)
        z = self.depression(z) * z
        z = z.transpose(1, 2)
        return z

    def depression(self, z):
        z = self.norm(z)

        pool, _ = z.max(1)
        pool = pool.unsqueeze(1)
        pool = self.max_pool(pool)

        z = z - pool
        return z.exp()


class InfluenceNetwork(nn.Module):
    cl = 0

    def __init__(
        self,
        *,
        n_motifs,
        w_motif,
        sparsity,
        transformer_heads=10,
        transformer_layers=6,
        spatial_softmax_width=25,
        out_layers=2,
        out_layer="lstm",
        cl=0,
    ):
        assert w_motif % 4 == 2
        assert transformer_layers % 2 == 0
        super().__init__()
        self.initial_embed = nn.Conv1d(4, n_motifs, 1)
        self.motifs_layer = ResidualUnit(n_motifs, w_motif // 2, ar=1)
        self.sparsity = SpatiallySparse(1 - sparsity, n_motifs, by_magnitude=False)
        self.positional_encoding = PositionalEncoding(n_motifs)
        self.transformer = nn.Transformer(
            n_motifs,
            nhead=transformer_heads,
            num_encoder_layers=transformer_layers // 2,
            num_decoder_layers=transformer_layers // 2,
        )
        self.cl = cl
        if spatial_softmax_width is not None:
            self.spatial_softmax = SpatialSoftmax(
                spatial_softmax_width, activation="abs"
            )
        else:
            self.spatial_softmax = nn.Identity()
        if out_layer == "lstm":
            self.out_layer = nn.LSTM(n_motifs, n_motifs, out_layers, bidirectional=True)
            self.final_projection = nn.Linear(2 * n_motifs, 3)
        else:
            raise RuntimeError(f"Bad out_layer {out_layer}")

    def forward(self, x, collect_intermediates=False):
        x = x.transpose(1, 2)
        x = self.initial_embed(x)
        x = self.motifs_layer(x)
        x = self.sparsity(x)
        x = x.transpose(1, 2)
        x = x.transpose(0, 1)
        x_pos = self.positional_encoding(x)
        influence = self.transformer(x_pos, x_pos)
        x = x * influence
        x = x.transpose(0, 1)
        x = self.spatial_softmax(x)
        x = x.transpose(0, 1)
        x, _ = self.out_layer(x)
        x = x.transpose(0, 1)
        x = x[:, self.cl // 2 : x.shape[1] - self.cl // 2]
        x = self.final_projection(x)
        if collect_intermediates:
            return dict(output=x)
        return x


class DownstreamModel(torch.nn.Module, ABC):
    def __init__(self, *, preprocess, starting_channels):
        super().__init__()
        self.preprocess = preprocess
        self.starting_channels = starting_channels

    def forward(self, x, collect_intermediates=False):
        x = x.transpose(1, 2)
        intermediates = {}
        x, extras = self.preprocess(x)
        intermediates.update(extras.items())
        y = self.downstream_forward(x)

        y = y.transpose(1, 2)
        if collect_intermediates:
            intermediates["output"] = y
            return intermediates

        return y

    @abstractmethod
    def downstream_forward(self, x):
        pass


class DownstreamSingleConvModel(DownstreamModel):
    def __init__(self, *, cl, hidden_size, layers, **kwargs):
        super().__init__(**kwargs)
        assert cl % 2 == 0, "even cl only to make the padding correct"
        self.conv = torch.nn.Conv1d(self.starting_channels, hidden_size, cl + 1)
        self.fcnet = torch.nn.ModuleList(
            [torch.nn.Conv1d(hidden_size, hidden_size, 1) for _ in range(layers)]
        )
        self.out = torch.nn.Conv1d(hidden_size, 3, 1)
        self.activation = torch.nn.ReLU()

    def downstream_forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        for layer in self.fcnet:
            x = layer(x)
            x = self.activation(x)
        x = self.out(x)
        return x


class DownstreamSingleAttentionModel(DownstreamModel):
    """
    Standard multi-head attention looks like

        e_{h,ij} = Q_h(m_j) K_h(m_i)^T

        m'_{h,j} = attention(e_{h,*j}, V_h(m_*))

    with positional embeddings being added to the inputs m_i

    There are two problems for this when it comes to our problem. First, we want to
        restrict the dataflow window of the model to CL and second we want to
        use a relative position embedding that is shift invariant.

    Using https://arxiv.org/pdf/2005.09940.pdf's approach to using a difference-based
        positional encoding, we can expand out our  definitions as so:

    q_{h,i} = m_i Q_h
    k_{h,i} = m_i K_h
    v_{h,i} = m_i V_h

    PE_{h,d} = p_d R_h

    e_{h,ij} = q_{h,i} k_{h,j}^T + q_{h,i} PE_{h,i-j}^T + alpha * k_{h,j} + beta * PE_{h,i-j}
    a_{h,ij} = exp(e_{h,ij}) / sum_j' exp(e_{h,ij'})
    b_{h,ij} = a_{h,ij} * 1(|i - j| < cl)
    m'_{h,i} = sum_{j} b_{h,ij} v_{h,j}

    we can simplify this somewhat by subsuming the bias values into making the q,k,v layers affine:


    e_{h,ij} = q_{h,j} k_{h,i}^T + q_{h,j} PE_{h,i-j}^T

    additionally, since we have a fairly small cl window, we can turn PE into a lookup table


    We can then do a change of variables j -> d as such

    e_{h,id} = q_{h,i} k_{h,i+d}^T + q_{h,i} PE_{h,d}^T
    a_{h,id} = exp(e_{h,id}) / sum_d' exp(e_{h,id'})
    m'_{h,i} = sum_{d} b_{h,id} v_{h,i+d}
    """

    def __init__(self, *, cl, embedding_size, hidden_size, layers, **kwargs):
        super().__init__(**kwargs)

        self.Q = torch.nn.Linear(self.starting_channels, hidden_size * embedding_size)
        self.K = torch.nn.Linear(self.starting_channels, hidden_size * embedding_size)
        self.V = torch.nn.Linear(self.starting_channels, hidden_size)
        self.PE = torch.nn.Parameter(torch.rand(cl, hidden_size, embedding_size))
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.cl = cl
        self.activation = torch.nn.ReLU()

        self.fcnet = torch.nn.ModuleList(
            [torch.nn.Conv1d(hidden_size, hidden_size, 1) for _ in range(layers)]
        )

        self.out = torch.nn.Conv1d(hidden_size, 3, 1)

    def downstream_forward(self, x):
        x = x.transpose(1, 2)
        N, L, C = x.shape
        cl = self.cl

        H = self.hidden_size
        E = self.embedding_size

        q = self.Q(x).reshape(N, L, H, E)
        k = self.K(x).reshape(N, L, H, E)
        v = self.V(x)

        i_idxs = torch.arange(L - cl)
        d_idxs = torch.arange(cl)
        off_idxs = i_idxs[:, None] + d_idxs[None, :]

        assert off_idxs.shape == (L - cl, cl)

        q_expanded = q[:, cl // 2 : L - cl // 2, None, :]
        k_expanded = k[:, off_idxs.reshape(-1), :].view(N, L - cl, cl, H, E)
        pe_expanded = self.PE[None, None, :, :]
        v_expanded = v[:, off_idxs.reshape(-1), :].view(N, L - cl, cl, H)

        e = (q_expanded * (pe_expanded + k_expanded)).sum(-1)

        del q_expanded
        del k_expanded

        a = e.softmax(2)

        a = a.transpose(2, 3)
        v_expanded = v_expanded.transpose(2, 3)

        out = batchdot(a, v_expanded)

        out = out.transpose(1, 2)

        out = self.activation(out)

        for layer in self.fcnet:
            out = layer(out)
            out = self.activation(out)
        out = self.out(out)
        return out


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


class StretchData(torch.utils.data.IterableDataset):
    def __init__(self, underlying_data, stretch_factor):
        self.data = underlying_data
        self.stretch_factor = stretch_factor

    def _stretch(self, x):
        return np.repeat(x, self.stretch_factor, axis=0)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x, y in self.data:
            yield self._stretch(x), self._stretch(y)


class RandomizeInputs(torch.utils.data.IterableDataset):
    def __init__(self, underlying_data, randomize_perc, seed=0):
        self.data = underlying_data
        self.r = randomize_perc
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x, y in self.data:
            x = x.copy()
            l = x.shape[0]
            indices = self.rng.choice(l, size=int(self.r * l), replace=False)
            rbases = self.rng.choice(4, size=indices.size, replace=True)
            x[indices] = np.eye(4)[rbases]
            yield x, y


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
        # print('yt', sum(yt), len(yt))
        # print('yp', yp)
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


def load_model(folder, attr, step=None):
    def hook(m):
        if hasattr(m, "_load_hook"):
            m._load_hook()
        return m

    kwargs = {}
    if not torch.cuda.is_available():
        kwargs = dict(map_location=torch.device("cpu"))

    if os.path.isfile(folder):
        return None, hook(torch.load(folder, **kwargs))

    model_dir = os.path.join(folder, f"model_{attr}")
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
