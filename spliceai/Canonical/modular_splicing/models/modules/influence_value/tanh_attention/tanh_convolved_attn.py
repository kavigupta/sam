import torch
import torch.nn as nn
from modular_splicing.models.modules.constant import Constant

from modular_splicing.models.modules.positional_encoding import PositionalEncoding
from modular_splicing.utils.construct import construct


class TanhConvolvedAttnLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        window,
        num_outputs,
        num_heads,
        forward_only,
        use_ocpe=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.window = window
        self.num_heads = num_heads
        self.num_outputs = num_outputs
        self.forward_only = forward_only

        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.proj_k = nn.Linear(embed_dim, embed_dim)

        if use_ocpe:
            self._ocpe = nn.Parameter(
                torch.zeros((window + 1, num_heads), dtype=torch.float32)
            )

    def forward(self, q, k, v):
        assert not self.forward_only, "Forward only not supported yet"
        q, k, v = (
            q.reshape(q.shape[0], q.shape[1], self.num_heads, self.embed_dim),
            k.reshape(k.shape[0], k.shape[1], self.num_heads, self.embed_dim),
            v.reshape(v.shape[0], v.shape[1], self.num_heads, self.num_outputs),
        )
        q = self.proj_q(q)
        k = self.proj_k(k)
        attn, out = compute_unnormalized_attn_within_window(
            q,
            k,
            v,
            ocpe=self._ocpe if hasattr(self, "_ocpe") else None,
            window=self.window,
            zero_value=self.zero_value(),
            activation=self.activation,
        )
        return attn, out

    def zero_value(self):
        return 0

    def activation(self, x):
        return torch.tanh(x)


class SoftmaxConvolvedAttnLayer(TanhConvolvedAttnLayer):
    def zero_value(self):
        return -1000

    def activation(self, x):
        return x.softmax(dim=2)


class SigmoidConvolvedAttnLayer(TanhConvolvedAttnLayer):
    def zero_value(self):
        return -1000

    def activation(self, x):
        return x.sigmoid()


class OperationOnAttentionLayers(nn.Module):
    def __init__(self, first_spec, second_spec, operation_spec, **kwargs):
        super().__init__()
        self.first = construct(attention_layer_types(), first_spec, **kwargs)
        self.second = construct(attention_layer_types(), second_spec, **kwargs)
        self.operation_spec = operation_spec

    def forward(self, q, k, v):
        attn1, out1 = self.first(q, k, v)
        attn2, out2 = self.second(q, k, v)
        out = construct(
            dict(sub=lambda x, y: x - y, mul=lambda x, y: x * y),
            self.operation_spec,
            x=out1,
            y=out2,
        )
        return dict(attn1=attn1, attn2=attn2), out


def attention_layer_types():
    return dict(
        TanhConvolvedAttnLayer=TanhConvolvedAttnLayer,
        SigmoidConvolvedAttnLayer=SigmoidConvolvedAttnLayer,
        SoftmaxConvolvedAttnLayer=SoftmaxConvolvedAttnLayer,
        OperationOnAttentionLayers=OperationOnAttentionLayers,
    )


def compute_unnormalized_attn_within_window(
    q, k, v, *, ocpe=None, zero_value, activation, window
):
    """
    Compute unnormalized attention within a window.

    Parameters:
        q: (batch_size, seq_len, num_heads, embed_dim)
        k: (batch_size, seq_len, num_heads, embed_dim)
        v: (batch_size, seq_len, num_heads, num_outputs)
        ocpe: (window, num_heads)
        zero_value: float
        activation: callable[torch.Tensor, torch.Tensor]
            takes in a tensor of shape (batch_size, seq_len, seq_len, num_heads)
                and returns a tensor of the same shape.
            examples can include pointwise computations, or something like `lambda x: x.softmax(dim=2)`
        window: int

    Returns:
        attn_flat: (batch_size, seq_len, window + 1, num_heads)
            computed as if:
                attn_unnorm, attn: (batch_size, seq_len, seq_len, num_heads)
                attn_unnorm[n, i, j, h] = q[n, i, h] @ k[n, j, h] + ocpe[j-i, h] if abs(i - j) <= window // 2 else zero_value
                attn = activation(attn_unnorm)
                attn_flat[n, i, d, h] = attn[n, i, i + (d - window//2), h] # out of bounds values are zero
        result: (batch_size, seq_len, num_outputs)
            computed as if:
                result[n, i] = sum_(j | abs(i - j) <= window // 2) attn[n, i, j - i] * v[n, j]
    """
    # let C be the size of the chunk used to compute the results
    # L = seq_len, W = window
    #
    # The total cost to compute one chunk is (C + W)^2
    # The total cost to compute all chunks is L / C * (C + W)^2
    # This is minimized when we minimize (C^2 + 2CW + W^2) / C
    # which can be expressed as C + 2W + W^2 / C
    # This is minimized when C = W

    assert window % 2 == 0
    assert q.shape == k.shape
    _, seq_len, _, _ = q.shape
    padding = window // 2

    assert ocpe is None or ocpe.shape == (window + 1, q.shape[2])
    # pad along the -2 axis
    q, k, v = [
        torch.nn.functional.pad(x, (0, 0, 0, 0, padding, padding)) for x in (q, k, v)
    ]

    chunk_size = window
    assert seq_len > 0
    attns = []
    results = []
    for out_start in range(0, seq_len, chunk_size):
        out_end = min(out_start + chunk_size, seq_len)
        chunk_start = out_start
        chunk_end = out_end + 2 * padding

        q_chunk = q[:, chunk_start:chunk_end]
        k_chunk = k[:, chunk_start:chunk_end]
        v_chunk = v[:, chunk_start:chunk_end]
        chunk_attn = torch.einsum("nxhd,nyhd->nxyh", q_chunk, k_chunk)
        idx_i, idx_j = (
            torch.arange(chunk_attn.shape[1], device=chunk_attn.device)[:, None],
            torch.arange(chunk_attn.shape[2], device=chunk_attn.device)[None],
        )
        within_window = (idx_i - idx_j).abs() <= window // 2
        if ocpe is not None:
            indices = idx_j - idx_i
            chunk_attn[:, within_window] += ocpe[window // 2 + indices[within_window]]
        # mask out values outside the window
        mask = ~within_window
        # mask out values originating from padding
        mask = (
            mask
            | (idx_i + chunk_start < padding)
            | (idx_i + chunk_start >= seq_len + padding)
        )
        mask = (
            mask
            | (idx_j + chunk_start < padding)
            | (idx_j + chunk_start >= seq_len + padding)
        )
        chunk_attn[:, mask] = zero_value

        chunk_attn = activation(chunk_attn)
        result = torch.einsum("nxyh,nyhd->nxd", chunk_attn, v_chunk)
        idxs_start = torch.arange(0, out_end - out_start)[:, None]
        idxs_window = torch.arange(window + 1)[None]
        chunk_attn = chunk_attn[:, idxs_start + window // 2, idxs_start + idxs_window]

        results.append(result[:, padding:-padding])
        attns.append(chunk_attn)

    results = torch.cat(results, dim=1)
    attns = torch.cat(attns, dim=1)
    return attns, results


class TanhConvolvedAttnOnIndices(nn.Module):
    def __init__(
        self,
        *,
        total_dim,
        input_idxs,
        output_idxs,
        embed_dim,
        num_outputs,
        window,
        max_len,
        num_heads=1,
        forward_only=False,
        tanh_attn_layer_spec,
        v_proj_spec=dict(type="Linear"),
        positional_encoding_spec=dict(type="PositionalEncoding", dropout=0),
        reproject_after_pe=False,
    ):
        super().__init__()

        self.total_dim = total_dim
        self.input_idxs = input_idxs
        self.output_idxs = output_idxs
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_outputs = num_outputs

        self.proj_q = nn.Linear(len(input_idxs), num_heads * embed_dim)
        self.proj_k = nn.Linear(len(input_idxs), num_heads * embed_dim)
        if reproject_after_pe:
            self.reproj_q = nn.Linear(num_heads * embed_dim, num_heads * embed_dim)
            self.reproj_k = nn.Linear(num_heads * embed_dim, num_heads * embed_dim)
        self.proj_v = construct(
            dict(
                Linear=lambda inp, out: nn.Linear(inp, out),
                Constant=lambda inp, out: Constant(out),
            ),
            v_proj_spec,
            inp=len(output_idxs),
            out=num_heads * num_outputs,
        )
        self.positional_encoding = construct(
            dict(
                PositionalEncoding=PositionalEncoding,
                Identity=lambda d_model, max_len: nn.Identity(),
            ),
            positional_encoding_spec,
            d_model=embed_dim,
            max_len=max_len,
        )
        self.attn_layer = construct(
            attention_layer_types(),
            tanh_attn_layer_spec,
            embed_dim=embed_dim,
            window=window,
            num_outputs=num_outputs,
            num_heads=num_heads,
            forward_only=forward_only,
        )

    def forward(self, x, collect_intermediates=False):
        del collect_intermediates
        q, k, v = (
            x[:, :, self.input_idxs],
            x[:, :, self.input_idxs],
            x[:, :, self.output_idxs],
        )
        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        L, N, CH = q.shape
        q = q.reshape(L, N, self.num_heads, self.embed_dim)
        k = k.reshape(L, N, self.num_heads, self.embed_dim)
        q = q.reshape(L, N * self.num_heads, self.embed_dim)
        k = k.reshape(L, N * self.num_heads, self.embed_dim)
        q = self.positional_encoding(q)
        k = self.positional_encoding(k)
        q = q.reshape(L, N, self.num_heads, self.embed_dim)
        k = k.reshape(L, N, self.num_heads, self.embed_dim)
        q = q.reshape(L, N, CH)
        k = k.reshape(L, N, CH)
        if hasattr(self, "reproj_q"):
            q = self.reproj_q(q)
        if hasattr(self, "reproj_k"):
            k = self.reproj_k(k)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)

        attn_matrix, out = self.attn_layer(q, k, v)
        return dict(attn_matrix=attn_matrix, output=out)
