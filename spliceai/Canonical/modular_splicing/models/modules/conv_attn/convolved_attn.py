import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiheadattention import CustomMultiheadAttention


class ConvolvedAttention(nn.Module):
    """
    Attention where the attention takes place over a window of finite size.

    This is done for two reasons:
        1. To reduce the computational cost of attention.
        2. To make the attention more interpretable.

    Also returns the attention matrix.

    Args:
    ------
    window: int
        The size of the window to use for attention.
    padding: str
        The padding to use for the window. Must be `same`.
    interpretable: bool
        Whether to make the attention more interpretable. If True, just does a batch matrix multiply of the
        attention matrix and the values.
    forward_only: bool
        Whether to only allow attention to the future. This is useful if we want to ensure that only
            upstream information is used.
    """

    def __init__(
        self,
        *args,
        window,
        padding="same",
        interpretable=False,
        forward_only=True,
        **kwargs
    ):
        super().__init__()
        assert window % 2 == 1
        assert padding == "same"
        self.attn = CustomMultiheadAttention(*args, **kwargs)
        self.padding = padding
        self.window = window
        self.interpretable = interpretable
        self.forward_only = forward_only

    def forward_unbatched(self, query, key, value, collect_intermediates):
        within_cl, allow = self._compute_masks(query, key)
        # this is weirdly backwards (False => use), see the docs.
        out, matr = self.attn(query, key, value, attn_mask=~allow)
        if len(matr.shape) == 3:
            matr = matr.unsqueeze(1)
        matr = matr.permute(0, 2, 3, 1)
        num_heads = matr.shape[-1]

        result = {}
        if collect_intermediates:
            full_matr = torch.zeros(
                (query.shape[1], query.shape[0], self.window, num_heads)
            ).to(query.device)
            source, target = torch.meshgrid(
                torch.arange(query.shape[0]),
                torch.arange(-(self.window // 2), 1 + self.window // 2),
                indexing="ij",
            )
            full_matr_idxs = source + target
            full_matr_pos = (full_matr_idxs >= 0) & (full_matr_idxs < value.shape[0])
            full_matr[:, full_matr_pos.to(query.device)] = matr[:, within_cl]
            result["attn_matr"] = full_matr

        if self.interpretable:
            # matr : (N, L, L, H)
            matr = matr.permute(0, 3, 1, 2)
            # value: (L, N, E); matr : (N, H, L, L)
            value = value.permute(1, 0, 2)
            # value: (N, L, E)
            value = value.reshape(
                value.shape[0], value.shape[1], value.shape[2] // num_heads, num_heads
            )
            # value: (N, L, E/H, H)
            value = value.permute(0, 3, 1, 2)
            # value: (N, H, L, E/H)
            out = self.multibatch_bmm(matr, value)
            # out: (N, H, L, E/H)
            out = out.permute(0, 2, 3, 1)
            # out: (N, L, E/H, H)
            out = out.reshape(out.shape[0], out.shape[1], -1)
            # out: (N, L, E)
            out = out.permute(1, 0, 2)
            # out : (L, N, E)

        result["output"] = out
        return result

    def _compute_masks(self, query, key):
        target, source = torch.meshgrid(
            torch.arange(query.shape[0]),
            torch.arange(key.shape[0]),
            indexing="ij",
        )
        allow = within_cl = (source - target).abs() <= self.window // 2

        if getattr(self, "forward_only", False):
            allow = allow & (target >= source)
        allow = allow.to(query.device)
        return within_cl, allow

    @staticmethod
    def multibatch_bmm(a, b):
        """
        Like torch.bmm but works with an arbitrary number of batch dimensions
        """
        batch_idxs = a.shape[:-2]
        assert batch_idxs == b.shape[:-2]
        a = a.reshape(-1, *a.shape[-2:])
        b = b.reshape(-1, *b.shape[-2:])
        ab = torch.bmm(a, b)
        return ab.reshape(*batch_idxs, *ab.shape[-2:])

    def forward_batched(self, query, key, value, collect_intermediates):
        L, N, E = query.shape
        assert query.shape == key.shape == value.shape
        W = self.window

        """
        Computational cost of chunks of length V:
            number of chunks: L / V
            cost per chunk:
                size of array created: V + W, V + W
            overall cost (V + W)^2 / V * L
        Minimize: num' den = den' num
            2(V + W) * V = (V + W)^2
            2V = (V + W)
            V = W
        """

        results = []
        results_attn = []

        V = W - 1
        n_chunks = (L + V - 1) // V
        for i in range(n_chunks):
            start, end = i * V, (i + 1) * V

            cl_left = min(start - 0, W // 2)
            cl_right = max(0, min(L - end, W // 2 + 1))

            selected = slice(start - cl_left, end + cl_right)

            full_result = self.forward_unbatched(
                query[selected], key[selected], value[selected], collect_intermediates
            )

            result = full_result.pop("output")

            idx_slice = slice(cl_left, result.shape[0] - cl_right)

            result = result[idx_slice]
            results.append(result)

            if collect_intermediates:
                results_attn.append(full_result.pop("attn_matr")[:, idx_slice])

            assert not full_result

        output = dict(output=torch.cat(results, dim=0))
        if collect_intermediates:
            output["attn_matrix"] = torch.cat(results_attn, dim=1)
        return output

    def forward(self, query, key, value, *, collect_intermediates):
        """
        Inputs
            query: (L, N, E)
            key: (L, N, E)
            value: (L, N, E)

        Outputs
            attn_output: (L, N, E)
        """
        return self.forward_batched(query, key, value, collect_intermediates)
