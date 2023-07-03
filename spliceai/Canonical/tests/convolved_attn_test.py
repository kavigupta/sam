import unittest
import torch

from modular_splicing.models.modules.influence_value.tanh_attention.tanh_convolved_attn import (
    compute_unnormalized_attn_within_window,
)


def compute_attn_directly(q, k, v, *, ocpe=None, zero_value, activation, window):
    """
    See compute_unnormalized_attn_within_window for documentation on what is going on here.
    """
    batch_size, seq_len, num_heads, embed_dim = q.shape
    assert q.shape == k.shape
    assert q.shape[:-1] == v.shape[:-1]
    out_dim = v.shape[-1]
    attn_unnorm = (
        torch.zeros((batch_size, seq_len, seq_len, num_heads), dtype=torch.float32)
        + zero_value
    )
    for n in range(batch_size):
        for i in range(seq_len):
            for j in range(seq_len):
                for h in range(num_heads):
                    if abs(i - j) <= window // 2:
                        attn_unnorm[n, i, j, h] = q[n, i, h] @ k[n, j, h]
                        if ocpe is not None:
                            assert ocpe.shape[0] == window + 1
                            attn_unnorm[n, i, j, h] += ocpe[window // 2 + j - i, h]
    attn = activation(attn_unnorm)
    attn_flat = torch.zeros(
        (batch_size, seq_len, window + 1, num_heads), dtype=torch.float32
    )
    assert attn_flat.shape == (batch_size, seq_len, window + 1, num_heads)
    for n in range(batch_size):
        for i in range(seq_len):
            for d in range(window + 1):
                for h in range(num_heads):
                    if i + d - window // 2 < 0 or i + d - window // 2 >= seq_len:
                        continue
                    attn_flat[n, i, d, h] = attn[n, i, i + d - window // 2, h]

    result = torch.zeros((batch_size, seq_len, out_dim), dtype=torch.float32)
    for n in range(batch_size):
        for i in range(seq_len):
            for j in range(seq_len):
                for h in range(num_heads):
                    if abs(i - j) <= window // 2:
                        result[n, i] += (
                            attn_flat[n, i, j - i + window // 2, h] * v[n, j, h]
                        )
    return attn_flat, result


class TestDirectConvolvedAttention(unittest.TestCase):
    """
    This test is to verify that the direct implementation of the convolved attention
    matches the specification we have in our head.
    """

    def verify_correct(self, q, k, v, attn_out, out, ocpe=None, **kwargs):
        # use a single batch
        q, k, v, attn_out, out = [
            torch.tensor(x, dtype=torch.float32)[None] for x in [q, k, v, attn_out, out]
        ]
        if ocpe is not None:
            ocpe = torch.tensor(ocpe, dtype=torch.float32)
        attn_out_direct, out_direct = compute_attn_directly(
            q, k, v, ocpe=ocpe, **kwargs
        )
        print("attn_out_direct", attn_out_direct)
        print("attn_out", attn_out)
        print("out_direct", out_direct)
        print("out", out)
        self.assertTrue((attn_out_direct - attn_out).max() < 1e-3)
        self.assertTrue((out_direct - out).max() < 1e-3)

    def test_1x1(self):
        # q, k, v : batch_size, seq_len, num_heads, *
        q = [[[1, 2, 3]]]
        k = [[[4, 5, 6]]]
        v = [[[1, -1, 2]]]
        attn_out = [[[[0], [32], [0]]]]
        out = [[32, -32, 64]]
        self.verify_correct(
            q, k, v, attn_out, out, zero_value=0, activation=lambda x: x, window=2
        )

    def test_nonzero_sequence(self):
        q = [[[1, 2, 3]], [[1, 0, 1]], [[-2, 1, 1]]]
        k = [[[4, 5, 6]], [[1, 2, 1]], [[1, 1, 1]]]
        v = [[[1, -1, 2]], [[1, 1, 1]], [[1, 1, 1]]]

        # 32 = [1, 2, 3] @ [4, 5, 6]
        # 8 = [1, 2, 3] @ [1, 2, 1]

        attn_out = [[[0], [32], [8]], [[10], [2], [2]], [[1], [0], [0]]]
        # [40, -24, 72] = 32 * [1, -1, 2] + 8 * [1, 1, 1]
        # [14, -6, 24] = 10 * [1, -1, 2] + 2 * [1, 1, 1] + 2 * [1, 1, 1]
        # [1, 1, 1] = 1 * [1, 1, 1] + 0 * [1, 1, 1]
        out = [[40, -24, 72], [14, -6, 24], [1, 1, 1]]
        self.verify_correct(
            q, k, v, attn_out, out, zero_value=0, activation=lambda x: x, window=2
        )

    def test_nonzero_sequence_ocpe(self):
        q = [[[1, 2, 3]], [[1, 0, 1]], [[-2, 1, 1]]]
        k = [[[4, 5, 6]], [[1, 2, 1]], [[1, 1, 1]]]
        v = [[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]]

        ocpe = [[-5], [7], [3]]

        # 39 = [1, 2, 3] @ [4, 5, 6] + 7
        # 11 = [1, 2, 3] @ [1, 2, 1] + 3

        attn_out = [[[0], [39], [11]], [[5], [9], [5]], [[-4], [7], [0]]]

        out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.verify_correct(
            q,
            k,
            v,
            attn_out,
            out,
            ocpe=ocpe,
            zero_value=0,
            activation=lambda x: x,
            window=2,
        )


class TestConvolvedAttn(unittest.TestCase):
    def verify_x(self, q, k, v, **kwargs):
        attn, out = compute_unnormalized_attn_within_window(q, k, v, **kwargs)

        attn_direct, out_direct = compute_attn_directly(q, k, v, **kwargs)

        print("attn", attn)
        print("attn_direct", attn_direct)

        print("attn shape", attn.shape)
        print("attn_direct shape", attn_direct.shape)

        self.assertTrue((attn - attn_direct).max() < 1e-3)

        print("out", out)
        print("out_direct", out_direct)

        print("out shape", out.shape)
        print("out_direct shape", out_direct.shape)

        self.assertTrue((out - out_direct).max() < 1e-3)

    def test_basic(self):
        q = torch.randn((3, 5, 2, 5))
        k = torch.randn((3, 5, 2, 5))
        v = torch.randn((3, 5, 2, 3))
        self.verify_x(q, k, v, zero_value=0, activation=torch.tanh, window=2)

    def test_multiple_windows(self):
        q = torch.randn((3, 20, 2, 5))
        k = torch.randn((3, 20, 2, 5))
        v = torch.randn((3, 20, 2, 3))
        for window in [2, 4, 6, 12, 14]:
            self.verify_x(q, k, v, zero_value=0, activation=torch.tanh, window=window)

    def test_attn(self):
        q = torch.randn((1, 3, 1, 1))
        k = torch.randn((1, 3, 1, 1))
        v = torch.randn((1, 3, 1, 1))
        self.verify_x(
            q,
            k,
            v,
            zero_value=-float("inf"),
            activation=lambda x: x.softmax(dim=2),
            window=2,
        )

    def test_ocpe(self):
        q = torch.randn((3, 20, 2, 5))
        k = torch.randn((3, 20, 2, 5))
        v = torch.randn((3, 20, 2, 3))
        ocpe = torch.randn((15, 2))
        self.verify_x(
            q,
            k,
            v,
            ocpe=ocpe,
            zero_value=0,
            activation=torch.tanh,
            window=14,
        )
