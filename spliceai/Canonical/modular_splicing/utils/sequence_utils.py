import numpy as np
import torch


def all_seqs(n, *, amount=4):
    """
    Generates all sequences of length n, with the given amount of nucleotides.
    """
    if n == 0:
        yield []
        return
    for s in all_seqs(n - 1, amount=amount):
        for i in np.eye(amount):
            yield [i] + s


def pack_sequences(seqs):
    """
    Encode the given sequences as single integers.
    """
    return seqs @ (4 ** np.arange(seqs.shape[-1]))


def decode_packed_seq(seqs, k):
    """
    Decode a packed sequences of length n
    """
    seqs = seqs[:, None]
    seqs = seqs % (4 ** (np.arange(k) + 1))
    seqs = seqs // 4 ** np.arange(k)
    return seqs


def slice_packed_seq(seqs, start, end):
    """
    Slice a packed sequence in low-bit-first order
    to get a subsequence from start (inclusive) to end (exclusive)
    """
    return (seqs % 4**end) // 4**start


def map_to_all_seqs_idx(x, motif_width):
    """
    Compute the index of every element in the given sequence, with the given motif width.

    I.e., each returned position is an encoding of the entire sequence of length motif_width
    centered at that position.
    """
    assert (
        motif_width * 2 <= 50
    ), "motif_width larger than what can fit in an double with some headroom"
    floats = torch.nn.functional.conv1d(
        x[:, None, :].double(),
        (4 ** torch.arange(motif_width, dtype=torch.double, device=x.device))[
            None, None
        ],
        padding=motif_width // 2,
    )
    floats = floats.squeeze(1)
    if motif_width * 2 <= 32:
        return floats.int()
    return floats.long()


def draw_bases(xs):
    if xs.dtype == np.int and 0 < xs.max() < 4:
        xs = np.eye(4)[xs]
    assert xs.shape[-1] == 4 and len(xs.shape) > 1
    if len(xs.shape) == 2:
        mask = (xs == 0).all(-1)
        xs = xs.argmax(-1)
        xs = np.array(list("ACGT"))[xs]
        xs[mask] = "N"
        return "".join(xs)
    return [draw_bases(x) for x in xs]


def parse_bases(x):
    x = x.upper()
    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "N": [0, 0, 0, 0],
    }
    return np.array([mapping[c] for c in x])
