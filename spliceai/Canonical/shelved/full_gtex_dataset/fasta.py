from functools import lru_cache
import os

import pickle
import tqdm.auto as tqdm

import numpy as np


def read_fasta_strings(fasta_path):
    with open(fasta_path) as f:
        lines = []
        for line in f:
            assert line[-1] == "\n"
            lines.append(line[:-1])
        lines.append(">")
    results = {}
    current_key = None
    current = []
    for line in tqdm.tqdm(lines):
        if line[0] == ">":
            if current_key is not None:
                results[current_key] = current
            current_key, current = line[1:], []
            continue
        current.append(line)
    assert current_key == "" and current == []
    results = {k: results[k] for k in results if "_" not in k and k != "chrM"}
    return results


def seq_to_one_hot(seq):
    seq = np.array(
        list(
            seq.upper()
            .encode("utf-8")
            .replace(b"A", b"\x01")
            .replace(b"C", b"\x02")
            .replace(b"G", b"\x03")
            .replace(b"T", b"\x04")
            .replace(b"N", b"\x00")
        )
    )
    return np.eye(4, dtype=np.uint8)[seq - 1] * (seq != 0)[:, None]


def _convert_to_one_hot(fasta_path, one_hot_path):
    results = read_fasta_strings(fasta_path)
    results_one_hot = {
        k: seq_to_one_hot("".join(results[k])) for k in tqdm.tqdm(results)
    }
    with open(one_hot_path, "wb") as f:
        pickle.dump(results_one_hot, f)
    return results_one_hot


@lru_cache(None)
def to_one_hot(fasta_path, one_hot_path):
    if not os.path.exists(one_hot_path):
        return _convert_to_one_hot(fasta_path, one_hot_path)
    with open(one_hot_path, "rb") as f:
        return pickle.load(f)


class OneHotFasta:
    def __init__(self, fasta_path, one_hot_path):
        self.data = to_one_hot(fasta_path, one_hot_path)

    def load(self, chrom, index_centers, cl, *, reverse_complement):
        index = index_centers[:, None] + np.arange(-cl // 2, cl // 2 + 1)
        datum = self.data[chrom]
        datum = datum[index]
        if reverse_complement:
            datum = datum[..., ::-1, ::-1]
        return datum
