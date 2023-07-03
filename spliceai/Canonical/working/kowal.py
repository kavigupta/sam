import ast
import csv
import numpy as np
import matplotlib.pyplot as plt

from modular_splicing.evaluation.predict_motifs import predict_motifs_binarized_sparse
from modular_splicing.motif_names import get_motif_names


def to_one_hot(s):
    return np.eye(4, dtype=np.float32)[["ACGT".index(c) for c in s]]


def load(path, motif_width=21):
    contents = [
        to_one_hot(ast.literal_eval(x[0])[0]) for x in list(csv.reader(open(path)))
    ]
    overall_length = max(x.shape[0] for x in contents)
    clip = motif_width // 2
    data = np.zeros((len(contents), overall_length, 4), dtype=np.float32)
    validity = np.zeros((len(contents), overall_length), dtype=bool)
    for i in range(len(contents)):
        data[i, : contents[i].shape[0]] = contents[i]
        validity[i, clip : contents[i].shape[0] - clip] = True
    return dict(data=data, validity=validity)


def predict(m, *, data, validity):
    mot = predict_motifs_binarized_sparse(m, data, bs=100).original
    mot = mot & validity[:, :, None]
    return mot.any(1)


def predict_correct_incorrect(m, contents):
    c = predict(m, **contents["correct"])
    i = predict(m, **contents["incorrect"])
    return dict(correct=c, incorrect=i)


def produce_statistics(fm, am, contents):
    fm_r = predict_correct_incorrect(fm, contents)
    am_r = predict_correct_incorrect(am, contents)

    def stat(fn):
        c = fn(fm_r["correct"], am_r["correct"]).mean(0)
        i = fn(fm_r["incorrect"], am_r["incorrect"]).mean(0)
        return c - i

    consensus = stat(lambda x, y: (x & y))
    fm_only = stat(lambda x, y: (x & ~y))
    am_only = stat(lambda x, y: (~x & y))

    return consensus, am_only - fm_only


def plot(xs, ys):
    names = get_motif_names("rbns")
    r = np.corrcoef(xs, ys)[0, 1]
    s = (np.sign(xs) == np.sign(ys)).mean()
    plt.scatter(xs, ys)
    plt.title(f"r={r:.2f}; s={s:.2f}")
    plt.xlabel("Consensus")
    plt.ylabel("AM only - FM only")
    for name in ["TRA2A", "HNRNPA1"]:
        plt.text(xs[names.index(name)], ys[names.index(name)], name)
