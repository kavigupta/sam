from collections import defaultdict

import numpy as np

from shelved.plot_base_importance import plot_base_importance, Motif


def load_from_paper():
    from_paper = """
    sequence: atggttgtactgatggcttgtttttcattttttttgtgctttttggtccatctattaa
    red     : --------*--**-------------*----------------------*---*---*
    blue    : ------------------**------------*--*--*-*-*---------------
    yellow  : ---------------------------------*-------*----------------
    """

    sequence, *patterns = [x.split(":")[1][1:] for x in from_paper.strip().split("\n")]
    patterns = np.array(
        [[x == "*" for x in pattern] for pattern in patterns], dtype=np.float
    )
    patterns[patterns == 0] = np.nan
    patterns *= np.arange(3)[:, None] + 1
    return sequence, patterns


def locate(data, gene):
    [[datafile_index]] = np.where(data.datafile["NAME"][:] == gene)
    return datafile_index


def line_up_gene(data, datafile_index, sequence, patterns):
    full_sequence = (
        data.datafile["SEQ"][datafile_index][5000:-5000].upper().decode("ascii")
    )
    off = full_sequence.index(sequence.upper())
    start = off
    end = off + len(sequence) + 1

    importances = np.zeros((3, end - start)) + np.nan
    importances[:, off - start : off + len(sequence) - start] = patterns
    sequence = np.eye(4)[undo_draw_bases(full_sequence)][start:end]

    return off, start, end, sequence, importances


def undo_draw_bases(xs):
    xs = xs.replace("A", "0").replace("C", "1").replace("G", "2").replace("T", "3")
    return np.array([int(x) for x in xs])


def draw_example(data, gene, sequence, patterns, branch_site_indices):
    datafile_index = locate(data, gene)
    off, start, end, sequence, importances = line_up_gene(
        data,
        datafile_index,
        sequence,
        patterns,
    )
    features = defaultdict(list)
    features.update(
        {
            idx: [Motif("B", "red")]
            for idx in branch_site_indices(datafile_index)
            - data.starts[datafile_index]
            - start
        }
    )
    features[off - start].append(Motif("START", "green"))
    features[off + len(sequence) - 1 - start].append(Motif("END", "green"))
    plot_base_importance(
        sequence,
        *importances,
        chunk_size=(end - start),
        features=features,
        style=[
            dict(marker="o", linestyle=" ", markersize=10, color=c)
            for c in ["red", "#2288ff", "orange"]
        ],
        labels=[
            "lariat sequencing & RNA seq",
            "lariant sequencing alone",
            "RNA sequencing alone",
        ],
        dpi=200,
        width=8,
        headroom=0.5,
    )
