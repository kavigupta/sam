import os

import numpy as np

import modular_splicing
from .maxent import FullTable

data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(modular_splicing.__file__))), "data"
)

path = os.path.join(data_dir, "splicepoint-maxent")
path_triplets = os.path.join(data_dir, "splicepoint-maxent-triplets")
path_triplets_old = os.path.join(data_dir, "splicepoint-maxent-triplets-old-style")

five_prime = [
    FullTable.from_file(f"{path}/me2x5", [-2, -1, 0, 3, 4, 5, 6]),
    FullTable(
        np.log([0.0040 / 0.27, 0.0032 / 0.23, 0.9896 / 0.23, 0.0032 / 0.27]), [1]
    ),
    FullTable(
        np.log([0.0034 / 0.27, 0.0039 / 0.23, 0.0042 / 0.23, 0.9884 / 0.27]), [2]
    ),
]

three_prime = [
    FullTable.from_file(f"{path}/me2x3acc9", [-6, -5, -4, -3]).invert(),
    FullTable.from_file(f"{path}/me2x3acc8", [-9, -8, -7]).invert(),
    FullTable.from_file(f"{path}/me2x3acc7", [-13, -12, -11, -10]).invert(),
    FullTable.from_file(f"{path}/me2x3acc6", [-16, -15, -14]).invert(),
    FullTable.from_file(f"{path}/me2x3acc5", [-9, -8, -7, -6, -5, -4, -3]),
    FullTable.from_file(f"{path}/me2x3acc4", [-16, -15, -14, -13, -12, -11, -10]),
    FullTable.from_file(f"{path}/me2x3acc3", [-6, -5, -4, -3, 0, 1, 2]),
    FullTable.from_file(f"{path}/me2x3acc2", [-13, -12, -11, -10, -9, -8, -7]),
    FullTable.from_file(f"{path}/me2x3acc1", [-20, -19, -18, -17, -16, -15, -14]),
    FullTable(
        np.log([0.9903 / 0.27, 0.0032 / 0.23, 0.0034 / 0.23, 0.0030 / 0.27]), [-2]
    ),
    FullTable(
        np.log([0.0027 / 0.27, 0.0037 / 0.23, 0.9905 / 0.23, 0.0030 / 0.27]), [-1]
    ),
]


# Triplet models are not used, since they seem to perform slightly worse than pairwise models
# in general. However, they are kept here for reference.
def five_prime_triplets(denom_adj):
    return [
        FullTable.from_npz(
            f"{path_triplets}/maxEnt5_prob", list(range(-2, 6 + 1)), denom_adj=denom_adj
        ),
    ]


def three_prime_triplets(denom_adj):
    return [
        FullTable.from_npz(
            f"{path_triplets}/maxEnt3_prob0",
            list(range(-20, -9 + 1)),
            denom_adj=denom_adj,
        ),
        FullTable.from_npz(
            f"{path_triplets}/maxEnt3_prob1",
            list(range(-8, 2 + 1)),
            denom_adj=denom_adj,
        ),
        FullTable.from_npz(
            f"{path_triplets}/maxEnt3_prob2",
            list(range(-14, -3 + 1)),
            denom_adj=denom_adj,
        ),
        FullTable.from_npz(
            f"{path_triplets}/maxEnt3_prob3",
            list(range(-14, -9 + 1)),
            denom_adj=denom_adj,
        ).invert(),
        FullTable.from_npz(
            f"{path_triplets}/maxEnt3_prob4",
            list(range(-8, -3 + 1)),
            denom_adj=denom_adj,
        ).invert(),
    ]


def three_prime_triplets_old_style(denom_adj):
    return [
        FullTable.from_npz(
            f"{path_triplets_old}/maxEnt3_prob0",
            list(range(-20, -14 + 1)),
            denom_adj=denom_adj,
        ),
        FullTable.from_npz(
            f"{path_triplets_old}/maxEnt3_prob1",
            list(range(-13, -7 + 1)),
            denom_adj=denom_adj,
        ),
        FullTable.from_npz(
            f"{path_triplets_old}/maxEnt3_prob2",
            list(range(-6, 2 + 1)),
            denom_adj=denom_adj,
        ),
        FullTable.from_npz(
            f"{path_triplets_old}/maxEnt3_prob3",
            list(range(-16, -10 + 1)),
            denom_adj=denom_adj,
        ),
        FullTable.from_npz(
            f"{path_triplets_old}/maxEnt3_prob4",
            list(range(-9, -3 + 1)),
            denom_adj=denom_adj,
        ),
        FullTable.from_npz(
            f"{path_triplets_old}/maxEnt3_prob5",
            list(range(-16, -14 + 1)),
            denom_adj=denom_adj,
        ).invert(),
        FullTable.from_npz(
            f"{path_triplets_old}/maxEnt3_prob6",
            list(range(-13, -10 + 1)),
            denom_adj=denom_adj,
        ).invert(),
        FullTable.from_npz(
            f"{path_triplets_old}/maxEnt3_prob7",
            list(range(-9, -7 + 1)),
            denom_adj=denom_adj,
        ).invert(),
        FullTable.from_npz(
            f"{path_triplets_old}/maxEnt3_prob8",
            list(range(-6, -3 + 1)),
            denom_adj=denom_adj,
        ).invert(),
    ]
