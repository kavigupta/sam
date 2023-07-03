import os
import numpy as np
from modular_splicing.data_pipeline.create_dataset import create_dataset


def select_to_match(lengths, small, large):
    [large_idxs] = np.where(large)
    np.random.RandomState(0).shuffle(large_idxs)
    lengths_ordered = lengths[large_idxs]
    lengths_cumulative = np.cumsum(lengths_ordered)
    split = np.where(lengths_cumulative < (small * lengths).sum())[0].max()
    large_idxs = large_idxs[:split]
    large_edited = np.zeros_like(large)
    large_edited[large_idxs] = 1
    return large_edited


def balanced_at(lengths, bar):
    below = lengths < bar
    above = ~below
    if (below * lengths).sum() > (above * lengths).sum():
        small, large = above, below
    else:
        small, large = below, above
    large_edited = select_to_match(lengths, small, large)

    assert (~(small & large_edited)).all()
    return small | large_edited


for folder, filter in [
    ("above-10k", lambda x: x > 10_000),
    ("below-10k", lambda x: x < 10_000),
    ("balanced-at-10k", lambda x: balanced_at(x, 10_000)),
    ("below-100k", lambda x: x < 100_000),
]:
    for suffix in "test_0", "train_all":
        print(folder, suffix)
        directory = f"../data/by-length/{folder}"
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass
        create_dataset(
            datafile_path=f"datafile_{suffix}.h5",
            dataset_path=f"{directory}/dataset_{suffix}.h5",
            SL=5000,
            CL_max=10_000,
            length_filter=filter,
        )
