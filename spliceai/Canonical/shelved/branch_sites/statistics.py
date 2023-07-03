import itertools
import tqdm.auto as tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from modular_splicing.dataset.h5_dataset import H5Dataset


def get_location_info(y, m, look_cl):
    overall = []
    splice_points = np.where(y)[0]
    for site in np.where(m[look_cl:-look_cl])[0]:
        site = site + look_cl
        if splice_points.size:
            nearest = splice_points[np.argmin(np.abs(splice_points - site))]
            dist = site - nearest
            nearest = {2: "D", 1: "A"}[y[nearest]]
        else:
            nearest = "N/A"
            dist = float("inf")
        overall.append([nearest, dist])
    return overall


def compute_statistics(amount, look_cl=500):
    dset = H5Dataset(
        path="dataset_train_all.h5",
        cl=0,
        cl_max=10_000,
        sl=5000,
        iterator_spec=dict(
            type="FullyRandomIter", shuffler_spec=dict(type="SeededShuffler", seed=0)
        ),
        datapoint_extractor_spec=dict(
            type="BasicDatapointExtractor",
            rewriters=[
                dict(
                    type="branch_site",
                    datafiles={
                        "True": "datafile_train_all.h5",
                        "False": "datafile_test_0.h5",
                    },
                ),
            ],
        ),
        post_processor_spec=dict(
            type="FlattenerPostProcessor",
            indices=[("inputs", "motifs"), ("outputs", "y")],
        ),
    )
    overall = []
    for m, y in tqdm.tqdm(itertools.islice(dset, amount), total=amount):
        m = m[:, 0]
        overall += get_location_info(y, m, look_cl=look_cl)

    overall = pd.DataFrame(overall, columns=["feature", "offset"])
    return overall


def histograms(overall, r=250):
    _, axs = plt.subplots(1, 2, dpi=200, figsize=(12, 5), sharey=True)
    for ax, feat in zip(axs, "AD"):
        ax.hist(
            np.array(overall[overall.feature == feat].offset),
            bins=np.arange(-r, r + 1, 10),
        )
        xs = ax.get_xticks()
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{feat}{x:+.0f}" for x in xs], rotation=90)
        ax.grid()
        span = {"A": (-r, 0), "D": (0, r)}[feat]
        ax.axvspan(*span, color="purple", alpha=0.1)
        ax.set_xlim(-r, r)
        ax.set_ylabel("Branch site frequency")
    plt.suptitle(
        f"{(np.abs(overall.offset) <= r).mean():.0%} of branchpoints are within {r} of some splicepoint"
    )
    plt.show()
