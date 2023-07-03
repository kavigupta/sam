from collections import defaultdict
import warnings

import attr
import numpy as np
import pandas as pd

from .well_understood_proteins import remap_motif, should_use_motif


@attr.s
class SingleVNRnaMap:
    by_cell_line = attr.ib()

    @property
    def mean(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(
                [x for xs in self.by_cell_line.values() for x in xs], axis=0
            )


@attr.s
class VNRnaMap:
    name = attr.ib()
    included = attr.ib()
    excluded = attr.ib()


def all_van_nostrand_data():
    included = read_van_nostrand_data(
        "../data/van_nostrand_rna_maps/20180817.included.nSEnorm_subtr_wnan.csv"
    )
    excluded = read_van_nostrand_data(
        "../data/van_nostrand_rna_maps/20180817.excluded.nSEnorm_subtr_wnan.csv"
    )
    assert included.keys() == excluded.keys()
    return {name: VNRnaMap(name, included[name], excluded[name]) for name in included}


def read_van_nostrand_data(path):
    table = pd.read_csv(path, sep="\t", header=None)
    results = defaultdict(lambda: defaultdict(list))
    for name, array in zip(table[table.columns[0]], np.array(table[table.columns[2:]])):
        name, cell_line = name.split()
        assert cell_line[0] == "(" and cell_line[-1] == ")"
        cell_line = cell_line[1:-1]
        results[name][cell_line].append(array)
    return {k: SingleVNRnaMap(v) for k, v in results.items()}


def van_nostrand_rna_maps():
    vn = {
        remap_motif(k): average_effect(el)
        for k, el in all_van_nostrand_data().items()
        if should_use_motif(k)
    }

    vn = {k: v for k, v in vn.items() if not np.isnan(v).any()}
    return vn


def average_effect(el):
    """
    Compute the average effect of the given VNRnaMap object
    """
    all_cell_lines = el.included.by_cell_line.keys() | el.excluded.by_cell_line.keys()
    mean_by_cell_line = [
        # mean across replicates. But first subtract the included from the excluded,
        # because this propagates nan values. If either replicate has no data, we
        # want to ignore it.
        np.nanmean(
            np.array(el.excluded.by_cell_line[k]) - el.included.by_cell_line[k],
            0,
        )
        for k in all_cell_lines
    ]
    # mean across cell lines
    return np.nanmean(np.array(mean_by_cell_line), 0)


def line_up_vn_with_ours(*, padding, ensure_contains, only_ours):
    """
    Compute the effects of a given SingleVNRnaMap object, lined up roughly with ours

    Data is from Extended Figure 6: https://www.nature.com/articles/s41586-020-2077-3/figures/13

    Compute the intronic and exonic effect of the given motif.

    The raw data is of the form of mean eclip read counts at each position for exons under the conditions
        - the exon ended up being excluded upon knockdown
        - the exon ended up being included upon knockdown

    If we subtract these two numbers, we can get a rough estimate of the exonic effect at that point.

    coordinates from paper (as seen in figure 5)
    https://www.nature.com/articles/s41586-020-2077-3/figures/5
    0-50: prev exon's last 50 bases
    50-350: first 300 bases of prev intron
    350-650: last 300 bases of prev intron
    650-700: first 50 bases of current exon
    700-750: last 50 bases of current exon
    750-1050: first 300 bases of next intron
    1050-1350: last 300 bases of next intron
    1350-1400: first 50 bases of next exon

    we are interested in the local effect on the current exon
    so we look at
        550-649, 650-699 as A-100 to A+50
        700-799, 800-899 as D-50 to D+100

    Parameters
    ----------
    padding: int
        How much padding to add between the A and D regions.
    ensure_contains: list
        List of names to ensure are in the returned effect.
    only_ours: bool
        If True, only return the names in ensure_contains.

    Returns
    -------
        effect: (M, 150 + padding + 150)
            The effect of the motifs on splicing, ready to plot as an impage
            with imshow.
        xaxis:
            kwargs for the xaxis ticks and labels.
    """

    vn = van_nostrand_rna_maps()

    if only_ours:
        vn = {k: vn[k] for k in vn if k in ensure_contains}

    length_original = len(list(vn.values())[0])

    for k in ensure_contains:
        if k not in vn:
            vn[k] = np.zeros(length_original) + np.nan

    names = sorted(vn)

    eff = np.array([vn[k] for k in names])

    xticks = [
        0,
        50,
        100,
        150,
        150 + padding,
        200 + padding,
        250 + padding,
        300 + padding,
    ]
    xnames = ["A-100", "A-50", "A", "A+50", "D-50", "D", "D+50", "D+100"]

    return (
        np.concatenate(
            [
                eff[:, 550:700],
                np.zeros((eff.shape[0], padding)) + np.nan,
                eff[:, 700:850],
            ],
            axis=1,
        ),
        names,
        dict(ticks=xticks, labels=xnames),
    )
