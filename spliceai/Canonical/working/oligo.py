from collections import Counter, defaultdict
import csv
import os
import re

import attr
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm.auto as tqdm
import scipy.stats

from permacache import permacache, stable_hash
from modular_splicing.data_pipeline.create_datafile import create_datafile
from modular_splicing.dataset.datafile_object import SpliceAIDatafile
from modular_splicing.models.modules.lssi_in_model import LSSI_MODEL_THRESH
from modular_splicing.motif_names import get_motif_names
from modular_splicing.utils.arrays import Sparse

from modular_splicing.utils.sequence_utils import parse_bases

DATABASE_PATH = "../data/ASO-HUMAN-DB.csv"


def from_database_gene_name(name):
    name = name.split(", ")[-1]
    canonicalizer = {"IL-1RAcP": "IL1RAP", "ataxin-3": "ATXN3"}
    return canonicalizer.get(name, name)


def datafile_all_all():
    path = "datafile_all_all.h5"
    if not os.path.exists(path):
        create_datafile(
            data_segment_to_use="all",
            data_chunk_to_use="all",
            sequence_path="canonical_sequence.txt",
            splice_table_path="canonical_dataset.txt",
            CL_max=10_000,
            outfile=path,
        )
    return path


@attr.s
class OligoDatabase:
    exons = attr.ib()
    sequences = attr.ib()
    positions = attr.ib()
    targets = attr.ib()
    value_map = attr.ib()
    gene_text = attr.ib()

    @staticmethod
    def for_gene(database_gene_name, database_path=DATABASE_PATH):
        gene_name = from_database_gene_name(database_gene_name)
        content = load_gene(datafile_all_all(), gene_name)
        if content is None:
            raise ValueError(
                f"Could not find gene {gene_name} (original {database_gene_name}) in database"
            )
        gene_text, intron_start, intron_end = content
        dmd_filter, meta_filter = database_for_gene(
            load_database(database_path),
            gene_name=database_gene_name,
            gene_text=gene_text,
        )
        db, meta_align = construct_database_for_gene(
            dmd_filter, gene_text, intron_start, intron_end
        )
        return db, meta_filter, meta_align

    def __len__(self):
        return len(self.sequences)

    def locate_exon(self, idx, sl, cl):
        seq = self.sequences[idx]
        pos = self.positions[idx]
        ex = self.exons[self.targets[idx]]

        val = self.value_map[seq, int(self.targets[idx])]
        val = [x for xs in val.values() for x in xs]
        try:
            val = [float(x) for x in val]
        except ValueError:
            return None

        pos, ex, start, end = compute_window(sl, cl, pos, ex)
        return seq, pos, ex, val, start, end

    def compute_with_and_without(
        self, mod, gene_text, idx, sl, cl, *, mot_rad, blank_mot, blank_spl
    ):
        over = self.locate_exon(idx, sl, cl)
        if over is None:
            return [np.nan, np.nan], [np.nan, np.nan], [np.nan]
        seq, pos, ex, val, start, end = over
        x = parse_bases(gene_text[start:end])
        yp_edited, yp_baseline = compute_with_without_oligo(
            mod,
            x,
            pos,
            len(seq),
            ex,
            cl,
            mot_rad=mot_rad,
            blank_mot=blank_mot,
            blank_spl=blank_spl,
        )
        return yp_edited, yp_baseline, val

    def dataset(self, mod, *, sl, cl, mot_rad, blank_mot, blank_spl):
        yp_editeds, yp_baselines, vals = compute_oligo_mask_for_all(
            self,
            mod,
            self.gene_text,
            sl=sl,
            cl=cl,
            mot_rad=mot_rad,
            blank_mot=blank_mot,
            blank_spl=blank_spl,
        )
        return OligoDataset(yp_editeds, yp_baselines, vals)

    def dataset_from_motifs(self, mod, effects):
        net_effects = np.array([effects[0] - effects[1], effects[2] - effects[3]])
        motifs, vals = compute_motifs_all(mod, self, sl=5000, cl=400)
        motifs = np.array(
            [
                -compute_effects(net_effects, *mot) if mot is not None else np.nan
                for mot in motifs
            ]
        )
        vals = np.array([np.mean(x) for x in vals])
        return OligoDatasetProcessed.from_runs(
            motifs, np.where(np.isnan(motifs), np.nan, 1), vals, self.categorize_runs()
        )

    def categorize_runs(self, *, max_gap=25, min_cluster_size=20):
        sorted_positions = np.array(sorted(self.positions))
        gaps = sorted_positions[1:] - sorted_positions[:-1]
        categories = np.cumsum([0, *(gaps > max_gap)])
        category_map = dict(zip(sorted_positions, categories))
        for_our_indices = np.array([category_map[x] for x in self.positions])
        index_sets = [
            np.where(for_our_indices == x)[0] for x in range(max(for_our_indices) + 1)
        ]
        index_sets = [x for x in index_sets if len(x) >= min_cluster_size]
        return index_sets


def compute_window(sl, cl, pos, ex):
    rad = (sl + cl) // 2

    centerpt = (ex[0] + ex[1]) // 2
    start, end = centerpt - rad, centerpt + rad

    pos = pos - start
    ex = ex - start
    return pos, ex, start, end


@permacache(
    "working/oligo/datasets_2", key_function=dict(models=stable_hash, dbs=stable_hash)
)
def datasets(models, dbs, settings):
    results = {}
    for gene in tqdm.tqdm(dbs):
        db, _, _ = dbs[gene]
        for setting in settings:
            results[gene, setting] = [
                db.dataset(mod, sl=5000, cl=400, **settings[setting])
                for mod in tqdm.tqdm(models)
            ]
    return results


@attr.s
class OligoDataset:
    yp_editeds_log = attr.ib()
    yp_baselines_log = attr.ib()
    vals_lists = attr.ib()

    def ours(self):
        yp_editeds_e = np.exp(self.yp_editeds_log)
        yp_baselines_e = np.exp(self.yp_baselines_log)
        ours = yp_editeds_e.mean(1) - yp_baselines_e.mean(1)
        confident = yp_baselines_e.mean(1)
        return ours, confident

    def theirs(self):
        return np.array([np.mean(x) for x in self.vals_lists])

    def processed(self, runs):
        ours, confident = self.ours()
        theirs = self.theirs()
        return OligoDatasetProcessed.from_runs(ours, confident, theirs, runs)


def mean_weighted(xs, weights, is_weighted):
    xs = np.array(xs)
    weights = np.array(weights)

    assert xs.shape == weights.shape
    assert len(xs.shape) == 2

    if not is_weighted:
        weights = np.ones_like(weights)

    return (weights * xs).sum(0) / (weights).sum(0)


def mean_weight_each(els, weights, is_weighted):
    return [mean_weighted(x, weights, is_weighted) for x in els]


@attr.s
class OligoDatasetProcessed:
    ours = attr.ib()
    theirs = attr.ib()
    confident = attr.ib()
    runs = attr.ib()

    @classmethod
    def from_runs(cls, ours, confident, theirs, runs):
        runs = [(ours[run], theirs[run], confident[run]) for run in runs]
        return cls(ours, theirs, confident, runs)

    @classmethod
    def concat(cls, datasets):
        assert all(isinstance(x, cls) for x in datasets)
        ours = np.concatenate([x.ours for x in datasets])
        theirs = np.concatenate([x.theirs for x in datasets])
        confident = np.concatenate([x.confident for x in datasets])
        runs = sum([x.runs for x in datasets], [])
        return cls(ours, theirs, confident, runs)

    @classmethod
    def mean(cls, datasets, *, is_weighted):
        ours, theirs, confidents = zip(
            *[(x.ours, x.theirs, x.confident) for x in datasets]
        )
        ours, theirs, confidents = mean_weight_each(
            (ours, theirs, confidents), confidents, is_weighted
        )
        runs = [
            np.array(xs).transpose(1, 0, 2) for xs in zip(*[x.runs for x in datasets])
        ]
        runs = [mean_weight_each(xs, xs[-1], is_weighted) for xs in runs]
        return cls(ours, theirs, confidents, runs)

    def ours_class(self, confidence_threshold):
        return self.ours[self.confident > confidence_threshold]

    def theirs_class(self, confidence_threshold):
        return self.theirs[self.confident > confidence_threshold]

    def score_histograms(self):
        plt.figure(facecolor="white")
        plt.hist(self.ours.astype(np.float32), bins=100)
        plt.xlabel("Our score")
        plt.ylabel("Frequency")
        plt.show()
        plt.figure(facecolor="white")
        plt.hist(self.theirs, bins=100)
        plt.xlabel("Database score")
        plt.ylabel("Frequency")
        plt.show()

    def predictor(self, confidence_threshold):
        return scipy.stats.linregress(
            self.ours_class(confidence_threshold),
            self.theirs_class(confidence_threshold),
        )

    def scatter(self, *, confidence_threshold, ax):
        ax.scatter(
            self.ours_class(confidence_threshold),
            self.theirs_class(confidence_threshold),
            alpha=0.1,
            marker=".",
        )
        ax.set_xlabel("Our Score")
        ax.set_ylabel("Database score")
        predictor = self.predictor(confidence_threshold)
        lim = ax.get_xlim()
        xs = np.linspace(*lim, 2)
        ys = predictor.slope * xs + predictor.intercept
        ax.plot(
            xs,
            ys,
            color="red",
            label=f"r={predictor.rvalue:.2f}, p={predictor.pvalue:.3f}",
        )
        ax.legend()
        ax.set_xlim(*lim)
        return predictor


class NoDataForGeneError(Exception):
    pass


@permacache(
    "working/oligo/compute_with_without_oligo_3",
    key_function=dict(mod=stable_hash, x=stable_hash),
)
def compute_with_without_oligo(
    mod, x, pos, seq_len, ex, cl, *, mot_rad, blank_mot, blank_spl
):
    def process(manip, manip_spl):
        with torch.no_grad():
            [yp] = (
                mod(
                    torch.tensor([x]).float().cuda(),
                    manipulate_post_sparse=manip,
                    manipulate_splicepoint_motif=manip_spl,
                )
                .log_softmax(-1)
                .cpu()
                .numpy()
            )
        return yp[ex - cl // 2, [1, 2] * (len(ex) // 2)]

    def manip_spl(x):
        if not blank_spl:
            return x
        assert x.shape[0] == 1
        x = x.clone()

        lays = [spl.conv_layers[0] for spl in mod.splicepoint_model.models]
        for i, lay in enumerate(lays):
            window = slice_for_radii(pos, lay.right, lay.left, seq_len, x)
            x[:, window, i] = LSSI_MODEL_THRESH
        return x

    def blanker(x):
        if not blank_mot:
            return x
        assert x.shape[0] == 1
        x = x.clone()
        x[:, slice_for_radii(pos, mot_rad, mot_rad, seq_len, x)] = 0
        return x

    yp_edited = process(blanker, manip_spl)
    yp_baseline = process(lambda x: x, lambda x: x)
    return yp_edited, yp_baseline


@permacache(
    "working/oligo/compute_for_all_2",
    key_function=dict(db=stable_hash, mod=stable_hash, gene_text=stable_hash),
)
def compute_oligo_mask_for_all(
    db, mod, gene_text, *, cl, sl, mot_rad, blank_mot, blank_spl
):
    amount = len(db)
    results = []
    for idx in tqdm.trange(amount):
        results.append(
            db.compute_with_and_without(
                mod,
                gene_text,
                idx,
                cl=cl,
                sl=sl,
                mot_rad=mot_rad,
                blank_mot=blank_mot,
                blank_spl=blank_spl,
            )
        )
    if not results:
        raise NoDataForGeneError()
    yp_editeds, yp_baselines, vals = zip(*results)
    return yp_editeds, yp_baselines, vals


def slice_for_radii(pos, rad_left, rad_right, seq_len, x):
    start, end = pos - rad_left, pos + seq_len + rad_right
    start, end = max(start, 0), min(end, x.shape[1])
    return slice(start, end)


def load_database(path):
    with open(path) as f:
        r = list(csv.reader(f))
    database = pd.DataFrame(r[1:], columns=r[0] + ["a", "b"])
    database = database.rename(
        columns={
            '"Target gene (RNA)"': "target_gene_rna",
            '"Oligo sequence /: Cocktail. -: weasel (connected)."': "oligo_seq",
            '"Target exon"': "target_exon",
            '"Standard type"': "standard_type",
            '"Standard value"': "standard_value",
            '"Standard relation"': "standard_relation",
            '"Oligo concentration"': "oligo_conc",
            '"Unit for oligo concentration"': "oligo_unit",
            '"cells used"': "cells_used",
            '"Figure/Table in literature"': "fig_lit",
        }
    )
    database = database.drop(columns=["a", "b"])
    return database


@permacache(
    "working/oligo/database_for_gene_2",
    key_function=dict(database=stable_hash, gene_text=stable_hash),
)
def database_for_gene(database, *, gene_name, gene_text):
    database = database[database.target_gene_rna == gene_name]
    oligos = sorted(set(database.oligo_seq))
    nucleic = np.array(
        [set(x.upper().replace("U", "T")) - set("ACGT") == set() for x in oligos]
    )
    target_in = np.array(
        [complement(x) in gene_text.upper() for x in tqdm.tqdm(oligos)]
    )

    target_in_map = dict(zip(oligos, target_in))
    found = database.oligo_seq.apply(lambda x: target_in_map[x])
    database = database[found].copy()
    num_found = database.shape[0]
    database = database[database.target_exon != "Null"]
    database = database[database.target_exon != "uncpcified"]
    database = database[database.target_exon != "unspecified"]
    num_with_exon = database.shape[0]
    database = database[database.standard_type == "Skip[%]"]
    database = database[database.standard_relation == "="]
    num_with_correct_relation = database.shape[0]

    meta = dict(
        non_nucleic_frac=np.mean(~nucleic),
        non_nucleic_examples=np.array(oligos)[~nucleic].tolist(),
        missing_frac=np.mean(~target_in & nucleic) / np.mean(nucleic),
        missing_examples=np.array(oligos)[~target_in & nucleic].tolist(),
        num_found=num_found,
        num_with_exon=num_with_exon,
        num_with_correct_relation=num_with_correct_relation,
    )
    return database, meta


def get_skip_values(database):
    """
    Get the skip [%] values for each oligo and target exon.

    Returns a dictionary mapping (oligo, target_exon) to a dictionary mapping
    (oligo_conc, oligo_unit, cells_used, fig_lit) to a list of standard_values.
    """
    value_map = defaultdict(lambda: defaultdict(list))
    for seq, tar, sv, oc, ocu, cu, t in zip(
        database.oligo_seq,
        database.target_exon,
        database.standard_value,
        database.oligo_conc,
        database.oligo_unit,
        database.cells_used,
        database.fig_lit,
    ):
        try:
            k1 = seq, int(tar) - 1
        except ValueError:
            continue
        k2 = (oc, ocu), cu, t
        value_map[k1][k2] += [sv]
    return value_map


@permacache(
    "working/oligo/collect_sequence_target_data_all",
    key_function=dict(dmd_filter=stable_hash, gene_text=stable_hash),
)
def collect_sequence_target_data_all(dmd_filter, gene_text):
    grouped_targets = (
        dmd_filter[["oligo_seq", "target_exon"]].groupby("oligo_seq").agg(set)
    )

    start_positions = [
        [x.start() for x in re.finditer(complement(x), gene_text)]
        for x in tqdm.tqdm(grouped_targets.index)
    ]

    sequences_all, targets_all, positions_all = [], [], []
    for seq, targets, positions in zip(
        grouped_targets.index, grouped_targets.target_exon, start_positions
    ):
        for target in targets:
            for position in positions:
                if "intron" in target.lower():
                    continue
                sequences_all.append(seq)
                targets_all.append(int(target) - 1)
                positions_all.append(position)
    sequences_all = np.array(sequences_all)
    positions_all = np.array(positions_all)
    targets_all = np.array(targets_all)
    return sequences_all, targets_all, positions_all


def construct_database_for_gene(dmd_filter, gene_text, intron_start, intron_end):
    sequences_all, targets_all, positions_all = collect_sequence_target_data_all(
        dmd_filter, gene_text
    )
    if len(sequences_all) == 0:
        return None, None
    value_map = get_skip_values(dmd_filter)
    exons = create_exons(len(gene_text), intron_start, intron_end)
    reordered_exons = exons[targets_all].T
    from_start, from_end = (positions_all - reordered_exons) / (
        reordered_exons[1] - reordered_exons[0]
    )
    valid = (np.abs(from_start) < 10) & (np.abs(from_end) < 10)
    sequences_valid, positions_valid, targets_valid = (
        sequences_all[valid],
        positions_all[valid],
        targets_all[valid],
    )
    db = OligoDatabase(
        exons, sequences_valid, positions_valid, targets_valid, value_map, gene_text
    )
    meta = dict(
        positions_all=positions_all,
        targets_all=targets_all,
        exons=exons,
        from_start=from_start,
        from_end=from_end,
        valid=valid,
        gene_text=gene_text,
    )
    return db, meta


def create_exons(gene_length, intron_start, intron_end):
    exons = np.array([[0, *intron_end], [*intron_start, gene_length]]).T
    return exons


def display_meta_filter(
    non_nucleic_frac,
    non_nucleic_examples,
    missing_frac,
    missing_examples,
    num_found,
    num_with_exon,
    num_with_correct_relation,
):
    print(f"{non_nucleic_frac:.2%} of oligos are not nucleic")
    print("Some examples:")
    for x in non_nucleic_examples[:5]:
        print(x)
    print()
    print(f"{missing_frac:.2%} of remaining nucleic oligos are missing from the gene")
    print("Some examples:")
    for x in missing_examples[:5]:
        print(x)
    print()
    print(f"{num_found} oligos found in the gene")
    print(f"{num_with_exon} oligos found in the gene with a target exon")
    print(
        f"{num_with_correct_relation} oligos found in the gene with a target exon and correct relation"
    )


def complement(oligo):
    return "".join(
        [
            {"A": "T", "G": "C", "C": "G", "T": "A", "U": "A"}.get(x, x)
            for x in oligo.upper()
        ]
    )[::-1]


# @permacache("working/oligo/load_gene")
def load_gene(dfile_path, gene_name, include_start=False):
    dfile = SpliceAIDatafile.load(dfile_path)
    gene_mask = dfile.names == gene_name
    if not np.any(gene_mask):
        print(f"Could not find: {gene_name}, {dfile_path}")
        return None
    [[gene_idx]] = np.where(gene_mask)
    gene_text = dfile.datafile["SEQ"][gene_idx].decode("utf-8")
    if dfile.strands[gene_idx] == "-":
        gene_text = complement(gene_text)
    cl = len(gene_text) - (dfile.ends[gene_idx] - dfile.starts[gene_idx] + 1)
    print(cl)
    [intron_start] = dfile.datafile["JN_START"][gene_idx]
    [intron_end] = dfile.datafile["JN_END"][gene_idx]

    intron_start, intron_end = [
        np.array([int(x) for x in x.decode("utf-8").strip(",").split(",")])
        - dfile.starts[gene_idx]
        + cl // 2
        for x in (intron_start, intron_end)
    ]

    if dfile.strands[gene_idx] == "-":
        intron_start, intron_end = [
            (len(gene_text) - 1 - x)[::-1] for x in (intron_start, intron_end)
        ]
        intron_start, intron_end = intron_end, intron_start
    results = [gene_text, intron_start, intron_end]
    if include_start:
        results.append(dfile.starts[gene_idx])
    return results


# def locate(dfile, oligo):
#     compl_oligo = complement(oligo)
#     return np.where(
#         [compl_oligo in x.decode("utf-8").upper() for x in dfile.datafile["SEQ"]]
#     )[0]


def plot_alignment_meta(
    positions_all, targets_all, exons, from_start, from_end, valid, gene_text
):
    plt.figure(dpi=200)
    plt.scatter(positions_all, targets_all, label="ASOs")
    for i, (start, end) in enumerate(exons):
        kwargs = dict(label="Exons") if i == 0 else dict()
        plt.plot([start, end], [i, i], marker=".", color="red", **kwargs)
    plt.xlabel("Position in DMD")
    plt.ylabel("Exon index")
    plt.grid()
    plt.legend()
    plt.show()
    _, axs = plt.subplots(2, 1, figsize=(4, 8), facecolor="white")

    o = np.argsort(positions_all)
    for i in range(2):
        axs[i].scatter(
            positions_all[o], from_start[o], marker=".", alpha=0.5, label="from 3'"
        )
        axs[i].scatter(
            positions_all[o], from_end[o], marker=".", alpha=0.5, label="from 5'"
        )
        axs[i].set_ylabel("Disp. from nearest splicepoint\n[exon lengths]")
    axs[1].set_ylim(-20, 20)
    axs[1].legend()
    axs[1].set_xlabel("Position in gene")
    plt.show()
    print(f"Percent of valid ASOs: {valid.mean():.2%}")
    to_seqs = lambda x: Counter("".join(x) for x in np.array(x).T)
    gt = to_seqs([np.array(list(gene_text))[exons[:-1, 1] + i] for i in (1, 2)])
    ag = to_seqs([np.array(list(gene_text))[exons[:-1, 0] + i] for i in (-2, -1)])
    print("GT")
    print(gt)
    print("AG")
    print(ag)


def r(x, y):
    mask = np.isnan(x)
    x, y = x[~mask], y[~mask]
    return np.corrcoef(x, y)[0, 1]


def compute_r_values(x):
    return [x.predictor(0).rvalue] + [r(x, y) for x, y, _ in x.runs]


def compute_mean_individual_r_values(results, means, dbs):
    labels = ["overall"] + [
        f"{k}'s sliding {i + 1}'"
        for k in dbs
        for i in range(len(dbs[k][0].categorize_runs()))
    ]
    individual = [compute_r_values(res) for res in results]
    individual_mean = np.mean(individual, 0)
    individual_min, individual_max = np.min(individual, 0), np.max(individual, 0)
    mean = compute_r_values(means)
    ebar = (
        (individual_max + individual_min) / 2,
        (individual_max - individual_min) / 2,
    )
    return labels, individual_mean, ebar, mean


def compute_motifs(m, relevant_gene_text, start, end):
    x = parse_bases(relevant_gene_text)
    with torch.no_grad():
        [mot] = (
            m(torch.tensor([x]).float().cuda(), only_motifs=True)[
                "post_sparse_motifs_only"
            ]
            .cpu()
            .numpy()
        )

    return Sparse.of(mot[start:end])


@permacache(
    "working/oligo/compute_motifs_all_2",
    key_function=dict(m=stable_hash, db=stable_hash),
)
def compute_motifs_all(m, db, sl, cl):
    motifs = []
    vals = []
    for idx in tqdm.trange(len(db)):
        over = db.locate_exon(idx, sl, cl)
        if over is None:
            motifs.append(None)
            vals.append(val)
            continue
        seq, pos, ex, val, start, end = over
        res = compute_motifs(m, db.gene_text[start:end], pos, pos + len(seq))
        mot_pos, mot_ids = res.where
        mot_pos = mot_pos[:, None] - (ex - pos)
        motifs.append((mot_pos, mot_ids))
        vals.append(val)
    return motifs, vals


def compute_effects(net_effects, mot_pos, mot_ids):
    radius = net_effects.shape[1] // 2

    acc_don = np.zeros_like(mot_pos)
    acc_don[:, 1] = 1
    mot_ids = np.zeros_like(mot_pos) + mot_ids[:, None]
    acc_don = acc_don.flatten()
    mot_pos = mot_pos.flatten()
    mot_ids = mot_ids.flatten()
    mask = np.abs(mot_pos) <= radius
    acc_don, mot_pos, mot_ids = acc_don[mask], mot_pos[mask], mot_ids[mask]
    return net_effects[acc_don, mot_pos + radius, mot_ids].sum()


def single_point_results(mod, oligo_len, gene_text, sl, cl, pos, exon, setting):
    pos, ex, x = to_model_friendly_window(gene_text, sl, cl, pos, exon)
    return compute_with_without_oligo(mod, x, pos, oligo_len, ex, cl=cl, **setting)


def to_model_friendly_window(gene_text, sl, cl, pos, exon):
    pos, ex, start, end = compute_window(sl=sl, cl=cl, pos=pos, ex=exon)
    x = parse_bases(gene_text[start:end])
    return pos, ex, x


@permacache(
    "working/oligo/sequential_point_results_3",
    key_function=dict(mod=stable_hash, gene_text=stable_hash),
)
def sequential_point_results(
    mod, oligo_len, gene_text, sl, cl, positions, exon, setting
):
    results = []
    for pos in tqdm.tqdm(positions):
        results.append(
            single_point_results(mod, oligo_len, gene_text, sl, cl, pos, exon, setting)
        )
    return results


def sequential_point_results_exon_mean(
    mod,
    oligo_len,
    gene_text,
    sl,
    cl,
    positions,
    exon,
    setting,
    summary=lambda x: np.mean(x, 1),
):
    edited, baseline = zip(
        *sequential_point_results(
            mod, oligo_len, gene_text, sl, cl, positions, exon, setting
        )
    )
    edited, baseline = np.array(edited), np.array(baseline)
    assert (baseline == baseline[-1]).all()
    baseline = summary(np.exp(baseline[:1]))[0]
    effects = summary(np.exp(edited))
    return effects, baseline


def sequential_point_results_several_mot_rad(
    mod, oligo_len, gene_text, sl, cl, positions, exon, setting, mot_rads, **kwargs
):
    editeds, baselines = zip(
        *[
            sequential_point_results_exon_mean(
                mod,
                oligo_len,
                gene_text,
                sl,
                cl,
                positions,
                exon,
                dict(**setting, mot_rad=mot_rad),
                **kwargs,
            )
            for mot_rad in mot_rads
        ]
    )
    return np.mean(editeds, 0), np.mean(baselines, 0)


def compute_motifs(mod, gene_text, positions, exon, sl, cl):
    pos, ex, start, end = compute_window(sl=sl, cl=cl, pos=positions[0], ex=exon)
    x = parse_bases(gene_text[start:end])
    with torch.no_grad():
        [mot] = mod.cpu()(torch.tensor([x]).float(), only_motifs=True)[
            "post_sparse_motifs_only"
        ].numpy()
    locs, ids = np.where(mot[pos : pos + len(positions)])
    return locs, ids


def positions_overlapped(a, b, l):
    """
    what start positions x does the given feature [a, b] intersect [x, x + l] at:
        starts at x + l = a; x = a - l
        ends at x = b
    """
    return a - l, b


def in_overlap(a, b, l, x):
    start, end = positions_overlapped(a, b, l)
    return start <= x <= end


def plot_exon_plot(
    positions,
    effects,
    baseline,
    *,
    exon,
    oligo_len,
    locs,
    ids,
    mot_rad,
    oligo_data,
):
    names = get_motif_names("rbns")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    window = [20, 2], [2, 6]

    plt.figure(dpi=200, figsize=(7, 7))
    plt.plot(positions - positions[0], 100 * effects, color="black")
    footprint_of_splice_site = np.zeros_like(positions, dtype=bool)
    for i, p in enumerate(exon - positions[0]):
        l, r = window[i % 2]
        start, end = positions_overlapped(p - l, p + r, oligo_len)
        footprint_of_splice_site[start : end + 1] = True
        plt.axvspan(
            start,
            end,
            color=["red", "green"][i % 2],
            label="AD"[i % 2],
            alpha=0.1,
        )

    for i in range(len(locs)):
        ys = np.arange(10, 100, 10)
        y = ys[i % len(ys)] + (i // len(ys)) * 3
        color = colors[i % len(colors)]
        start, end = positions_overlapped(
            locs[i] - mot_rad, locs[i] + mot_rad, oligo_len
        )
        plt.plot([start, end], [y, y], color=color)
        plt.text(x=end, y=y, s=names[ids[i]], verticalalignment="center", color=color)
    if oligo_data is not None:
        oligo_data = oligo_data.sort_values("x").set_index("x")
        for c in oligo_data:
            plt.plot(
                oligo_data.index - positions[0], oligo_data[c], label=c, marker="."
            )
    plt.axhline(100 * baseline, color="black", label="Baseline")
    plt.xlabel("Oligo binding starts at position")
    plt.ylabel("Predicted Exon splicing %")
    plt.legend()
    plt.ylim(0, 140)
    plt.yticks(np.arange(0, 101, 20))
    return footprint_of_splice_site


def get_footprint_of_splice_sites(positions, *, exon, oligo_len):
    window = [20, 2], [2, 6]

    footprint_of_splice_site = np.zeros_like(positions, dtype=bool)
    for i, p in enumerate(exon - positions[0]):
        l, r = window[i % 2]
        start, end = positions_overlapped(p - l, p + r, oligo_len)
        footprint_of_splice_site[start : end + 1] = True

    return footprint_of_splice_site


def produce_table_of_relevant_sites(
    relevant_sites,
    *,
    positions,
    exon,
    oligo_len,
    gene_text,
    mot_rad,
    versus_baseline,
    locs_all,
    ids_all,
):
    all_names = np.array(get_motif_names("rbns"))
    rows = []
    for site in relevant_sites:
        row = []

        row.append(positions[0] + site)
        row.append(site - (exon[0] - positions[0]))
        row.append(
            complement(
                gene_text[positions[0] : positions[-1] + 1][site : site + oligo_len]
            )
        )
        scores = versus_baseline[:, site]
        row.append(scores.mean())
        row.extend(scores)
        for seed in range(len(locs_all)):
            locs, ids = locs_all[seed], ids_all[seed]
            start, end = positions_overlapped(site - mot_rad, site + mot_rad, oligo_len)
            mask = (start <= locs) & (locs <= end)
            locs, ids = locs[mask], ids[mask]
            disp, names = (locs - site).tolist(), all_names[ids].tolist()
            row.append("; ".join([f"{n}:{d:+d}" for d, n in zip(disp, names)]))
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=["Gene Position", "Offset from 5N's 3'", "Oligo", "Mean Score"]
        + [f"Score #{i + 1}" for i in range(len(locs_all))]
        + [f"Motifs #{i + 1}" for i in range(len(locs_all))],
    )


def curve_for_all_models(
    models, *, gene_text, positions, exon, oligo_len, max_mot_rad, summary
):
    mot_rads = range(1 + max_mot_rad)

    baselines_all = []
    effects_all = []
    locs_all = []
    ids_all = []
    for m in models:
        mod = m.model
        effects, baseline = sequential_point_results_several_mot_rad(
            mod,
            oligo_len,
            gene_text,
            sl=5000,
            cl=400,
            positions=positions,
            exon=exon,
            setting=dict(blank_mot=True, blank_spl=True),
            mot_rads=mot_rads,
            summary=summary,
        )
        effects_all.append(effects)
        baselines_all.append(baseline)
        locs, ids = compute_motifs(mod, gene_text, positions, exon, sl=5000, cl=400)
        locs_all.append(locs)
        ids_all.append(ids)

    return dict(
        effects_all=effects_all,
        baselines_all=baselines_all,
        locs_all=locs_all,
        ids_all=ids_all,
    )


def coords(table, i):
    row = table.iloc[i]
    start = row["Offset from 5N's 3'"]
    end = start + len(row["Oligo"])
    return start, end


def plot_created_oligos(table, exon, positions):
    mask = get_footprint_of_splice_sites(positions, exon=exon, oligo_len=0)
    [els] = np.where(mask[1:] != mask[:-1])
    els = els.reshape(2, 2, 2) - (exon[0] - positions[0])
    for i, row in table.iterrows():
        start, end = coords(table, i)
        plt.plot([start, end], [row["Mean Score"]] * 2)
    plt.xlabel("Offset from 5N's 3'")
    plt.ylabel("Mean Score")
    for i in range(2):
        for ad in range(2):
            plt.axvspan(*els[i, ad], alpha=0.25, color=["red", "green"][ad])


def compute_relevant_sites(versus_baseline, *, positions, exon):
    relevant_sites = {}
    for ml in versus_baseline:
        [rs] = np.where(
            ((versus_baseline[ml] > 0.01).sum(0) >= 3)
            & ~get_footprint_of_splice_sites(positions, exon=exon, oligo_len=ml)
        )
        rs = rs[rs + ml < positions.size]
        relevant_sites[ml] = rs
    return relevant_sites


def produce_full_table(
    versus_baseline, *, gene_text, positions, exon, max_mot_rad, all_model_curve
):
    relevant_sites = compute_relevant_sites(
        versus_baseline, positions=positions, exon=exon
    )
    relevant_text = gene_text.upper()[positions[0] : positions[-1] + 1]
    tables = {}
    for ml in versus_baseline:
        table = produce_table_of_relevant_sites(
            relevant_sites[ml],
            positions=positions,
            exon=exon,
            oligo_len=ml,
            gene_text=gene_text,
            mot_rad=max_mot_rad,
            versus_baseline=versus_baseline[ml],
            locs_all=all_model_curve[ml]["locs_all"],
            ids_all=all_model_curve[ml]["ids_all"],
        )
        tables[ml] = table
    table = pd.concat(tables.values())
    table = table[
        table["Oligo"].apply(
            lambda x: len(re.findall(complement(x), relevant_text)) == 1
        )
    ]
    table = table.sort_values("Mean Score")[::-1].reset_index()
    del table["index"]
    return table


def filter_table_for_overlaps(table):
    def overlap(table, i, j):
        s1, e1 = coords(table, i)
        s2, e2 = coords(table, j)
        return max(0, min(e2, e1) - max(s2, s1)) / (max(e2, e1) - min(s2, s1))

    selected = []
    for i, row in tqdm.tqdm(list(table.iterrows())):
        if any(overlap(table, i, j) >= 2 / 3 for j in selected):
            continue
        selected.append(i)
    table = table.iloc[selected].reset_index()
    del table["index"]
    return table
