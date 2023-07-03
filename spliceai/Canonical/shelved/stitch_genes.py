from abc import ABC, abstractmethod

from permacache import permacache, stable_hash

import numpy as np
import tqdm.auto as tqdm

from shelved.hsmm.gene_dataset import (
    get_from_genes,
    run_on_all_genes,
    run_on_all_genes_with_threshold,
    run_on_entire_gene,
)
from modular_splicing.utils.construct import construct


gene_modification_specs = {
    "short_to_long_20nt": dict(
        type="ReplaceIntrons",
        genes_to_modify_spec=dict(type="shorter_than", thresh=10_000),
        genes_to_extract_spec=dict(type="longer_than", thresh=100_000),
        intron_selection_spec=dict(type="require_longer"),
        intron_replacement_spec=dict(type="stitch_from_sides", size=20),
    ),
    "short_to_short_20nt": dict(
        type="ReplaceIntrons",
        genes_to_modify_spec=dict(type="shorter_than", thresh=10_000),
        genes_to_extract_spec=dict(type="shorter_than", thresh=10_000),
        intron_selection_spec=dict(type="require_within", percent=25),
        intron_replacement_spec=dict(type="stitch_from_sides", size=20),
    ),
    "long_to_long_20nt": dict(
        type="ReplaceIntrons",
        genes_to_modify_spec=dict(type="longer_than", thresh=100_000),
        genes_to_extract_spec=dict(type="longer_than", thresh=100_000),
        intron_selection_spec=dict(type="require_within", percent=25),
        intron_replacement_spec=dict(type="stitch_from_sides", size=20),
    ),
    "long_to_short_20nt": dict(
        type="ReplaceIntrons",
        genes_to_modify_spec=dict(type="longer_than", thresh=100_000),
        genes_to_extract_spec=dict(type="shorter_than", thresh=10_000),
        intron_selection_spec=dict(type="require_shorter"),
        intron_replacement_spec=dict(type="stitch_from_sides", size=20),
    ),
}


def gene_filter_types():
    return dict(
        shorter_than=lambda thresh, xs: [len(x) < thresh for x in xs],
        longer_than=lambda thresh, xs: [len(x) >= thresh for x in xs],
    )


def keep_old(old_intron, new_intron):
    return old_intron


def stitch_in_middle(old_intron, new_intron):
    assert new_intron.shape[0] >= old_intron.shape[0]
    clip_total = old_intron.shape[0]
    clip_left, clip_right = clip_total // 2, clip_total - clip_total // 2
    center = new_intron[clip_left : new_intron.shape[1] - clip_right]
    return np.concatenate(
        [
            old_intron[:clip_left],
            center,
            old_intron[clip_left:],
        ]
    )


def stitch_from_sides(old_intron, new_intron, size):
    if old_intron.shape[0] < size * 2:
        return old_intron
    assert old_intron.shape[0] >= 2 * size
    return np.concatenate(
        [
            old_intron[:size],
            new_intron[size : new_intron.shape[0] - size],
            old_intron[-size:],
        ]
    )


def gene_modify_types():
    return dict(ReplaceIntrons=ReplaceIntrons)


class ModifyGenes(ABC):
    @abstractmethod
    def genes_to_modify(self, xs):
        pass

    @abstractmethod
    def genes_to_extract(self, xs):
        pass


class ReplaceIntrons(ModifyGenes):
    def __init__(
        self,
        *,
        genes_to_modify_spec,
        genes_to_extract_spec,
        intron_selection_spec,
        intron_replacement_spec,
        tries=100,
    ):
        self.genes_to_modify_spec = genes_to_modify_spec
        self.genes_to_extract_spec = genes_to_extract_spec
        self.intron_selection_spec = intron_selection_spec
        self.intron_replacement_spec = intron_replacement_spec
        self.tries = tries

    def genes_to_modify(self, xs):
        return construct(gene_filter_types(), self.genes_to_modify_spec, xs=xs)

    def genes_to_extract(self, xs):
        return construct(gene_filter_types(), self.genes_to_extract_spec, xs=xs)

    def intron_works(self, intron, length):
        return construct(
            dict(
                require_longer=lambda intron, length: len(intron) > length,
                require_shorter=lambda intron, length: len(intron) < length,
                require_within=lambda intron, length, percent: abs(len(intron) - length)
                / length
                < percent / 100,
            ),
            self.intron_selection_spec,
            intron=intron,
            length=length,
        )

    def construct_replacement(self, intron, new_intron):
        return construct(
            dict(
                keep_old=keep_old,
                stitch_in_middle=stitch_in_middle,
                stitch_from_sides=stitch_from_sides,
            ),
            self.intron_replacement_spec,
            old_intron=intron,
            new_intron=new_intron,
        )

    def locate_insertion(self, xs, ys, genes_to_extract_from_idxs, rng, length):
        for _ in range(self.tries):
            long_gene_idx = rng.choice(genes_to_extract_from_idxs)
            long_gene_introns = introns(ys[long_gene_idx])
            intron_idx = rng.choice(len(long_gene_introns))
            start_replace_intron, end_replace_intron = long_gene_introns[intron_idx]
            new_intron = xs[long_gene_idx][start_replace_intron:end_replace_intron]
            if self.intron_works(new_intron, length):
                return new_intron
        return None

    def replace_gene(
        self, xs, ys, *, gene_to_modify_idx, genes_to_extract_from_idxs, rng
    ):
        assert gene_to_modify_idx < len(ys)
        don, acc = introns(ys[gene_to_modify_idx]).T
        exons = []
        introns_ = []
        for start_exon, end_exon in zip([0, *acc], [*don, len(ys[gene_to_modify_idx])]):
            exons.append(xs[gene_to_modify_idx][start_exon:end_exon])
        for start_intron, end_intron in zip(don, acc):
            introns_.append(xs[gene_to_modify_idx][start_intron:end_intron])

        for i in range(len(introns_)):
            new_intron = self.locate_insertion(
                xs, ys, genes_to_extract_from_idxs, rng, len(introns_[i])
            )
            if new_intron is None:
                continue
            introns_[i] = self.construct_replacement(introns_[i], new_intron)

        xs_chunks = []
        ys_chunks = []

        def add(chunk, starting):
            xs_chunks.append(chunk)
            y = np.zeros((chunk.shape[0], 3), dtype=np.int32)
            y[:, 0] = 1
            y[0] = starting
            ys_chunks.append(y)

        for exon, intron in zip(exons, introns_):
            add(exon, [0, 1, 0])
            add(intron, [0, 0, 1])
        add(exons[-1], [0, 1, 0])

        xs_chunks = np.concatenate(xs_chunks)
        ys_chunks = np.concatenate(ys_chunks)

        ys_chunks[0] = [1, 0, 0]
        return xs_chunks, ys_chunks


@permacache(
    "hsmm/gene_dataset/evaluate_on_modified_genes_3", key_function=dict(m=stable_hash)
)
def evaluate_on_modified_genes(m, path, *, gene_modification_spec, seed):
    print(stable_hash(m), path, gene_modification_spec, seed)

    _, thresholds = run_on_all_genes_with_threshold(m, path)

    ysc, ypsc = run_on_modified_genes(
        m,
        path,
        gene_modification_spec=gene_modification_spec,
        seed=seed,
    )

    ypsc = [yp[:, 1:] > thresholds for yp in ypsc]
    return np.concatenate([yp[y[:, 1:].astype(np.bool)] for yp, y in zip(ypsc, ysc)])


def run_on_modified_genes(m, path, *, gene_modification_spec, seed):
    xs, ys = get_from_genes(path)
    gene_modification = construct(gene_modify_types(), gene_modification_spec)

    [gene_idxs_to_modify] = np.where(gene_modification.genes_to_modify(xs))

    assert (gene_idxs_to_modify <= len(ys)).all()

    [genes_to_extract_from_idxs] = np.where(gene_modification.genes_to_extract(xs))

    ysc, yps = [], []
    for gene_to_modify_idx in tqdm.tqdm(gene_idxs_to_modify):
        x, y = gene_modification.replace_gene(
            xs,
            ys,
            gene_to_modify_idx=gene_to_modify_idx,
            genes_to_extract_from_idxs=genes_to_extract_from_idxs,
            rng=np.random.RandomState((gene_to_modify_idx, seed)),
        )
        assert (ys[gene_to_modify_idx].sum(0)[1:] == y.sum(0)[1:]).all()
        yp = run_on_entire_gene(m, x)
        ysc.append(y)
        yps.append(yp)
    return ysc, yps


def gene_stitch_analysis(model, seed=0):
    return {
        spec_name: {
            k: evaluate_on_modified_genes(
                model[k],
                "datafile_test_0.h5",
                gene_modification_spec=gene_modification_specs[spec_name],
                seed=seed,
            )
            for k in tqdm.tqdm(model)
        }
        for spec_name in gene_modification_specs
    }


def introns(y):
    pos, is_don = np.where(y[:, 1:])
    assert (is_don[::2] == 1).all() and (is_don[1::2] == 0).all()
    return pos.reshape(-1, 2)
