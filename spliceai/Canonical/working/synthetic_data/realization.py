import attr

import numpy as np

from working.synthetic_data.mutation.codon_mutation import CodonMutation
from working.synthetic_data.splicing_mechanism.place_splice_sites import (
    add_splice_sites,
)


def codon_realization(codon_table, slpd, rng):
    """
    Realize codons from a splicing length pattern distribution. This approach
        produces codons for both exons and introns.

        TODO allow intron codons to be sampled from a different distribution.
    """
    exon_lengths, intron_lengths = slpd.sample_splicing_length_pattern(rng)
    exon_lengths, exon_aminos = sample_codons(
        codon_table, exon_lengths, rng, pseudo=False
    )
    intron_lengths, intron_aminos = sample_codons(
        codon_table, intron_lengths, rng, pseudo=True
    )
    return exon_lengths, exon_aminos, intron_lengths, intron_aminos


def sample_codons(codon_table, lengths, rng, *, pseudo):
    """
    Enforce that the lengths allow for a full codon at the end of the sequence
        by adding to the last length in the sequence, and then sample codons.
    """
    lengths = lengths[:]
    if len(lengths) > 0:
        lengths[-1] += (-sum(lengths)) % 3
    num_aminos = sum(lengths) // 3
    aminos = codon_table.sample_protein(rng, num_aminos=num_aminos, pseudo=pseudo)
    return lengths, aminos


@attr.s
class Realization:
    codon_table = attr.ib()
    splicing_mechanism = attr.ib()
    exon_indices = attr.ib()
    intron_indices = attr.ib()
    protected_sites = attr.ib()
    exon_aminos = attr.ib()
    intron_aminos = attr.ib()  # mutable
    rna = attr.ib()  # mutable
    motifs = attr.ib()  # mutable
    motifs2 = attr.ib()  # mutable
    pred_splicing_pattern = attr.ib()  # mutable
    real_splicing_pattern = attr.ib()  # mutable

    mutation_log = attr.ib(default=attr.Factory(list))  # mutable
    score_value = attr.ib(default=None)  # mutable

    @classmethod
    def initialize(
        cls,
        codon_table,
        splicing_mechanism,
        exon_lengths,
        exon_aminos,
        intron_lengths,
        intron_aminos,
        rng,
        frac_fake,
    ):
        donors = []
        acceptors = []
        exon_indices = []
        intron_indices = []
        start = 0
        for length, index_list, pos_list in [
            x
            for (e, i) in zip(exon_lengths, [*intron_lengths, None])
            for x in [(e, exon_indices, donors), (i, intron_indices, acceptors)]
        ][:-1]:
            index_list.append(np.arange(start, start + length))
            start += length
            pos_list.append(start)
        exon_indices = np.concatenate(exon_indices).reshape(-1, 3)
        if len(intron_indices) > 0:
            intron_indices = np.concatenate(intron_indices).reshape(-1, 3)
        else:
            intron_indices = exon_indices[:0]
        rna = np.zeros(start, dtype=np.int8) - 1
        for idxs, aminos in [
            (exon_indices, exon_aminos),
            (intron_indices, intron_aminos),
        ]:
            for amino in range(codon_table.num_aminos()):
                ct = codon_table.codons_for(amino)
                mask = aminos == amino
                rna[idxs[mask]] = ct[rng.choice(ct.shape[0], size=mask.sum())]
        assert (rna != -1).all()
        real_splicing_pattern = (
            np.array(donors, dtype=np.int64)[:-1],
            np.array(acceptors, dtype=np.int64) - 1,
        )
        rna, protected_sites = add_splice_sites(
            codon_table,
            rna=rna,
            splicing_mechanism=splicing_mechanism,
            exon_indices=exon_indices,
            intron_indices=intron_indices,
            splice_sites=real_splicing_pattern,
            seed=rng.choice(2**32),
            frac_place=frac_fake,
        )
        rna = np.copy(rna)
        motifs = splicing_mechanism.predict_motifs(rna)
        motifs2 = splicing_mechanism.processed_motifs(motifs)
        pred_splicing_pattern = splicing_mechanism.predict_splicing_pattern_from_motifs(
            motifs2
        )
        return cls(
            codon_table=codon_table,
            splicing_mechanism=splicing_mechanism,
            exon_indices=exon_indices,
            intron_indices=intron_indices,
            protected_sites=set(protected_sites),
            exon_aminos=exon_aminos,
            intron_aminos=intron_aminos,
            rna=rna,
            motifs=motifs,
            motifs2=motifs2,
            pred_splicing_pattern=pred_splicing_pattern,
            real_splicing_pattern=real_splicing_pattern,
        )

    def is_done(self):
        return all(
            np.all(x == y)
            for x, y in zip(self.pred_splicing_pattern, self.real_splicing_pattern)
        )

    def score(self):
        if self.score_value is None:
            self.score_value = self.compute_score()
        return self.score_value

    def compute_score(self):
        x, y = self.compute_scores()
        return x - y * 0.5

    def compute_scores(self):
        motifs2 = self.splicing_mechanism.processed_motifs(self.motifs)
        real_score = self.splicing_mechanism.score_splicing_pattern_from_motifs(
            motifs2, self.real_splicing_pattern
        )
        # return real_score
        if self.is_done():
            return real_score, 0
        pred_score = self.splicing_mechanism.score_splicing_pattern_from_motifs(
            motifs2, self.pred_splicing_pattern
        )
        return real_score, pred_score

    def sample_mutation(self, rng):
        while True:
            # intron or exon
            if rng.rand() < self.exon_indices.shape[0] / (
                self.exon_indices.shape[0] + self.intron_indices.shape[0]
            ):
                mode = "same_amino"
                indices = self.exon_indices
            else:
                mode = "any_amino"
                indices = self.intron_indices
            # codon position
            pos = rng.choice(indices.shape[0])
            if set(indices[pos]) & self.protected_sites:
                continue
            new_sequence = self.codon_table.sample_mutation(
                self.rna[indices[pos]], mode=mode, rng=rng
            )
            if new_sequence is not None:
                break
        return CodonMutation(indices[pos], new_sequence)

    def perform(self, mutation):
        rna_footprint = mutation.footprint()
        ranges = compute_ranges(rna_footprint)
        radius = self.splicing_mechanism.motif_width() // 2

        stored_state = dict(
            mutation=mutation,
            rna_footprint=rna_footprint,
            rna=self.rna[rna_footprint].copy(),
            motif_undo=None,
            motif2_undo=None,
            splicing_pattern=self.pred_splicing_pattern,
            score_value=self.score_value,
        )

        self.mutation_log.append(stored_state)

        mutation.perform(self.rna)

        stored_state["motif_undo"], modified_mot, mot_ranges = apply_local(
            self.splicing_mechanism.predict_motifs,
            self.rna,
            self.motifs,
            radius,
            ranges,
        )

        if modified_mot:
            stored_state["motif2_undo"], modified_mot_2, _ = apply_local(
                self.splicing_mechanism.processed_motifs,
                self.motifs,
                self.motifs2,
                self.splicing_mechanism.motif_processing_width() // 2,
                mot_ranges,
            )
            if modified_mot_2:
                self.pred_splicing_pattern = (
                    self.splicing_mechanism.predict_splicing_pattern_from_motifs(
                        self.motifs2
                    )
                )
                self.score_value = None
                self.score()

    def undo(self, mutation):
        stored = self.mutation_log.pop()
        mutation_ = stored.pop("mutation")
        rna_footprint = stored.pop("rna_footprint")
        rna = stored.pop("rna")
        motif_undo = stored.pop("motif_undo")
        motif2_undo = stored.pop("motif2_undo")
        pred_splicing_pattern = stored.pop("splicing_pattern")
        score_value = stored.pop("score_value")
        assert stored == {}
        assert mutation_ is mutation
        self.rna[rna_footprint] = rna
        motif_undo()
        if motif2_undo is not None:
            motif2_undo()
        self.pred_splicing_pattern = pred_splicing_pattern
        self.score_value = score_value

    def __permacache_hash__(self):
        return {
            x.name: getattr(self, x.name)
            for x in self.__attrs_attrs__
            if x.name != "mutation_log"
        }


def compute_ranges(footprint):
    """
    Convert a sorted array of indices into a list of (exclusive) ranges

    E.g., [1, 2, 3, 5, 6, 8] -> [(1, 4), (5, 6), (8, 9)]
    """
    breaks = np.where(np.diff(footprint) > 1)[0]
    return list(
        zip(
            np.concatenate([[footprint[0]], footprint[breaks + 1]]),
            np.concatenate([footprint[breaks] + 1, [footprint[-1] + 1]]),
        )
    )


def apply_local(fn, inp, out, radius, ranges):
    """
    Apply a function to local neighborhoods of an input sequence.

    Parameters
    ----------
    fn: function
        (N, A) -> (N, B). The function to apply. Must have an effective radius
        of `radius`.
    inp: ndarray
        (L, A). The input sequence.
    out: ndarray
        (L, B). The output sequence.
    radius: int
        The radius of `fn`.
    ranges: list of (int, int)
        The ranges of the input ranges that have been changed

    Performs
    -------
    out[ranges[i]:ranges[i+1]] = fn(inp[ranges[i]-radius:ranges[i+1]+radius])

    Returns
    -------
    undo. A function that undoes the operation.
    changed. Whether any of the output ranges were changed.
    """

    output_ranges = [
        (max(0, start - radius), min(inp.shape[0], end + radius))
        for start, end in ranges
    ]
    old = [out[start:end].copy() for start, end in output_ranges]

    undo = ApplyLocalUndo(out, output_ranges, old)

    extended_input_ranges = [
        (max(0, start - radius), min(inp.shape[0], end + radius))
        for start, end in output_ranges
    ]
    changed = False
    for (start, end), (ext_start, ext_end) in zip(output_ranges, extended_input_ranges):
        res = fn(inp[ext_start:ext_end])
        clip_left, clip_right = start - ext_start, ext_end - end
        new = res[clip_left : res.shape[0] - clip_right]
        if (new != out[start:end]).any():
            changed = True
        out[start:end] = new

    return undo, changed, output_ranges


@attr.s
class ApplyLocalUndo:
    out = attr.ib()
    output_ranges = attr.ib()
    old = attr.ib()

    def __call__(self):
        for (start, end), o in zip(self.output_ranges, self.old):
            self.out[start:end] = o
