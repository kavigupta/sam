from abc import ABC, abstractmethod
import os

import attr
from matplotlib import pyplot as plt
import numpy as np


class SLPD(ABC):
    """
    Splicing length pattern distribution. Represents the distribution of
        splicing length patterns (i.e. the lengths of the exons and introns
        in a splicing pattern, along with the number of exons and introns).
    """

    @abstractmethod
    def sample_splicing_length_pattern(self, rng):
        """
        Returns two lists (lengths_exons, lengths_introns)
            where lengths_exons[i] is the length of the i-th exon
            and lengths_introns[i] is the length of the i-th intron

        We have len(lengths_exons) = len(lengths_introns) + 1
        """
        pass

    def plot_samples(self, amount=32):
        """
        Plot a few samples from the distribution
        """
        plt.figure(figsize=(5, 5), dpi=200)
        h = 0.5
        for seed in range(amount):
            exons, introns = self.sample_splicing_length_pattern(
                np.random.RandomState(seed)
            )
            start = 0
            for label, length in [
                x for (e, i) in zip(exons, introns + [None]) for x in zip("ei", (e, i))
            ][:-1]:
                plt.fill_between(
                    [start, start + length],
                    [seed, seed],
                    [seed + h, seed + h],
                    color={"e": "red", "i": "black"}[label],
                )
                start += length
        plt.xlabel("Length [nt]")
        plt.xticks(rotation=30)
        plt.ylabel("seed")
        plt.ylim(amount + 1, -1)


@attr.s
class HSMM_SLPD(SLPD):
    """
    Splicing length pattern distribution, represented as an HSMM

    The HSMM has 5 states:
        SingleExon
        FirstExon
        MiddleExon
        LastExon
        Intron

    and the following graph
        start -> SingleExon -> end

        start -> FirstExon -> Intron -> MiddleExon -> [back to Intron]
                                     -> LastExon -> end

    Length distributions are specified by cumulative sum distributions
    """

    p_single_exon = attr.ib()
    p_last_exon = attr.ib()
    length_single_exon = attr.ib()
    length_first_exon = attr.ib()
    length_middle_exon = attr.ib()
    length_last_exon = attr.ib()
    length_intron = attr.ib()

    @classmethod
    def from_directory(cls, directory):
        REMAP_TABLE = {
            "length_single_exon": "lengthSingleExons.npy",
            "length_first_exon": "lengthFirstExons.npy",
            "length_middle_exon": "lengthMiddleExons.npy",
            "length_last_exon": "lengthLastExons.npy",
            "length_intron": "lengthIntrons.npy",
            "p_single_exon": "p1E.npy",
            "p_last_exon": "pEO.npy",
        }

        data = {k: np.load(os.path.join(directory, v)) for k, v in REMAP_TABLE.items()}

        def process(v):
            idx_last = np.where(v > 0)[0][-1]
            v = v[: idx_last + 1]
            v = v / v.sum()
            v = np.cumsum(v)
            return v

        data = {k: process(v) if k.startswith("length") else v for k, v in data.items()}
        return cls(**data)

    @staticmethod
    def sample_from(probs, rng):
        """
        Sample from a distribution specified by a cumulative sum distribution
        """
        return np.searchsorted(probs, rng.uniform())

    def sample_splicing_length_pattern(self, rng):
        if rng.uniform() < self.p_single_exon:
            return [self.sample_from(self.length_single_exon, rng)], []

        lengths_exons = [self.sample_from(self.length_first_exon, rng)]
        lengths_introns = []
        while True:
            lengths_introns.append(self.sample_from(self.length_intron, rng))
            if rng.uniform() < self.p_last_exon:
                lengths_exons.append(self.sample_from(self.length_last_exon, rng))
                break
            lengths_exons.append(self.sample_from(self.length_middle_exon, rng))
        return lengths_exons, lengths_introns
