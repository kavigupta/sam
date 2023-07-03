import attr
from matplotlib import pyplot as plt

from permacache import permacache

import numpy as np
import scipy.special

from modular_splicing.utils.plots.plot_psam import render_psams

from .motif import Motif


@attr.s
class PSAMMotif(Motif):
    log_psams = attr.ib()  # log-affinities; (num_motifs, width, 4)
    activation_point_value = attr.ib()  # point where the motif is activated
    clip_threshold = attr.ib()  # threshold to subtract and clip (in log space)

    @classmethod
    def sample(
        cls,
        rng,
        *,
        density,
        num_psams,
        motif_width,
        range,
        activation_point=None,
    ):
        center_psam = np.log(sample_psam(rng, motif_width) + 1e-10)
        psams = center_psam[None].repeat(num_psams, 0)
        psams += rng.randn(num_psams, motif_width, 4) * range
        psams = np.exp(psams)
        if activation_point is None:
            activation_point = rng.randint(motif_width)
        return cls.from_motifs(psams, activation_point, density)

    @classmethod
    def from_motifs(cls, psams, activation_point, density):
        psams = np.array(psams)
        assert psams.ndim == 3 and psams.shape[2] == 4
        psams /= np.max(psams, axis=2, keepdims=True)
        psams = np.log(psams)
        return cls(psams, activation_point, 0).calibrate_to(density)

    def calibrate_to(self, density):
        """
        Calibrates the motif to the given density, using a binary search.

        Only varies the clip_threshold.
        """
        motif = self
        low, high = -100, 0
        while high - low > 1e-3:
            mid = (low + high) / 2
            motif = attr.evolve(motif, clip_threshold=mid)
            if motif.empirical_density(100_000) < density:
                high = mid
            else:
                low = mid
        return motif

    def score(self, rna, pad=True, cut_off=True):
        if len(rna.shape) == 1:
            # no batch dimension
            rna = rna[None]
            has_batch_dim = False
        else:
            assert len(rna.shape) == 2
            has_batch_dim = True
        # cut off i from the left, and psam_width - i from the right
        affinities_each_base = np.array(
            [
                self.log_psams[
                    :, i, rna[:, i : rna.shape[1] - (self.log_psams.shape[1] - 1 - i)]
                ]
                for i in range(self.log_psams.shape[1])
            ]
        )
        # affinities_each_base[base_in_motif, psam, batch, start_pos_of_motif]
        affinities_across_base = affinities_each_base.sum(0)
        # affinities_across_base[psam, batch, start_pos_of_motif]
        affinities = scipy.special.logsumexp(affinities_across_base, axis=0)
        # affinities[batch, start_pos_of_motif]
        affinities = affinities - self.clip_threshold
        if cut_off:
            affinities[affinities < 0] = 0
        # need to pad with psam_width - 1 total, but split such that the activation point
        # is in the middle
        pad_left = self.activation_point()
        pad_right = self.log_psams.shape[1] - 1 - pad_left
        result = affinities
        if pad:
            result = np.concatenate(
                [
                    np.zeros(
                        (
                            result.shape[0],
                            pad_left,
                        )
                    ),
                    result,
                    np.zeros(
                        (
                            result.shape[0],
                            pad_right,
                        )
                    ),
                ],
                axis=1,
            )
        if not has_batch_dim:
            assert result.shape[0] == 1
            result = result[0]
        return result

    def radii_each(self):
        radius_right = self.log_psams.shape[1] - 1 - self.activation_point()
        radius_left = self.activation_point()
        return radius_left, radius_right

    def activation_point(self):
        return self.activation_point_value

    @property
    def psams(self):
        return np.exp(self.log_psams)

    def psams_to_render(self, prefix):
        psams = [self.empirical_psam(), *self.psams]
        names = ["empirical", *range(len(self.psams))]
        names = [f"{prefix} [{name}]" for name in names]
        return psams, names


def plot_demo(r_values):
    all_psams, all_names = [], []
    for r in r_values:
        rng = np.random.RandomState(0)
        motif = PSAMMotif.sample(
            rng,
            density=0.01,
            num_psams=3,
            motif_width=7,
            range=r,
        )
        psams, names = motif.psams_to_render(f"r={r}")
        all_psams += psams
        all_names += names
    render_psams(
        all_psams,
        names=all_names,
        psam_mode="info",
        axes_mode="just_y",
        ncols=len(psams),
    )
    plt.show()


@permacache(
    "working/synthetic_data/splicing_mechanism/motif/psam_motif/known_psam_columns"
)
def known_psam_columns():
    from modular_splicing.psams.sources import read_rbns_v2p1_motifs

    return np.concatenate(
        [
            np.concatenate([p.matrix for p in ps])
            for ps in read_rbns_v2p1_motifs().values()
        ]
    )


def sample_psam(rng, motif_width):
    all_psam_columns = known_psam_columns()

    columns = all_psam_columns[
        rng.choice(all_psam_columns.shape[0], replace=False, size=motif_width)
    ]
    for i in range(columns.shape[0]):
        rng.shuffle(columns[i])
    return columns
