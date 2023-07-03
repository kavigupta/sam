import fire
import numpy as np

from permacache import permacache, stable_hash

from working.synthetic_data.codon_table import CodonTable
from working.synthetic_data.linear_approx_mcmc import (
    linear_approx_mcmc,
    multiple_try_linear_approx_mcmc,
)
from working.synthetic_data.splicing_length_pattern_distribution import HSMM_SLPD
from working.synthetic_data.splicing_mechanism.example import mechanism
from working.synthetic_data.realization import (
    codon_realization,
    Realization,
)
from working.synthetic_data.codon_addressing import CodonAddress

codon_table = CodonTable.standard()
slpd = HSMM_SLPD.from_directory("../data/scfg-params")


def realize(seed):
    rng = np.random.RandomState(seed)
    while True:
        cr = codon_realization(codon_table, slpd, rng)
        realization = Realization.initialize(
            codon_table, mechanism(), *cr, rng, frac_fake=0.001
        )
        rna = realization.rna.copy()
        true_splice = realization.real_splicing_pattern

        protected = np.array(sorted(realization.protected_sites), dtype=np.int64)
        protected_mask = np.zeros(rna.shape[0], dtype=np.bool_)
        protected_mask[protected] = 1

        unchangable_motifs = mechanism().unchangable_motifs(protected_mask)
        codon_address = CodonAddress.of(
            realization.exon_indices, realization.intron_indices, protected_mask
        )
        return rna, true_splice, unchangable_motifs, codon_address


@permacache(
    "working/synthetic_data/experiment/run_realization_2",
    multiprocess_safe=True,
)
def run_realization(
    seed, num_tries, max_steps, temperature=50, temperature_mut=50, negativity_bias=0
):
    print(seed, "start")
    rna, true_splice, unchangable_motifs, codon_address = realize(seed)
    result = multiple_try_linear_approx_mcmc(
        rna=rna,
        codon_table=codon_table,
        codon_address=codon_address,
        mech=mechanism(),
        true_splice=true_splice,
        unchangable_motifs=unchangable_motifs,
        temperature=temperature,
        temperature_mut=temperature_mut,
        max_steps=max_steps,
        negativity_bias=negativity_bias,
        num_tries=num_tries,
    )
    print(seed, "end")
    return result


@permacache(
    "working/synthetic_data/experiment/run_experiment_2",
    multiprocess_safe=True,
)
def run_experiment(
    seed, temperature, temperature_mut, negativity_bias, max_steps=50_000
):
    rng = np.random.RandomState(seed)
    seed_1 = rng.randint(2**32)
    seed_2 = rng.randint(2**32)
    rna, true_splice, unchangable_motifs, codon_address = realize(seed_1)
    return linear_approx_mcmc(
        seed=seed_2,
        rna=rna,
        codon_table=codon_table,
        codon_address=codon_address,
        mech=mechanism(),
        true_splice=true_splice,
        unchangable_motifs=unchangable_motifs,
        temperature=temperature,
        temperature_mut=temperature_mut,
        max_steps=max_steps,
        negativity_bias=negativity_bias,
    )


def plot_results_by_temperature(table_relevant, ax, m=50_000):
    table_relevant = table_relevant.pivot(
        index="temp", columns="temp_mut", values="steps"
    )
    original = table_relevant.copy()
    table_relevant = table_relevant.fillna(m)

    ax.imshow(table_relevant, clim=(0, m), cmap="gray_r")
    ax.set_xticks(np.arange(len(table_relevant.columns)))
    ax.set_xticklabels(table_relevant.columns)
    ax.set_xlabel("Temp of mut")
    ax.set_yticks(np.arange(len(table_relevant.index)))
    ax.set_yticklabels(table_relevant.index)
    ax.set_ylabel("Temp")
    ax.set_title("Steps to convergence")
    for i in range(len(table_relevant.index)):
        for j in range(len(table_relevant.columns)):
            # print(i, j, original.iloc[i, j], np.isnan(original.iloc[i, j]))
            ax.text(
                j,
                i,
                "×"
                if np.isnan(original.iloc[i, j])
                else f"{table_relevant.iloc[i, j]:.0f}"
                if table_relevant.iloc[i, j] < m
                else "∞",
                ha="center",
                va="center",
                color="w" if table_relevant.iloc[i, j] > m / 2 else "k",
            )


def main(seed, temperature, temperature_mut, negativity_bias, max_steps=50_000):
    run_experiment(seed, temperature, temperature_mut, negativity_bias, max_steps)


if __name__ == "__main__":
    fire.Fire()
