import attr
import tqdm.auto as tqdm

import numpy as np

from permacache import permacache, stable_hash

from working.synthetic_data.gradient_mcmc import select_independent_mutations

from working.synthetic_data.saliency_map import SaliencyMap


@attr.s
class StaticComponents:
    codon_table = attr.ib()
    codon_address = attr.ib()
    mech = attr.ib()
    unchangable_motifs = attr.ib()
    temperature = attr.ib()
    temperature_mut = attr.ib()
    negativity_bias = attr.ib()
    true_splice = attr.ib()


@permacache(
    "working/synthetic_data/linear_approx_mcmc/linear_approx_mcmc_7",
    key_function=dict(
        rna=stable_hash,
        codon_table=stable_hash,
        codon_address=stable_hash,
        mech=stable_hash,
        unchangable_motifs=stable_hash,
    ),
    multiprocess_safe=True,
)
def linear_approx_mcmc(
    *,
    rna,
    codon_table,
    codon_address,
    mech,
    true_splice,
    unchangable_motifs,
    seed,
    temperature,
    temperature_mut,
    max_steps,
    negativity_bias,
):
    static_components = StaticComponents(
        codon_table=codon_table,
        codon_address=codon_address,
        mech=mech,
        unchangable_motifs=unchangable_motifs,
        temperature=temperature,
        temperature_mut=temperature_mut,
        negativity_bias=negativity_bias,
        true_splice=true_splice,
    )
    rng = np.random.RandomState(seed)
    all_steps = []
    all_scores = []
    all_new_scores = []
    for i, k in enumerate(split_chunks(max_steps, 1000)):
        print(i, k)
        rna, steps, scores, new_scores, rng = several_steps_linear_approx_mcmc(
            rna, true_splice, k, static_components, rng
        )
        done = len(scores) <= k
        if i > 0:
            steps = steps[1:]
            scores = scores[1:]
            new_scores = new_scores[1:]
        all_steps.extend(steps)
        all_scores.extend(scores)
        all_new_scores.extend(new_scores)
        if done:
            break
    return rna, np.cumsum(all_steps).tolist(), all_scores, all_new_scores


def split_chunks(max_steps, chunk_size):
    result = []
    result += [chunk_size] * (max_steps // chunk_size)
    if max_steps % chunk_size != 0:
        result += [max_steps % chunk_size]
    assert sum(result) == max_steps
    assert all(x > 0 for x in result)
    return result


@permacache(
    "working/synthetic_data/linear_approx_mcmc/several_steps_linear_approx_mcmc_3",
    key_function=dict(
        rna=stable_hash,
        static_components=stable_hash,
        rng=lambda rng: stable_hash(rng.get_state()),
    ),
    multiprocess_safe=True,
)
def several_steps_linear_approx_mcmc(
    rna, true_splice, max_steps, static_components, rng
):
    motifs_uncut = static_components.mech.predict_motifs(rna, cut_off=False)
    motifs = np.maximum(motifs_uncut, 0)
    score = static_components.mech.score_from_motifs(motifs, true_splice)
    steps = [0]
    scores = [score]
    new_scores = [score]
    for _ in tqdm.trange(max_steps):
        (
            done,
            rna,
            motifs,
            motifs_uncut,
            score,
            new_score,
            num_steps,
        ) = single_step_linear_approx_mcmc(
            static_components=static_components,
            rna=rna,
            motifs=motifs,
            motifs_uncut=motifs_uncut,
            score=score,
            rng=rng,
        )
        if done:
            break
        steps.append(num_steps)
        new_scores.append(new_score)
        scores.append(score)
    return rna, steps, scores, new_scores, rng


def single_step_linear_approx_mcmc(
    *, static_components, rna, motifs, motifs_uncut, score, rng
):
    saliency_map = SaliencyMap.of(
        static_components.unchangable_motifs,
        static_components.codon_address,
        static_components.mech.motif_saliency_map(
            motifs, static_components.true_splice
        ),
        0.01,
    )
    muts = select_independent_mutations(
        saliency_map,
        static_components.mech,
        static_components.codon_table,
        rna,
        motifs_uncut,
        rng=rng,
        temperature=static_components.temperature_mut,
        negativity_bias=static_components.negativity_bias,
    )
    new_rna = rna.copy()
    for mut in muts:
        mut.perform(new_rna)
    new_motifs_uncut = static_components.mech.predict_motifs(new_rna, cut_off=False)
    new_motifs = np.maximum(new_motifs_uncut, 0)
    new_score = static_components.mech.score_from_motifs(
        new_motifs, static_components.true_splice
    )
    accept = rng.uniform() < np.exp(static_components.temperature * (new_score - score))
    done = False
    if accept:
        rna, motifs, motifs_uncut, score = (
            new_rna,
            new_motifs,
            new_motifs_uncut,
            new_score,
        )
        pred_splice = static_components.mech.predict_splicing_pattern_from_motifs(
            static_components.mech.processed_motifs(motifs)
        )
        if all(
            x.tolist() == y.tolist()
            for x, y in zip(pred_splice, static_components.true_splice)
        ):
            done = True

    return done, rna, motifs, motifs_uncut, score, new_score, len(muts)


def multiple_try_linear_approx_mcmc(*, num_tries, max_steps, **kwargs):
    overall = []
    for seed in range(num_tries):
        rna, steps, scores, new_scores = linear_approx_mcmc(
            max_steps=max_steps,
            **kwargs,
            seed=seed,
        )
        overall.append(
            dict(
                rna=rna,
                steps=steps,
                scores=scores,
                new_scores=new_scores,
                true_splice=kwargs["true_splice"],
            )
        )
        if len(scores) < max_steps:
            break
    return overall
