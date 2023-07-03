import os
import re
from matplotlib import pyplot as plt
import numpy as np

from permacache import permacache

from modular_splicing.models_for_testing.load_model_for_testing import step_for_density

from modular_splicing.utils.io import load_model, model_steps

from modular_splicing.motif_perturbations.summarize_effect import (
    positional_effects_near_splicepoints,
)
from modular_splicing.motif_perturbations.perturbations_on_standardized_sample import (
    motif_perturbations_individual_on_standardized_sample,
)

spm_model_settings = {
    "236a2": dict(
        name="13x4-nsp",
        original="model/rbnsp-80-adj13x4-nsp",
        original_sparsity=0.178e-2,
        seeds=(1, 2, 3, 4),
    ),
    "237a2": dict(
        name="21x2",
        original="model/rbnsp-80-adj2",
        original_sparsity=0.178e-2,
        seeds=(1, 2, 3, 4),
    ),
}


def spm_models():
    for path, setting in spm_model_settings.items():
        for seed in setting["seeds"]:
            original = f"{setting['original']}_{seed}"
            yield f'{setting["name"]}_{seed}', dict(
                path=original,
                step=step_for_density(original, setting["original_sparsity"]),
            ), f"model/msp-{path}_{seed}/"


def single_perturbed_models(path):
    count = len([x for x in os.listdir(path) if re.match(r"dropping_(.*)", x)])
    if count <= 1:
        return
    max_step = max(model_steps(f"{path}/dropping_0"))
    for i in range(count):
        p = f"{path}/dropping_{i}"
        if max_step in model_steps(p):
            yield i, (p, max_step)


def perturbations_for_model(*, p, max_step, sl):
    step, m = load_model(p, max_step)
    assert step == max_step
    return motif_perturbations_individual_on_standardized_sample(
        m.eval(), path="dataset_train_all.h5", is_binary=True, value_range=None, sl=sl
    )


@permacache("training/single_perturbed/analysis/sensitivity_for_model")
def sensitivity_for_model(path, step, *, sl, num_motifs, effect_radius):
    pert = perturbations_for_model(p=path, max_step=step, sl=sl)
    return positional_effects_near_splicepoints(
        pert,
        num_motifs=num_motifs,
        blur_radius=1,
        effect_radius=effect_radius,
        normalize_mode="by_total_effect",
    ).sum((0, 1))


def single_perturbed_model_sensitivity(path, *, sl, num_motifs, effect_radius):
    result = {}
    for i, (p, max_step) in single_perturbed_models(path):
        res = sensitivity_for_model(
            path=p,
            step=max_step,
            sl=sl,
            num_motifs=num_motifs,
            effect_radius=effect_radius,
        )
        assert res[-1] == 0
        res = res[:-1]
        res = np.concatenate([res[:i], [0], res[i:]])
        result[i] = res
    return result


def spm_table(sl=1000, num_motifs=79):
    ops = []
    ups = []
    names = []
    for name, original, path in spm_models():
        orig_p = sensitivity_for_model(
            path=original["path"],
            step=original["step"],
            sl=sl,
            num_motifs=num_motifs,
            effect_radius=200,
        )
        updated_p = single_perturbed_model_sensitivity(
            path=path, sl=sl, num_motifs=num_motifs, effect_radius=200
        )
        updated_p = np.array([updated_p[i] for i in range(num_motifs)])
        ops.append(orig_p)
        ups.append(updated_p)
        names.append(name)
    ops, ups = np.array(ops), np.array(ups)
    return ops, ups, names


def draw_average(ops, names):
    xs = np.arange(ops.shape[1])
    ordering = np.argsort(-ops.mean(0))
    ord_ops = ops[:, ordering] * 100
    low, high = np.percentile(ord_ops, 25, axis=0), np.percentile(ord_ops, 75, axis=0)
    plt.figure(figsize=(15, 5), dpi=200)
    plt.plot(xs, ord_ops.T, color="black", alpha=0.5, marker=".", linestyle=" ")
    plt.fill_between(xs, low, high, color="blue", alpha=0.25)
    plt.plot(xs, ord_ops.mean(0), color="blue")
    plt.ylabel("Overall Effect [%]")
    plt.xticks(xs, np.array(names)[ordering], rotation=90)
    plt.grid()
    plt.show()
