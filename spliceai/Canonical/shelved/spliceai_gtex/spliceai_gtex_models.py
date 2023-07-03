import pandas as pd
import tqdm.auto as tqdm
import numpy as np

import attr
import torch

from modular_splicing.evaluation.evaluation_criterion import EvaluationCriterion

from modular_splicing.models_for_testing.load_model_for_testing import step_for_density
from modular_splicing.evaluation.standard_e2e_eval import evaluate_on_checkpoint
from modular_splicing.utils.io import load_model
from modular_splicing.utils.construct import construct

settings = [
    dict(
        setting="/Canonical",
        get_path=dict(
            type="arch_lookup",
            tables={
                "FM": "rbnsp-67",
                "AM_21x2": "rbnsp-80-adj2",
                "AM_13x4-nsp": "rbnsp-80-adj13x4-nsp",
                "spliceai-400": "msp-249zaa2",
                "spliceai-10k": "msp-249zba2",
            },
        ),
        outs=["Canonical"],
    ),
    dict(
        setting="/GTEx",
        get_path=dict(
            type="path_fmt", path_fmt="msp-{number}{multiplier_letter}a{version}_{seed}"
        ),
        outs=["GTEx"],
    ),
    dict(
        setting="/Cano+GTEx",
        get_path=dict(
            type="path_fmt",
            path_fmt="msp-{number}m{multiplier_letter}a{version}_{seed}",
        ),
        outs=["Canonical", "GTEx"],
    ),
    # dict(
    #     setting="/GT-C",
    #     get_path=dict(
    #         type="path_fmt",
    #         path_fmt="msp-{number}o{multiplier_letter}a{version}_{seed}",
    #     ),
    #     outs=["GTEx"],
    # ),
    dict(
        setting="/C-GT",
        get_path=dict(
            type="path_fmt",
            path_fmt="msp-{number}p{multiplier_letter}a{version}_{seed}",
        ),
        outs=["Cano"],
    ),
    # dict(
    #     setting="/Can+GTEx; to GTEx",
    #     get_path=dict(
    #         type="path_fmt", path_fmt="msp-{number}n{multiplier_letter}a1_{seed}"
    #     ),
    #     outs=["Canonical", "GTEx"],
    # ),
]


def spliceai_arch(label, multiplier_letter):
    return dict(
        number=249,
        multiplier_letter=multiplier_letter,
        arch=label,
        get_model=dict(
            type="with_step",
            step=3254400,
        ),
        version={
            "/Canonical": 2,
            "/GTEx": 2,
            "/Cano+GTEx": 2,
            "/GT-C": 2,
            "/C-GT": 3,
        },
        plot_lc=False,
    )


archs = [
    spliceai_arch("spliceai-400", "a"),
    spliceai_arch("spliceai-10k", "b"),
    dict(
        number=246,
        multiplier_letter="a",
        arch="FM",
        get_model=dict(type="at_sparsity", sparsity=0.178e-2),
        plot_lc=True,
        version={"/C-GT": 2},
    ),
    dict(
        number=247,
        multiplier_letter="b",
        arch="AM_21x2",
        get_model=dict(type="at_sparsity", sparsity=0.178e-2),
        plot_lc=True,
        version={"/C-GT": 2},
    ),
    dict(
        number=248,
        multiplier_letter="d",
        arch="AM_13x4-nsp",
        get_model=dict(type="at_sparsity", sparsity=0.178e-2),
        plot_lc=True,
        version={"/C-GT": 2},
    ),
]

path_prefixes = {
    "Cano": ".",
    "GTEx": "../data/gtex_dataset",
    "GT-C": "../data/canonical_and_gtex_dataset",
    "C-GT": "../data/canonical_and_gtex_dataset",
}


def data_spec(eval_on):
    if eval_on in {"GT-C", "C-GT"}:
        return dict(
            type="NonConflictingAlternativeDataset",
            post_processor_spec=dict(type="IdentityPostProcessor"),
            underlying_ordering=["spliceai_canonical", "spliceai_gtex"],
            outcome_to_pick={"GT-C": "spliceai_gtex", "C-GT": "spliceai_canonical"}[
                eval_on
            ],
            channels_per_outcome=3,
            mask_channel_offsets=[1, 2],
            always_keep_picked={"GT-C": False, "C-GT": True}[eval_on],
        )
    elif eval_on in ["GTEx", "Cano"]:
        return dict(
            type="H5Dataset",
            datapoint_extractor_spec=dict(type="BasicDatapointExtractor"),
            post_processor_spec=dict(type="IdentityPostProcessor"),
        )
    else:
        raise ValueError(f"Unknown eval_on: {eval_on}")


def path(setting, arch, seed):
    return construct(
        dict(
            path_fmt=lambda path_fmt, arch, seed: path_fmt.format(
                number=arch["number"],
                multiplier_letter=arch["multiplier_letter"],
                seed=seed,
                version=arch.get("version", {}).get(setting["setting"], 1),
            ),
            arch_lookup=lambda tables, arch, seed: f"{tables[arch['arch']]}_{seed}",
        ),
        setting["get_path"],
        arch=arch,
        seed=seed,
    )


def display(setting, arch, seed):
    return (
        arch["arch"] + setting["setting"] + ("; seed " + str(seed) if seed > 1 else "")
    )


def get_criterion(setting, eval_for, eval_on):
    idx = setting["outs"].index(eval_for)
    channels = list(range(idx * 3, idx * 3 + 3))
    if eval_on in ["GT-C", "C-GT"]:
        typ = EvaluateSubsetOfChannelsCriterionOneHot
    else:
        assert eval_on in ["GTEx", "Cano"]
        typ = EvaluateSubsetOfChannelsCriterion
    return typ(
        model_channels=channels,
        subset_to_evaluate=channels[1:],
    )


@attr.s
class EvaluateSubsetOfChannelsCriterion(EvaluationCriterion):
    model_channels = attr.ib()
    subset_to_evaluate = attr.ib()
    version = attr.ib(default="1.3")

    def loss_criterion(self):
        raise NotImplementedError

    def reorient_data_for_classification(self, y, yp, mask, weights):
        raise NotImplementedError

    def mask_for(self, y):
        return torch.ones_like(y, dtype=np.bool)

    def evaluation_channels(self, yp):
        return self.subset_to_evaluate

    def for_channel(self, y, c):
        return y == self.model_channels.index(c)

    def actual_eval_indices(self):
        raise NotImplementedError


class EvaluateSubsetOfChannelsCriterionOneHot(EvaluateSubsetOfChannelsCriterion):
    version = attr.ib(default="1.4")

    def mask_for(self, y):
        return torch.ones(y.shape[:-1], dtype=np.bool, device=y.device)

    def for_channel(self, y, c):
        return y[:, self.model_channels.index(c)]


def evaluate_model(setting, arch, *, seed, eval_on, limit=float("inf")):
    p = f"model/{path(setting, arch, seed)}"
    step = construct(
        dict(
            at_sparsity=lambda sparsity: step_for_density(p, sparsity, err=False),
            with_step=lambda step: load_model(p, step)[0],
        ),
        arch["get_model"],
    )
    if step is None:
        return {eval_for: np.nan for eval_for in setting["outs"]}
    return {
        eval_for: np.mean(
            evaluate_on_checkpoint(
                path=p,
                step=step,
                limit=limit,
                bs=32,
                pbar=tqdm.tqdm,
                evaluation_criterion=get_criterion(setting, eval_for, eval_on),
                data_spec=dict(**data_spec(eval_on), SL=5000),
                data_path=f"{path_prefixes[eval_on]}/dataset_test_0.h5",
            )
        )
        for eval_for in setting["outs"]
    }


def table_of_accuracies():
    results = []
    names = []
    for arch in archs:
        for setting in settings:
            all_evaluations = {
                (eval_on, eval_for): res
                for eval_on in path_prefixes
                for eval_for, res in evaluate_model(
                    setting,
                    arch,
                    seed=1,
                    eval_on=eval_on,
                ).items()
            }
            for eval_for in sorted(set(y for _, y in all_evaluations)):
                names.append(display(setting, arch, seed=1) + f"; out={eval_for[:4]}")
                results.append(
                    {
                        eval_on: all_evaluations[eval_on, y]
                        for eval_on, y in all_evaluations
                        if y == eval_for
                    }
                )
    results = pd.DataFrame(results, index=names) * 100
    results = results[sorted(results)]
    return results
