import functools
import os

from permacache import permacache, drop_if_equal
import tqdm.auto as tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

from modular_splicing.dataset.basic_dataset import basic_dataset
from modular_splicing.evaluation.standard_e2e_eval import evaluate_on_checkpoint
from modular_splicing.models_for_testing.load_model_for_testing import (
    achieved_target_acc,
)
from modular_splicing.utils.entropy_calculations import hbern
from modular_splicing.utils.io import load_model, model_steps

from shelved.auto_minimize_motifs.utils import dropped_motifs

from modular_splicing.utils.construct import construct

normal_spec = dict(
    type="H5Dataset",
    sl=5000,
    datapoint_extractor_spec=dict(
        type="BasicDatapointExtractor",
    ),
    post_processor_spec=dict(type="IdentityPostProcessor"),
)


@functools.lru_cache(None)
def len_training_data():
    data = basic_dataset("dataset_train_all.h5", 400, 10_000, sl=1)
    return len(data)


@permacache(
    "spliceai/statistics/model_accuracy_thresh",
    key_function=dict(path=os.path.abspath, default=drop_if_equal(80)),
)
def model_accuracy_thresh(path, step, default=80):
    print(path, step)
    _, m = load_model(path, step, map_location=torch.device("cpu"))
    manager = getattr(m, "_adaptive_sparsity_threshold_manager", None)
    if manager is None:
        return default
    return manager.current_threshold


class LearningCurvePlot:
    def __init__(
        self,
        size=None,
        *,
        ax=None,
        y_metric="sparsity",
        x_metric="epoch",
        annotation_functions=[],
        dpi_scale=1,
    ):
        assert (size is not None) != (ax is not None)
        if size is None:
            self.ax = ax
            self.full_figure = False
        else:
            plt.figure(figsize=(size, size), dpi=500 * dpi_scale // size)
            self.ax = plt.gca()
            self.full_figure = True
        self.colors = [
            x
            for _ in range(100)
            for x in plt.rcParams["axes.prop_cycle"].by_key()["color"]
        ][::-1]
        self.y_metric = y_metric
        self.x_metric = x_metric
        self.annotation_functions = annotation_functions

    def plot(
        self,
        name,
        desc,
        *,
        sl=5000,
        motifs=100,
        mode=1,
        w=None,
        only_epochs=True,
        produce_annotation_values=False,
        only_in_annotation=False,
        evaluate_kwargs={},
        evaluate_acc_spec=dict(type="topk_accuracy"),
        **kwargs,
    ):
        from modular_splicing.models.entire_model.reconstruct_sequence import (
            evaluate_reconstruction,
        )

        if mode == "inc":
            mode = self.mode
            self.mode += 1

        path = f"model/{name}"

        annotations = [kv for fn in self.annotation_functions for kv in fn(path)]
        if self.y_metric.startswith("accuracy"):
            steps = model_steps(path)
            if only_epochs:
                steps = np.array(steps)[
                    np.where(np.array(steps)[1:] - np.array(steps)[:-1] == steps[0])
                ].tolist()
            if only_in_annotation:
                steps = sorted(set(steps) & set(x for x, _ in annotations))
            steps = np.array(steps)
            accuracies = [
                100
                * construct(
                    dict(
                        topk_accuracy=lambda **kwargs: np.mean(
                            evaluate_on_checkpoint(
                                limit=8000,
                                bs=128,
                                pbar=tqdm.tqdm,
                                split="val",
                                **evaluate_kwargs,
                                **kwargs,
                            )
                        ),
                        evaluate_reconstruction=evaluate_reconstruction,
                    ),
                    evaluate_acc_spec,
                    path=path,
                    step=step,
                    data_spec=normal_spec,
                )
                for step in steps
            ]
            steps = np.array(steps)

        else:
            steps, sparsities = achieved_target_acc(path)
            sparsities = 1 - sparsities
            sparsities = np.array(sparsities)
        items = steps * sl
        mode_kwargs = {
            0: dict(linestyle="-"),
            1: dict(marker="*", linestyle="-"),
            2: dict(marker=".", linestyle="--"),
            3: dict(marker="o", linestyle=":"),
        }
        if self.y_metric == "accuracy":
            y_values = accuracies
        elif self.y_metric == "entropy":
            true_motifs = self.true_motifs(motifs, path, steps)
            true_sparsity = sparsities * motifs / true_motifs
            true_sparsity = np.minimum(true_sparsity, 0.5)
            y_values = true_motifs * hbern(true_sparsity)
        elif self.y_metric == "k":
            y_values = sparsities * motifs * (2 * w + 1)
        else:
            y_values = dict(
                sparsity=sparsities * 100,
                covered_positions=sparsities * 100 * motifs,
            )[self.y_metric]
        if self.x_metric.startswith("epoch"):
            x_values = items / len_training_data()
        elif self.x_metric == "num_motifs":
            x_values = self.true_motifs(motifs, path, steps)
        else:
            assert self.x_metric == "accuracy-bound"
            x_values = [model_accuracy_thresh(path, step) for step in steps]
        label_kwargs = dict(label=desc) if self.group_count > 0 else dict()
        self.group_count -= 1
        self.ax.plot(
            x_values,
            y_values,
            **label_kwargs,
            color=self.pc,
            **mode_kwargs[mode],
            **kwargs,
        )
        annotations = list(
            self.get_annotations(items // sl, x_values, y_values, annotations)
        )
        if produce_annotation_values:
            return annotations
        else:
            self.plot_annotations(annotations)

    def true_motifs(self, motifs, path, steps):
        return np.array([motifs - len(dropped_motifs(path, step)) for step in steps])

    def get_annotations(self, steps, xs, ys, annots):
        sx_map = dict(zip(steps, xs))
        sy_map = dict(zip(steps, ys))
        for step, annot in annots:
            x = sx_map[step]
            y = sy_map[step]
            yield x, y, annot

    def plot_annotations(self, annotations):
        for x, y, annot in annotations:
            self.ax.text(x, y, annot, fontsize=8, color=self.pc, rotation=90)

    def group(self, group_count=float("inf")):
        self.pc = self.colors.pop()
        self.group_count = group_count
        self.mode = 1

    def close(self, title=None, ylim=None, ticks=None):
        if ticks is None:
            ticks = self.get_ticks()
        if not self.y_metric.startswith("accuracy"):
            self.ax.set_yscale("log")
        self.ax.set_yticks(ticks, [str(y) for y in ticks])
        if self.y_metric.startswith("accuracy"):
            if self.x_metric == "epoch":
                self.ax.set_xlabel("Epoch")
            else:
                assert self.x_metric == "num_motifs"
                self.ax.set_xlabel("Number of motifs")
        elif self.x_metric.startswith("epoch"):
            self.ax.set_xlim(0, self.ax.set_xlim()[1])
            if self.x_metric == "epoch":
                self.ax.set_xlabel("Epoch at which 80% accuracy was achieved")
            elif self.x_metric == "epoch-generic-acc":
                self.ax.set_xlabel("Epoch at which target accuracy was achieved")
            else:
                raise RuntimeError(f"Bad x metric {self.x_metric}")
        else:
            assert self.x_metric == "accuracy-bound"
            self.ax.set_xlabel("Accuracy")

        label = dict(
            sparsity="Sparsity [%]",
            covered_positions="Positions covered in sequence [%]",
            entropy="Entropy estimate [b]",
            k="Mean activations in 2w+1 window (k)",
            accuracy="Accuracy [%]",
        )[self.y_metric]
        self.ax.set_ylabel(label)
        self.ax.grid()
        self.ax.legend()
        if title is not None:
            self.ax.set_title(title)
        if ylim is not None:
            self.ax.set_ylim(*ylim)
        if self.full_figure:
            plt.show()

    def get_ticks(self):
        ticks = np.array([100, 50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01])
        if self.y_metric == "accuracy":
            ticks = np.arange(50, 101, 2.5)
        if self.y_metric == "covered_positions":
            ticks *= 100
        if self.y_metric == "entropy":
            ticks = [y for y in ticks if y > 0.1]
        if self.y_metric == "k":
            ticks = ticks * 100
        return ticks
