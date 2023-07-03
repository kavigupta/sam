from functools import lru_cache
import numpy as np
import torch
import torch.nn as nn
import tqdm.auto as tqdm

from permacache import permacache

from modular_splicing.dataset.data_rewriter import DataRewriter
from modular_splicing.evaluation.evaluation_criterion import EvaluationCriterion
from modular_splicing.models.modules.spliceai import SpliceAIModule

from modular_splicing.utils.io import load_model
from modular_splicing.utils.run_batched import run_batched
from modular_splicing.utils.construct import construct

from modular_splicing.evaluation import standard_e2e_eval


class ReconstructSequenceModel(nn.Module):
    """
    Run a reconstruction experiment on an existing model's motifs.

    Arguments:
        original_model_path: Path to the model to reconstruct.
        original_model_step: Step of the model to reconstruct.
        downstream_spec: Specification of the downstream model.
    """

    def __init__(
        self,
        *,
        original_model_path,
        original_model_step,
        downstream_spec,
        input_size=4,
        sparsity=None,
        num_motifs,
        cl,
    ):
        super().__init__()
        _, self.original_model = load_model(original_model_path, original_model_step)
        for param in self.original_model.parameters():
            param.requires_grad = False

        self.downstream = construct(
            dict(SpliceAIModule=SpliceAIModule),
            downstream_spec,
            window=cl,
            input_size=num_motifs,
            output_size=4,
        )

    def forward(self, x, collect_losses=False):
        x = self.original_model(x, only_motifs=True)["post_sparse_motifs_only"]
        x = self.downstream(x)
        if collect_losses:
            return dict(output=x)
        return x


class ReconstructSequenceDataRewriter(DataRewriter):
    """
    Sets the output to be the input, but clipped, since the
    goal here is autoencoding.
    """

    def rewrite_datapoint(self, *, el, i, j, dset):
        x, y = el["inputs"]["x"], el["outputs"]["y"]
        trim = x.shape[0] - y.shape[0]
        x = x[trim // 2 : x.shape[0] - trim // 2]
        el["outputs"]["y"] = x.astype(np.float32)
        return el


class ReconstructSequenceEvaluationCriterion(EvaluationCriterion):
    """
    Evaluation criterion for the reconstruction experiment.

    Main difference is that we use BCEWithLogitsLoss instead of
    CrossEntropyLoss, just so we can get the per-base loss.
    """

    def __init__(self, only_train=None):
        assert only_train is None

    def loss_criterion(self):
        return torch.nn.BCEWithLogitsLoss(reduction="none")

    def reorient_data_for_classification(self, y, yp, mask, weights):
        assert y.shape == yp.shape
        assert len(yp.shape) == 3
        assert len(mask.shape) == len(weights.shape) == len(y.shape) - 1 == 2
        mask = mask.unsqueeze(-1).repeat(1, 1, yp.shape[-1])
        weights = weights.unsqueeze(-1).repeat(1, 1, yp.shape[-1])
        return (
            y.flatten(),
            yp.flatten(),
            mask.flatten(),
            weights.flatten(),
        )

    def mask_for(self, y):
        return torch.ones_like(y[..., 0], dtype=np.bool)

    def evaluation_channels(self, yp):
        """
        We evaluate on all indices, since we're predicting ACGT rather than null/A/D.
        """
        assert yp.shape[-1] == 4, str(yp.shape)
        return range(4)

    def for_channel(self, y, c):
        assert y.shape[-1] == 4
        return y[:, c]

    def actual_eval_indices(self):
        return [0, 1, 2, 3]


@lru_cache(None)
def get_data(cl):
    """
    Get the data for the reconstruction experiment, just as `xs` values.
    """
    normal_spec = dict(
        type="H5Dataset",
        sl=5000,
        datapoint_extractor_spec=dict(
            type="BasicDatapointExtractor",
        ),
        post_processor_spec=dict(type="IdentityPostProcessor"),
    )
    data = list(tqdm.tqdm(standard_e2e_eval.test_data(data_spec=normal_spec, cl=cl)))
    xs = np.array([x["inputs"]["x"] for x in data])
    return xs


@permacache("modules/reconstruct_sequence/evaluate_reconstruction")
def evaluate_reconstruction(*, path, step, cl):
    """
    Evaluate a reconstruction model. The path and step here are for the `ReconstructSequenceModel`,
    not the original model.

    Computes the accuracy of how well the reconstruction model can predict the original sequence,
    e.g., [A, C, G, T] -> [A, C, G, T] is 100% and [A, C, G, T] -> [A, T, T, T] is 50%.
    """
    _, model = load_model(path, step)
    model = model.eval()
    xs = get_data(cl)
    xs_predicted = run_batched(model, xs, 32, pbar=tqdm.tqdm)
    return (xs_predicted.argmax(-1) == xs[:, cl // 2 : -cl // 2].argmax(-1)).mean()
