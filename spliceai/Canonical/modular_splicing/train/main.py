import json
import numpy as np
import torch
import os
from modular_splicing.dataset.generic_dataset import dataset_types
from modular_splicing.models.entire_model.entire_model_types import entire_model_types

from .arguments import get_args

from modular_splicing.train.standard_train import train_model

from shelved.single_perturbed import single_perturbed_trainer
from modular_splicing.utils.construct import construct
from shelved.auto_minimize_motifs import train_loop

from .adaptive_sparsity_threshold_manager import AdaptiveSparsityThresholdManager


def main():
    torch.set_num_threads(1)
    args = get_args()
    assert args.seed is not None
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(args.model_path)

    assert not os.path.isfile(
        args.model_path
    ), "provided model_path must be a folder if it exists"
    path = args.model_path

    n_epochs = args.n_epochs
    bs = args.batch_size

    dtrain = construct(
        dataset_types(),
        args.dataset_spec,
        path=add_suffix(args.data_dir, "dataset_train_all.h5"),
        cl=args.window,
        cl_max=args.CL_max,
        sl=args.SL,
        iterator_spec=dict(
            type="FastIter", shuffler_spec=dict(type="UnseededShuffler")
        ),
    )

    eval_split = "0" if args.data_chunk_to_use is None else args.data_chunk_to_use
    deval = construct(
        dataset_types(),
        args.dataset_spec,
        path=add_suffix(args.data_dir, f"dataset_test_{eval_split}.h5"),
        cl=args.window,
        cl_max=args.CL_max,
        sl=args.SL,
        iterator_spec=dict(type="FastIter", shuffler_spec=dict(type="DoNotShuffle")),
    )

    def architecture():
        spec = json.loads(args.msp_architecture_spec)
        return construct(
            entire_model_types(),
            spec,
            input_size=4,
            sparsity=args.learned_motif_sparsity,
            cl=args.window,
        )

    def update_callback(m, epoch, acc):
        if args.learned_motif_sparsity_update is None:
            return
        sparse_class = AdaptiveSparsityThresholdManager.setup(
            m,
            maximal_threshold=args.learned_motif_sparsity_threshold_initial,
            minimal_threshold=args.learned_motif_sparsity_threshold,
            decrease_per_epoch=args.learned_motif_sparsity_threshold_decrease_per_epoch,
        )
        if sparse_class.passes_accuracy_threshold(acc, epoch):
            m.update_sparsity(args.learned_motif_sparsity_update)

    def model_done_training(m):
        if args.stop_at_density is None:
            return False
        density = 1 - m.get_sparsity()
        return density <= args.stop_at_density

    if args.train_mode == "standard-supervised":
        train_model(
            path=path,
            dtrain=dtrain,
            deval=deval,
            architecture=architecture,
            bs=bs,
            n_epochs=n_epochs,
            lr=args.lr,
            evaluation_criterion_spec=args.evaluation_criterion_spec,
            only_train=args.only_train,
            decay_start=args.decay_start,
            decay_amount=args.decay_amount,
            report_frequency=args.report_frequency,
            update_callback=update_callback,
            eval_limit=args.eval_limit,
            train_limit=args.train_limit,
            cuda=args.train_cuda,
            model_done_training=model_done_training,
        )
    elif args.train_mode == "from-spec":
        construct(
            dict(
                auto_minimizer=train_loop.train,
                single_perturbed=single_perturbed_trainer.train,
            ),
            args.train_spec,
            path=path,
            dtrain=dtrain,
            deval=deval,
            architecture=architecture,
            bs=bs,
            lr=args.lr,
        )
    else:
        raise ValueError(f"Unknown train mode: {args.train_mode}")


def add_suffix(path, suffix):
    if isinstance(path, str):
        return path + suffix
    assert isinstance(path, list)
    return [add_suffix(p, suffix) for p in path]


if __name__ == "__main__":
    main()
