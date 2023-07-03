from datetime import datetime

import torch
import numpy as np

from modular_splicing.utils.io import load_model, save_model
from modular_splicing.utils.construct import construct

from modular_splicing.evaluation.evaluation_criterion import (
    evaluation_criteria_specs,
)
from modular_splicing.evaluation.run_evaluation import evaluate_model_on_data
from .utils import get_loss


def train_model(
    *,
    path,
    dtrain,
    deval,
    architecture,
    bs,
    n_epochs,
    lr=1e-3,
    evaluation_criterion_spec,
    only_train=None,
    decay_start=6,
    decay_amount=0.5,
    report_frequency=1000,
    update_callback=lambda *args, **kwargs: None,
    eval_limit=float("inf"),
    train_limit=float("inf"),
    cuda=True,
    print_model=True,
    model_done_training=lambda m: False,
):
    """
    Train the given model. This function can be used to train a model from scratch,
        or to continue training a model that has already been trained. Will just
        continue training from the last saved batch.

    Saves the model after each `report_frequency` steps, and at the end of each
        epoch. Whenever the model is saved, the model is also evaluated on the
        validation set and the whole test set.

    The callback `update_callback` is called after each reporting step (but not
        at the end of each epoch). It is called with the following arguments:
            m: the model
            fractional_epoch: the fractional epoch (i.e., step 500/10000 of epoch 2 is
                fractional epoch 2.05)
            callback_acc: the accuracy on the validation set
        It can be used to change the sparsity of the model, for example.

    Arguments:
        path: the path to save the model to
        dtrain: the training set
        deval: the validation set
        architecture: the architecture to use
        bs: the batch size
        n_epochs: the number of epochs to train for
        lr: the learning rate
        evaluation_criterion_spec: the evaluation criterion to use
        only_train: if not None, only train the given outputs. Passed as an argument to
            construct the evaluation criterion
        decay_start: the epoch to start decaying the learning rate
        decay_amount: the amount to decay the learning rate by (e.g., 0.9 is -10% decay)
        report_frequency: the number of steps between reporting steps, in batches.
        eval_limit: the maximum number of examples (not batches!) to evaluate on
        train_limit: the maximum number of examples (not batches!) to train on
        cuda: whether to use CUDA. If False, will use CPU.
        print_model: whether to print the model after constructing it
        model_done_training: a function that takes the model as an argument and returns
            True if the model is done training and should stop training. This is useful
            for models where we want to stop training after it reaches a certain
            sparsity.
    """
    evaluation_criterion = construct(
        evaluation_criteria_specs(),
        evaluation_criterion_spec,
        only_train=only_train,
    )
    skip_to_step, m = load_model(path)
    if m is None:
        m = architecture()
        skip_to_step = 0
    if not torch.cuda.is_available():
        cuda = False
    if cuda:
        m = m.cuda()
    else:
        m = m.cpu()
        print("WARNING: CUDA not available, using CPU")
    if print_model:
        print(m)
    else:
        print(f"[model output elided]")
    print("Step", skip_to_step)

    if n_epochs == 0 and skip_to_step == 0:
        save_model(m, path, 0)

    if skip_to_step >= len(dtrain) * n_epochs - 1:
        return

    step = 0

    optimizer = torch.optim.Adam(m.parameters(), lr=lr)

    for e in range(n_epochs):
        l = torch.utils.data.DataLoader(dtrain, num_workers=0, batch_size=bs)
        for i, xy in enumerate(l):
            # if we have already done this step, skip it
            if step < skip_to_step:
                step += bs
                continue
            step += bs
            loss_display = training_step(
                m=m,
                xy=xy,
                evaluation_criterion=evaluation_criterion,
                optimizer=optimizer,
            )
            periodic_report_batch = i % report_frequency == 0
            end_of_epoch_batch = i == len(l) - 1
            if periodic_report_batch or end_of_epoch_batch:
                save_model(m, path, step)

                if model_done_training(m):
                    return m

                callback_acc = reporting_step(
                    m=m,
                    deval=deval,
                    bs=bs,
                    eval_limit=eval_limit,
                    evaluation_criterion=evaluation_criterion,
                    loss_display=loss_display,
                    step_display=f"s={step}, e={e}/{n_epochs}, it={i}/{len(l)},",
                )
            if periodic_report_batch:
                update_callback(m, e + i / len(l), callback_acc)
            # if we have done enough steps, stop
            if step > train_limit:
                return m
        # decay the learning rate
        if e >= decay_start:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= decay_amount
    return m


def reporting_step(
    *, m, deval, bs, eval_limit, evaluation_criterion, loss_display, step_display
):
    """
    Run a reporting step, which includes evaluating the model on the validation and whole
        test set and printing the results.

    Returns the accuracy of the model on the validation set.
    """

    callback_acc, acc_display = accuracy_and_display_for_dataset(
        m,
        deval,
        evaluation_criterion=evaluation_criterion,
        bs=bs,
        limit=min(eval_limit, len(deval) // 2),
        prefix="val_acc",
    )

    print(
        f"[{datetime.now()}] {step_display}",
        f"loss={loss_display}, {acc_display}",
    )

    return callback_acc


def training_step(*, m, xy, optimizer, evaluation_criterion):
    """
    Run a single training step on the given batch

    Arguments:
        m: the model
        xy: the batch
        optimizer: the optimizer
        evaluation_criterion: the evaluation criterion

    Returns:
        a string representing the loss
    """
    loss, weight = get_loss(
        m=m,
        xy=xy,
        evaluation_criterion=evaluation_criterion,
    )
    weight = weight.item()
    if loss is not None:
        loss.backward()
        take_step(optimizer, weight)
        loss_display = f"{loss.item():.4e}"
    else:
        loss_display = "-"
    return loss_display


def take_step(optimizer, weight):
    """
    Weights the learning rates by the given weight, then takes a step.
    Zeros the gradients afterwards.

    Arguments:
        optimizer: the optimizer to use
        weight: the weight to use
    """
    original_lrs = [param_group["lr"] for param_group in optimizer.param_groups]
    updated_lrs = [lr * weight for lr in original_lrs]
    for param_group, updated_lr in zip(optimizer.param_groups, updated_lrs):
        param_group["lr"] = updated_lr
    optimizer.step()
    optimizer.zero_grad()
    for param_group, original_lr in zip(optimizer.param_groups, original_lrs):
        param_group["lr"] = original_lr


def accuracy_and_display_for_dataset(
    m, deval, *, evaluation_criterion, bs, limit, prefix
):
    """
    Compute accuracy and accuracy display for a model on a dataset.

    Args:
        m: model
        deval: dataset
        evaluation_criterion: evaluation criterion
        bs: batch size
        limit: limit on number of examples to evaluate
        prefix: prefix for accuracy display

    Returns (accuracy, accuracy_display)
        accuracy is a float from 0 to 100 representing the accuracy percentage
        accuracy_display is a string of the form "acc=XX.XX% {XX.XX%, XX.XX%}"
            where the first number is the mean accuracy percentage and the rest are
            the accuracy percentages for each class
    """
    accuracies = evaluate_model_on_data(
        m,
        deval,
        limit=limit,
        bs=bs,
        evaluation_criterion=evaluation_criterion,
    )

    accuracies = np.array(accuracies) * 100
    accuracy_mean = np.mean(
        np.array(accuracies)[evaluation_criterion.actual_eval_indices()]
    )
    individual = "{" + "; ".join(f"{a:.2f}%" for a in accuracies) + "}"

    acc_display = f"{prefix}={accuracy_mean:.2f}% {individual}"

    return accuracy_mean, acc_display
