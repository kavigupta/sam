import torch

from modular_splicing.dataset.for_model import reformat_argument_for_model


def get_loss(*, m, xy, evaluation_criterion):
    """
    Get the loss for the given model on the given batch.

    Arguments
    ---------
    m: torch.nn.Module
        The model being trained.
    xy: dict
        The batch of data to evaluate on.
    evaluation_criterion: EvaluationCriterion
        The EvaluationCriterion used for this model.
        Used to determine the loss criterion and to reorient the data.

    Returns
    -------
    loss: torch.Tensor
        The loss for this batch.
    weight: torch.Tensor
        The weight for this batch
    """
    # get the input and output for the given model
    input_argument, y, mask = reformat_argument_for_model(
        m, xy, evaluation_criterion=evaluation_criterion
    )

    # make sure the mask is on the same device as the output
    assert y.device == mask.device, str((y.device, mask.device))

    # if the mask is all false, then there is no data to train on
    if not mask.any():
        return None

    criterion = evaluation_criterion.loss_criterion().to(y.device)

    # compute additional losses
    out = m(input_argument, collect_losses=True)
    yps = [out["output"]]
    yps_weights = [1]  # everything is relative to the true output
    for k in out:
        if not k.startswith("output_to_evaluate"):
            continue
        k_weight = k.replace("output_to_evaluate", "weight_of_output_to_evaluate", 1)
        yps.append(out[k])
        yps_weights.append(out[k_weight])

    extra_loss = sum(v for k, v in out.items() if k.split(".")[0] == "loss")

    # get weights and reformat
    if "weights" in xy["outputs"]:
        weights = xy["outputs"]["weights"]
        assert weights.shape[-1] == 1
        weights = weights.squeeze(-1)
        weights = weights.to(mask.device)
    else:
        weights = torch.ones(mask.shape, dtype=torch.float32, device=mask.device)

    main_losses = [
        compute_main_loss(
            y,
            yp,
            mask,
            weights,
            evaluation_criterion=evaluation_criterion,
            criterion=criterion,
        )
        * w
        for yp, w in zip(yps, yps_weights)
    ]
    # a little hokey in order to ensure that we keep the exact same tree
    # structure when there's only one loss. Not sure if this is actually
    # necessary but might as well.
    main_loss = main_losses[0]
    for l in main_losses[1:]:
        main_loss = main_loss + l
    # add the auxiliary losses
    loss = main_loss + extra_loss

    return loss, weights.mean()


def compute_main_loss(y, yp, mask, weights, *, evaluation_criterion, criterion):
    # check that the data is on a common device
    assert yp.device == y.device == mask.device == weights.device, str(
        (yp.device, y.device, mask.device, weights.device)
    )
    # reorient the data
    y, yp, mask, weights = evaluation_criterion.reorient_data_for_classification(
        y, yp, mask, weights
    )
    # double check that the data is on a common device
    assert yp.device == y.device == mask.device == weights.device, str(
        (yp.device, y.device, mask.device, weights.device)
    )

    # weighted loss by position
    main_loss = criterion(yp, y).flatten() * weights
    # average over the positions that are not masked
    main_loss = main_loss[mask].mean() * mask.float().mean()
    return main_loss
