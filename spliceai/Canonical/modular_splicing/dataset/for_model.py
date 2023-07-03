def reformat_argument_for_model(m, xy, evaluation_criterion):
    """
    Produce the reformatted argument for the model, given the original argument.

    This handles data in batch form, so it has shape (N, L, ...).

    Also places the arguments in the correct device.

    If the argument is a dictionary, the input will be unchanged, and the output will be
        key ["outputs"]["y"]. The mask value will be computed as ["outputs"]["mask"]
        if it exists, otherwise by `evaluation_criterion.mask_for`.

    If the argument is a tuple, it will be interpreted as x, y, motifs, where
        motifs is optional. Motifs will be passed to the input.


    Arguments:
        m {torch.nn.Module} -- The model
        xy {tuple or dict} -- The input to the model
        evaluation_criterion {EvaluationCriterion} -- The evaluation criterion

    Returns a tuple (input_argument, output, mask)
    """

    device = next(m.parameters()).device

    if isinstance(xy, dict):
        input_argument = xy["inputs"]
        input_argument = {k: v.to(device) for k, v in input_argument.items()}
        y = xy["outputs"]["y"].to(device)
        return (
            input_argument,
            y,
            xy["outputs"].get("mask", evaluation_criterion.mask_for(y)).to(y.device),
        )

    x, y, *_ = xy
    x = x.to(device)
    y = y.to(device)

    input_argument = x
    if getattr(m, "_input_dictionary", False):
        input_argument = dict(x=x)
        if len(xy) > 2:
            input_argument["motifs"] = xy[2].to(device)
    return input_argument, y, evaluation_criterion.mask_for(y)
