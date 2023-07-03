import os

import torch

from modular_splicing.legacy.remapping_pickle import load_with_remapping_pickle


def load_model(folder, step=None, **kwargs):
    """
    Load a model from the given path. Uses `load_with_remapping_pickle` to load the model
        to allow for backwards compatibility.

    Has different behavior depending on the arguments.
        If step is None, we use the last step.
        If folder is a file, we load that file.
        If folder is a directory, we load the model from that directory, with the given step.
            The pattern is {folder}/model/{step}

    Arguments:
        folder: path to the model directory or model file
        step: step to load
        **kwargs: additional arguments to pass to the torch.load function

    Returns (step, model)
        step: the step that was loaded. None if the model was not found or if folder is a file.
        model: the model that was loaded. None if the model was not found.
    """

    if not torch.cuda.is_available():
        kwargs.update(dict(map_location=torch.device("cpu")))

    if os.path.isfile(folder):
        # If folder is a file, load that file
        return None, load_with_remapping_pickle(folder, **kwargs)

    model_dir = os.path.join(folder, "model")
    if not os.path.exists(model_dir):
        # If the model directory does not exist, return None
        return None, None

    if step is None and os.listdir(model_dir):
        # If step is None, use the last step
        step = max(os.listdir(model_dir), key=int)

    path = os.path.join(model_dir, str(step))
    if not os.path.exists(path):
        # If the model file does not exist, return None
        return None, None

    # Load the model
    return int(step), load_with_remapping_pickle(path, **kwargs)


def save_model(model, folder, step):
    """
    Save model to folder, with the current step.

    Saves the model to a file named {folder}/{model}/{step}

    Parameters
    ----------
    model : torch.nn.Module
        Model to save.
    folder : str
        Folder to save model to.
    step : int
        Current step.
    """
    path = os.path.join(folder, "model", str(step))
    try:
        os.makedirs(os.path.dirname(path))
    except FileExistsError:
        pass
    torch.save(model, path)


def model_steps(folder):
    """
    Get the steps that are available for the given model.

    Parameters
    ----------
    folder : str
        Folder where the model is saved

    Returns
    -------
    list of int
    """
    return sorted([int(x) for x in os.listdir(os.path.join(folder, "model"))])
