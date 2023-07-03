from collections import defaultdict
import re
import os

from permacache import permacache

import pandas as pd
import torch
from modular_splicing.utils.io import load_model, model_steps

from .utils import dropped_motifs


def motif_substitutability(path):
    accuracies = motif_change_accuracies(path)
    actions = list(accuracies)
    relative_accuracies = accuracies - accuracies.mean()
    effects = pd.DataFrame(
        {
            a: (relative_accuracies[a] - relative_accuracies[b])
            for a, b in zip(actions, actions[1:])
        }
    )
    effects = process_rows_and_columns(effects)
    return effects


def motif_change_accuracies(path):
    actions = []
    res = defaultdict(dict)
    for step, action in dropped_motifs_annotation(path):
        actions.append(action)
        info = selection_information(path, step)
        res[action] = dict(zip(info["names"], info["accuracies"]))
    accuracies = pd.DataFrame(res)
    return accuracies


def flip_if_necessary(x):
    add_or_drop, motif = re.match(r"(drop|add)\('?([^\)']+)'?\)", x).groups()
    return dict(add=1, drop=-1)[add_or_drop], motif


def process_columns(frame):
    frame = frame.copy()
    new_columns = []
    for col in frame:
        flip, new_col = flip_if_necessary(col)
        frame[col] *= flip
        new_columns.append(new_col)
    frame.columns = new_columns
    return frame


def process_rows(frame):
    frame = frame.T
    frame = process_columns(frame)
    frame = frame.T
    return frame


def process_rows_and_columns(frame):
    frame = process_columns(frame)
    frame = process_rows(frame)
    return frame


@permacache(
    "spliceai/statistics/selection_information",
    key_function=dict(path=os.path.abspath),
)
def selection_information(path, step):
    _, mod = load_model(path, step, map_location=torch.device("cpu"))
    return mod.model.selection_information


def dropped_motifs_annotation(path):
    steps = model_steps(path)
    for sprev, scurr in zip(steps, steps[1:]):
        prev, curr = dropped_motifs(path, sprev, named=True), dropped_motifs(
            path, scurr, named=True
        )
        annots = []
        annots += [f"drop({x})" for x in sorted(set(curr) - set(prev))]
        annots += [f"add({x})" for x in sorted(set(prev) - set(curr))]
        if len(annots) > 0:
            yield scurr, ";".join(annots)
