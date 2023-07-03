import os
import sys
import numpy as np
import torch

import tqdm

from modular_splicing.models_for_testing.load_model_for_testing import (
    achieved_target_acc,
)
from modular_splicing.utils.io import model_steps

sys.path.insert(0, ".")


def delete_additional_checkpoints(path, keep_epoch_steps=True):
    if not path.startswith("model/"):
        path = f"model/{path}"

    steps = model_steps(path)
    try:
        target_acc_steps = set(achieved_target_acc(path, pbar=tqdm.tqdm)[0])
    except AttributeError:
        target_acc_steps = set()

    epoch_steps = set(
        np.array(steps)[
            np.where(np.array(steps)[1:] - np.array(steps)[:-1] == steps[0])
        ].tolist()
    )
    keep_steps = target_acc_steps | set(steps[-5:]) | set(steps[:5])
    if keep_epoch_steps and len(target_acc_steps) < 5:
        keep_steps = keep_steps | epoch_steps

    for i in range(len(steps)):
        before_and_after = set(steps[max(i - 1, 0) : i + 2])
        if before_and_after & keep_steps:
            print(steps[i], "kept")
        else:
            print(steps[i], "removed")
            os.remove(os.path.join(f"{path}/model", str(steps[i])))


def main():
    for argument in sys.argv[1:]:
        print(argument)
        delete_additional_checkpoints(argument, keep_epoch_steps=True)


if __name__ == "__main__":
    main()
