import itertools
import numpy as np
import pandas as pd
import tqdm.auto as tqdm

from modular_splicing.utils.io import model_steps
from modular_splicing.evaluation import standard_e2e_eval


pattern = "model/msp-266a1-{splice},{left},{right}_{seed}"
CL_VALUES = list(range(0, 1 + 20, 2))


def accuracy_results(splice, seed):
    """
    Produce accuracy results for the given splicepoint and seed.

    The result is a pandas table where you can index

    result[left][right] to get the accuracy for the given
        right and left window.
    """
    results = {left: {} for left in CL_VALUES}
    for left, right in tqdm.tqdm(list(itertools.product(CL_VALUES, CL_VALUES))):
        path = pattern.format(splice=splice, left=left, right=right, seed=seed)
        try:
            steps = model_steps(path)
        except FileNotFoundError:
            continue
        # fully trained
        if max(steps) != 1627500:
            results[left][right] = np.nan
            continue
        results[left][right] = standard_e2e_eval.evaluate_on_checkpoint(
            path=path,
            step=max(steps),
            limit=float("inf"),
            bs=128,
            pbar=tqdm.tqdm,
            data_spec=dict(
                type="H5Dataset",
                sl=5000,
                datapoint_extractor_spec=dict(type="BasicDatapointExtractor"),
                post_processor_spec=dict(type="IdentityPostProcessor"),
            ),
            force_cl=40,
        )[{"acceptor": 0, "donor": 1}[splice]]
    return pd.DataFrame(results)
