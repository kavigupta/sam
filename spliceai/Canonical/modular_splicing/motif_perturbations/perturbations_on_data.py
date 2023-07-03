from permacache import permacache, stable_hash

import tqdm.auto as tqdm

from .compute_perturbations import motif_perturbations_individual


@permacache(
    "modular_splicing/motif_perturbations/perturbations_on_data/motif_perturbations_individual_on_data",
    key_function=dict(m=stable_hash, xs=stable_hash, ys=stable_hash, bs=None),
)
def motif_perturbations_individual_on_data(m, xs, ys, bs=100, num_output_channels=3):
    """
    Runs `motif_perturbbations_individual` on the given data. Does the clipping for you.
    """
    cl_max = xs.shape[1] - ys.shape[1]
    clip = (cl_max - m.cl) // 2
    xs = xs[:, clip : xs.shape[1] - clip]

    return [
        motif_perturbations_individual(
            m,
            x,
            y,
            threshold_info=None,
            pbar=lambda x: x,
            include_threshold=0.001,
            bs=bs,
            num_output_channels=num_output_channels,
        )
        for x, y in tqdm.tqdm(list(zip(xs, ys)))
    ]
