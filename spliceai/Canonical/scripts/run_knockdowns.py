import fire
import numpy as np
import torch
import tqdm.auto as tqdm

from modular_splicing.motif_names import get_motif_names
from modular_splicing.knockdown.models import short_donor_models
from modular_splicing.knockdown.experimental_setting import experimental_settings
from modular_splicing.knockdown.compute_knockdown import (
    compute_with_and_without_knockouts,
)
from modular_splicing.knockdown.pipeline import dataset_for_cell_line

limit = None

CELL_LINES = [
    "HepG2",
    "K562",
]


def get_parameters():
    ms = short_donor_models()
    for cell_line in CELL_LINES:
        dset = dataset_for_cell_line(cell_line=cell_line)

        for motif in sorted(dset.index):
            for mod, (m, source) in ms.items():
                assert len(m) == 1
                m = m[0]
                assert isinstance(m, torch.nn.Module)
                motifs = get_motif_names(motif_names_source=source)
                if motif not in motifs:
                    continue
                for k in experimental_settings:
                    if k != "SE":
                        # FOR NOW, ONLY COMPUTE SE
                        continue
                    yield dict(
                        k=k,
                        m=m,
                        motif=motif,
                        source=source,
                        mod=mod,
                        cell_line=cell_line,
                    )


def run_for_seeds(*, split_param=1, num_splits, split):
    parameters = list(get_parameters())

    np.random.RandomState(split_param).shuffle(parameters)

    assert split in range(num_splits)
    parameters = parameters[split::num_splits]

    for param in tqdm.tqdm(parameters):
        print(f"{param['mod']} {param['motif']} {param['k']} {param['cell_line']}")
        compute_with_and_without_knockouts(*get_args_for_params(param))


def get_args_for_params(param):
    dset = dataset_for_cell_line(cell_line=param["cell_line"])
    motifs = get_motif_names(motif_names_source=param["source"])
    args = (
        experimental_settings[param["k"]],
        param["m"],
        dset.loc[param["motif"]][param["k"]][:limit],
        "datafile_train_all.h5",
        motifs.index(param["motif"]),
    )
    return args


def main(num_splits, split, split_param=1):
    run_for_seeds(num_splits=num_splits, split=split, split_param=split_param)


if __name__ == "__main__":
    fire.Fire(main)
