from functools import lru_cache
import os
from types import SimpleNamespace
import attr

import numpy as np
import h5py
import tqdm.auto as tqdm
from permacache import permacache

from .gene_dataset import create_gene_dataset
from .hsmm_model import HSMM
from .substring import substring_hsmm
from .fill_gaps import fill_gaps
from .transformed_models import (
    ClippedModel,
    FlattenedModel,
    OriginalModel,
    ShortenModel,
)
from modular_splicing.utils.arrays import run_length_encoding


STATES = sorted(["ES", "EF", "EM", "EL", "5'", "3'", "I", "done"])


@attr.s
class KGMConfiguration:
    datafile_path = attr.ib()
    tune_indices = attr.ib()
    test_indices = attr.ib()
    ell = attr.ib()

    @classmethod
    def standard(cls):
        datafile_path = "datafile_train_all.h5"
        with h5py.File(create_gene_dataset(datafile_path), "r") as f:
            all_indices = range(len(f) // 2)
        tune_indices = [x for x in all_indices if x % 2 == 0]
        test_indices = [x for x in all_indices if x % 2 == 1]
        return cls(
            datafile_path=datafile_path,
            tune_indices=tune_indices,
            test_indices=test_indices,
            ell=10_000,
        )

    def iterate_annotations(self, use, limit=None):
        for x, y in self._iterate_raw(use, limit=limit):
            yield annotate_y(y[:])

    def iterate_data(self, use, limit=None):
        for x, y in self._iterate_raw(use, limit=limit):
            yield x[:], y[:], annotate_y(y[:])

    def _iterate_raw(self, use, limit=None):
        with h5py.File(create_gene_dataset(self.datafile_path), "r") as f:
            indices = getattr(self, use)
            for i in tqdm.tqdm(indices[:limit]):
                yield f[f"x{i}"], f[f"y{i}"]

    @property
    def initial_distribution(self):
        return substring_initial_distribution(
            self.datafile_path, tune_indices=self.tune_indices, ell=self.ell
        )


@lru_cache(None)
def original_model(do_fill=True):
    folder = "../data/scfg-params"
    params = SimpleNamespace(
        **{
            f.split(".")[0]: np.load(os.path.join(folder, f))
            for f in os.listdir(folder)
        }
    )
    hsmm = HSMM(
        initial={
            "ES": params.p1E.item(),
            "EF": 1 - params.p1E.item(),
            "EM": 0,
            "EL": 0,
            "5'": 0,
            "3'": 0,
            "I": 0,
            "done": 0,
        },
        distance_distributions={
            "ES": params.lengthSingleExons,
            "EF": params.lengthFirstExons,
            "EM": params.lengthMiddleExons,
            "EL": params.lengthLastExons,
            "5'": [1],
            "3'": [1],
            "I": params.lengthIntrons,
            "done": [1],
        },
        transition_distributions={
            "ES": {"done": 1},
            "EF": {"5'": 1},
            "EM": {"5'": 1},
            "EL": {"done": 1},
            "5'": {"I": 1},
            "3'": {"EM": 1 - params.pEO.item(), "EL": params.pEO.item()},
            "I": {"3'": 1},
            "done": {"done": 1},
        },
    )
    if do_fill:
        hsmm, _ = fill_gaps(hsmm)
    return OriginalModel(hsmm)


@lru_cache(None)
def clipped_model():
    return ClippedModel.of(original_model())


@lru_cache(None)
def geometricized_model(k):
    return ShortenModel.of(clipped_model(), ell=1000, count=5, K=k)


@lru_cache(None)
def geometricized_and_flattened_model(k):
    return FlattenedModel.of(geometricized_model(k))


def all_transformed_hsmm():
    return dict(
        original=original_model,
        clipped=clipped_model,
        geometricized=geometricized_model,
        geometricized_and_flattened=geometricized_and_flattened_model,
    )


def annotate_y(y):
    positions, which = np.where(y[:, 1:])
    # check donor/acceptor/donor/acceptor...
    if not is_valid(which):
        return np.zeros(y.shape[0], dtype=np.uint8)

    annotated = np.zeros(y.shape[0], dtype=np.int8)

    if positions.shape[0] == 0:
        annotated = STATES.index("ES")
        return annotated

    annotated[positions[::2]] = STATES.index("5'")
    annotated[positions[1::2]] = STATES.index("3'")
    annotated[: positions[0]] = STATES.index("EF")
    annotated[positions[-1] + 1 :] = STATES.index("EL")
    for intron_start, intron_end in zip(positions[::2], positions[1::2]):
        annotated[intron_start + 1 : intron_end] = STATES.index("I")
    for intron_start, intron_end in zip(positions[1::2], positions[2::2]):
        annotated[intron_start + 1 : intron_end] = STATES.index("EM")
    return annotated


def is_valid(which):
    return (which == (np.arange(which.shape[0]) + 1) % 2).all() and which.shape[
        0
    ] % 2 == 0


@permacache(
    "hsmm/kayla_gene_model/subset_initial_distribution",
    key_function=dict(datafile_path=os.path.abspath),
)
def substring_initial_distribution(datafile_path, *, tune_indices, ell):
    with h5py.File(create_gene_dataset(datafile_path), "r") as z:
        pa = np.zeros((len(STATES)), dtype=np.int)
        pf = np.zeros((len(STATES)), dtype=np.int)
        for i in tqdm.tqdm(tune_indices):
            rlengths, _, values = run_length_encoding(annotate_y(z[f"y{i}"][:]))
            np.add.at(pa, values[rlengths <= ell], 1)
            np.add.at(pf, values[rlengths > ell], 1)
    count = pa.sum() + pf.sum()
    pa = pa / count
    pf = pf / count
    new_pi = {}
    new_pi.update({("A", STATES[i]): pa[i] for i in range(len(STATES))})
    new_pi.update({("F", STATES[i]): pf[i] for i in range(len(STATES))})

    return {k: v for k, v in new_pi.items() if v != 0}
