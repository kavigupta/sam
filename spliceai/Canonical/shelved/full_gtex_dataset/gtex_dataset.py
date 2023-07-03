from abc import ABC, abstractmethod
import collections
from functools import lru_cache
import json
import os
import pickle

import numpy as np
from modular_splicing.dataset.data_rewriter import DataRewriter
from modular_splicing.dataset.h5_dataset import H5Dataset

from modular_splicing.utils.construct import construct


class GTExDataset(H5Dataset):
    def __init__(
        self, *, path, gtex_index_rewriter_spec, psi_calculation_width, **kwargs
    ):
        folder = os.path.dirname(os.path.abspath(path))
        psis_pkl = os.path.join(folder, "psis.pkl")
        tissue_names = os.path.join(folder, "tissue_names.json")
        assert os.path.exists(psis_pkl), f"{psis_pkl} does not exist"
        assert os.path.exists(tissue_names), f"{tissue_names} does not exist"
        self.psis = load_pickle_cached(psis_pkl)
        with open(tissue_names, "r") as f:
            self.tissue_names = json.load(f)
        self.psi_calculation_width = psi_calculation_width
        super().__init__(
            path=path,
            datapoint_extractor_spec=dict(
                type="BasicDatapointExtractor",
                run_argmax=False,
                rewriters=[gtex_index_rewriter_spec],
            ),
            **kwargs,
        )


class GTExIndexRewriter(DataRewriter):
    def rewrite_datapoint(self, *, el, i, j, dset):
        el = el.copy()
        el["outputs"] = el["outputs"].copy()

        el["outputs"]["y"] = self.rewrite_indices(el["outputs"]["y"], dset)
        return el

    def rewrite_indices(self, y, dset):
        params = self.compute_params(dset)

        assert len(y.shape) == 1

        result = np.zeros((y.shape[0], 3), dtype=np.float)

        result[y == 0, 0] = 1

        for idx in np.where(y > 0)[0]:
            splicepoint_idx = y[idx] - 1

            psi = dset.psis["psi_values"][splicepoint_idx]
            _, strand, _, is_end = dset.psis["all_keys"][splicepoint_idx]

            if idx < 0 or idx >= result.shape[0]:
                continue

            which_splicepoint = {
                # + strand is 5' to 3'
                ("+", 0): 2,
                ("+", 1): 1,
                # - strand is 3' to 5'
                ("-", 0): 1,
                ("-", 1): 2,
            }[(strand, is_end)]
            psi_overall = self.compute_psi(dset, params, psi)
            result[idx, which_splicepoint] = psi_overall
            result[idx, 0] = 1 - result[idx, 1] - result[idx, 2]
        return result

    @abstractmethod
    def compute_params(self, dset):
        pass

    @abstractmethod
    def compute_psi(self, dset, params, psi):
        pass


class GTExIndexRewriterAveragedPsi(GTExIndexRewriter):
    def __init__(self, *, weighted_average_spec, zero_empty_keys):
        self.weighted_average_spec = weighted_average_spec
        self.zero_empty_keys = zero_empty_keys

    def compute_params(self, dset):
        weights = construct(
            weighted_average_types(), self.weighted_average_spec, dset=dset
        )
        return dict(weights=weights)

    def compute_psi(self, dset, params, psi):
        result = sum(
            psi[k][dset.psi_calculation_width] * params["weights"][k] for k in psi
        )
        if not self.zero_empty_keys:
            result = result / sum(params["weights"][k] for k in psi)
        return result


class GTExIndexRewriterAveragedPsiSimpleMean(GTExIndexRewriter):
    def compute_params(self, dset):
        return dict()

    def compute_psi(self, dset, params, psi):
        return psi.mean()


def gtex_index_rewriter_types():
    return dict(
        GTExIndexRewriterAveragedPsi=GTExIndexRewriterAveragedPsi,
        GTExIndexRewriterAveragedPsiSimpleMean=GTExIndexRewriterAveragedPsiSimpleMean,
    )


def uniform_weighted_average(dset):
    amount = len(dset.tissue_names)
    return {tissue: 1 / amount for tissue in dset.tissue_names}


def per_tissue_weighted_average(dset):
    tissue_dict = collections.defaultdict(list)
    for tissue in dset.tissue_names:
        tissue_dict[tissue_prefix(tissue)].append(tissue)
    tissue_count = {k: len(v) for k, v in tissue_dict.items()}
    return {
        tissue: 1 / (tissue_count[tissue_prefix(tissue)] * len(tissue_count))
        for tissue in dset.tissue_names
    }


def tissue_prefix(tissue):
    tissue_prefixes = [
        "Adipose",
        "Adrenal",
        "Artery",
        "Brain",
        "Breast",
        "Cells",
        "Colon",
        "Esophagus",
        "Heart",
        "Kidney",
        "Liver",
        "Lung",
        "Minor_Salivary_Gland",
        "Muscle",
        "Nerve",
        "Ovary",
        "Pancreas",
        "Pituitary",
        "Prostate",
        "Skin",
        "Small_Intestine",
        "Spleen",
        "Stomach",
        "Testis",
        "Thyroid",
        "Uterus",
        "Vagina",
        "Whole_Blood",
    ]

    [prefix] = [prefix for prefix in tissue_prefixes if tissue.startswith(prefix)]
    return prefix


def per_tissue_type_weighted_average(dset):
    pass


def weighted_average_types():
    return dict(
        uniform_weighted_average=uniform_weighted_average,
        per_tissue_weighted_average=per_tissue_weighted_average,
    )


@lru_cache(None)
def load_pickle_cached(path):
    with open(path, "rb") as f:
        return pickle.load(f)
