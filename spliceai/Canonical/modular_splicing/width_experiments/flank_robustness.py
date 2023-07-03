"""
Contains methods for the computation of flank robustness of motifs, effectively
to what extent given motifs can have their flanks clipped.
"""

from collections import Counter
import itertools

from more_itertools import chunked

from permacache import permacache
import numpy as np
import pandas as pd
import torch
import tqdm.auto as tqdm


from modular_splicing.dataset.basic_dataset import basic_dataset
from modular_splicing.motif_names import get_motif_names

from modular_splicing.models_for_testing.list import AM
from modular_splicing.legacy.hash_model import hash_model

from ..utils.sequence_utils import (
    map_to_all_seqs_idx,
    slice_packed_seq,
)
from modular_splicing.utils.construct import construct


def data_from_genome(data_path, sl, amount, *, random=False):
    """
    Get data from the genome, using the given data path, and the given amount of sequences.
    """
    # cl=0 to avoid overlaps
    dset = basic_dataset(
        data_path,
        cl=0,
        cl_max=10_000,
        sl=sl,
        iterator_spec=dict(
            type="FullyRandomIter", shuffler_spec=dict(type="SeededShuffler", seed=0)
        )
        if random
        else dict(type="FastIter", shuffler_spec=dict(type="DoNotShuffle")),
    )
    dset = (x for x, _ in itertools.islice(dset, amount))
    return dset, amount


def data_from_random(seed, sl, amount):
    """
    Generate data purely randomly.
    """
    rng = np.random.RandomState(seed)
    dset = (np.eye(4, dtype=np.float32)[rng.choice(4, size=sl)] for _ in range(amount))
    return dset, amount


@permacache(
    "validation/motif_pattern/compute_binding_sites_for_model_1",
    key_function=dict(mod=hash_model, bs=None),
)
def compute_binding_sites_for_model(
    mod, data_spec, *, num_motifs, motif_true_width, bs=16
):
    """
    Get all the binding sites of all motifs in the given model. The binding sites are in the form
        bit-packed sequences, with two bits per base, stored in the format of map_to_all_seqs_idx.

    The binding site is at position motif_true_width // 2, which is the center.
    """
    assert motif_true_width % 2 == 1
    motif_radius = (motif_true_width - 1) // 2

    data, amount = construct(
        dict(genome=data_from_genome, random=data_from_random), data_spec
    )
    bound_sites = [[] for _ in range(num_motifs)]
    for xs in chunked(tqdm.tqdm(data, total=amount), bs):
        xs = np.array(xs)
        xs = torch.tensor(xs).cuda()
        valid_xs = xs.sum(-1) > 0
        valid_xs = torch.stack(
            [valid_xs[:, i : -(motif_radius * 2 - i)] for i in range(motif_radius * 2)]
        ).all(0)[:, :, None]
        seq_ids = map_to_all_seqs_idx(xs.argmax(-1), motif_true_width)[
            :, motif_radius:-motif_radius
        ]
        if mod is not None:
            with torch.no_grad():
                valid_xs = (
                    valid_xs
                    & (
                        mod(xs, only_motifs=True)["post_sparse_motifs_only"][
                            :, motif_radius:-motif_radius
                        ]
                        > 0
                    )[:, :, :num_motifs]
                )
        assert valid_xs.shape[-1] == num_motifs
        for motif_id in range(num_motifs):
            bound_sites[motif_id].append(
                seq_ids[valid_xs[:, :, motif_id]].cpu().numpy()
            )
    bound_sites = [np.concatenate(x) for x in bound_sites]
    return bound_sites


def train_shrunken_motif_models(
    mod, data_spec_train, *, motif_true_width, num_motifs, a, b
):
    """
    Train a shrunken motif model for each motif in the current model on the given training data set.

    The shrunken motif model is a model that tries to match the binding sites of the model, but not
    using the left a and right b bases. It does so while mantaining unbiasedness by being probabilistic
    and using the fraction of sites that are active that contain the shrunken sequence of bases.

    We return a dictionary with keys
        models: a list of shrunken motif models. Each one consists of a dictionary from packed sequence
            (of length num_motifs - a - b) to the probability of that sequence being active. 0-probability
            sequences are not included.
        domain: the set of all possible sequences that we considered. Any sequence that is not in this
            set is one we have no information on.
    """
    print("training model", hash_model(mod), "on", data_spec_train, a, b)
    bound_sites = compute_binding_sites_for_model(
        mod, data_spec_train, motif_true_width=motif_true_width, num_motifs=num_motifs
    )
    all_sites = compute_binding_sites_for_model(
        None, data_spec_train, motif_true_width=motif_true_width, num_motifs=1
    )[0]
    bound_sites_shrunk = [
        Counter(slice_packed_seq(bs, a, motif_true_width - b))
        for bs in tqdm.tqdm(bound_sites, desc="shrinking")
    ]
    all_sites_shrunk = Counter(slice_packed_seq(all_sites, a, motif_true_width - b))
    bound_sites_shrunk = [
        {k: v / all_sites_shrunk[k] for k, v in bs.items()} for bs in bound_sites_shrunk
    ]
    return dict(models=bound_sites_shrunk, domain=set(all_sites_shrunk))


@permacache(
    "validation/motif_pattern/compute_robustness_to_flank_shrinkage_2",
    key_function=dict(mod=hash_model),
    multiprocess_safe=True,
)
def compute_robustness_to_flank_shrinkage(
    mod, data_spec_train, data_spec_val, *, motif_true_width, num_motifs, a, b
):
    """
    Produces robustness estimates given the given training and validation dataset.
    For each motif, returns two values
        acc: the estimated top-k accuracy, ignoring any binding sites whose shrunken sequence is
            not in the domain of the trained shrunken motif model.
        bad_sites_frac: the fraction of binding sites whose shrunken sequence is not in the domain
    """
    model = train_shrunken_motif_models(
        mod,
        data_spec_train,
        motif_true_width=motif_true_width,
        num_motifs=num_motifs,
        a=a,
        b=b,
    )
    sites_to_test = compute_binding_sites_for_model(
        mod, data_spec_val, motif_true_width=motif_true_width, num_motifs=num_motifs
    )
    sites_to_test = [
        Counter(slice_packed_seq(bs, a, motif_true_width - b)) for bs in sites_to_test
    ]
    results = []
    for motif_id in range(num_motifs):
        m = model["models"][motif_id]
        sites = sites_to_test[motif_id]
        relevant_sites = [x for x in sites if x in model["domain"]]
        bad_sites_frac = (
            (len(sites) - len(relevant_sites)) / len(sites) if len(sites) else np.nan
        )
        acc = np.mean([m.get(s, 0) for s in relevant_sites])
        results.append(dict(bad_sites_frac=bad_sites_frac, acc=acc))
    return results


def compute_max_robustness_to_flank_shrinkage(
    mod, data_spec_train, data_spec_val, *, motif_true_width, num_motifs, core_size
):
    """
    Produce the maximum robustness estimate for the given model on the given dataset.
    Takes all possible combinations of a and b that mantain the core size.
    """
    assert core_size <= motif_true_width
    total_flank_size = motif_true_width - core_size

    by_combination = [
        compute_robustness_to_flank_shrinkage(
            mod,
            data_spec_train,
            data_spec_val,
            motif_true_width=motif_true_width,
            num_motifs=num_motifs,
            a=a,
            b=total_flank_size - a,
        )
        for a in range(total_flank_size + 1)
    ]
    return [
        max(x, key=lambda x: x["acc"] - x["bad_sites_frac"])
        for x in list(zip(*by_combination))
    ]


def plausible_range(res):
    good_sites = 1 - res["bad_sites_frac"]
    acc = res["acc"]
    return acc, acc * good_sites, acc * good_sites + (1 - good_sites)


def models_to_analyze():
    mods = [AM.non_binarized_model(seed) for seed in (1, 2, 3)]
    models = {mod.name: mod.model for mod in mods}
    return models


def compute_all_flank_robustnesses(models, data, widths):
    result = {
        mod: pd.DataFrame(
            {
                width: [
                    np.array(plausible_range(res))
                    for res in compute_max_robustness_to_flank_shrinkage(
                        models[mod],
                        **data,
                        motif_true_width=21,
                        num_motifs=79,
                        core_size=width,
                    )
                ]
                for width in widths
            },
            index=get_motif_names("rbns"),
        )
        for mod in models
    }
    result["mean"] = sum(result.values()) / len(result)
    return result


genome_data = dict(
    data_spec_train=dict(
        type="genome", data_path="dataset_train_all.h5", amount=20_000, sl=5000
    ),
    data_spec_val=dict(
        type="genome", data_path="dataset_test_0.h5", amount=1000, sl=5000
    ),
)
random_data = dict(
    data_spec_train=dict(type="random", seed=0, amount=20_000, sl=5000),
    data_spec_val=dict(type="random", seed=1, amount=1000, sl=5000),
)
