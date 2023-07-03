import itertools
from types import SimpleNamespace

import numpy as np
from permacache import permacache, drop_if_equal
import tqdm.auto as tqdm

from modular_splicing.dataset.h5_dataset import H5Dataset


@permacache("eclip/count_eclip_by_intron_or_exon_2")
def count_eclip_by_intron_or_exon(
    path, eclip_params, amount, max_dist_from_splice=float("inf")
):
    dset = H5Dataset(
        path=path,
        cl=0,
        sl=5000,
        cl_max=10_000,
        iterator_spec=dict(
            type="FullyRandomIter", shuffler_spec=dict(type="SeededShuffler", seed=0)
        ),
        datapoint_extractor_spec=dict(
            type="BasicDatapointExtractor",
            rewriters=[
                dict(
                    type="AdditionalChannelDataRewriter",
                    out_channel=["inputs", "motifs"],
                    data_provider_spec=dict(type="eclip", params=eclip_params),
                )
            ],
        ),
        post_processor_spec=dict(
            type="FlattenerPostProcessor",
            indices=[("inputs", "motifs"), ("outputs", "y")],
        ),
    )
    in_exon, total_exon, in_intron, total_intron = 0, 0, 0, 0
    for m, y in tqdm.tqdm(itertools.islice(dset, amount), total=amount):
        m = m.astype(np.uint64)
        is_intron = np.zeros(y.shape, dtype=np.bool)
        is_exon = np.zeros(y.shape, dtype=np.bool)
        [positions] = np.where(y)
        ad = y[positions]
        for i in range(positions.shape[0] - 1):
            signature = (ad[i], ad[i + 1])
            region = list(range(positions[i] + 1, positions[i + 1]))
            region = [
                x
                for x in region
                if min(abs(x - positions[i]), abs(x - positions[i + 1]))
                < max_dist_from_splice
            ]
            if signature == (1, 2):
                is_exon[region] = 1
            elif signature == (2, 1):
                is_intron[region] = 1
            else:
                1 / 0
        in_exon += m[is_exon].sum(0)
        in_intron += m[is_intron].sum(0)
        total_exon += is_exon.sum()
        total_intron += is_intron.sum()
    return SimpleNamespace(
        in_exon=in_exon,
        in_intron=in_intron,
        total_exon=total_exon,
        total_intron=total_intron,
    )


@permacache(
    "eclip/count_eclips_across_and_internal_exon_4",
    key_function=dict(use_introns=drop_if_equal(False)),
)
def count_eclips_across_and_internal_exon(
    path,
    eclip_params,
    amount,
    use_introns=False,
    position_count_distance_limit=float("inf"),
):
    dset = H5Dataset(
        path=path,
        cl=0,
        cl_max=10_000,
        sl=5000,
        iterator_spec=dict(
            type="FullyRandomIter", shuffler_spec=dict(type="SeededShuffler", seed=0)
        ),
        datapoint_extractor_spec=dict(
            type="BasicDatapointExtractor",
            rewriters=[
                dict(
                    type="AdditionalChannelDataRewriter",
                    out_channel=["inputs", "motifs"],
                    data_provider_spec=dict(type="eclip", params=eclip_params),
                )
            ],
        ),
        post_processor_spec=dict(
            type="FlattenerPostProcessor",
            indices=[("inputs", "motifs"), ("outputs", "y")],
        ),
    )
    across = 0
    internal = 0
    across_position_count = 0
    internal_position_count = 0
    for m, y in tqdm.tqdm(itertools.islice(dset, amount), total=amount):
        if use_introns:
            y[y != 0] = 3 - y[y != 0]  # flip 1 and 2

        across += np.zeros(m.shape[1], dtype=np.long)
        internal += np.zeros(m.shape[1], dtype=np.long)
        across_position_count += np.zeros(m.shape[1], dtype=np.long)
        internal_position_count += np.zeros(m.shape[1], dtype=np.long)
        [positions] = np.where(y)
        if positions.size == 0:
            continue
        if y[positions[0]] == 2:
            positions = positions[1:]
        if positions.size % 2 == 1:
            positions = positions[:-1]
        for acc, don in positions.reshape(-1, 2):
            mexon = m[acc : don + 1]
            for mot in range(mexon.shape[1]):
                for i in range(mexon.shape[2]):
                    idxs, pattern = np.where(mexon[:, mot, i])
                    if pattern.size == 0:
                        continue
                    if pattern[0] == 1:
                        across[mot] += 1
                        across_position_count[mot] += min(
                            idxs[0], position_count_distance_limit
                        )
                        pattern = pattern[1:]
                        idxs = idxs[1:]
                    if pattern.size == 0:
                        continue
                    if pattern[-1] == 0:
                        across[mot] += 1
                        across_position_count[mot] += min(
                            mexon.shape[0] - idxs[-1], position_count_distance_limit
                        )
                        pattern = pattern[:-1]
                        idxs = idxs[:-1]
                    if pattern.size == 0:
                        continue
                    for u in range(idxs.shape[0] // 2):
                        left, right = idxs[u], idxs[u + 1]
                        total = 0
                        total += min(right, position_count_distance_limit) - min(
                            left, position_count_distance_limit
                        )
                        total += min(
                            mexon.shape[0] - left, position_count_distance_limit
                        ) - min(mexon.shape[0] - right, position_count_distance_limit)
                        internal_position_count[mot] += total
                    internal[mot] += pattern.size // 2
    return SimpleNamespace(
        across=across,
        internal=internal,
        across_position_count=across_position_count,
        internal_position_count=internal_position_count,
    )
