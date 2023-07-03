from abc import ABC, abstractmethod

import attr
from permacache import permacache, stable_hash
from cached_property import cached_property

from modular_splicing.utils.arrays import run_length_encoding

from .flatten_distance_distributions import flatten_hsmm
from .shorten_distance_distributions import shorten_states
from .substring import substring_hsmm


class TransformedModel(ABC):
    @property
    @abstractmethod
    def chunks(self):
        pass

    @property
    @abstractmethod
    def underlying_hsmm(self):
        pass

    @abstractmethod
    def forward_process_rle(self, values, lengths):
        pass

    def score_sequence(self, a):
        return run_transformed_model(self, a)

    @cached_property
    def stable_hash(self):
        return stable_hash(self)


@permacache(
    "hsmm/transformed_models/run_transformed_model",
    key_function=dict(tm=lambda x: x.stable_hash, a=stable_hash),
)
def run_transformed_model(tm, a):
    if tm.chunks:
        chunks = [a[i : i + tm.chunks] for i in range(0, a.size, tm.chunks)]
    else:
        chunks = [a]
    result = 0
    for chunk in chunks:
        lengths, _, values = run_length_encoding(chunk)
        values, lengths = tm.forward_process_rle(values, lengths)
        result += tm.underlying_hsmm.score_rle_sequence(lengths, values)
    return result


@attr.s
class OriginalModel(TransformedModel):
    hsmm = attr.ib()

    @property
    def chunks(self):
        return False

    @property
    def underlying_hsmm(self):
        return self.hsmm

    def forward_process_rle(self, values, lengths):
        values = [self.hsmm.states[v] for v in values]
        return values, lengths


@attr.s
class ClippedModel(TransformedModel):
    hsmm = attr.ib()
    prev_transformed_model = attr.ib()
    split_lengths = attr.ib()

    @classmethod
    def of(cls, transformed_model):
        hsmm, split_lengths = transformed_model.underlying_hsmm.clip_distances()
        return cls(
            hsmm=hsmm,
            prev_transformed_model=transformed_model,
            split_lengths=split_lengths,
        )

    @property
    def chunks(self):
        return self.prev_transformed_model.chunks

    @property
    def underlying_hsmm(self):
        return self.hsmm

    def forward_process_rle(self, values, lengths):
        values, lengths = self.prev_transformed_model.forward_process_rle(
            values, lengths
        )
        n_values, n_lengths = [], []
        for v, l in zip(values, lengths):
            if v in self.split_lengths:
                l_to_use = min(self.split_lengths[v], l - 1)
                n_values += [("prefix", v), ("suffix", v)]
                n_lengths += [l_to_use, l - l_to_use]
            else:
                n_values.append(v)
                n_lengths.append(l)
        return n_values, n_lengths


@attr.s
class ShortenModel(TransformedModel):
    hsmm = attr.ib()
    prev_transformed_model = attr.ib()
    bms = attr.ib()

    @classmethod
    def of(cls, transformed_model, ell, count, K=10):
        hsmm, bms_original = shorten_states(
            transformed_model.underlying_hsmm, ell, count, K=K
        )
        return cls(
            hsmm=hsmm,
            prev_transformed_model=transformed_model,
            bms=bms_original,
        )

    @property
    def chunks(self):
        return self.prev_transformed_model.chunks

    @property
    def underlying_hsmm(self):
        return self.hsmm

    def forward_process_rle(self, values, lengths):
        values, lengths = self.prev_transformed_model.forward_process_rle(
            values, lengths
        )
        results = []
        new_lengths = []
        for v, l in zip(values, lengths):
            if v in self.bms:
                results.extend([("geom", self.bms[v](l), v)] * l)
                new_lengths.extend([1] * l)
            else:
                results.append(v)
                new_lengths.append(l)
        return results, new_lengths


@attr.s
class SubstringModel(TransformedModel):
    hsmm = attr.ib()
    prev_transformed_model = attr.ib()
    ell = attr.ib()

    @classmethod
    def of(cls, hsmm, initial_distribution, ell):
        new_hsmm = substring_hsmm(hsmm.underlying_hsmm, initial_distribution, ell)
        return cls(new_hsmm, hsmm, ell)

    @property
    def chunks(self):
        return self.ell

    @property
    def underlying_hsmm(self):
        return self.hsmm

    def forward_process_rle(self, values, lengths):
        values, lengths = self.prev_transformed_model.forward_process_rle(
            values, lengths
        )
        if len(values) == 1:
            [f] = values
            return [("F", f)] * lengths[0], [1] * lengths[0]
        a, *bs, c = values
        return [("A", a)] + [("B", b) for b in bs] + [("C", c)], lengths


@attr.s
class FlattenedModel(TransformedModel):
    hsmm = attr.ib()
    split = attr.ib()
    prev_transformed_model = attr.ib()

    @classmethod
    def of(cls, hsmm):
        split, new_hsmm = flatten_hsmm(hsmm.hsmm)
        return cls(new_hsmm, split, hsmm)

    @property
    def chunks(self):
        return self.prev_transformed_model.chunks

    @property
    def underlying_hsmm(self):
        return self.hsmm

    def forward_process_rle(self, values, lengths):
        values, lengths = self.prev_transformed_model.forward_process_rle(
            values, lengths
        )
        new_values, new_lengths = [], []
        for value, length in zip(values, lengths):
            if value in self.split:
                new_values.extend([("flattened", i, value) for i in range(length)])
                new_lengths.extend([1] * length)
                continue
            assert length == 1
            new_values.append(value)
            new_lengths.append(length)
        return new_values, new_lengths


def unwrap_state(s):
    if isinstance(s, tuple):
        return unwrap_state(s[-1])
    return s
