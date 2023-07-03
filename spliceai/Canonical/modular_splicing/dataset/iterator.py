from .shuffler import shuffler_types
from modular_splicing.utils.construct import construct


class FastIter:
    """
    Shuffle the data chunks and then shuffle the contents of each data chunk.

    Fast, for training. Not ideal for when you want a real random sample.
    """

    def __init__(self, shuffler_spec):
        self.shuffler_spec = shuffler_spec

    def iterate(self, dset, data):
        shuffler = construct(shuffler_types(), self.shuffler_spec)
        lengths = dset.length_each(data)
        i_s = list(range(len(lengths)))
        shuffler.shuffle(i_s)
        for i in i_s:
            data_for_i = dset.data_for_chunk(data, i)
            j_s = list(range(lengths[i]))
            shuffler.shuffle(j_s)
            for j in j_s:
                yield dset.get_datapoint(data_for_i, i, j)


class FullyRandomIter:
    """
    Shuffle the data chunks and points fully randomly.
    """

    def __init__(self, shuffler_spec):
        self.shuffler_spec = shuffler_spec

    def iterate(self, dset, data):
        shuffler = construct(shuffler_types(), self.shuffler_spec)
        ijs = [(i, j) for i, l in enumerate(dset.length_each(data)) for j in range(l)]
        shuffler.shuffle(ijs)
        data = dset.cached_data(data)
        for i, j in ijs:
            yield dset.get_datapoint(data, i, j)


class SkipFirst:
    """
    Skip the first `skip_first_frac` of the data.

    Useful for evaluating on the second half of spliceai.
    """

    def __init__(self, iterator_spec, skip_first_frac):
        self.iterator = construct(iterator_types(), iterator_spec)
        self.skip_first_frac = skip_first_frac

    def iterate(self, dset, data):
        for i, el in enumerate(self.iterator.iterate(dset, data)):
            if i < int(self.skip_first_frac * len(dset)):
                continue
            yield el


def iterator_types():
    return dict(
        FastIter=FastIter,
        FullyRandomIter=FullyRandomIter,
        SkipFirst=SkipFirst,
    )
