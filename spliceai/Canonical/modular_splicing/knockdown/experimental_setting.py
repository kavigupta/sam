import attr

from modular_splicing.utils.construct import construct


@attr.s
class ExperimentalSetting:
    feature_map = attr.ib()
    start_end = attr.ib()
    feature_powers = attr.ib()

    def query(self, probability_by_feature, combinator_spec):
        """
        Produce a single probability from the probabilities of the various features.
        """
        probs, powers = [], []
        for feature, power in self.feature_powers.items():
            probs.append(probability_by_feature[feature])
            powers.append(power)

        return construct(
            dict(multiply_with_powers=multiply_with_powers, max_for_exon=max_for_exon),
            combinator_spec,
            probs=probs,
            powers=powers,
        )

    def middle(self, names, coord):
        """
        Return the middle of the region specified by the given coordinates.
        """
        coord = coord[[names.index(x) for x in self.start_end]]
        mi, ma = coord.min(), coord.max()
        return (mi + ma) // 2

    def check(self, coord):
        """
        Check whether the given coordinates contain all necessary features.
        """
        return set(self.feature_powers) - set(coord) == set()


def multiply_with_powers(probs, powers):
    """
    Use the given powers to compute a single score for the exon.
    """
    result = 1
    for p, pw in zip(probs, powers):
        result *= p**pw
    return result


def max_for_exon(probs, powers):
    """
    Take the maximum of the probabilities. Only works on skip exons.
    """
    assert powers == [1, 1]
    return max(probs)


## The names in the following correspond to the names in the underlying data.


def a_ss(k):
    return ExperimentalSetting(
        feature_map={
            "longExonStart_0base": ("exstart", 0),
            "longExonEnd": ("exend", 0),
            "shortES": ("exstart", 1),
            "shortEE": ("exend", 1),
            "flankingES": ("exstart", 2),
            "flankingEE": ("exend", 2),
        },
        start_end=["longExonStart_0base", "longExonEnd", "shortES", "shortEE"],
        feature_powers={(k, 0): 1, (k, 1): -1},
    )


experimental_settings = {
    "SE": ExperimentalSetting(
        feature_map={
            "exonStart_0base": ("exstart", 0),
            "exonEnd": ("exend", 0),
            "upstreamES": ("exstart", 1),
            "upstreamEE": ("exend", 1),
            "downstreamES": ("exstart", 2),
            "downstreamEE": ("exend", 2),
        },
        start_end=["exonStart_0base", "exonEnd"],
        feature_powers={("3'", 0): 1, ("5'", 0): 1},
    ),
    "A3SS": a_ss("3'"),
    "A5SS": a_ss("5'"),
}
