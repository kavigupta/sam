from modular_splicing.utils.construct import construct

import numpy as np


class InSilicoDeltaPsiAnnotator:
    """
    Annotates the results of a knockdown analysis with a delta psi value.
    """

    def __init__(self, *, filter_spec=dict(type="DoNotFilter"), agg_spec):
        self.filter = construct(
            dict(DoNotFilter=DoNotFilter),
            filter_spec,
        )
        self.agg = construct(
            dict(StandardAggregator=StandardAggregator),
            agg_spec,
        )

    def annotate(self, setting, results):
        """
        Returns both the statistics and mask on the results.
        """
        return dict(
            stat=self.agg(setting, results),
            mask=self.filter(setting, results),
        )


class StandardAggregator:
    """
    Aggregator that computes the overall delta psi from the results of a knockdown analysis,
        using the given query combinator and difference function.

    Args:
        query_combinator_spec: the specification for a query combinator.
            This combines the probabilities of the various features in the
            experimental setting to produce a single probability.
            See `ExperimentalSetting.query` for more information.

        difference_spec: the specification for a difference function.
            Either "log_ratio" or "difference".
    """

    def __init__(self, *, query_combinator_spec, difference_spec):
        self.query_combinator_spec = query_combinator_spec
        self.difference_spec = difference_spec

    def __call__(self, setting, results):
        """
        Returns the overall delta psi from each result.

        Takes the mean across different models in the ensemble.
        """
        pred = [result["pred"] for result in results]
        perturbed_pred = [result["perturbed_pred"] for result in results]

        pred, perturbed_pred = mean_dicts(pred), mean_dicts(perturbed_pred)

        return np.array(
            [
                self.difference(
                    setting.query(x, self.query_combinator_spec),
                    setting.query(y, self.query_combinator_spec),
                )
                for x, y in zip(perturbed_pred, pred)
            ]
        )

    def difference(self, x, y):
        return construct(
            dict(log_ratio=lambda x, y: np.log(x / y), difference=lambda x, y: x - y),
            self.difference_spec,
            x=x,
            y=y,
        )


class DoNotFilter:
    """
    Allows all results
    """

    def __call__(self, setting, results):
        [length] = {len(result["pred"]) for result in results}
        return np.ones(length, dtype=bool)


def mean_dicts(preds):
    """
    Take the mean of several prediction values.
    """
    preds = list(zip(*preds))
    results = []
    for pred in preds:
        ks = pred[0].keys()
        assert [x.keys() == ks for x in pred]
        results.append({k: np.mean([x[k] for x in pred]) for k in ks})
    return results


def annotator_types():
    return dict(
        InSilicoDeltaPsiAnnotator=InSilicoDeltaPsiAnnotator,
    )
