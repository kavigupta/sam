from abc import ABC, abstractmethod
import attr

import numpy as np
import torch

ONLY_TRAIN_MAP = {"acceptor": 1, "donor": 2}


class EvaluationCriterion(ABC):
    @abstractmethod
    def loss_criterion(self):
        """
        Returns the loss criterion to use.
        E.g., an instance of `torch.nn.CrossEntropyLoss`.

        Always set `reduction="none"`.
        """
        pass

    @abstractmethod
    def reorient_data_for_classification(self, y, yp, mask, weights):
        """
        Reorient the data so that it can be used in computing loss.

        Some of the exact shapes depend on the underlying data source,
            those are marked with *. * Can be 0, 1, or more dimensions.

        Parameters
        ----------
        y : torch.Tensor[N, L, *]
            The true labels.
        yp : torch.Tensor[N, L, *]
            The predicted labels.
        mask : torch.Tensor[N, L]
            The mask to use.
        weights : torch.Tensor[N, L]
            The weights to use.

        Returns (y, yp, mask, weights)
        -------
        such that criterion(y, yp), mask, and weights are all of shape (NL,).
        """
        pass

    @abstractmethod
    def mask_for(self, y):
        """
        Produce the default mask for the given y values.

        Parameters
        ----------
        y : torch.Tensor[N, L, *]
            The true labels.

        Returns
        -------
        mask : torch.Tensor[N, L]
            The mask to use.
        """
        pass

    @abstractmethod
    def evaluation_channels(self, yp):
        """
        Produce the channels to use for evaluation. yp is provided for shape.

        Parameters
        ----------
        yp : torch.Tensor[N, L, *]
            The predicted labels.

        Returns
        -------
        channels : list[int]
            The channels to use in evaluation. These will be indexed into the last index of yp.
        """
        pass

    @abstractmethod
    def for_channel(self, y, c):
        """
        Produce a mask of when y matches the given channel

        Parameters
        ----------
        y : torch.Tensor[N, L, *]
            The true labels.
        c : int
            The channel to use.

        Returns
        -------
        mask : torch.Tensor[N, L]
            Whether y matches the given channel at the given position.
        """
        pass

    @abstractmethod
    def actual_eval_indices(self):
        """
        Of the evaluation channels, which ones should be thrown in the average
            to compute the final accuracy?

        These indices should be in the range [0, len(self.evaluation_channels())),
            *not* a subset of the evaluation channels, rather an index into
            the list of evaluation channels.
        """
        pass


@attr.s
class DefaultEvaluationCriterion(EvaluationCriterion):
    only_train = attr.ib(default=None, kw_only=True)

    def loss_criterion(self):
        """
        If we are only training one of the two models, then we need to
            ignore the other one.
        """
        if self.only_train is not None:
            only_train = ONLY_TRAIN_MAP[self.only_train]
            weight = np.array([1.0, 0, 0])
            weight[only_train] = 1
            return torch.nn.CrossEntropyLoss(
                torch.tensor(weight).float().cuda(), reduction="none"
            )
        else:
            return torch.nn.CrossEntropyLoss(reduction="none")

    def reorient_data_for_classification(self, y, yp, mask, weights):
        """
        Flatten out the y, yp, mask, and weights tensors, but
            ensure that yp still has its last dimension.
        """
        assert y.shape == yp.shape[:-1]
        assert len(yp.shape) == 3
        return (
            y.reshape([-1]),
            yp.reshape([-1, yp.shape[-1]]),
            mask.flatten(),
            weights.flatten(),
        )

    def mask_for(self, y):
        """
        By default, all positions are valid.
        """
        return torch.ones_like(y, dtype=np.bool)

    def evaluation_channels(self, yp):
        """
        The 0 channel isn't actually used for evaulation.
        """
        return range(1, yp.shape[-1])

    def for_channel(self, y, c):
        """
        The data is by default stored in integer mode.
        """
        return y == c

    def actual_eval_indices(self):
        """
        Both A and D are used for evaluation. only_train is potentially relevant
        here but it is never used in an adaptive sparsity context anyway.
        """
        if self.only_train is not None:
            return [ONLY_TRAIN_MAP[self.only_train] - 1]
        return [0, 1]


class MultiEvaluationCriterion(EvaluationCriterion):
    """
    Represents a situation where there are multiple channels of potential outputs.

    In this context, `y` has shape (N, L, num_predictions * num_channels_per_prediction),
        where effectively it represents several independent output variables' one hot
        encodings concatenated together.
    """

    def __init__(
        self,
        *,
        num_channels_per_prediction,
        num_predictions,
        eval_indices=[0, 1],
        only_train=None,
    ):
        assert only_train is None
        self.num_channels_per_prediction = num_channels_per_prediction
        self.num_predictions = num_predictions
        self.eval_indices = eval_indices

    def loss_criterion(self):
        return torch.nn.CrossEntropyLoss(reduction="none")

    def reorient_data_for_classification(self, y, yp, mask, weights):
        """
        We need to first ensure that y is in integer mode (it starts off in one-hot mode).

        We do this via a reshape. Additionally, we need to ensure that the mask and weights
        are properly expanded out to include the multiple predictions.

        Finally, we flatten everything out.
        """
        assert y.shape == yp.shape
        assert len(yp.shape) == 3
        assert yp.shape[-1] == self.num_predictions * self.num_channels_per_prediction

        yp = yp.reshape([-1, self.num_predictions, self.num_channels_per_prediction])
        y = y.reshape([-1, self.num_predictions, self.num_channels_per_prediction])

        y = y.argmax(axis=-1)

        mask = mask.reshape([-1])
        mask = mask[:, None].repeat(1, self.num_predictions)

        weights = weights.reshape([-1])
        weights = weights[:, None].repeat(1, self.num_predictions)

        yp = yp.reshape([-1, self.num_channels_per_prediction])
        y = y.reshape([-1])
        mask = mask.reshape([-1])
        weights = weights.reshape([-1])

        return y, yp, mask, weights

    def mask_for(self, y):
        """
        Make sure to cut off the last axis of y, since it is one-hot encoded.
        """
        return torch.ones_like(y[..., 0], dtype=np.bool)

    def evaluation_channels(self, yp):
        """
        The first channel in every case is the null channel, which is ignored. Return
            the rest.
        """
        assert yp.shape[-1] == self.num_predictions * self.num_channels_per_prediction
        channels = np.arange(yp.shape[-1])
        channels = channels.reshape(
            [self.num_predictions, self.num_channels_per_prediction]
        )
        channels = channels[:, 1:]
        return channels.flatten()

    def for_channel(self, y, c):
        """
        One hot encoding.
        """
        return y[:, c]

    def actual_eval_indices(self):
        """
        As specified in constructor
        """
        return self.eval_indices

    def __permacache_hash__(self):
        return self.__dict__


class EvaluateMultiplePredictionOutputOnSingle(EvaluationCriterion):
    """
    Evaluates multi-prediction output model on a single prediction y value.

    Returns the results as a concatenated list of results, one per channel.

    Only usable for evaluation, not training.
    """

    def __init__(
        self,
        *,
        one_hot,
        num_channels_per_prediction,
        only_train=None,
    ):
        assert only_train is None
        self.one_hot = one_hot
        self.num_channels_per_prediction = num_channels_per_prediction
        self._version = 2

    def loss_criterion(self):
        raise NotImplementedError()

    def reorient_data_for_classification(self, y, yp, mask, weights):
        raise NotImplementedError()

    def mask_for(self, y):
        """
        Make sure to cut off the last axis of y, if it is one-hot encoded.
        """
        if self.one_hot:
            return torch.ones_like(y[..., 0], dtype=np.bool)
        else:
            return torch.ones_like(y, dtype=np.bool)

    def evaluation_channels(self, yp):
        """
        The first channel in every case is the null channel, which is ignored. Return
            the rest.
        """
        channels = np.arange(yp.shape[-1])
        channels = channels.reshape([-1, self.num_channels_per_prediction])
        channels = channels[:, 1:]
        return channels.flatten()

    def for_channel(self, y, c):
        """
        Handle one-hot encoding.
        """
        if self.one_hot:
            return y[:, c % self.num_channels_per_prediction]
        else:
            return y == (c % self.num_channels_per_prediction)

    def actual_eval_indices(self):
        """
        As specified in constructor
        """
        raise NotImplementedError()

    def __permacache_hash__(self):
        return self.__dict__


def evaluation_criteria_specs():
    from modular_splicing.models.entire_model.reconstruct_sequence import (
        ReconstructSequenceEvaluationCriterion,
    )

    return dict(
        DefaultEvaluationCriterion=DefaultEvaluationCriterion,
        MultiEvaluationCriterion=MultiEvaluationCriterion,
        ReconstructSequenceEvaluationCriterion=ReconstructSequenceEvaluationCriterion,
        EvaluateMultiplePredictionOutputOnSingle=EvaluateMultiplePredictionOutputOnSingle,
    )
