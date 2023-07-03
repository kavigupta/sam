from abc import ABC, abstractmethod
import itertools
from datetime import datetime

import numpy as np

import torch
import tqdm.auto as tqdm
from modular_splicing.evaluation.evaluation_criterion import DefaultEvaluationCriterion
from modular_splicing.evaluation.run_evaluation import accuracy_from_batches
from modular_splicing.utils.io import save_model
from modular_splicing.evaluation.run_model import run_model_on_single_batch

from modular_splicing.train.utils import get_loss


class AutoMinimizingTrainer(ABC):
    """
    Contains methods to help in the automatic minimization of a trainer
    """

    def __init__(self, architecture, *, lr, batch_size):
        self.model = architecture()
        self.lr = lr
        self.batch_size = batch_size
        self.train_data_idx = 0
        self._actual_train_data_used = 0

    @abstractmethod
    def inital_model_setup(self):
        """
        Set up the initial model
        """
        pass

    @abstractmethod
    def candidate_models(self):
        """
        Return all possible candidate continuations of the current model

        Returns (names, sub_models) where
            names is a list of strings, each one representing a name for the model
            sub_models is a list of models, each one representing a possible continuation

        Returns [], [] if we are done and there are no more possible continuations
        """
        pass

    def print_with_time(self, x):
        print(f"[{datetime.now()}] {x}")

    def start(self):
        if self._actual_train_data_used == self.train_data_idx == 0:
            self.inital_model_setup()

    def select_model(self, training_data, train_batches, test_batches):
        names, sub_models = self.candidate_models()
        if not sub_models:
            return True

        self._train_all_models(sub_models, training_data, train_batches)
        test_accuracies = self._test_all_models(
            sub_models, training_data, test_batches, on_train=True
        )
        if test_accuracies is None:
            return False
        best_idx = np.argmax(test_accuracies)
        self.print_with_time(
            f"Selecting {names[best_idx]} with train accuracy {test_accuracies[best_idx]:.2%}"
        )
        self.model = sub_models[best_idx]
        self.model.selection_information = dict(
            names=names,
            accuracies=test_accuracies,
            best_idx=best_idx,
        )
        return False

    def train_current_model(
        self, *, training_data, train_batches, testing_data, test_batches
    ):
        assert isinstance(test_batches, int), str(test_batches)
        self._train_all_models([self.model], training_data, train_batches)
        test_accuracies = self._test_all_models(
            [self.model], testing_data, test_batches, on_train=False
        )
        if test_accuracies is not None:
            self.print_with_time(f"Test accuracy {test_accuracies[0]:.2%}")
        return test_accuracies

    def save(self, path):
        if self._actual_train_data_used == self.train_data_idx:
            save_model(self, path, self.train_data_idx)

    def _train_all_models(self, sub_models, training_data, train_batches):
        if self._actual_train_data_used < self.train_data_idx:
            self._skip_data(training_data, train_batches * len(sub_models))
            assert self._actual_train_data_used <= self.train_data_idx
            return

        for m in sub_models:
            optimizer = torch.optim.Adam(m.parameters(), lr=self.lr)
            for xy in tqdm.tqdm(
                itertools.islice(training_data, train_batches),
                total=train_batches,
                desc="Training",
            ):
                self._increment_train_data_idx()
                loss, weight = get_loss(
                    m=m,
                    xy=xy,
                    evalutaion_criterion=DefaultEvaluationCriterion(),
                )
                assert weight.item() == 1.0, "does not support weighted loss"
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def _skip_data(self, training_data, train_batches):
        self._actual_train_data_used += train_batches * self.batch_size
        for _ in range(train_batches):
            next(training_data)

    def _increment_train_data_idx(self):
        self.train_data_idx += self.batch_size
        self._actual_train_data_used += self.batch_size

    def _test_all_models(self, sub_models, training_data, test_batches, *, on_train):
        if on_train and self._actual_train_data_used < self.train_data_idx:
            self._skip_data(training_data, test_batches)
            assert self._actual_train_data_used <= self.train_data_idx
            return

        test_outputs = []
        for xy in tqdm.tqdm(
            itertools.islice(training_data, test_batches),
            total=test_batches,
            desc="Testing",
        ):
            if on_train:
                self._increment_train_data_idx()
            test_outputs.append(
                [
                    run_model_on_single_batch(
                        m=m, xy=xy, evalutaion_criterion=DefaultEvaluationCriterion()
                    )
                    for m in sub_models
                ]
            )
        # test_outputs[batch_index][model_index]
        # zip(*test_outputs)[model_index][batch_index]
        test_accuracies = [
            np.mean(accuracy_from_batches(outputs))
            # outputs[batch_index]
            for outputs in zip(*test_outputs)
        ]

        return test_accuracies
