import attr


@attr.s
class AdaptiveSparsityThresholdManager:
    """
    Manages the sparsity threshold of a model.

    The threshold is decreased by a fixed amount every epoch, but never below the minimal threshold

    The threshold is increased to the accuracy of the model if the accuracy is higher than the current threshold,
    but never above the maximal threshold.
    """

    maximal_threshold = attr.ib()
    minimal_threshold = attr.ib()
    decrease_per_epoch = attr.ib()

    previous_epoch = attr.ib()
    current_threshold = attr.ib()

    @classmethod
    def setup(cls, m, *, maximal_threshold, minimal_threshold, decrease_per_epoch):
        """
        Sets up the sparsity threshold manager for a model.

        If the model already has a sparsity threshold manager, it is returned.

        Should only be called on step 0 of the training loop, as it assumes that the previous epoch is 0

        :param m: The model
        :param maximal_threshold: The maximal threshold. If this is set to None, the maximal threshold is set to the
            minimal threshold, and the decay does not happen. The current threshold is also set to this value.
        :param minimal_threshold: The minimal threshold. Must not be None.
        :param decrease_per_epoch: The amount by which the threshold is decreased every epoch.
        :return: The sparsity threshold manager
        """
        minimal_threshold = float(minimal_threshold)
        decrease_per_epoch = float(decrease_per_epoch)
        if maximal_threshold is None:
            maximal_threshold = minimal_threshold
        if not hasattr(m, "_adaptive_sparsity_threshold_manager"):
            m._adaptive_sparsity_threshold_manager = cls(
                maximal_threshold,
                minimal_threshold,
                decrease_per_epoch,
                0,
                maximal_threshold,
            )
        return m._adaptive_sparsity_threshold_manager

    def passes_accuracy_threshold(self, acc, epoch):
        """
        Updates the threshold as specified above.

        See the test for examples.
        """
        new_thresh = self.current_threshold
        new_thresh -= (epoch - self.previous_epoch) * self.decrease_per_epoch
        new_thresh = max(new_thresh, acc)
        new_thresh = max(new_thresh, self.minimal_threshold)
        new_thresh = min(new_thresh, self.maximal_threshold)
        print(f"Threshold: {new_thresh}")
        self.previous_epoch = epoch
        self.current_threshold = new_thresh

        return acc >= self.current_threshold
