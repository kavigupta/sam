from abc import ABC, abstractmethod
import attr

import torch


class LocalSplicepointModelResidualPropagator(ABC):
    """
    Represents a technique for propagating the LSSI residual through a model.
    """

    @abstractmethod
    def propagate_residuals(self, output, splicepoint_results_residual):
        """
        Propagate the residuals to the output.

        Input:
            output: (N, L, C)
            splicepoint_results_residual: (N, L, 2)
        Output:
            output: (N, L, C)
        """


def lsmrp_types():
    return dict(
        EarlyChannelLSMRP=EarlyChannelLSMRP,
        MultiOutputLSMRP=MultiOutputLSMRP,
        NoSplicepointLSMRP=NoSplicepointLSMRP,
        DoNotPropagateLSMRP=DoNotPropagateLSMRP,
    )


@attr.s
class EarlyChannelLSMRP(LocalSplicepointModelResidualPropagator):
    """
    Propagates the residuals to the early few channels of the model.

    Does not touch later channels. Uses the sum of the log softmaxes to do a masking in the non-log space.
    """

    def propagate_residuals(self, output, splicepoint_results_residual):
        output, extra_outputs = output[:, :, :3], output[:, :, 3:]

        output = output.log_softmax(-1) + splicepoint_results_residual.log_softmax(-1)

        output = torch.cat([output, extra_outputs], dim=-1)
        return output


@attr.s
class MultiOutputLSMRP(LocalSplicepointModelResidualPropagator):
    """
    Treats the model as producing several different outputs, and propagates the residuals to each of them.
    """

    def propagate_residuals(self, output, splicepoint_results_residual):
        num_channels_each = splicepoint_results_residual.shape[-1]

        chunks = [
            output[..., i * num_channels_each : (i + 1) * num_channels_each]
            for i in range(output.shape[-1] // num_channels_each)
        ]

        chunks = [
            chunk.log_softmax(-1) + splicepoint_results_residual.log_softmax(-1)
            for chunk in chunks
        ]

        output = torch.cat(chunks, dim=-1)

        return output


@attr.s
class NoSplicepointLSMRP(LocalSplicepointModelResidualPropagator):
    """
    Ensure that the splicepoint model does not produce any residuals, and just returns the original output.
    """

    def propagate_residuals(self, output, splicepoint_results_residual):
        assert (splicepoint_results_residual == 0).all()
        return output


@attr.s
class DoNotPropagateLSMRP(LocalSplicepointModelResidualPropagator):
    """
    Just returns the original output.
    """

    def propagate_residuals(self, output, splicepoint_results_residual):
        return output
