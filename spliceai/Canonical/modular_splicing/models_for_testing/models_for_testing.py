from abc import ABC, abstractmethod
import attr

from modular_splicing.utils.io import load_model, model_steps
from .load_model_for_testing import step_for_density

STANDARD_DENSITY = 0.178e-2
STANDARD_DENSITY_FLY = 0.317e-2
# what ten epochs are for the given batch size
TEN_EPOCHS = {10: 1627100, 15: 1627200}
TWENTY_EPOCHS = {k: v * 2 for k, v in TEN_EPOCHS.items()}


@attr.s
class ModelForTesting(ABC):
    """
    Represents a model to use in testing. Contains a name, path and seed.

    The step to load the model is left to the child class.
    """

    name_prefix = attr.ib()
    path_prefix = attr.ib()
    seed = attr.ib()
    before_prefix = attr.ib(default="_")

    @property
    def name(self):
        return f"{self.name_prefix}_{self.seed}"

    @property
    def path(self):
        return f"{self.path_prefix}{self.before_prefix}{self.seed}"

    @property
    @abstractmethod
    def step(self):
        pass

    @property
    def model(self):
        path = self.path
        step = self.step
        assert step is not None, "Step must be set"
        _, model = load_model(path, step)
        return model.eval()


@attr.s
class ModelForTestingWithStep(ModelForTesting):
    """
    Represents a model to use in testing, with a known step.
    """

    step_value = attr.ib(kw_only=True)

    @property
    def step(self):
        return self.step_value


@attr.s
class ModelForTestingWithDensity(ModelForTesting):
    """
    Represents a model to use in testing, with a known density.
    """

    density = attr.ib(kw_only=True)

    @property
    def step(self):
        return step_for_density(self.path, self.density)


@attr.s
class ModelForTestingLastStep(ModelForTesting):
    """
    Represents a model to use in testing, with the last step.
    """

    @property
    def step(self):
        return model_steps(self.path)[-1]


@attr.s
class EndToEndModelsForTesting:
    """
    Represents a family of end-to-end models to use in testing.

    Each can be binarized.
    """

    name = attr.ib()
    path_prefix = attr.ib()
    seeds = attr.ib()
    density = attr.ib()
    binarized_seeds = attr.ib(default=())
    binarized_step = attr.ib(default=None)
    binarization_suffix = attr.ib(default=".x1")

    def binarized_model(self, seed):
        """
        Get the binarized model for the given seed.
        """
        return ModelForTestingWithStep(
            name_prefix=self.name,
            path_prefix=self.path_prefix + self.binarization_suffix,
            seed=seed,
            step_value=self.binarized_step,
        )

    def non_binarized_model(self, seed, density_override=None):
        """
        Get the non-binarized model for the given seed.

        If density_override is given, use that density instead of the default.
            This is useful for getting earlier checkpoints of a given model, that
            have a greater density and thus can be used as a baseline more effectively
            for comparsion.
        """
        density = density_override if density_override is not None else self.density
        return ModelForTestingWithDensity(
            name_prefix=self.name,
            path_prefix=self.path_prefix,
            seed=seed,
            density=density,
        )

    def binarized_models(self):
        return [self.binarized_model(seed) for seed in self.binarized_seeds]

    def non_binarized_models(self, density_override=None):
        return [
            self.non_binarized_model(seed, density_override=density_override)
            for seed in self.seeds
        ]

    def rename(self, new_name):
        """
        Rename this model.
        """
        return attr.evolve(self, name=new_name)
