from abc import ABC, abstractmethod


class Mutation(ABC):
    @abstractmethod
    def perform(self, rna):
        """
        Perform the mutation on the given rna.
        """
        pass

    @abstractmethod
    def footprint(self):
        """
        Returns the footprint of this mutation, i.e.,
            the array of indices that are affected by this mutation.
        """
        pass
