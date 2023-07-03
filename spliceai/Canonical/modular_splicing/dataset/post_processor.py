from abc import ABC, abstractmethod


class PostProcessor(ABC):
    @abstractmethod
    def post_process(self, element):
        """
        Post process the element, producing a sequence of elements
        """
        pass


class FlattenerPostProcessor(PostProcessor):
    def __init__(self, *, indices):
        self.indices = indices

    def post_process(self, element):
        return tuple(element[a][b] for a, b in self.indices)


class IdentityPostProcessor(PostProcessor):
    def post_process(self, element):
        return element


def post_processor_types():
    return dict(
        IdentityPostProcessor=IdentityPostProcessor,
        FlattenerPostProcessor=FlattenerPostProcessor,
    )
