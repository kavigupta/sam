from abc import abstractmethod
from modular_splicing.motif_names import get_motif_names

from .trainer import AutoMinimizingTrainer
from modular_splicing.utils.construct import construct


class MotifSelector(AutoMinimizingTrainer):
    def __init__(self, *args, names_spec, **kwargs):
        super().__init__(*args, **kwargs)

        self.names = construct(dict(rbnsp=lambda: get_motif_names("rbns")), names_spec)

    def inital_model_setup(self):
        self.model.sparse_layer.dropped_motifs = self.initial_dropped_motifs()

    def candidate_models(self):
        names = []
        models = []
        for name, dropped in self.candidate_dropped():
            names.append(name)
            models.append(copy_model(self.model))
            models[-1].sparse_layer.dropped_motifs = dropped
        return names, models

    @abstractmethod
    def initial_dropped_motifs(self):
        pass

    @abstractmethod
    def candidate_dropped(self):
        pass


class MotifIncrementalDropper(MotifSelector):
    def initial_dropped_motifs(self):
        return []

    def candidate_dropped(self):
        dropped_so_far = self.model.sparse_layer.dropped_motifs
        extra_drop = sorted(set(range(len(self.names))) - set(dropped_so_far))
        for i in extra_drop:
            yield f"drop({self.names[i]!r})", dropped_so_far + [i]


class MotifIncrementalAdder(MotifSelector):
    def initial_dropped_motifs(self):
        return list(range(len(self.names)))

    def candidate_dropped(self):
        dropped_so_far = self.model.sparse_layer.dropped_motifs
        for i in dropped_so_far:
            yield f"add({self.names[i]!r})", [x for x in dropped_so_far if x != i]


def copy_model(model):
    f = io.BytesIO()
    torch.save(model, f)
    f.seek(0)
    return load_with_remapping_pickle(f)
