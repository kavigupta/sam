from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import IterableDataset
from modular_splicing.utils.construct import construct

from modular_splicing.utils.io import load_model

from .motif_selector import (
    MotifIncrementalAdder,
    MotifIncrementalDropper,
)


def train(
    *,
    path,
    dtrain,
    deval,
    architecture,
    bs,
    lr,
    train_steps,
    train_select_steps,
    test_steps,
    trainer_spec,
):
    train_batches = train_steps // bs
    train_select_batches = train_select_steps // bs
    test_batches = test_steps // bs

    idx, trainer = load_model(path)

    dtrain_repeat = iter(DataLoader(InfinitelyIterate(dtrain), batch_size=bs))

    if idx is None:
        trainer = construct(
            dict(
                MotifIncrementalDropper=MotifIncrementalDropper,
                MotifIncrementalAdder=MotifIncrementalAdder,
            ),
            trainer_spec,
            architecture=architecture,
            lr=lr,
            batch_size=bs,
        )
        print("not found, starting from scratch")
    else:
        print("loading", path, idx)
    trainer.start()
    while True:
        trainer.train_current_model(
            training_data=dtrain_repeat,
            train_batches=train_batches,
            testing_data=DataLoader(deval, batch_size=bs),
            test_batches=test_batches,
        )
        trainer.save(path)
        if trainer.select_model(dtrain_repeat, train_select_batches, test_batches):
            break
        trainer.save(path)


class InfinitelyIterate(IterableDataset):
    def __init__(self, underlying_dataset):
        super().__init__()
        self.underlying_dataset = underlying_dataset

    def __iter__(self):
        while True:
            yield from self.underlying_dataset
