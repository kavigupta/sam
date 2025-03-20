import numpy as np
import torch
import torch.nn as nn
import tqdm

from modular_splicing.eclip.test_motifs.names import get_testing_names
from modular_splicing.eclip.trained_on_eclip.model import EclipMatchingModelAM
from modular_splicing.legacy.remapping_pickle import permacache_with_remapping_pickle
from modular_splicing.models.motif_model_stub import EclipMatchingModelMotifModelStub
from modular_splicing.utils.construct import construct

from .training_data import EclipMotifsDataset


def extendable_dict(**kwargs):
    return lambda **x: dict(**kwargs, **x)


def get_model_specs():
    """
    Get a full dictionary of model specs.

    Currently is just the one.
    """
    model_specs = {}
    w = 21
    k = 2
    name = f"am_{w}x{k}_178_post_sparse_scale"
    model_specs[name] = extendable_dict(
        type="EclipMatchingModelAM",
        k=k,
        w=w,
        psam_source="rbrc",
        post_sparse_scale=True,
        sparsity=0.178e-2,
    )
    return model_specs


model_specs = get_model_specs()


train_data_spec = dict(
    type="EclipMotifsDataset",
    path="dataset_train_all.h5",
    path_annotation="dataset_intron_exon_annotations_train_all.h5",
    mode=("from_5'", 50),
)


class PretrainedEclipMotifModel(nn.Module):
    """
    Pretrained eclip motif model. Good for use in end-to-end models.
    """

    def __init__(
        self,
        input_size,
        channels,
        stub_spec,
        num_motifs,
        set_eclip_models_to_eval=False,
    ):
        super().__init__()
        assert input_size == 4
        assert set_eclip_models_to_eval, (
            "set_eclip_models_to_eval must be True. "
            "This is here to ensure that experiments are updated and models are trained apppropriately."
        )
        del channels
        self.full_model_stub = construct(
            dict(model_trained_on_eclip=model_trained_on_eclip), stub_spec
        )
        for param in self.full_model_stub.parameters():
            param.requires_grad = False
        assert len(self.full_model_stub.models) == num_motifs, str(
            (len(self.full_model_stub.models), num_motifs)
        )

    def forward(self, sequence):
        with torch.no_grad():
            self.full_model_stub.eval()
            res = self.full_model_stub(sequence, only_motifs=True)
        return res["post_sparse_motifs_only"].float()

    def notify_sparsity(self, sparsity):
        pass


def models_trained_on_eclip(*, motif_names_source):
    """
    Produce a full list of models trained on eclip data.

    All are stubs.
    """

    return {
        f"trained_on_eclip_{x}_{seed}": model_trained_on_eclip(
            motif_names_source=motif_names_source, eclip_model_spec=x, seed=seed
        )
        for x in model_specs
        for seed in (1, 2, 3)
    }


def model_trained_on_eclip(*, motif_names_source, eclip_model_spec, seed):
    """
    Produce a model trained on eclip data, as a stub.
    """
    return (
        EclipMatchingModelMotifModelStub(
            400,
            [
                eclip_model_from_name(eclip_model_spec, motif, seed=seed).eval().cuda()
                for motif in get_testing_names(
                    motif_names_source=motif_names_source
                ).common_names
            ],
            # each should have sparsity already enforced.
            sparsity=None,
        )
        .cuda()
        .eval()
    )


def eclip_model_from_name(name, motif, *, seed):
    """
    Produce an eclip model from the given name.
    """
    return train_eclip_model(
        eclip_model_spec=model_specs[name](motif=motif),
        data_spec=train_data_spec,
        seed=seed,
        motif=motif,
        batch_size=128,
        epochs=20,
        pbar=tqdm.tqdm,
    )


@permacache_with_remapping_pickle(
    "eclip/train_eclip_model_3",
    key_function=dict(batch_size=None, pbar=None),
    multiprocess_safe=True,
)
def train_eclip_model(
    *, eclip_model_spec, data_spec, motif, batch_size, epochs, pbar, seed
):
    """
    Train an eCLIP matching model, and return the trained model. Cached
        on the disk.

    Parameters
    ----------
    eclip_model_spec : dict
        The specification for the eCLIP matching model to be trained
    data_spec : dict
        The specification for the data to be used for training
    motif : str
        The motif to use, e.g., "TRA2A"
    batch_size : int
        The batch size to use for training
    epochs : int
        The number of epochs to train for
    pbar :
        A progress bar to use for training
    seed : int
        The random seed to use for training. Will be used for both initializing the
            model and for the data loader.
    """
    seed_torch, seed_data = np.random.RandomState(seed).randint(0, 2**32, 2)
    torch.manual_seed(seed_torch)
    data = construct(
        dict(EclipMotifsDataset=EclipMotifsDataset), data_spec, seed=seed_data
    )
    model = construct(
        dict(EclipMatchingModelAM=EclipMatchingModelAM),
        eclip_model_spec,
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(epochs):
        print(motif, epoch)
        for idx, (xs, ys) in enumerate(data.batched_data(motif, batch_size, pbar=pbar)):
            xs = np.eye(4)[xs]
            xs, ys = [torch.tensor(d).float().cuda() for d in (xs, ys)]
            optimizer.zero_grad()
            loss = model.loss(xs, ys)
            if idx % 1000 == 0:
                print(idx, loss.item())
            loss.backward()
            optimizer.step()
    return model
