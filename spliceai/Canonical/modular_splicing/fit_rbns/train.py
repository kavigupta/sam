import torch

import tqdm

from modular_splicing.utils.io import load_model
from modular_splicing.motif_names import get_motif_names
from modular_splicing.utils.io import save_model

from .rbns_data import RBNSData, MotifModelBasedClassifer

# TODO document how to get this data
data_path = "../data/500k_binary_rbns_dataset/"
rbnsp_names = get_motif_names("rbns")


def psam_motif_spec(motif):
    return {
        "type": "PSAMMotifModel",
        "motif_spec": {"type": "rbns"},
        "exclude_names": ("3P", "5P") + tuple(set(rbnsp_names) - {motif}),
    }


def train(
    model_path,
    *,
    motif,
    initial_density,
    n_epochs=10,
    bs=1000,
    adjustment_model_spec,
    k=2,
    seed=0,
):
    """
    Train a model on the RBNS dataset. Cached in the model directory.

    While this appears to train an AdjustedMotifModel, it is in practice
        just training the sum of a PSAM and a learned model.
        The only purpose of this is to ensure that we keep the same architecture as AMs.
        Actually trying to be similar to FMs is of secondary importance, since we
        are training on the same dataset.


    Parameters
    ----------
    model_path: the path to save the model to
    motif: the motif to train on
    initial_density: the initial density of the model. This is typically
        set to a high value (high density) and we do not do any sparsity updates.
    n_epochs: the number of epochs to train for
    bs: the batch size
    adjustment_model_spec: the specification of the adjustment model
    k: the sparsity multiplier. Like initial sparsity, largely irrelevant.
    seed: the random seed to use in training
    """
    model_path = model_path.format(motif=motif)
    motif_model_spec = {
        "type": "AdjustedMotifModel",
        "model_to_adjust_spec": psam_motif_spec(motif),
        "adjustment_model_spec": adjustment_model_spec,
        "sparsity_enforcer_spec": {
            "type": "SparsityEnforcer",
            "sparse_spec": {
                "type": "SpatiallySparseAcrossChannels",
                "sparsity": 1 - initial_density,
            },
        },
        "sparsity_multiplier": k,
    }

    data = RBNSData(data_path)
    assert motif in rbnsp_names and motif in data.names

    step, model = load_model(model_path)
    if step is None:
        # set up initial training
        step = 0
        model = MotifModelBasedClassifer(motif_model_spec).cuda()

    for i in range(step, n_epochs):
        print(motif, i)
        # set up the optimizer, we use a new one per epoch
        # learning rate is set to decay by 10% every epoch
        opt = torch.optim.Adam(model.parameters(), 1e-3 * 0.9 ** (-i))

        pbar = tqdm.tqdm(
            data.batched_data(motif, bs, seed=i + seed * 1000, is_test=False),
            total=(data.length(motif) + bs - 1) // bs,
        )

        # train for one epoch
        for batch in pbar:
            _, loss = model.loss(batch.x, batch.y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_description(f"{loss.item():.2f}")
        save_model(model, model_path, i + 1)
    return model


def train_21x2_model(motif, seed=0):
    """
    Train a 21"x2" model (the x2 doesn't really mean anything since we set the
    density to 100%, so we are "letting in 200% of the information").
    """
    return train(
        "model/rbns-binary-model-{motif}-21x2" + ("_{}".format(seed) if seed else ""),
        motif=motif,
        initial_density=1,
        bs=10**4,
        adjustment_model_spec={
            "type": "LearnedMotifModel",
            "motif_width": 21,
            "motif_fc_layers": 5,
            "motif_feature_extractor_spec": {"type": "ResidualStack", "depth": 5},
        },
        seed=seed,
    )
