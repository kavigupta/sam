from math import ceil

from modular_splicing.train.standard_train import train_model


def train(
    *,
    path,
    dtrain,
    deval,
    architecture,
    bs,
    lr,
    epochs_per_motif,
    num_motifs,
):
    train_limit = int(len(dtrain) * epochs_per_motif)

    for i in range(num_motifs):
        print("training motif", i)

        train_model(
            path=f"{path}/dropping_{i}",
            dtrain=dtrain,
            deval=deval,
            evaluation_criterion_spec=dict(type="DefaultEvaluationCriterion"),
            architecture=modified_architecture(architecture, i),
            bs=bs,
            n_epochs=int(ceil(epochs_per_motif)),
            lr=lr,
            report_frequency=500,
            train_limit=train_limit,
            cuda=True,
            print_model=False,
        )


def modified_architecture(architecture, i):
    def updated():
        model = architecture()
        model.sparse_layer.dropped_motifs = [i]
        return model

    return updated
