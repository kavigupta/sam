import json

import argparse


def even(x):
    x = int(x)
    if x % 2 != 0:
        raise RuntimeError("Expected even number but received {}".format(x))
    return x


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--CL_max",
        default=10000,
        type=even,
        help="Should be the same as the CL_max you used when generating the dataset.",
    )

    p.add_argument(
        "--SL",
        default=5000,
        type=int,
        help="Length of the predicted sequence. Should be a factor of the SL you used when generating the dataset.",
    )

    # Output details
    p.add_argument(
        "--data_dir",
        default="./",
        type=json.loads,
        help="Directory where the dataset is stored. Should be the same as the one you used when generating the dataset.",
    )
    p.add_argument(
        "--window",
        type=int,
        help="the context length of the model. Not necessarily the actual context length of the model,"
        " but rather the amount that is clipped by the model. Used to set x vs y.",
    )
    p.add_argument("--seed", type=int, help="Random seed.")

    # Data Segments
    p.add_argument(
        "--data_chunk_to_use",
        choices=["0", "1", "all"],
        default=None,
        help=(
            "override the data chunk to use in the evaluation set during training."
            " By default it is '0' since this is the chunk (no paralogs) that is used for evaluation normally"
        ),
    )

    # Train arguments
    p.add_argument(
        "--train-mode",
        choices=["standard-supervised", "from-spec"],
        default="standard-supervised",
        help="Mode for training",
    )
    p.add_argument(
        "--train-spec",
        type=json.loads,
        default=None,
        help="specification for the training function. Only used if train-mode is 'from-spec'",
    )

    p.add_argument(
        "--report-frequency",
        type=int,
        default=1000,
        help="Report every K batches. Default is 1000.",
    )
    p.add_argument(
        "--model-path",
        help="provide the model folder to store the model. Can also resume from this folder",
    )
    p.add_argument("--batch-size", default=32, type=int, help="batch size for training")
    p.add_argument(
        "--dataset-spec",
        type=json.loads,
        help="the dataset settings to use for training and evaluation. "
        "cl/sl/path/iteration parameters are given by the code, you need to provide the other parameters",
        default=dict(
            type="H5Dataset",
            datapoint_extractor_spec=dict(type="BasicDatapointExtractor"),
            post_processor_spec=dict(type="IdentityPostProcessor"),
        ),
    )

    p.add_argument(
        "--learned-motif-sparsity",
        type=float,
        default=0.01,
        help="Initial sparsity of the motifs to train.",
    )
    p.add_argument(
        "--learned-motif-sparsity-update",
        type=float,
        default=None,
        help=(
            "how much to reduce density when it gets to an accuracy threshold."
            " E.g., 0.75 -> 25% density reduction"
        ),
    )
    p.add_argument(
        "--learned-motif-sparsity-threshold",
        type=float,
        default=None,
        help="threshold accuracy at which point to reduce the sparsity. "
        "Serves as the minimum possible threshold, and the actual threshold cannot fall below this.",
    )
    p.add_argument(
        "--learned-motif-sparsity-threshold-initial",
        type=float,
        default=None,
        help="initial threshold accuracy. If not provided, it is set to the same value as --learned-motif-sparsity-threshold",
    )
    p.add_argument(
        "--learned-motif-sparsity-threshold-decrease-per-epoch",
        type=float,
        default=1.0,
        help="how much to decrease the threshold per epoch of not hitting it. Irrelevant if --learned-motif-sparsity-threshold is not provided",
    )
    p.add_argument(
        "--learned-motif-sparsity-drop-motif-frequency",
        type=float,
        default=None,
        help="how often to drop a motif. For example if you pass 0.5,"
        " every time the sparsity crosses 0.5, 0.25, 0.125, etc, a motif is removed.",
    )

    p.add_argument(
        "--msp-architecture-spec",
        default=None,
        help="the architecture for the msp model, expressed as a json of a spec.",
    )
    p.add_argument(
        "--evaluation-criterion-spec",
        type=json.loads,
        default=dict(type="DefaultEvaluationCriterion"),
        help="The specification of the evaluation criterion. This is used both for training and evaluation",
    )
    p.add_argument(
        "--only-train",
        choices=["acceptor", "donor"],
        default=None,
        help="If specified, will only train the given type of splicepoint. Either 'acceptor' or 'donor'",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate to use",
    )
    p.add_argument(
        "--decay-start",
        type=int,
        default=6,
        help="Sets the iteration on which the learning rate decay starts",
    )
    p.add_argument(
        "--decay-amount",
        type=float,
        default=0.5,
        help="Sets the amount by which the learning rate decays. E.g., 0.9 means it will reduce by 10% every iteration",
    )
    p.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Number of epochs to train the model",
    )
    p.add_argument(
        "--stop-at-density",
        type=float,
        default=None,
        help="Stop training when the density reaches this value",
    )
    p.add_argument(
        "--eval-limit",
        help="Amount of samples to limit the evaluation to:"
        " this impacts the validation accuracy too, so use with caution when training a sparse model",
        type=int,
        default=float("inf"),
    )
    p.add_argument(
        "--train-limit",
        help="Amount of samples to limit the training to",
        type=int,
        default=float("inf"),
    )
    p.add_argument(
        "--train-cuda",
        help="Whether to use cuda for training",
        type=int,
        default=1,
    )
    return p


def get_args():
    return get_parser().parse_args()
