import os
from math import log

import argparse


def even(x):
    x = int(x)
    if x % 2 != 0:
        raise RuntimeError("Expected even number but received {}".format(x))
    return x


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 'y', 't'):
        return True
    elif v.lower() in ('false', 'n', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--CL_max",
        default=10000,
        type=even,
        help="Maximum nucleotide context length (CL_max/2 on either side of the "
        "position of interest. "
        "CL_max should be an even number.",
    )

    p.add_argument(
        "--SL",
        default=5000,
        type=int,
        help="Sequence length of SpliceAIs (SL+CL will be the input length and "
        "SL will be the output length)",
    )

    p.add_argument(
        "--stretch",
        default=1,
        type=int,
        help="the amount of 'stretch' to put into the input. A stretch > 1 will lead to the bases being 'stretched' out,"
        " for example stretch=2 means that AGAACT will becmoe AAGAAAACCTT. This halves the effective window of the model",
    )

    p.add_argument("--splice_table", default="canonical_dataset.txt")
    # Input details
    p.add_argument("--ref_genome")
    # Output details
    p.add_argument("--data_dir", default="./")
    p.add_argument("--sequence", default="canonical_sequence.txt")
    p.add_argument("--window", type=int, help="the width of the model")
    p.add_argument("--seed", type=int)
    p.add_argument("--load-seeds", type=int, nargs="+")
    p.add_argument("--model-paths", nargs="+")
    p.add_argument("--organism", default="canonical")

    # Data Segments
    p.add_argument("--data_segment_to_use", choices=["train", "test", "all"])
    p.add_argument("--data_chunk_to_use", choices=["0", "1", "all"])

    # Train arguments
    p.add_argument(
        "--train_no_eval",
        action="store_true",
        help="Do not evaluate as part of the train process",
    )
    p.add_argument(
        "--report-frequency",
        type=int,
        default=1000,
        help="Report every K iterations",
    )
    p.add_argument(
        "--null_weight",
        type=float,
        default=1.0,
        help="How much to weight the 0 class (no splice)",
    )
    p.add_argument(
        "--downstream",
        choices=["spliceai", "convolution", "attention"],
        default="spliceai",
        help="just for torch, the downstream model to be used",
    )
    p.add_argument(
        "--num-channels",
        type=int,
        default=32,
        help="just for torch, the number of channels (L) to use at each layer of the convolution",
    )
    p.add_argument(
        "--model-path",
        nargs="+",
        help="just for torch, provide the model folder for training or multiple model folders/paths for evaluation",
    )
    p.add_argument("--batch_size", default=32, type=int, help="just for torch")
    p.add_argument(
        "--fuzz-inputs",
        type=float,
        default=0,
        help="just for torch. If set to a value > 0 will randomize that fraction of the input to make the model more robust",
    )
    p.add_argument(
        "--spatial-sparse",
        type=float,
        help="just for torch. If set to a value V will clip out V fraction of the values from each conv layer, spatially",
    )
    p.add_argument(
        "--motifs",
        choices=["none", "learned", "given"],
        default="none",
        help="just for torch. If set to 'learned', will learn motifs, if set to 'given', will use the given motifs.",
    )
    p.add_argument(
        "--learned-motifs-architecture",
        choices=["psam", "residual", "convolutional"],
        default="psam",
        help="type of motif to be learned",
    )
    p.add_argument(
        "--num-learned-motifs",
        type=int,
        default=100,
        help="number of motifs to be learned (only active if --motifs learned is passed)",
    )
    p.add_argument(
        "--learned-motifs-entropy-loss",
        type=float,
        default=0,
        help="weight applied to the mean entropy of the learned PSAM distributions."
        " A higher weight means they will be forced to have lower entropy",
    )
    p.add_argument(
        "--learned-motifs-entropy-upper-bound",
        type=float,
        default=0,
        help="upper bound to the mean entropy of the learned PSAM distributions."
        " If the mean entropy is below this threshold, the loss will not be used."
        " By default, this is set to 0, which means the entropy loss is always on."
        " If set to log(4), it would mean the loss is always off.",
    )
    p.add_argument(
        "--psams-per-learned-motif",
        type=int,
        default=5,
        help="number of psams per learned motif (only active if --motifs learned is passed)",
    )
    p.add_argument(
        "--learned-motif-length",
        type=int,
        default=13,
        help="length of learned motif, must be odd for padding to work properly (only active if --motifs learned is passed)",
    )
    p.add_argument(
        "--learned-motif-depth",
        type=int,
        default=None,
        help="depth of the learned motif learned",
    )
    p.add_argument(
        "--learned-motif-no-conv",
        action="store_true",
        help="if set, depth must be set to 0. "
        "Uses no convolutional computational layers, just a single conv from the full patch --> channels and then fc",
    )
    p.add_argument(
        "--learned-motif-no-conv-extra-compute",
        type=int,
        default=1,
        help="If set to a value > 1, will increase the number of channels, then apply a relu, then project back",
    )
    p.add_argument(
        "--learned-motif-fc-layers",
        type=int,
        default=0,
        help="additional fc layers downstream",
    )
    p.add_argument(
        "--learned-motif-fc-layer-type",
        choices=["relu", "resblock"],
        default="resblock",
        help="type of additional fc layers downstream",
    )
    p.add_argument(
        "--learned-motif-calculation-multiplier",
        type=int,
        default=1,
        help="how much extra calculation for the learned motifs. E.g., passing 2 makes 200 layers instead of 100",
    )
    p.add_argument(
        "--learned-motif-sparsity",
        type=float,
        default=0.01,
        help="sparsity of learned motif.",
    )
    p.add_argument(
        "--learned-motif-sparsity-no-magnitude",
        action="store_true",
        help="if set, uses value rather than magnitude for the learned motifs.",
    )
    p.add_argument(
        "--learned-motif-sparsity-discontinuity-mode",
        choices=["none", "subtraction"],
        default="none",
        help="how to handle the discontinuity for the learned motif sparsity.",
    )
    p.add_argument(
        "--learned-motif-propagate-sparsity",
        choices=["mask", "product", "sum"],
        default=None,
        help="whether to propagate the sparsity of the learned motifs.",
    )
    p.add_argument(
        "--learned-motif-sparsity-technique",
        choices=[
            "percentile-by-channel",
            "percentile-across-channels",
            "percentile-across-channels-drop-motifs",
            "gaussian-noise",
            "l1",
        ],
        default="percentile-by-channel",
        help="technique for specifying the sparsity of the matrix to use.",
    )
    p.add_argument(
        "--learned-motif-sparsity-update",
        type=float,
        default=None,
        help="how much to reduce sparsity when it gets to a certain threshold.",
    )
    p.add_argument(
        "--learned-motif-sparsity-threshold",
        type=float,
        default=None,
        help="threshold accuracy at which point to start reducing the sparsity",
    )
    p.add_argument(
        "--learned-motif-sparsity-threshold-initial",
        type=float,
        default=None,
        help="initial threshold accuracy",
    )
    p.add_argument(
        "--learned-motif-sparsity-threshold-decrease-per-epoch",
        type=float,
        default=1.0,
        help="how much to decrease the threshold per epoch of not hitting it",
    )
    p.add_argument(
        "--learned-motif-sparsity-drop-motif-frequency",
        type=float,
        default=None,
        help="how often to drop a motif. For example if you pass 0.5,"
        " every time the sparsity crosses 0.5, 0.25, 0.125, etc, a motif is removed.",
    )

    p.add_argument(
        "--learned-motif-sparsity-use-train-threshold",
        action="store_true",
        help="whether to use the training data to compute the threshold",
    )
    p.add_argument(
        "--learned-motif-sparsity-presparse-norm",
        action="store_true",
        help="normalize pre-sparse layer",
    )
    p.add_argument(
        "--learned-motif-aux-loss-given",
        type=float,
        default=0,
        help="if set, adds an auxilary loss of Xentropy between the first K learned motifs"
        " and the given motifs. The remaining (num-learned-motifs) - K will not be tested"
        " (only active if --motifs learned is passed)",
    )
    p.add_argument(
        "--learned-motif-fix-motifs",
        default=None,
        help="if set, fixes the original motifs. Specify a valid path to use a splicepoint model"
        " (only active if --motifs learned is passed)",
    )
    p.add_argument(
        "--output-model-depth",
        type=int,
        default=1,
        help="depth of the output model. Only used in the attention influence model.",
    )
    p.add_argument(
        "--influence-model-only-influence-splicepoints",
        action="store_true",
        help="if set to true, allows the influence model to only influence splicepoint motifs.",
    )
    p.add_argument(
        "--influence-model-local-conflicts",
        choices=["none", "nonzero-counting", "softmax", "exponential-depression"],
        default="none",
        help="technique used for handling local conflicts in attention influence model.",
    )
    p.add_argument(
        "--influence-model-local-conflicts-width",
        type=int,
        help="width of local conflicts model.",
    )
    p.add_argument(
        "--local-conflict-loss",
        type=float,
        default=0,
        help="weight (lambda) on local conflicts loss.",
    )
    p.add_argument(
        "--local-conflict-loss-width",
        type=lambda x: x if x == "adaptive" else int(x),
        default=None,
        help="width of local conflicts loss.",
    )
    p.add_argument(
        "--local-conflict-loss-type",
        choices=["lq", "scad"],
        default="lq",
        help="local conflicts loss type. See scad.py for more info on scad",
    )
    p.add_argument(
        "--local-conflict-loss-norm-q",
        type=float,
        default=None,
        help="Lq norm of local conflicts loss. Active if lq loss is used",
    )
    p.add_argument(
        "--local-conflict-loss-alambda",
        type=float,
        default=10,
        help="a * lambda for local conflicts loss. Active if scad loss is used",
    )
    p.add_argument(
        "--local-conflict-with-splicepoint-extra",
        type=float,
        default=0,
        help="how much extra should conflicts with splicepoints be weighed",
    )
    p.add_argument(
        "--influence-model-n-heads",
        type=int,
        default=1,
        help="number of heads to use.",
    )
    p.add_argument(
        "--influence-model-activation",
        choices=["identity", "sigmoid"],
        default="identity",
        help="The influence model's activation function",
    )
    p.add_argument(
        "--influence-model-positional-encoding",
        action="store_true",
        help="Whether to use a positional encoding",
    )
    p.add_argument(
        "--influence-model-residual-splicepoint-models",
        default=None,
        nargs=2,
        help="Influence model should be residual against the given splicepoint model. "
        "Used as a motif (skipping sparsity) as well as added to the outputs",
    )
    p.add_argument(
        "--influence-model-splicepoint-no-thresh-for-residual",
        action="store_true",
        help="If set, will not set a threshold for the residual connection in the splicepoint models."
        " Will still use a threshold for injection into the main model to prevent informational leakage.",
    )
    p.add_argument(
        "--influence-model-splicepoint-transform-motif",
        action="store_true",
        help="If set, will linearly transform the output of the models to prepare it for the motifs layer.",
    )
    p.add_argument(
        "--influence-model-motif-residual",
        default="none",
        choices=["none", "rbns_psams", "ensembled_latent"],
        help="Influence model motifs should be residual against this source of motifs, which are kept fixed.",
    )
    p.add_argument(
        "--influence-model-motif-residual-args",
        default=[],
        nargs="*",
        help="Influence model motifs should be residual against this source of motifs, which are kept fixed.",
    )
    p.add_argument(
        "--no-pre-sparse-gru",
        action="store_true",
        help="whether we should disable the pre-sparse gru. Only used in the attention influence model.",
    )
    p.add_argument(
        "--post-sparse-type",
        default="gru",
        help="The type of the post-sparse-gru. Either gru or conv<layers>,<kernel_size>",
    )
    p.add_argument(
        "--post-influence-type",
        default="none",
        help="The type of the post-influence-gru. Either none or conv<layers>,<kernel_size>",
    )
    p.add_argument(
        "--final-layer-type",
        default="lstm",
        help="The type of the final layer. Either lstm or none",
    )
    p.add_argument(
        "--real-attention",
        action="store_true",
        help="whether we should use real attention or the more interpretable product model."
        "Only used in the attention influence model.",
    )
    p.add_argument(
        "--influence-model-extra-attn-layers",
        type=int,
        default=0,
        help="How many additional attention layers to add.",
    )
    p.add_argument(
        "--influence-model-pcfg-output",
        choices=["none", "forward-backward"],
        default="none",
        help="How to handle the pcfg output.",
    )
    p.add_argument(
        "--attention-cl",
        type=int,
        default=None,
        help="the CL of the attention layer. Only used in the atteention influence model.",
    )
    p.add_argument(
        "--model-type",
        choices=[
            "spliceai",
            "residual-splicepoint-motif-model",
            "splicepoint-motif-model",
            "influence-model",
            "attention-influence-model",
        ],
        default="spliceai",
        help="just for torch. "
        "If set to 'splicepoint-motif-model', will use the motif model, which is designed to find just the splicepoint motifs.",
    )
    p.add_argument(
        "--transformer-layers",
        type=int,
        default=6,
        help="number of layers to use in the transformer",
    )
    p.add_argument(
        "--asymmetric-window",
        type=int,
        nargs=2,
        default=None,
        help="Only for splicepoint motif model. If used, will make the window for the splicepoint motif model asymmetric.",
    )
    p.add_argument(
        "--only-train",
        choices=["acceptor", "donor"],
        default=None,
        help="If specified, will only train the given type of splicepoint. Either acceptor or donor",
    )
    p.add_argument(
        "--load-preprocess",
        help="path to load the preprocess model from for pretraining",
    )
    p.add_argument(
        "--freeze-preprocess",
        action="store_true",
        help="freeze the preprocessing layers. Only applicable if --load-preprocess is passed",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="just for torch. Sets the learning rate",
    )
    p.add_argument(
        "--decay-start",
        type=int,
        default=6,
        help="just for torch. Sets the iteration on which the decay starts",
    )
    p.add_argument(
        "--decay-amount",
        type=float,
        default=0.5,
        help="just for torch. Sets the amount by which it decays. E.g., 0.9 means it will reduce by 10% every iteration",
    )
    p.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="just for torch. Number of epochs",
    )
    # Test arguments
    p.add_argument(
        "--test_individual",
        action="store_true",
        help="Evaluate each model in the ensemble individually",
    )
    p.add_argument(
        "--output-path",
        help="Path to save outputs to",
    )

    # generate synthetic dataset
    p.add_argument(
        "--step_num",
        default=None,
        help="step_num in Synthetic mechanism"
    )
    p.add_argument(
        "--real_acceptor",
        default=False,
        type=str2bool,
        help="use real acceptor"
    )
    p.add_argument(
        "--num_motifs",
        default=81,
        type=int, 
        help="num of motifs to use in synthetic mechanism"
    )
    p.add_argument(
        "--mode",
        default='None',
        help=f"define the mode to generate the potential splice site set"
    )
    p.add_argument(
        "--splice_site_details", 
        default=None,
        help="details of splice site mode"
    )
    p.add_argument(
        "--s_site", 
        default=0.01,
        type=float,
        help="the pre-sparsity of splice site",
    )
    p.add_argument(
        "--splicesite_range", 
        default=160,
        type=int,
        help="The range to search for splicesite points",
    )
    p.add_argument(
        "--simple_sparsity",
        default=False,
        type=str2bool,
        help="only use sparsity that allowed",
    )
    p.add_argument(
        "--influence_mechanism",
        default=None,
        help="which influence mechanism to use",
    )
    
    return p


def get_args():
    return get_parser().parse_args()
