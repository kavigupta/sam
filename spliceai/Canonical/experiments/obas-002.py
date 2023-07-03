# This current file represents a more-or-less identical experiment to the one
# we have been using, with the exception being it uses dynamic accuracy bar
# instead of static accuracy bar.

# I think in general this tends to have a slightly negative effect on the
# performance of the model, but within 0.5-1% in accuracy. I will be able
# to give an exact number for this model soon as I am training one right now.


from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

# this should probably stay the same. this is the maximum initial
# accuracy bar to start with, it will be gradually lowered over time.
STARTING_ACCURACY_BAR = 85
# The number of channels to use in the post-sparse convolution
# This is already pretty high, so might not be worth increasing
POST_SPARSE_CONVOLUTION_CHANNELS = 200
# These are the parameters for the post-sparse convolution.
# The depth is the number of residual blocks, each of which contains 2 convolutions
# So it can be provided as an integer or an integer + 0.5, in that case
# an extra single convolution will be used.
#
# Either way, these numbers must satisfy (POST_SPARSE_CONVOLUTION_WIDTH - 1) % (2 * POST_SPARSE_CONVOLUTION_DEPTH) == 0
POST_SPARSE_CONVOLUTION_WIDTH = 49
POST_SPARSE_CONVOLUTION_DEPTH = 4
# The number of channels to use in the attention layer.
# Currently its set to the number of motifs. Might be useful to increase
INFLUENCE_VALUE_CALCULATOR_ATTENTION_CHANNELS = 82
# The width of the attention layer. Might be worth increasing this.
ATTENTION_WINDOW_SIZE = 400

# Same idea as above with the post-sparse convolution layer, but this time for the
# post-influence layer. It serves the same purpose, to unsparsify the motifs and
# allow all channels to be used more easily by the neural networks.

POST_INFLUENCE_CONVOLUTION_CHANNELS = 200
POST_INFLUENCE_CONVOLUTION_WIDTH = 49
POST_INFLUENCE_CONVOLUTION_DEPTH = 4

# The number of channels that the LSTM should take as input and produce as an output
# The lstm is a bilstm so the output channels will be divided by 2 to get the output of each direction
# Additionally, you can make the LSTM have more layers. In general you need
# LSTM_OUT_CHANNELS % (2 * LSTM_LAYERS) == 0
LSTM_IN_CHANNELS = 82
LSTM_OUT_CHANNELS = 82
LSTM_LAYERS = 1

msp.architecture["motif_model_spec"]["motif_width"] = 21
msp.architecture["motif_model_spec"]["motif_feature_extractor_spec"] = dict(
    type="ResidualStack",
    depth=5,
)

msp.architecture["num_motifs"] = 82
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")
msp.architecture["motif_model_spec"] = dict(
    type="AdjustedMotifModel",
    model_to_adjust_spec=dict(
        type="PSAMMotifModel",
        motif_spec=dict(type="rbns"),
        exclude_names=("3P", "5P"),
    ),
    adjustment_model_spec=msp.architecture["motif_model_spec"],
    sparsity_enforcer_spec=dict(
        type="SparsityEnforcer", sparse_spec=msp.architecture["sparse_spec"]
    ),
)
msp.architecture["affine_sparsity_enforcer"] = True

msp.architecture["influence_calculator_spec"] = dict(
    type="InfluenceValueCalculator",
    post_sparse_spec=dict(
        type="ResidualStack",
        width=POST_SPARSE_CONVOLUTION_WIDTH,
        depth=POST_SPARSE_CONVOLUTION_DEPTH,
        hidden_channels=POST_SPARSE_CONVOLUTION_CHANNELS,
    ),
    long_range_reprocessor_spec=dict(type="SingleAttentionLongRangeProcessor"),
    intermediate_channels=INFLUENCE_VALUE_CALCULATOR_ATTENTION_CHANNELS,
    selector_spec=dict(type="Linear"),
)

msp.architecture["final_processor_spec"] = dict(
    type="FinalProcessor",
    post_influence_spec=dict(
        type="ResidualStack",
        width=POST_INFLUENCE_CONVOLUTION_WIDTH,
        depth=POST_INFLUENCE_CONVOLUTION_DEPTH,
        hidden_channels=POST_INFLUENCE_CONVOLUTION_CHANNELS,
    ),
    long_range_final_layer_spec=dict(
        type="LSTMLongRangeFinalLayer", layers=LSTM_LAYERS
    ),
    long_range_in_channels=LSTM_IN_CHANNELS,
    long_range_out_channels=LSTM_OUT_CHANNELS,
)

msp.window = ATTENTION_WINDOW_SIZE

msp.acc_thresh = 0
msp.extra_params += (
    f" --learned-motif-sparsity-threshold-initial {STARTING_ACCURACY_BAR}"
)

msp.run()
