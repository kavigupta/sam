import numpy as np


def get_hparams(window, CL_max):
    if window == 80:
        W = np.asarray([11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1])
        BATCH_SIZE = 18
    elif window == 400:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
        BATCH_SIZE = 18
    elif window == 2000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11, 21, 21, 21, 21])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10])
        BATCH_SIZE = 12
    elif window == 10000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11, 21, 21, 21, 21, 41, 41, 41, 41])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10, 25, 25, 25, 25])
        BATCH_SIZE = 6
    else:
        raise AssertionError("Invalid window: {}".format(window))

    # Hyper-parameters:
    # L: Number of convolution kernels
    # W: Convolution standard_args.window size in each residual unit
    # AR: Atrous rate in each residual unit

    CL = 2 * np.sum(AR * (W - 1))
    assert CL <= CL_max and CL == window

    return W, AR, BATCH_SIZE, CL
