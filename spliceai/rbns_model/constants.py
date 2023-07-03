import os
import argparse

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
    p.add_argument("--protein", default=1, type=int, help="target protein id")
    p.add_argument("--bs", default=512, type=int, help="batch size")
    p.add_argument("--model_path", default="result/")
    p.add_argument("--data_dir", default="data/")
    p.add_argument("--n_epochs", default=20, type=int)
    p.add_argument("--lr", default=1e-2, type=float)
    p.add_argument("--protein_test", default=47, type=int, help="the protein number")
    p.add_argument("--each_class_train_size", default=100000, type=int, help="the size of training")
    p.add_argument("--l", default=16, type=int, help="hidden size")
    p.add_argument("--window_size", default=20, type=int, help="the window size")

    # training
    p.add_argument("--save", default=True, type=str2bool, help="save models or not")
    p.add_argument("--partition", default=None, type=int, help="the n-th partition for protein models")
    p.add_argument("--threshold", default=0.1, type=float, help="the threshold to filter out small signals")
    p.add_argument("--sparsity", default=0.0005, type=float, help="sparsity for the model")
    p.add_argument("--protein_name", default=None, help="rbns protein name")
    p.add_argument("--num_motifs", default=104, type=int, help="starting channels")
    p.add_argument("--psam_only", default=False, type=str2bool, help="to match PSAM number")
    p.add_argument("--use_motifs", default=False, type=str2bool, help=" decide whether use motifs")
    p.add_argument("--use_splice_site", default=False, type=str2bool, help="whether to use splice site in SpliceAI")

    # testing
    p.add_argument("--attr", default=0.00055, help="for testing")
    p.add_argument("--evaluate_result", default=False, type=str2bool, help="parameter to measure evaluate result or not")

    # dataset
    p.add_argument("--shuffle_by_group", default=False, type=str2bool, help="whether shuffle by group")
    p.add_argument("--organism", default="canonical", help="give the organism to use")

    # dataset format
    p.add_argument("--window", default=5000, type=int, help="the width of the model")
    p.add_argument("--CL", default=5000, type=int, help="CL")
    p.add_argument("--CL_max", default=5000, type=int, help="CL_max")
    p.add_argument("--SL", default=5000, type=int, help="Sequence Length")

    # draw psam
    p.add_argument("--motif_name", default="46", help="in plot for motifs")

    return p


def get_args():
    return get_parser().parse_args()
