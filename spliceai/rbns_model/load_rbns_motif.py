import os
import numpy as np

from binary_motif_model import (
    load_model as load_motif_model, 
)

from map_protein_name import (
    get_map_protein,
    get_rbns_name_psam_idx_map,
)

import splice_point_identifier

from utils import prod

import torch

RBNS_MOTIFS_DIRECTORY = os.path.join(
    os.path.dirname(__file__), "single_binary_model_w_11"
)
SPLICE_MOTIFS_DIRECTORY = os.path.join(
    os.path.dirname(__file__), "splicepoint_model"
)

idx_map = {
    "5P": 2,
    "3P": 1,
}

def read_rbns_motif(dir, rbns_name, psam_idx):
    # return the single rbns motif model
    # if psam_idx < 0.0:
    #     protein = rbns_name
    # else:
    #     protein = psam_idx
    protein = rbns_name
    
    _, m = load_motif_model(dir, protein)
    m.cuda()
    m.eval()

    return m # use_as_motif=True


def read_splicesite_motif(dir, site):
    _, m = load_motif_model(dir, site)

    return m


def rbns_motifs(
    directory=RBNS_MOTIFS_DIRECTORY,
    splice_motifs_directory=SPLICE_MOTIFS_DIRECTORY,
    psam_only=False,
    splice_site_only=False,
):
    # load all rbns motif models in <m1, m2, ..., m102>
    # store them as dict, name: m
    if splice_site_only:
        results = {
            "3P": read_splicesite_motif(splice_motifs_directory, "acceptor"),
            "5P": read_splicesite_motif(splice_motifs_directory, "donor"),
        }
    else:
        rbns_name_psam_idx_dict = get_rbns_name_psam_idx_map()
        if psam_only:
            results = dict()
            for protein_rbns_name, protein_psam_idx in sorted(rbns_name_psam_idx_dict.items()):
                if protein_psam_idx != -1 or protein_rbns_name in ['RALYL']:
                    results[protein_rbns_name] = read_rbns_motif(directory, protein_rbns_name, protein_psam_idx)
        else:
            results = {
                protein_rbns_name: read_rbns_motif(directory, protein_rbns_name, protein_psam_idx)
                for protein_rbns_name, protein_psam_idx in sorted(rbns_name_psam_idx_dict.items())
            }

        results.update(
        {
            "3P": read_splicesite_motif(splice_motifs_directory, "acceptor"),
            "5P": read_splicesite_motif(splice_motifs_directory, "donor"),
        }   
        )
        # print(f"results: {results}")
        print(len(results))

    return results


def use_rbns_motifs(rbns_motifs, x):
    # run each motifs on x
    # concatenate results
    # pad till the same shape
    all_out_non_splicesite = list()
    all_out_splice_site = list()

    for protein_name in rbns_motifs:
        m = rbns_motifs[protein_name]
        if protein_name in idx_map:
            out = m(x, use_as_motif=True)[:, idx_map[protein_name], :]
            out = out.detach().cpu().numpy()
            out = np.pad(out, pad_width=[(0, 0), (0, x.shape[-1] - out.shape[-1])])
            all_out_splice_site.append(out)
        else:
            # print(protein_name)
            # print(m.window_size)
            out = m(x, use_as_motif=True)[:, 1, :]
            out = out.detach().cpu().numpy()
            out = np.pad(out, pad_width=[(0, 0), (0, x.shape[-1] - out.shape[-1])])
            all_out_non_splicesite.append(out)
    
    return all_out_non_splicesite, all_out_splice_site


def rbns_motifs_for(motifs, x, use_splice_site):
    all_out_non_splicesite, all_out_splice_site = use_rbns_motifs(motifs, x)
    if use_splice_site:
        splice_site_motifs = np.array(all_out_splice_site).transpose(1, 0, 2)
        return None, torch.tensor(splice_site_motifs).cuda(), {}
    non_splice_site_motifs = np.array(all_out_non_splicesite).transpose(1, 0, 2)
    splice_site_motifs = np.array(all_out_splice_site).transpose(1, 0, 2)

    return torch.tensor(non_splice_site_motifs).cuda(), torch.tensor(splice_site_motifs).cuda(), {}



