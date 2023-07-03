import tqdm
from modular_splicing.motif_names import get_motif_names

from modular_splicing.fit_rbns.train import train_21x2_model
from modular_splicing.fit_rbns.rbns_data import RBNSData

psam_names = get_motif_names("rbns")
motifs = RBNSData("../data/500k_binary_rbns_dataset/").names
for seed in range(5):
    for motif in tqdm.tqdm(motifs):
        if motif not in psam_names:
            continue
        train_21x2_model(motif, seed)
