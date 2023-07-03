import matplotlib.pyplot as plt


def render_motif_saliency_map(realization):
    mech = realization.splicing_mechanism
    true_splice = realization.real_splicing_pattern
    plt.figure(figsize=(6, 6), dpi=200)
    motifs = mech.predict_motifs(realization.rna)
    salience = mech.motif_saliency_map(motifs, true_splice)
    plt.plot(salience[:, 0], color="red", label="3'")
    plt.plot(salience[:, 1], color="blue", label="5'")
    for motif_idx in range(salience.shape[1] - 2):
        plt.plot(salience[:, motif_idx + 2], alpha=0.5, label=f"Motif {motif_idx + 1}")
    plt.xlabel("Position in gene")
    plt.ylabel("Gradient at position")
    plt.legend()
