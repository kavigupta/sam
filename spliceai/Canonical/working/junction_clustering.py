import matplotlib.pyplot as plt
import numpy as np
from modular_splicing.gtex_data.annotation.compute_optimal_sequence import (
    choose_optimal_sequence_for_gene,
)


def render_gene(sites, psis, junctions, junction_tpms, annotations_chosen, xlim=None):
    """
    Render the given gene.

    Parameters
    ----------
    sites : list of strings
        The sites in the gene, either "A" or "D", in order.
    psis : list of floats
        The psi values for each site in the gene, averaged across all tissues.
    junctions : list of lists of ints
        Each element is a two element list (start, end),
            representing the start and end of a junction,
            in the index of the sites list.
    junction_tpms : list of floats
        The tpm of each junction in the gene, averaged across all tissues.
    annotations_chosen : list of Annotations
        The annotations that were chosen for this gene.
    """
    x_arb = np.arange(len(sites))
    sites = np.array(sites)
    for s in "AD":
        plt.scatter(x_arb[sites == s], psis[sites == s], label=s, alpha=0.5)
    for i, (t, (s, e)) in enumerate(zip(junction_tpms, junctions)):
        if xlim is not None:
            if e < xlim[0] or s > xlim[1]:
                continue
        plt.plot([s, e], [t] * 2)
        plt.text((s + e) / 2, t, s=str(i))
    plt.xticks(x_arb, rotation=90)
    plt.grid(axis="x")

    y_center = 1.2
    y_rad = 0.05

    for annot in annotations_chosen:
        if xlim is not None:
            if annot.sites[-1] < xlim[0] or annot.sites[0] > xlim[1]:
                continue
        annot.render(plt.gca(), y_center=y_center, y_rad=y_rad)

    plt.ylim(-0.1, y_center + y_rad + 0.1)
    yt = np.arange(0, y_center + 0.2, 0.2)
    plt.yticks(yt, [*[f"{x:.0%}" for x in yt[:-1]], "Class"])
    if xlim is not None:
        plt.xlim(*xlim)


def plot_example(
    genes,
    juncs,
    gene_ensg,
    name,
    png_path,
    cost_params,
    xlim=None,
    size=None,
    psis=None,
):
    intermediates, _, annotations = choose_optimal_sequence_for_gene(
        genes, juncs, gene_ensg, cost_params
    )
    if psis is None:
        psis = intermediates["psis"]
    plt.figure(figsize=(0.2 * len(psis), 6) if size is None else size)
    render_gene(
        intermediates["sites"],
        psis,
        intermediates["index_juncs"],
        intermediates["tpm_juncs"],
        annotations,
        xlim=xlim,
    )
    plt.title(name)
    if png_path is not None:
        plt.savefig(png_path)
        plt.close()
