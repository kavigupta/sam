from collections import Counter

from permacache import permacache
import numpy as np
import pandas as pd


def load_interactions():
    """
    Loads the interactions from the Lang Paper. Specifically, supplementary
        table 9.

    Returns
    -------
    dictionary from interaction type to a table guaranteed to have columns
        - protein_1, protein_2: the two interacting proteins
        - known: whether the interaction was previously known or not
    """
    interactions = pd.ExcelFile(
        "../data/lang-rbp-relationships/Table S9_Identified_interactions_sorted_by_process.xlsx"
    )
    interactions = {x: pd.read_excel(interactions, x) for x in interactions.sheet_names}
    interactions["stress_g_processing_b"] = interactions[
        "stress_g_processing_b"
    ].rename(columns={"protein1": "protein_1", "protein2": "protein_2"})
    return interactions


def count_interactions(interaction, names):
    """
    Counts the number of interactions for each name in the given table.

    Note that a protein interacting with itself is counted just once.

    Parameters
    ----------
    interaction: a table with columns protein_1 and protein_2
    names: a list of names to count interactions for

    Returns
    -------
    an array with the counts for each name
    """
    interaction = {(row.protein_1, row.protein_2) for _, row in interaction.iterrows()}
    interaction |= {(y, x) for x, y in interaction}
    counts = Counter(x for x, _ in interaction)
    return np.array([counts[x] for x in names])


def has_interaction(interaction, names):
    """
    Returns a boolean array indicating whether each name has an interaction.

    See count_interactions for more details.
    """
    return count_interactions(interaction, names) != 0


def weighted_sums(interactions, robustness, technique):
    """
    Produce a series with the weighted sum of robustness for each interaction type,
        including the `overall` interaction, which refers to the result of
        concatenating all interactions.

    Parameters
    ----------
    interactions: a dictionary from interaction type to a table with columns
        protein_1 and protein_2
    robustness: a dictionary from interaction type to a table with columns
        motif_name and robustness
    technique: function that takes [interaction, names] and produces an array
        of weights.

    Returns
    -------
    a series with the weighted sum of robustness for each interaction type
    """
    result = {}
    for k, interaction in [
        ("overall", pd.concat(list(interactions.values()))),
        *interactions.items(),
    ]:
        weights = technique(interaction=interaction, names=robustness.index)
        result[k] = (robustness * weights).mean() / weights.mean()
    return pd.Series(result)


def robustness_analysis(robustness):
    """
    Return several series with the weighted sum of robustness for each interaction
        type, using different techniques, and using either all or just previously
        known interactions.
    """
    universal = universal_set()
    robustness = robustness.loc[[x in universal for x in robustness.index]]
    interactions = load_interactions()
    known_interactions = {k: v[v.known == 1] for k, v in interactions.items()}
    table = pd.DataFrame(
        {
            f"{weight_type} [{inter_name}]": weighted_sums(
                inter, robustness, weight_technique
            )
            for (weight_type, weight_technique) in [
                ("by count", count_interactions),
                ("by presence/absence", has_interaction),
            ]
            for (inter_name, inter) in [
                ("all", interactions),
                ("known", known_interactions),
            ]
        }
    )
    return table, robustness.mean()


@permacache("modular_splicing/width_experiments/lang_rbp_complex/universal_set")
def universal_set():
    """
    Produce the universal set of motifs in this experiment. Using table 2,
        which is a full set of experiments.
    """
    result = pd.read_excel(
        "../data/lang-rbp-relationships/Table S2_Screen_results.xlsx"
    )
    return {*result["Protein A"], *result["Protein B"]}
