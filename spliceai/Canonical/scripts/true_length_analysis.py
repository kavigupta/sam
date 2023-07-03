from modular_splicing.width_experiments.flank_robustness import (
    models_to_analyze,
    compute_all_flank_robustnesses,
    genome_data,
    random_data,
)

models = models_to_analyze()

results_genome = compute_all_flank_robustnesses(
    models, genome_data, list(range(11, 1 + 21))
)
results_random = compute_all_flank_robustnesses(
    models, random_data, list(range(11, 1 + 21))
)
