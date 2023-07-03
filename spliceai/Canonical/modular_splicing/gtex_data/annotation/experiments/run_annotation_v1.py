import logging
from modular_splicing.gtex_data.pipeline.splice_table import produce_splice_table
from modular_splicing.gtex_data.annotation.compute_optimal_sequence import (
    compute_optimal_sequences_all,
)

logging.getLogger("filelock").setLevel(logging.WARNING)

cost_params = dict(annot_cost=1e-4, other_cost=0.1)

splice_table, name_to_ensg, frac_kept = produce_splice_table()
compute_optimal_sequences_all(sorted(name_to_ensg.values()), cost_params=cost_params)
