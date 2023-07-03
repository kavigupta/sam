from shelved.spliceai_gtex.alternative_dataset import write_alternative_dataset


write_alternative_dataset(
    files_for_gene_intersection=["spliceai_gtex"],
    data_segment_chunks_to_use=[("train", "all"), ("test", "0")],
    ref_genome_path="/scratch/kavig/hg19.fa",
    out_prefix="../data/canonical_and_gtex_dataset/dataset",
    include_splice_tables_from=["spliceai_canonical", "spliceai_gtex"],
)
