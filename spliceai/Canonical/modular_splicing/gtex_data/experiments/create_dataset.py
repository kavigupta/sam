import fire

from modular_splicing.gtex_data.pipeline.marginal_psis_dataset import (
    generate_probabilistic_gtex_dataset,
)


class MarginalPsiV3:
    data_path_folder = "../data/gtex_direct/marginal_psis/v3/"

    @classmethod
    def create(cls):
        return generate_probabilistic_gtex_dataset(
            data_path_folder=cls.data_path_folder,
            fasta_path="/scratch/kavig/hg38.fa",
            ensembl_version=89,
            tissue_id_function=lambda x: x,
            CL_max=10_000,
            SL=5000,
            segment_chunks=[("train", "all"), ("test", "0")],
            psi_computation_spec=dict(
                type="compute_psis_from_annotation_sequence",
                cost_params=dict(annot_cost=1e-4, other_cost=0.1),
            ),
        )


DATASETS = dict(
    MarginalPsiV3=MarginalPsiV3,
)


def create_dataset(name):
    DATASETS[name].create()


if __name__ == "__main__":
    fire.Fire(create_dataset)
