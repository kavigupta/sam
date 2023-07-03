import unittest
from modular_splicing.evaluation.evaluation_criterion import MultiEvaluationCriterion
from modular_splicing.dataset.generic_dataset import dataset_types
from modular_splicing.evaluation.run_evaluation import evaluate_model_on_data

from modular_splicing.models_for_testing.list import AM, FM
from modular_splicing.dataset.basic_dataset import basic_dataset
from modular_splicing.models.entire_model.reconstruct_sequence import (
    ReconstructSequenceEvaluationCriterion,
)
from modular_splicing.utils.io import load_model
from modular_splicing.utils.construct import construct


class TestEvaluation(unittest.TestCase):
    def test_basic_evaluation(self):
        """
        Test using the default evaluation criterion.
        """
        data = basic_dataset("dataset_test_0.h5", 400, 10_000, sl=5000)

        fm_result = [0.6, 0.7049180327868853]
        fm = FM.non_binarized_model(1).model
        self.assertEqual(
            fm_result,
            evaluate_model_on_data(fm, data, limit=100, bs=10),
        )
        self.assertEqual(
            fm_result,
            evaluate_model_on_data(fm, data, limit=100, bs=50),
        )

        am_result = [0.8, 0.8688524590163934]
        am = AM.non_binarized_model(1).model
        self.assertEqual(
            am_result,
            evaluate_model_on_data(am, data, limit=100, bs=10),
        )
        self.assertEqual(
            am_result,
            evaluate_model_on_data(am, data, limit=100, bs=50),
        )

    def test_reconstruction_evaluation_criterion(self):
        """
        Test using the reconstruction evaluation criterion.
        """
        data = construct(
            dataset_types(),
            dict(
                type="H5Dataset",
                datapoint_extractor_spec=dict(
                    type="BasicDatapointExtractor",
                    rewriters=[dict(type="ReconstructSequenceDataRewriter")],
                    run_argmax=False,
                ),
                post_processor_spec=dict(type="IdentityPostProcessor"),
            ),
            path="dataset_test_0.h5",
            cl=10_000,
            cl_max=10_000,
            sl=5000,
            iterator_spec=dict(
                type="FastIter", shuffler_spec=dict(type="DoNotShuffle")
            ),
        )
        recon_result = [
            0.6346660938611045,
            0.583687642422389,
            0.6402718537614249,
            0.6370083375545912,
        ]
        recon = load_model("model/msp-284a1_1")[1].eval()
        predicted_result = evaluate_model_on_data(
            recon,
            data,
            limit=100,
            bs=10,
            evaluation_criterion=ReconstructSequenceEvaluationCriterion(),
        )
        # assert that they are within 1e-3 of each other
        self.assertEqual(len(predicted_result), len(recon_result))
        self.assertTrue(
            all(
                [
                    abs(predicted_result[i] - recon_result[i]) < 1e-3
                    for i in range(len(predicted_result))
                ]
            )
        )

    def multi_evaluation_criterion_test(self):
        """
        Test using a few multi-evalutaiton criterion.

        These are all used for the GTEx datasets.
        """
        data = construct(
            dataset_types(),
            dict(
                type="UnifiedAlternativeDataset",
                post_processor_spec=dict(type="IdentityPostProcessor"),
                underlying_ordering=["gtex_const_and_alt", "gtex_const"],
            ),
            path="../data/gtex_derived_alt_const_split/1000/uniform/dataset_test_0.h5",
            cl=400,
            cl_max=10_000,
            sl=5000,
            iterator_spec=dict(
                type="FastIter", shuffler_spec=dict(type="DoNotShuffle")
            ),
        )
        mod = load_model("model/msp-271b1_1")[1].eval()

        out = [
            0.5248868778280543,
            0.5211864406779662,
            0.32786885245901637,
            0.2608695652173913,
        ]

        # should not matter what the eval indices are

        def check(indices):
            self.assertEqual(
                out,
                evaluate_model_on_data(
                    mod,
                    data,
                    limit=100,
                    bs=10,
                    evaluation_criterion=MultiEvaluationCriterion(
                        num_channels_per_prediction=3,
                        num_predictions=2,
                        eval_indices=indices,
                    ),
                ),
            )

        check([0, 1])
        check([2, 3])
