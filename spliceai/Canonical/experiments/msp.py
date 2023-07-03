import json
import os
import shlex

import numpy as np

with open("../data/rbnsp_motifs_split.json") as f:
    rbnsp_split_half = json.load(f)

with open("../data/rbnsp_motifs_split_90.json") as f:
    rbnsp_split_90 = json.load(f)

with open("../data/rbnsn_motifs_split_90.json") as f:
    rbnsn_split_90 = json.load(f)

with open("../data/am_1-to-3-motif_widths_above_90_shorter_donor.json") as f:
    am_one_to_3_motif_widths_above_90_shorter_donor = json.load(f)

am_one_to_3_motif_widths_above_90_shorter_donor_shuffled = (
    am_one_to_3_motif_widths_above_90_shorter_donor[:]
)
np.random.RandomState(0).shuffle(
    am_one_to_3_motif_widths_above_90_shorter_donor_shuffled
)


class MSP:
    seed = 1
    file = None
    SL = 5000
    window = 400
    lr = "5e-5"
    batch_size = 15
    n_epochs = 20
    decay_per_epoch = 0.9
    sparsity_start = 0.75
    sparsity_update = 0.75

    acc_thresh = 80

    data_dir = f"./"

    data_spec = None

    train_spec = None

    evaluation_criterion_spec = None

    stop_at_density = None

    architecture = dict(
        type="ModularSplicePredictor",
        num_motifs=100,
        channels=200,
        splicepoint_model_spec=dict(
            type="BothLSSIModels",
            acceptor="model/splicepoint-model-acceptor-1",
            donor="model/splicepoint-donor2-2.sh",
        ),
        motif_model_spec=dict(
            type="LearnedMotifModel",
            motif_width=21,
            motif_fc_layers=5,
            motif_feature_extractor_spec=dict(type="ResidualStack", depth=5),
        ),
        sparse_spec=dict(type="SpatiallySparseByChannel"),
        influence_calculator_spec=dict(
            type="InfluenceValueCalculator",
            post_sparse_spec=dict(type="ResidualStack", width=49, depth=4),
            long_range_reprocessor_spec=dict(type="SingleAttentionLongRangeProcessor"),
        ),
        final_processor_spec=dict(
            type="FinalProcessor",
            post_influence_spec=dict(type="ResidualStack", width=49, depth=4),
        ),
    )

    def sparse_adaptation_params(self):
        return (
            f"--learned-motif-sparsity {self.sparsity_start} --learned-motif-sparsity-update {self.sparsity_update}"
            f" --learned-motif-sparsity-threshold {self.acc_thresh}"
        )

    def train_params(self):
        if self.train_spec is not None:
            return f"--train-mode from-spec --train-spec {shlex.quote(json.dumps(self.train_spec))}"
        else:
            return f"--decay-start 0 --decay-amount {self.decay_per_epoch} --n-epochs {self.n_epochs} --report-frequency 500"

    def motif_params(self):
        if self.data_spec is None:
            return ""
        return f"--dataset-spec" f" {shlex.quote(json.dumps(self.data_spec))}"

    def data_dir_params(self):
        data_dir = self.data_dir
        if "MSP_DATA_DIR" in os.environ:
            data_dir_new = os.environ["MSP_DATA_DIR"]
            if data_dir_new != data_dir:
                print(
                    f"WARNING: overriding intrinsic data dir {data_dir} with {os.environ['MSP_DATA_DIR']}"
                )
                data_dir = data_dir_new
        return f"--data_dir {shlex.quote(json.dumps(data_dir))}"

    def evalutaion_criterion_params(self):
        if self.evaluation_criterion_spec is None:
            return ""
        return f"--evaluation-criterion-spec {shlex.quote(json.dumps(self.evaluation_criterion_spec))}"

    extra_params = ""

    extra_name_for_path = ""

    def name(self):
        assert self.file is not None
        name = os.path.splitext(os.path.basename(self.file))[0]
        return f"{name}{self.extra_name_for_path}_{self.seed}"

    def stop_at_density_params(self):
        if self.stop_at_density is None:
            return ""
        return f"--stop-at-density {self.stop_at_density}"

    def a(self):
        return " ".join(
            [
                f"--seed {self.seed}",
                f"--SL {self.SL}",
                f"--window {self.window}",
                f"--model-path model/{self.name()}",
                self.sparse_adaptation_params(),
                f"--lr {self.lr}",
                self.train_params(),
                self.extra_params,
                f"--msp-architecture-spec {shlex.quote(json.dumps(self.architecture))}",
                self.motif_params(),
                self.data_dir_params(),
                self.evalutaion_criterion_params(),
                self.stop_at_density_params(),
            ]
        )

    def train(self):
        return f"PYTHONPATH=. python -u -m modular_splicing.train.main {self.a()} --batch-size {self.batch_size}"

    def run(self):
        import sys

        technique = sys.argv[2] if len(sys.argv) > 2 else "train"
        result = getattr(self, technique)()
        assert not os.system(result)

    def __setattr__(self, item, value):
        if hasattr(self, item):
            super().__setattr__(item, value)
        else:
            raise RuntimeError("Invalid attribute {}".format(item))

    def run_binarizer_generic(self, compute_step):
        fname = os.path.basename(self.file)
        *items, xk, suffix = fname.split(".")
        assert suffix == "py"
        assert xk[0] == "x"
        model_to_binarize = ".".join(items)
        model_to_binarize = f"model/{model_to_binarize}_{self.seed}"

        step = compute_step(model_to_binarize)

        self.architecture = dict(
            type="DiscretizeMotifModel",
            original_model_path=model_to_binarize,
            original_model_step=step,
            reset_adaptive_sparsity_threshold_manager=True,
        )
        self.run()

    def run_binarizer(self, target_density):
        self.run_binarizer_generic(
            lambda model_to_binarize: step_for_density(
                model_to_binarize, target_density
            )
        )

    def run_binarizer_with_target_num_motifs_dropped(self, target_num_motifs_dropped):
        return self.run_binarizer_generic(
            lambda model_to_binarize: step_for_num_motifs_dropped(
                model_to_binarize, target_num_motifs_dropped
            )
        )

    def run_reconstruction(self, *, path, sparsity, cl):
        self.architecture = dict(
            type="ReconstructSequenceModel",
            original_model_path=path,
            original_model_step=step_for_density(path, sparsity),
            downstream_spec=dict(type="SpliceAIModule"),
            num_motifs=80,
        )

        self.data_spec = dict(
            type="H5Dataset",
            datapoint_extractor_spec=dict(
                type="BasicDatapointExtractor",
                rewriters=[dict(type="ReconstructSequenceDataRewriter")],
                run_argmax=False,
            ),
            post_processor_spec=dict(type="IdentityPostProcessor"),
        )
        self.evaluation_criterion_spec = dict(
            type="ReconstructSequenceEvaluationCriterion"
        )

        self.acc_thresh = 100

        self.window = cl


def setup_as_am(msp):
    msp.architecture["splicepoint_model_spec"] = dict(
        type="BothLSSIModels",
        acceptor="model/splicepoint-model-acceptor-1",
        donor="model/splicepoint-model-donor-1",
    )

    msp.architecture["num_motifs"] = 82
    msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")
    msp.architecture["motif_model_spec"] = dict(
        type="AdjustedMotifModel",
        model_to_adjust_spec=dict(
            type="PSAMMotifModel",
            motif_spec=dict(type="rbns"),
            exclude_names=("3P", "5P"),
        ),
        adjustment_model_spec=msp.architecture["motif_model_spec"],
        sparsity_enforcer_spec=dict(
            type="SparsityEnforcer", sparse_spec=msp.architecture["sparse_spec"]
        ),
    )

    msp.architecture["affine_sparsity_enforcer"] = True
    msp.stop_at_density = 0.17e-2

    return msp


def step_for_density(path, target_density):
    import sys

    sys.path.insert(0, ".")
    from modular_splicing.models_for_testing.load_model_for_testing import (
        step_for_density,
    )

    return step_for_density(path, target_density)


def step_for_num_motifs_dropped(path, target_num_motifs_dropped):
    import sys

    sys.path.insert(0, ".")
    from shelved.auto_minimize_motifs.utils import amm_get_step_with_dropped_motifs

    return amm_get_step_with_dropped_motifs(path, target_num_motifs_dropped)


def starting_point_for_different_number_motifs(*args):
    import sys

    sys.path.insert(0, ".")
    from modular_splicing.utils.entropy_calculations import (
        starting_point_for_different_number_motifs,
    )

    return starting_point_for_different_number_motifs(*args)


amm_mean40v1 = [
    "SRSF2",
    "TRA2A",
    "ZCRB1",
    "ZNF326",
    "RBM6",
    "HNRNPA1",
    "CNOT4",
    "DAZAP1",
    "SRSF8",
    "HNRNPA2B1",
    "SRSF11",
    "IGF2BP1",
    "CELF1",
    "HNRNPK",
    "NUPL2",
    "PTBP3",
    "ILF2",
    "RBM15B",
    "RBM23",
    "DAZ3",
    "TARDBP",
    "RBM25",
    "ESRP1",
    "SFPQ",
    "HNRNPD",
    "KHSRP",
    "EIF4G2",
    "FUS",
    "MSI1",
    "SRSF9",
    "A1CF",
    "RBM45",
    "HNRNPF",
    "RBM24",
    "RBM41",
    "FUBP1",
    "RBMS3",
    "SF1",
    "TIA1",
    "RBFOX3",
]


eclip_91 = [
    "A1CF",
    "BOLL",
    "CELF1",
    "CNOT4",
    "CPEB1",
    "CPEB4",
    "DAZ3",
    "DAZAP1",
    "EIF4G2",
    "ELAVL4",
    "ESRP1",
    "EWSR1",
    "FMR1",
    "FUBP1",
    "FUBP3",
    "FUS",
    "FXR1",
    "FXR2",
    "HNRNPA0",
    "HNRNPA1",
    "HNRNPA2B1",
    "HNRNPC",
    "HNRNPCL1",
    "HNRNPD",
    "HNRNPDL",
    "HNRNPF",
    "HNRNPH2",
    "HNRNPK",
    "HNRNPL",
    "IGF2BP1",
    "IGF2BP2",
    "ILF2",
    "KHDRBS1",
    "KHDRBS2",
    "KHDRBS3",
    "KHSRP",
    "MATR3",
    "MBNL1",
    "MSI1",
    "NOVA1",
    "NUPL2",
    "PABPC4",
    "PABPN1L",
    "PCBP1",
    "PCBP2",
    "PCBP4",
    "PRR3",
    "PTBP1",
    "PTBP3",
    "PUF60",
    "PUM1",
    "QKI",
    "RALY",
    "RBFOX2",
    "RBFOX3",
    "RBM15B",
    "RBM22",
    "RBM23",
    "RBM24",
    "RBM25",
    "RBM4",
    "RBM41",
    "RBM45",
    "RBM47",
    "RBM4B",
    "RBM6",
    "RBMS2",
    "RBMS3",
    "RC3H1",
    "SF1",
    "SFPQ",
    "SNRPA",
    "SRSF1",
    "SRSF10",
    "SRSF11",
    "SRSF2",
    "SRSF4",
    "SRSF5",
    "SRSF7",
    "SRSF8",
    "SRSF9",
    "TAF15",
    "TARDBP",
    "TIA1",
    "TRA2A",
    "TRNAU1AP",
    "U2AF2",
    "UNK",
    "ZCRB1",
    "ZFP36",
    "ZNF326",
]
