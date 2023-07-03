import os
import re
import shutil
import subprocess
import tempfile
import unittest

from permacache import stable_hash

from modular_splicing.utils.io import load_model


def regression_hash(experiment, seed, do_eval_limit):
    path = f"model/just_for_testing_{seed}"
    try:
        os.makedirs(path + "/model")

        source_step = 0
        if experiment == "msp-262da5":
            source_step = 75150 - 150
            shutil.copy(
                f"model/msp-262da5_2/model/{source_step + 150}",
                path + f"/model/{source_step}",
            )

        with open(f"experiments/{experiment}.py") as f:
            code = f.read()
        code = replace_guaranteed(code, "argv[1]", str(seed))
        code = replace_guaranteed(code, "__file__", "'just_for_testing'")
        code = code.strip().split("\n")
        assert code.pop() == "msp.run()"
        eval_limit = "--eval-limit 1" if do_eval_limit else ""
        code.append(
            f'msp.extra_params += " --train-limit 1 {eval_limit} --train-cuda 0"'
        )
        code.append("msp.run()")
        code = "\n".join(code)
        with tempfile.NamedTemporaryFile(suffix=".py") as f:
            with open(f.name, "w") as f:
                f.write(code)
            x = subprocess.check_output(
                f"PYTHONPATH=experiments python {f.name}", shell=True
            )
            x = x.decode("utf-8")
            x = x.split("\n")
            print(x)
            x = x[x.index(f"Step {source_step}") :]
            x = "\n".join(x)
        result = stable_hash(load_model(path))
        return result, x
    finally:
        shutil.rmtree(path)


def replace_guaranteed(s, x, y):
    assert x in s
    return s.replace(x, y)


class TrainRegressionTest(unittest.TestCase):
    def test_hashes(self):
        results = {
            "msp-163b1": (
                "90f1ecdca1e5a3d202e476875949a218c9ac2dc48185e514e429d688c9ff421d",
                "Step 0\n[2023-05-17 10:57:11.189940] s=15, e=0/20, it=0/10848, loss=3.9256e-01, val_acc=0.00% {0.00%; 0.00%}\nThreshold: 80.0\n",
            ),
            "msp-226aa1": (
                "6ff094c79c9cd9ee0a950f51d1df2795c6f0947a8ae6936189acfe44b4342cc1",
                "Step 0\n[2023-05-17 10:58:26.302671] s=15, e=0/20, it=0/10848, loss=4.2165e-01, val_acc=0.00% {0.00%; 0.00%}\nThreshold: 80.0\n",
            ),
            "msp-226bbz1": (
                "8b6b039c4e835232445d09e7da0e5f6cede8ffb909cda3d24c6eb8441174ee14",
                "Step 0\n[2023-05-17 10:59:35.839382] s=15, e=0/20, it=0/10848, loss=1.9067e+00, val_acc=0.00% {0.00%; 0.00%; 0.00%}\nThreshold: 100.0\n",
            ),
            "msp-262da5": (
                "70658a7193e65a064cfd26d84950f457da91c7046c217fd3b6d083fb5c8d8fd2",
                "Step 75000\n[2023-05-17 11:03:57.735948] s=75150, e=0/10, it=500/1085, loss=6.8358e-04, val_acc=24.47% {0.04%; 24.47%}\nThreshold: 80.0\n",
            ),
            "msp-277ab.895a3": (
                "f887ac1e03db8331a9434d68cc3bfc457134dd29fa503ca8d905877ad99d287d",
                "Step 0\n[2023-05-17 11:05:22.954016] s=15, e=0/40, it=0/10848, loss=1.6918e+00, val_acc=0.00% {0.00%; 0.00%}\nThreshold: 89.5\n",
            ),
        }

        date_match = r"\[\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d\.\d\d\d\d\d\d\]"

        actual = {
            model: regression_hash(
                model, 113847, do_eval_limit=model not in {"msp-262da5"}
            )
            for model in results
        }
        print(actual)
        for model in results:
            self.assertEqual(actual[model][0], results[model][0], model)
            self.assertEqual(
                re.sub(date_match, "[DATE_AND_TIME]", actual[model][1].strip()),
                re.sub(date_match, "[DATE_AND_TIME]", results[model][1].strip()),
                model,
            )
