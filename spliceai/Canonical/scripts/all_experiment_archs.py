import os
import tempfile
import subprocess
import json
import tqdm.auto as tqdm


def load_architecture(path):
    overall_path = f"experiments/{path}"
    if not path.endswith(".py"):
        return None
    if path in {"msp-266a1.py", "msp.py"}:
        return None
    if not os.path.exists(overall_path):
        return None
    with open(overall_path) as f:
        x = f.read().strip().split("\n")
    last = x.pop()
    if "msp.run_binarizer" in last:
        return None
    assert last == "msp.run()"
    x.append("import json; print(json.dumps(msp.architecture))")
    x = "\n".join(x)

    with tempfile.NamedTemporaryFile() as f:
        f.write(x.encode("utf-8"))
        f.flush()
        result = subprocess.check_output(
            f"PYTHONPATH=experiments python {f.name} 1", shell=True
        )

    result = result.decode("utf-8").strip()
    return json.loads(result)


def all_subdictionaries(x):
    if isinstance(x, dict):
        yield x
        for v in x.values():
            for y in all_subdictionaries(v):
                yield y


def bad(d):
    return d.get("type", None) == "BothLSSIModels" and "sparse_spec" in d


paths = os.listdir("experiments")
bads = []
for path in tqdm.tqdm(paths):
    print(path)
    try:
        arch = load_architecture(path)
    except Exception as e:
        print(e)
        bads.append(path)
        continue
    if arch is None:
        continue
    if any(bad(d) for d in all_subdictionaries(arch)):
        print("BAD")
        bads.append(path)

print(bads)
