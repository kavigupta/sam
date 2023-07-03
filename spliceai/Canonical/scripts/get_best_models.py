import os
import glob
import sys
import subprocess

_, prefix = sys.argv

results = glob.glob(prefix + "-*")
try:
    os.mkdir(prefix)
except FileExistsError:
    pass

assert os.listdir(prefix) == []

for result in results:
    seed = result.split("-")[-1]
    result = os.path.join(result, "model")
    step = max(os.listdir(result), key=int)
    result = os.path.join(result, step)
    command = ["cp", result, os.path.join(prefix, seed)]
    subprocess.run(command, check=True)
