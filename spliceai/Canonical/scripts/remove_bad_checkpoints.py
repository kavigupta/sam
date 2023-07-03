import os
from sys import argv

from spliceai_torch import load_model

for tag in argv[1:]:
    path = f"model/{tag}"
    print(path)
    step = max(int(x) for x in os.listdir(path + "/model"))
    try:
        load_model(path, step)
        print("good!")
        bad = False
    except:
        bad = True
    if bad == True:
        print("removing")
        os.remove(f"{path}/model/{step}")
