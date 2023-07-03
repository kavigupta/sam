from datetime import datetime
import numpy as np
import torch
import os
import time

from constants import get_args

from motif_model import (
    MotifModel,
    load_model, 
    save_model,
    MotifModelDataset,
    evaluate_model,
)

from load_psam_motif import read_motifs
from map_protein_name import get_map_protein

def equal_length(x):
    # print(x.shape)
    # print(x)
    unique_x = torch.unique(x)
    res = unique_x.shape[0]
    # print(x)
    # print(unique_x)
    # print(res)
    if res == 1 and x[0] >= 20:
        return True
    else:
        return False

def main(bs, path, data_dir, name, n_epochs, map_index, lr):
    # args = get_args()
    # bs = args.bs
    # path = args.model_path
    # name = args.protein
    # data_dir = args.data_dir

    # path = 'result_raw_20'
    torch.set_num_threads(1)

    skip_to_step, m = load_model(path, name)

    if m is None:
        m = MotifModel(num_proteins=2)
        skip_to_step = 0
    m = m.cuda()
    print(m)
    # exit(0)

    dtrain = MotifModelDataset(
        path=f"{data_dir}/rbns_train.h5",
        protein=name,
        map_index=map_index,
    )
    deval = MotifModelDataset(
        path=f"{data_dir}/rbns_test.h5",
        protein=name,
        map_index=map_index,
    )

    if skip_to_step >= len(dtrain)* n_epochs - 1:
        return 

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)

    step = 0

    print('start training')
    start_time = time.time()
    for e in range(n_epochs):
        d = dtrain.loader(bs)

        for i, (x, y, f, l) in enumerate(d):

            x = x.cuda()
            f = f.cuda()
            y = y.cuda()
            l = l.cuda()

            if step < skip_to_step:
                step += bs
                continue

            step += bs
            yp = m(x, f, l)

            loss = criterion(yp, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 10000 == 0 or i == len(d) - 1:
                save_model(m, path, name, step)
                print('saved')
                # print(y)
                train_acc = evaluate_model(m, dtrain, name, limit=10000, quiet=True)
                print(f"train acc: {train_acc}")
                test_acc = 0.0 # evaluate_model(m, deval, name, limit=float("inf"), quiet=True)
                print(
                        "[{}] s={}, e={}/{}, it={}/{}, loss={:.4f}, train-acc={:.4f}, test-acc={:.4f}%".format(
                        datetime.now(),
                        step,
                        e,
                        n_epochs,
                        i,
                        len(d),
                        loss.item(),
                        train_acc * 100,
                        test_acc * 100,
                    )
                )
                with open(f"rbns_model_binary_results.txt", 'a') as f:
                    f.write(
                            "[{}] s={}, e={}/{}, it={}/{}, loss={:.4f}, train-acc={:.4f}, test-acc={:.4f}%".format(
                        datetime.now(),
                        step,
                        e,
                        n_epochs,
                        i,
                        len(d),
                        loss.item(),
                        train_acc * 100,
                        test_acc * 100,
                        )
                    )
                    f.write("\n")
                    f.close()
        if e >=6:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.5


# main()

def run():
    args = get_args()
    bs = args.bs
    path = args.model_path
    data_dir = args.data_dir
    n_epochs = args.n_epochs
    lr = args.lr
    protein_test = args.protein_test

    map_index = get_map_protein()

    with open(f"rbns_model_binary_results.txt", 'w') as f:
        f.close()

    for i in range(81):
        if i != protein_test:
            continue
        with open(f"rbns_model_binary_results.txt", 'a') as f:
            f.write(f"The protein idx: {i}.\n")
            f.close()
        main(bs=bs, path=path, name=i, data_dir=data_dir, n_epochs=n_epochs, map_index=map_index, lr=lr)


if __name__ == "__main__":
    run()



