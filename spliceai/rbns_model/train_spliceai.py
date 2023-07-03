from datetime import datetime
import numpy as np
import torch
import os
import time

from constants import get_args

from spliceai_torch import (
    SpliceAI,
    load_model, 
    save_model,
    SpliceAIDataset,
    evaluate_model,
)

from load_psam_motif import read_motifs
from map_protein_name import get_map_protein

from hyperparams import get_hparams

from load_rbns_motif import (
    rbns_motifs,
    rbns_motifs_for, 
)


def main(
    window, 
    CL_max, 
    l,
    lr, 
    bs, 
    motifs, 
    sparsity,
    data_dir,
    organism,
    SL, 
    path,
    n_epochs,
    num_motifs,
    use_motifs, 
    use_splice_site,
    ):

    torch.set_num_threads(1)
    attr = f"{sparsity}_{bs}_{lr}_{l}_{organism}"

    skip_to_step, m = load_model(path, attr)

    if m is None:
        w, ar, _, cl = get_hparams(window, CL_max)
        if use_motifs:
            if use_splice_site:
                m = SpliceAI(l=l, w=w, ar=ar, motifs=motifs, preprocess=rbns_motifs_for, \
                    starting_channels=6, use_splice_site=use_splice_site)
            else:
                m = SpliceAI(l=l, w=w, ar=ar, motifs=motifs, preprocess=rbns_motifs_for, sparsity=sparsity, starting_channels=num_motifs)
        else:
            m = SpliceAI(l=l, w=w, ar=ar)
        skip_to_step = 0
    m = m.cuda()
    print(m)

    dtrain = SpliceAIDataset.of(
        data_dir + "/" + organism + "_" + "dataset" + "_" + "train" + "_" + "all" + ".h5",
        cl=window,
        cl_max=CL_max,
        sl=SL,
    )

    deval = SpliceAIDataset.of(
        data_dir + "/" + organism + "_" + "dataset" + "_" + "test" + "_" + "all" + ".h5",
        cl=window,
        cl_max=CL_max,
        sl=SL,
    )


    if skip_to_step >= len(dtrain)* n_epochs - 1:
        return 

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)

    step = 0

    start_time = time.time()
    max_test_acc = [0.0, 0.0]
    for e in range(n_epochs):
        d = torch.utils.data.DataLoader(dtrain, num_workers=0, batch_size=bs)

        for i, (x, y) in enumerate(d):

            x = x.cuda()
            y = y.cuda()

            if step < skip_to_step:
                step += bs
                continue

            step += bs
            yp = m(x)
            # print(e, i, x.shape, y.shape, yp.shape)

            loss = criterion(yp.reshape([-1, 3]), y.reshape([-1]))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 1000 == 0 or i == len(d) - 1:
                # print(y)
                train_acc = evaluate_model(m, dtrain, bs=128, limit=60000, quiet=True)
                # print(f"train acc: {train_acc}")
                test_acc = evaluate_model(m, deval, bs=128, limit=float("inf"), quiet=True)
                print(
                        "[{}] s={}, e={}/{}, it={}/{}, loss={:.4f}, train-acc={}, test-acc={}".format(
                        datetime.now(),
                        step,
                        e,
                        n_epochs,
                        i,
                        len(d),
                        loss.item(),
                        train_acc,
                        test_acc,
                    )
                )
                if sum(test_acc)/len(test_acc) > sum(max_test_acc)/len(max_test_acc):
                    save_model(m, path, attr, step)
                    print('saved')
                    max_test_acc = test_acc

        if e >= 6 and e % 3 == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.5


def run():
    args = get_args()

    main(
    window=args.window, 
    CL_max=args.CL_max, 
    l=args.l,
    lr=args.lr, 
    bs=args.bs, 
    motifs=rbns_motifs(psam_only=args.psam_only, splice_site_only=args.use_splice_site), 
    sparsity=args.sparsity,
    data_dir=args.data_dir,
    organism=args.organism,
    SL=args.SL,
    path=args.model_path,
    n_epochs=args.n_epochs,
    num_motifs=args.num_motifs,
    use_motifs=args.use_motifs,
    use_splice_site=args.use_splice_site,
    )


if __name__ == "__main__":
    run()



