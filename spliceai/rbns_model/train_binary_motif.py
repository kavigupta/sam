from datetime import datetime
import numpy as np
import torch
import os
import time

from constants import get_args

from binary_motif_model import (
    MotifModel,
    load_model, 
    save_model,
    MotifModelDataset,
    evaluate_model,
)

from map_protein_name import (
    get_map_protein,
    get_rbns_name_psam_idx_map,
)


def main(
    bs, 
    path, 
    data_dir, 
    protein_psam_idx,
    protein_rbns_name,
    n_epochs, 
    lr,
    l, 
    window_size, 
    save, 
    shuffle_by_group,
    acc_path,
    protein_name,
    ):
    if protein_psam_idx < 0.0:
        protein = protein_rbns_name
    else:
        protein = protein_psam_idx


    torch.set_num_threads(1)

    skip_to_step, m = load_model(path, protein_name)

    if m is None:
        m = MotifModel(hidden_size=l, window_size=window_size)
        skip_to_step = 0
    # else:
    #     print(f"trained.")
    #     f = open(acc_path, 'a')
    #     f.write(f"trained.\n")
    #     f.close()
    #     return 
    m = m.cuda()
    # print(m)
    # exit(0)

    dtrain = MotifModelDataset(
        path=f"{data_dir}/protein_{protein}/rbns_train.h5",
        shuffle_by_group=shuffle_by_group,
    )
    deval = MotifModelDataset(
        path=f"{data_dir}/protein_{protein}/rbns_test.h5",
        shuffle_by_group=shuffle_by_group,
    )

    if skip_to_step >= len(dtrain)* n_epochs - 1:
        return 

    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(m.parameters(), lr=lr)
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)

    step = 0

    start_time = time.time()
    train_end_flag = False

    max_train_acc = 0.0
    max_test_acc = 0.0
    for e in range(n_epochs):
        if train_end_flag:
            break
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

            if i == len(d) - 1:
                #     save_model(m, path, protein, step)
                # print('saved')
                # print(y)
                train_acc = evaluate_model(m, dtrain, limit=float("inf"), quiet=True)
                test_acc = evaluate_model(m, deval, limit=float("inf"), quiet=True)
                # print(
                #         "[{}] s={}, e={}/{}, it={}/{}, loss={:.4f}, train-acc={:.4f}, test-acc={:.4f}%".format(
                #         datetime.now(),
                #         step,
                #         e,
                #         n_epochs,
                #         i,
                #         len(d),
                #         loss.item(),
                #         train_acc * 100,
                #         test_acc * 100,
                #     )
                # )
                if save:
                    if train_acc >= max_train_acc and test_acc >= max_test_acc:
                        print(f"saved.")
                        max_train_acc, max_test_acc = train_acc, test_acc
                        save_model(m, path, protein_name, 0)

                # for time saving
                # if abs(last_train_acc-train_acc) < 0.002 and train_acc > 0.60:
                #     if save:
                #         save_model(m, path, protein, step)
                #         train_end_flag = True
                #         f = open(acc_path, 'a')
                #         f.write("protein: {protein}; train: {train_acc}; test: {test_acc}\n")
        if e >= 6 and e % 3 == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.5
    
    if train_end_flag == False:
        if save:
            # save_model(m, path, protein_name, 0)
            train_end_flag = True
            f = open(acc_path, 'a')
            f.write(f"protein: {protein_name}; train: {max_train_acc}; test: {max_test_acc}\n")
            print(f"protein: {protein_name}; train: {max_train_acc}; test: {max_test_acc}")


# main()

def run():
    args = get_args()
    bs = args.bs
    path = args.model_path
    data_dir = args.data_dir
    n_epochs = args.n_epochs
    lr = args.lr
    l = args.l
    window_size = args.window_size
    save = args.save
    shuffle_by_group = args.shuffle_by_group
    partition = args.partition
    protein_name = args.protein_name

    # left = partition * 17
    # right = (partition + 1) * 17

    rbns_name_psam_idx_dict = get_rbns_name_psam_idx_map()
    
    i = 0
    for protein_rbns_name, protein_psam_idx in sorted(rbns_name_psam_idx_dict.items()):
        # if protein_name is not None:
        #     if protein_name != protein_rbns_name:
        #         continue
        # elif i != partition:
        #     i += 1
        #     continue
        # if protein_rbns_name == "RBFOX3":
        #     i += 1
        #     continue
        # if i < right and i >= left:
        #     i += 1
        # else:
        #     i += 1
        #     continue

        acc_path = f"rbns_model_binary_results/single_proteins_w_11/rbns_model_binary_results_{window_size}_{data_dir}_{protein_rbns_name}.txt"
        if os.path.isfile(acc_path):
            continue
        with open(acc_path, 'w') as f:
            f.close()

        print(f"{protein_rbns_name}, {protein_psam_idx}")
        f = open(acc_path, 'a')
        f.write(f"{protein_rbns_name}, {protein_psam_idx}\n")
        f.close()

        main(
            bs=bs, 
            path=path, 
            protein_psam_idx=protein_psam_idx,
            protein_rbns_name=protein_rbns_name,
            data_dir=data_dir, 
            n_epochs=n_epochs, 
            lr=lr, 
            l=l, 
            window_size=window_size, 
            save=save,
            shuffle_by_group=shuffle_by_group,
            acc_path=acc_path,
            protein_name=protein_rbns_name,
        )
        i += 1

if __name__ == "__main__":
    run()



