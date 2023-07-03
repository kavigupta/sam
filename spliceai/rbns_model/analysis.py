import torch
from spliceai_torch import (
    SpliceAI,
    load_model, 
    save_model,
    SpliceAIDataset,
    evaluate_model,
)


def load_dataset(
    data_dir, 
    organism, 
    window, 
    CL_max, 
    SL,
    bs,
    ):

    dtrain = SpliceAIDataset.of(
        data_dir + "/" + organism + "_" + "dataset" + "_" + "train" + "_" + "all" + ".h5",
        cl=window,
        cl_max=CL_max,
        sl=SL,
    )
    d = torch.utils.data.DataLoader(dtrain, num_workers=0, batch_size=bs)

    return d


def read_average_sparsity(d):
    base_num = 0
    acceptor_num = 0
    donor_num = 0

    for i, (x, y) in enumerate(d):
        y_count_acceptor = torch.sum(y==1)
        y_count_donor = torch.sum(y==2)
        y_count_none = torch.sum(y==0)
        y_count_padding = torch.sum(y==-1)
        # print(f"In {y.shape[0]*y.shape[1]} bases, acceptor: {y_count_acceptor}, \
        #     donor: {y_count_donor}, none: {y_count_none}, padding: {y_count_padding}")
        # exit(0)
        base_num += y.shape[0]*y.shape[1]
        donor_num += y_count_donor
        acceptor_num += y_count_acceptor
    
    print(f"In {base_num} bases, acceptor: {acceptor_num}, \
        #     donor: {donor_num}, acceptor_per: {acceptor_num/base_num}, donor_per: {donor_num/base_num}")



if __name__ == "__main__":
    data_dir = 'organism/canonical/400'
    organism = 'canonical'
    window=400
    CL_max=400
    SL=5000
    bs=1
    d = load_dataset(
        data_dir,
        organism,
        window,
        CL_max,
        SL,
        bs,
    )
    read_average_sparsity(d)

