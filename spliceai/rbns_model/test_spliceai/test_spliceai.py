import torch

from constants import get_args

from spliceai_torch import (
    load_model, 
    SpliceAIDataset,
    evaluate_model,
)



def main(
    window, 
    CL_max, 
    bs, 
    data_dir,
    SL, 
    data_file,
    path,
    ):

    torch.set_num_threads(1)

    skip_to_step, m = load_model(path)
    # print(m)

    deval = SpliceAIDataset.of(
        data_dir + "/" + data_file, # Probably needs to CHANGE the file name, or text here.
        cl=window,
        cl_max=CL_max,
        sl=SL,
    )

    evaluate_model(m, deval, bs=bs, limit=float("inf"), quiet=False)


def run():
    args = get_args()

    main(
    window=args.window, 
    CL_max=args.CL_max,
    bs=args.batch_size,
    data_dir=args.data_dir,
    SL=args.SL,
    data_files=args.data_file, # CHANGE this is the path to the .h5 file
    path=args.model_path, # CHANGE
    )


if __name__ == "__main__":
    run()