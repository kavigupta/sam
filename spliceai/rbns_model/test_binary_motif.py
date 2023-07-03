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

from utils import write_permute_res

def main(
    bs,
    path,
    data_dir,
    protein_psam_idx,
    protein_rbns_name,
    evaluate_result,
    window_size,
):
    # if window_size == 11:
    #     protein = protein_rbns_name
    # else:
    if protein_psam_idx < 0.0:
        protein = protein_rbns_name
    else:
        protein = protein_psam_idx
    
    # print(path, protein)
    if window_size == 11:
        _, m = load_model(path, protein_rbns_name)
    else:
        _, m = load_model(path, protein)

    if m is None:
        print(f"No model for protein:{protein_rbns_name}, {protein_psam_idx}")
        return 

    deval = MotifModelDataset(
        path=f"{data_dir}/protein_{protein}/rbns_test.h5",
    )

    if evaluate_result:
        evalute_time = time.time()
        acc, permute_dict = evaluate_model(m, deval, limit=float("inf"), quiet=True, evaluate_result=evaluate_result)
        print(acc)
        print(f"finish evaluation: {time.time() - evalute_time} sec")
        permute_path = f"intermediate_data/protein_binding_distribution/nn_{window_size}_{protein_rbns_name}_{protein_psam_idx}.txt"
        write_permute_res(permute_dict, permute_path, acc=acc)
        print(f"finish writing: {protein_rbns_name}, {protein_psam_idx}")
    else:
        evaluate_model(m, deval, limit=float("inf"), quiet=True, evaluate_result=evaluate_result)

    return 


def run():
    args = get_args()
    bs = args.bs
    path = args.model_path
    evaluate_result = args.evaluate_result
    data_dir = args.data_dir
    window_size = args.window_size

    rbns_name_psam_idx_dict = get_rbns_name_psam_idx_map()

    for protein_rbns_name, protein_psam_idx in sorted(rbns_name_psam_idx_dict.items()):
        if  protein_rbns_name != 'RBFOX2':
            continue
        print(protein_rbns_name)
        main(
            bs=bs,
            path=path,
            data_dir=data_dir,
            protein_psam_idx=protein_psam_idx,
            protein_rbns_name=protein_rbns_name,
            evaluate_result=evaluate_result,
            window_size=window_size,
        )
    
    return 




if __name__ == "__main__":
    run()