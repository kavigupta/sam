from tqdm import tqdm

from psam_model import load_model, evaluate_model, MotifModel, MotifModelDataset
from constants import get_args
from load_psam_motif import read_motifs
from map_protein_name import get_map_protein

from utils import write_permute_res

def main(bs, protein, data_dir, motifs, map_index, threshold_list, evaluate_result):
    bs = bs
    # p = 'result/'

    m = MotifModel(motifs=motifs, num_proteins=81, protein=protein) # load_model(p)[1]

    deval = MotifModelDataset(
        path=f"{data_dir}/protein_{protein}/rbns_test.h5",
    )

    # evaluate_model(m, deval, limit=float("inf"), bs=bs, quiet=False)
    if evaluate_result:
        acc, permute_dict = evaluate_model(m, deval, map_index, threshold_list=threshold_list, limit=float("inf"), bs=bs, quiet=False, evaluate_result=evaluate_result)
        permute_path = f"intermediate_data/protein_binding_distribution/psam_{protein}.txt"
        write_permute_res(permute_dict, permute_path, acc=acc)
        print(f"finish writing: {protein}")
    else:
        evaluate_model(m, deval, map_index, threshold_list=threshold_list, limit=float("inf"), bs=bs, quiet=False, evaluate_result=evaluate_result)
    
    return


def run(varied_threshold_list):
    args = get_args()
    bs = args.bs
    path = args.model_path
    name = args.protein
    data_dir = args.data_dir
    evaluate_result = args.evaluate_result

    print(data_dir)

    motifs = read_motifs()
    map_index = get_map_protein()
    # threshold_list = [0.0, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.3]
    threshold_list = [0.001]

    with open(f"binary_all_acc_tmp.txt", 'w') as f:
        f.close()
    # protein_spec = 46
    for i in range(81):
        if i < 2:
            continue
        if i!=46:
            continue
        # threshold_list[0] = varied_threshold_list[i-2]
        f = open(f"binary_all_acc_tmp.txt", 'a')
        f.write(f"{i}\n")
        f.close()
        print(f"protein: {i}")
        #  if i < protein_spec:
        #    continue
        # if i > protein_spec:
        #    continue
        main(bs=bs, 
        protein=i, 
        data_dir=data_dir, 
        motifs=motifs, 
        map_index=map_index, 
        threshold_list=threshold_list,
        evaluate_result=evaluate_result)


if __name__ == "__main__":
    varied_threshold_list = [0.26501592993736267, 0.3667528033256531, 0.9034467339515686, 1.3102374076843262, 0.7271395921707153, 0.7946707606315613, 1.0418727397918701, 0.13571062684059143, 0.8928019404411316, 0.7193368673324585, 0.10848090797662735, 1.0110220909118652, 0.9353128671646118, 3.3182919025421143, 0.6896530389785767, 0.5995594263076782, 2.4036920070648193, 0.7480984926223755, 0.8586286902427673, 0.5647818446159363, 0.679750382900238, 0.6035876274108887, 0.3247525691986084, 0.4051118493080139, 0.19034485518932343, 0.5185756087303162, 0.13056281208992004, 0.6890026926994324, 0.5814595222473145, 0.26342496275901794, 0.9011789560317993, 0.14632147550582886, 0.3209635615348816, 0.315451443195343, 0.40171322226524353, 1.2246934175491333, 0.12088800966739655, 0.18452097475528717, 0.1965116411447525, 0.0934765487909317, 0.9689454436302185, 0.4177393615245819, 0.39528438448905945, 0.8432025909423828, 0.36221903562545776, 0.492397278547287, 0.7940261960029602, 0.07172971218824387, 1.2466919422149658, 0.32893791794776917, 0.9762769937515259, 0.9464859962463379, 0.3818933665752411, 0.17098087072372437, 0.5645045042037964, 0.21834401786327362, 0.33137068152427673, 0.37143951654434204, 1.3053789138793945, 0.122153639793396, 0.12602993845939636, 0.7502365708351135, 1.1324546337127686, 0.6503979563713074, 1.9860022068023682, 1.242436170578003, 1.3302764892578125, 0.8792383074760437, 0.7416897416114807, 0.9289566278457642, 0.1711372286081314, 1.0214877128601074, 0.976638913154602, 0.17266397178173065, 0.9239301681518555, 0.8019489645957947, 1.1550496816635132, 0.7689653635025024, 0.2402070015668869, 0.04873685538768768]
    run(varied_threshold_list)
