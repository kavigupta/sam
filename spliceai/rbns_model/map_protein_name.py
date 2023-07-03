import re

def get_map_protein():
    protein_psam_name_dict = {}
    protein_rbns_name_dict = {}
    
    with open('intermediate_data/protein_psam_name.txt', 'r') as f:
        # f.readline()
        idx = 0
        for line in f:
            psam_name = re.split(',|\n', line)[:-1][0]
            # print(psam_name)
            protein_psam_name_dict[psam_name] = idx
            idx += 1
    
    with open('intermediate_data/protein_rbns_name.txt', 'r') as f:
        # f.readline()
        idx = 1 # 0 rbns starts from 1
        for line in f:
            protein_rbns_name_dict[re.split(',|\n', line)[:-1][0]] = idx
            idx += 1
    
    map_dict = dict()
    
    for psam_name in protein_psam_name_dict:
        flag = 0
        for rbns_name in protein_rbns_name_dict:
            if psam_name == rbns_name:
                flag = 1
                # print(psam_name, rbns_name)
                if protein_rbns_name_dict[rbns_name] in map_dict:
                    map_dict[protein_rbns_name_dict[rbns_name]].append(protein_psam_name_dict[psam_name])
                else:
                    # map_dict[protein_rbns_name_dict[rbns_name]] = protein_psam_name_dict[psam_name]
                    map_dict[protein_rbns_name_dict[rbns_name]] = [protein_psam_name_dict[psam_name]]
            # elif psam_name in rbns_name or rbns_name in psam_name:
            #     print(f"NNNNN---- {psam_name}, {rbns_name}")
            # else:
            #     continue
        # if flag == 0:
        #     print(f"not found! {psam_name}")
    
    # print(len(map_dict))
    # print(4 in map_dict)
    # print(map_dict)
    return map_dict


def get_rbns_name_psam_idx_map():
    f = open("intermediate_data/psam_idx_rank_by_enrichment.txt", 'r')
    f.readline()
    rbns_name_psam_idx_dict = dict()

    for line in f:
        content = line[:-1].split(", ")
        rbns_name, psam_idx = content[0], int(content[1])
        rbns_name_psam_idx_dict[rbns_name] = psam_idx
    
    f.close()
    return rbns_name_psam_idx_dict


if __name__ == "__main__":
    # map_dict = get_map_protein()

    enrichment_dict = {}
    with open('intermediate_data/enrichment_pentamer.txt', 'r') as f:
        for line in f:
            content = line[:-1].split(', ')
            enrichment_dict[content[0]] = float(content[3])
        f.close()
    # print(enrichment_dict)

    protein_psam_name_dict = {}
    protein_rbns_name_dict = {}
    
    with open('intermediate_data/intermediate_dataprotein_psam_name.txt', 'r') as f:
        # f.readline()
        idx = 0
        for line in f:
            psam_name = re.split(',|\n', line)[:-1][0]
            # print(psam_name)
            protein_psam_name_dict[psam_name] = idx
            idx += 1
    
    # with open('protein_rbns_name.txt', 'r') as f:
    #     # f.readline()
    #     idx = 1 # 0 rbns starts from 1
    #     for line in f:
    #         protein_rbns_name_dict[re.split(',|\n', line)[:-1][0]] = idx
    #         idx += 1
    
    f = open("intermediate_data/psam_idx_rank_by_enrichment.txt", 'w')
    f.write("rbns_name, psam_idx\n")
    psam_idx_enrichment = dict()
    for k, v in sorted(enrichment_dict.items(), key=lambda item: item[1], reverse=True):
        rbns_name = k.split('-')[0]
        idx = -1
        for psam_name in protein_psam_name_dict:
            if psam_name == rbns_name:
                idx = protein_psam_name_dict[psam_name]
        f.write(f"{rbns_name}, {str(idx)}\n")
    f.close()




    


