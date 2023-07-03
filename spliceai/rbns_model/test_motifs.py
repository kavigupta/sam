from tqdm import tqdm

from motif_model import load_model, evaluate_model, MotifModel, MotifModelDataset

def main():
    bs = 512
    p = 'result_raw_20/'

    m = load_model(p)[1]

    deval = MotifModelDataset(
        'data/rbns_test.h5'
    )

    evaluate_model(m, deval, limit=float("inf"), bs=bs, quiet=False)


main()
