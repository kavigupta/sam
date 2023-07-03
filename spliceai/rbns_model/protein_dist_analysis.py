# analysis for prediction results of PSAMs and NN


def evaluate_model_all(m, d, limit=float("inf"), bs=500, map_index=None):
    try:
        m.eval()
        dataset = d.loader(bs)

        for i, (x, y, f, l) in tqdm(enumerate(dataset)):
            for x, y, f, l in pbar(DataLoader(d, batch_size=bs)):