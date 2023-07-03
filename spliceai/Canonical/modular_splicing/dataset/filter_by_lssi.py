import numpy as np
import torch
from modular_splicing.dataset.additional_data import AdditionalData
from modular_splicing.example_figure.run_models import compute_thresholds
from modular_splicing.models.modules.lssi_in_model import BothLSSIModelsJustSplicepoints


class FilterByIncorrectLSSI(AdditionalData):
    """
    Filter out all the sites which are predicted correctly by the LSSI model.

    This is useful for training models to predict only on hard sites. Should
        be used as a mask.


    Parameters:
        path_acceptor, path_donor: paths to the LSSI models
        species: species of the LSSI models
        multiplier: multiplier for the LSSI threshold, applied in log-space.
            Higher multiplier = looser threshold.
        clip_real_outputs: whether to clip the real outputs to 0 or 1
    """

    def __init__(
        self, path_acceptor, path_donor, species, multiplier, clip_real_outputs=True
    ):
        self.spm = BothLSSIModelsJustSplicepoints.from_paths(
            path_acceptor, path_donor, 0
        ).eval()
        self.thresholds = multiplier * compute_thresholds(self.spm.cuda(), species)
        self.clip_real_outputs = clip_real_outputs

    def compute_additional_input(self, original_input, path, i, j):
        raise NotImplementedError

    def __call__(self, el, path, i, j, *, cl_max, target):
        x = el["inputs"]["x"]
        y = el["outputs"]["y"][:, 1:]
        device = next(self.spm.parameters()).device
        x = torch.from_numpy(x).to(device)
        with torch.no_grad():
            splicepoints = self.spm(x[None])[0, :, 1:].cpu().numpy()
        yps = splicepoints > np.log(self.thresholds)
        yps = yps[cl_max // 2 : yps.shape[0] - cl_max // 2]

        if self.clip_real_outputs:
            y = y[:, : yps.shape[1]]

        mask = np.zeros(yps.shape[0], dtype=np.bool)
        # keep all the sites that are not real splice sites
        mask |= ~y.any(-1)
        # keep all the sites that are mispredicted
        mask |= (y != yps).any(-1)

        return mask
