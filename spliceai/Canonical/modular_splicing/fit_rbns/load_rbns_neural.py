import os
import tqdm.auto as tqdm
from permacache import stable_hash

from modular_splicing.motif_names import get_motif_names
from modular_splicing.legacy.remapping_pickle import permacache_with_remapping_pickle
from modular_splicing.models.motif_model_stub import EclipMatchingModelMotifModelStub
from modular_splicing.utils.io import load_model


def load_rbns_models_for_evaluation(just_nfm):
    """
    Load the models we will be using for evaluation.

    Specifically, the rbnsn 21x2 models (NFM) as well as the directly trained models (labeled `chenxi`).

    Returns:
        A dict mapping model names to models.
    """
    models = {
        f"NFM_{i}": load_nfm_model(f"model/rbns-binary-model-{{motif}}-21x2_{i}")
        for i in (1, 2, 3, 4)
    }
    if not just_nfm:
        models.update(
            {
                "chenxi_sbm_w11_1": load_chexni_rbns_models(
                    "model/single_binary_model_w_11_all79/"
                ),
                "chenxi_sbm_w14_updated_1": load_chexni_rbns_models(
                    "model/single_binary_model_w_14_updated_all79/"
                ),
            }
        )
    return models


@permacache_with_remapping_pickle(
    "modular_splicing/fit_rbns/load_rbns_model/load_nfm_model_4"
)
def load_nfm_model(path_format, out_cl=400, sparsity=0.178e-2):
    print("Loading NFM model from", path_format)

    def load_fn(mot):
        mot_model_path = path_format.format(motif=mot)
        step, model = load_model(mot_model_path)
        assert step is not None, mot_model_path
        return model.motif_model.eval().cpu().cuda()

    return load_with_by_name_function(
        "rbns_using_am", load_fn, out_cl, sparsity, names_source="rbns"
    )


def load_chexni_rbns_models(root_path, out_cl=400, sparsity=0.178e-2):
    """
    Load the rbns models directly from the chenxi models at the given path.
    """
    mod = load_rbns_models(root_path=root_path)
    mod = {k: v.eval() for k, v in mod.items()}
    for m in mod.values():
        # added when we added the condidtion that they are in eval mode
        m.version = 2
    return _load_chenxi_rbns_models_from_models(mod, out_cl, sparsity)


@permacache_with_remapping_pickle(
    "modular_splicing/fit_rbns/_load_chenxi_rbns_models_from_models_3",
    key_function=dict(mod=stable_hash),
)
def _load_chenxi_rbns_models_from_models(mod, out_cl, sparsity):
    """
    Like load_chenxi_rbns_models_from_models, but with a permacache.
    """
    return load_with_by_name_function(
        "rbns_chenxi_fixed",
        get_am=lambda k: mod[k],
        out_cl=out_cl,
        sparsity=sparsity,
        names_source="rbns",
    )


def load_with_by_name_function(model_type, get_am, out_cl, sparsity, *, names_source):
    """
    Load the models using the given function to get the models by name.
    """
    mnames = get_motif_names(names_source)
    amms = [get_am(k) for k in mnames]
    return EclipMatchingModelMotifModelStub(
        out_cl,
        amms,
        sparsity,
        model_type=model_type,
        pbar=tqdm.tqdm,
    ).cuda()


def load_rbns_models(root_path, *, load=True, exclude_names=()):
    """
    Load all the models from the given folder, associate each with the key it appears under the folder by.

    Disables the gradient for each model.

    Parameters
    ----------
    root_path : str
        The path to the folder containing the models.
    load : bool
        Whether to load the models from disk. If not, just return the names of each, associated with None.
    exclude_names : List[str]
        The names of models to exclude from the returned dictionary.
        All of these must exist in the folder.
    """

    modules = {}
    removed = set()
    for path in os.listdir(root_path):
        assert path.startswith("model_")
        key = path[len("model_") :]
        if key in exclude_names:
            removed.add(key)
            continue
        path = os.path.join(root_path, path, "0")
        if load:
            _, m = load_model(path)
            modules[key] = m
            for p in m.parameters():
                p.requires_grad = False
        else:
            modules[key] = None
    if set(exclude_names) - set(removed) != set():
        raise RuntimeError(f"Extra names: {set(exclude_names) - set(removed)}")
    return modules
