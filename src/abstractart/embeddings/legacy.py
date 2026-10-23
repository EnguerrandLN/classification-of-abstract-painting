"""Loaders for the v1 (school project) feature caches, re-aligned on filenames.

They are only kept as baselines for the benchmark; the v1 notebooks that produced
them live in the `v1-projet-ecole` git branch. Row order of the `.npy` caches was
never stored: like the v1 fusion notebook, we assume `sorted(os.listdir)`.
"""

import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .. import ROOT
from ..data import list_images

V1_DIR = ROOT / "legacy" / "v1_features"
CNN_GRAM_PATH = V1_DIR / "gram_vgg16_pca50.npy"      # cnn/cnn_features_pca.npy in v1
VAE_RESNET_PATH = V1_DIR / "vae_resnet50.npy"        # latent_space/fused_vectors.npy in v1
HANDCRAFTED_PATH = V1_DIR / "handcrafted.pkl"        # features/features_brutes.pkl in v1

# Column layout of the 1924 handcrafted features (order of extract_all_features in v1 features.ipynb)
HOG_COLS = range(285, 1853)
# taille_octets, taille_par_pixel, taille_log, ratio_compression: JPEG/file properties, not style
FILE_SIZE_COLS = [1856, 1857, 1858, 1921]


def _reindex(X: np.ndarray, names: list[str], filenames: list[str]) -> np.ndarray:
    pos = {n: i for i, n in enumerate(names)}
    return X[[pos[f] for f in filenames]]


def load_legacy(filenames: list[str]) -> dict[str, np.ndarray]:
    """Return the v1 representations, rows ordered like `filenames`."""
    sorted_names = list_images()
    cnn = _reindex(np.load(CNN_GRAM_PATH), sorted_names, filenames)
    fused = _reindex(np.load(VAE_RESNET_PATH), sorted_names, filenames)

    with open(HANDCRAFTED_PATH, "rb") as f:
        feats, names = pickle.load(f)
    hand = _reindex(np.nan_to_num(np.stack(feats)), list(names), filenames)
    drop = set(HOG_COLS) | set(FILE_SIZE_COLS)
    hand_clean = hand[:, [i for i in range(hand.shape[1]) if i not in drop]]

    # fused_vectors.npy = [0.7 * VAE (128) | 1.3 * ResNet50 PCA (128)]
    vae, resnet = fused[:, :128], fused[:, 128:]

    # Exact reproduction of the shipped v1 fusion (melange_features_CNN.ipynb)
    n = cnn.shape[1]
    geo_red = PCA(n_components=n).fit_transform(StandardScaler().fit_transform(hand))
    vae_red = PCA(n_components=n).fit_transform(StandardScaler().fit_transform(fused))
    v1_fusion = np.hstack((geo_red * 0.2, cnn * 0.6, vae_red * 0.2))

    return {
        "v1/gram_vgg16": cnn,
        "v1/resnet50": resnet,
        "v1/beta_vae": vae,
        "v1/handcrafted_raw": hand,
        "v1/handcrafted_clean": hand_clean,
        "v1/fusion_shipped": v1_fusion,
    }
