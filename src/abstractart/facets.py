"""Facet-conditioned spaces: the same painting seen through its colour, texture, form or composition.

SigLIP 2 embeds images and text in one space. For each facet, the embeddings of its vocabulary
terms (centred, so that the shared "an abstract painting with" part cancels out) span a small
subspace; projecting the centred image embeddings on an orthonormal basis of that subspace keeps
only what the facet's words can express. Neighbours in the colour space are paintings with similar
colours, whatever their brushwork.

The facet spaces are built from text only, so they can be checked against the measured
descriptors, which play no part in building them (`facet_specificity`).
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .vocabulary import FACETS, IMAGE_DIM, TERMS, encode_texts, term_prompts

# Measured descriptors that quantify each facet (src/abstractart/descriptors.py)
DESCRIPTOR_GROUPS = {
    "colour": ["lightness", "lightness_contrast", "chroma", "colourfulness", "chromatic_share",
               "warm_ratio", "hue_diversity", "palette_size"],
    "texture": ["edge_density", "local_entropy", "flat_ratio", "spectral_slope", "fractal_dimension"],
    "form": ["rectilinearity", "orientation_entropy", "periodicity"],
    "composition": ["symmetry_lr", "centre_focus", "horizontal_bias", "aspect_ratio"],
}

# Candidate views: vocabulary facets used to build the space, descriptor groups it should capture.
# Form and texture terms turned out to capture the same thing (their neighbours agree equally on
# texture and form descriptors), so they are merged into one "structure" view.
CANDIDATE_VIEWS = {
    "colour": {"facets": ["colour"], "groups": ["colour"]},
    "structure": {"facets": ["form", "texture"], "groups": ["texture", "form"]},
    "composition": {"facets": ["composition"], "groups": ["composition"]},
}


def view_passes(agreement: dict[str, float], own_groups: list[str]) -> bool:
    """A view is kept if its neighbours agree more on its own descriptors than on any other group."""
    own = np.mean([agreement[g] for g in own_groups])
    return all(own > v for g, v in agreement.items() if g not in own_groups)


def facet_spaces(global_descriptor: np.ndarray, views: dict[str, list[str]] | None = None,
                 exclude: frozenset = frozenset()) -> dict[str, np.ndarray]:
    """view -> (n_images, n_terms) standardised coordinates in the text subspace of the view's facets.

    `views` maps a view name to the vocabulary facets it uses (default: one view per facet);
    `exclude` lists terms that failed the grounding check and must not define an axis.
    """
    views = views or {facet: [facet] for facet in FACETS}
    img = global_descriptor[:, :IMAGE_DIM]
    img = img / np.linalg.norm(img, axis=1, keepdims=True)
    img = img - img.mean(0)
    text = encode_texts(term_prompts())
    text = text - text.mean(0)
    spaces = {}
    for view, facets in views.items():
        rows = [i for i, (f, t) in enumerate(TERMS) if f in facets and t not in exclude]
        basis, _ = np.linalg.qr(text[rows].T)  # (768, n_terms), orthonormal
        coords = img @ basis
        spaces[view] = (coords - coords.mean(0)) / coords.std(0)
    return spaces


def neighbour_agreement(space: np.ndarray, values: np.ndarray, k: int = 10, metric: str = "euclidean",
                        seed: int = 0) -> float:
    """How much closer the k nearest neighbours are on `values` than random paintings.

    1 - mean |difference| to the neighbours / mean |difference| to random paintings:
    0 = no better than random, 1 = identical values. `values` must be z-scored (NaN allowed).
    """
    _, idx = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(space).kneighbors(space)
    idx = idx[:, 1:]
    rand = np.random.default_rng(seed).integers(0, len(space), size=idx.shape)
    neighbour = np.nanmean(np.abs(values[:, None, :] - values[idx]))
    random = np.nanmean(np.abs(values[:, None, :] - values[rand]))
    return float(1 - neighbour / random)


def facet_specificity(spaces: dict[str, np.ndarray], descriptors_z: dict[str, np.ndarray],
                      k: int = 10) -> dict[str, dict[str, float]]:
    """space -> {descriptor group -> neighbour agreement}. A working facet space wins on its own group."""
    out = {}
    for name, space in spaces.items():
        metric = "cosine" if name == "default" else "euclidean"
        out[name] = {g: neighbour_agreement(space, v, k, metric) for g, v in descriptors_z.items()}
    return out
