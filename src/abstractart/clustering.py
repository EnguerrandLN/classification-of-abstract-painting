"""Hierarchical density-based clustering of the painting embeddings.

The corpus is a continuum (cosine silhouettes < 0.15 for every representation), so
instead of forcing K spherical clusters we look for dense regions at several
granularities and let sparse paintings be "noise" first:

- EVoC (Tutte Institute): kNN graph -> node embedding -> density clustering, which
  natively returns several layers, from fine to coarse;
- UMAP -> HDBSCAN: the classic pipeline, used as a baseline;
- K-means: the v1 approach, used as a baseline.

Noise points are then attached to the cluster voted by their nearest clustered
neighbours, so that every painting gets a label, with a `core` flag telling whether
it was a member of a dense region or attached afterwards.

Hyperparameters are selected without labels, by stability under subsampling.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors


@dataclass
class Layer:
    labels: np.ndarray          # -1 = noise
    strengths: np.ndarray | None = None

    @property
    def n_clusters(self) -> int:
        return len(set(self.labels[self.labels >= 0]))

    @property
    def coverage(self) -> float:
        return float((self.labels >= 0).mean())


@dataclass
class Clustering:
    method: str
    params: dict
    layers: list[Layer] = field(default_factory=list)  # fine -> coarse


def fit_evoc(Z: np.ndarray, noise_level: float = 0.5, base_min_cluster_size: int = 5,
             n_neighbors: int = 15, seed: int = 0) -> Clustering:
    import evoc
    model = evoc.EVoC(noise_level=noise_level, base_min_cluster_size=base_min_cluster_size,
                      n_neighbors=n_neighbors, random_state=seed).fit(Z.astype(np.float32))
    layers = [Layer(np.asarray(l), np.asarray(s))
              for l, s in zip(model.cluster_layers_, model.membership_strength_layers_)]
    params = {"noise_level": noise_level, "base_min_cluster_size": base_min_cluster_size, "n_neighbors": n_neighbors}
    return Clustering("evoc", params, layers)


def fit_umap_hdbscan(Z: np.ndarray, min_cluster_sizes: tuple = (5, 10, 25), min_samples: int = 5,
                     n_neighbors: int = 15, n_components: int = 10, seed: int = 0) -> Clustering:
    """One HDBSCAN layer per `min_cluster_size`, on a shared UMAP embedding."""
    import hdbscan
    import umap
    U = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=0.0,
                  metric="cosine", random_state=seed).fit_transform(Z)
    layers = []
    for mcs in min_cluster_sizes:
        model = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=min_samples).fit(U)
        layers.append(Layer(model.labels_, model.probabilities_))
    params = {"min_cluster_sizes": min_cluster_sizes, "min_samples": min_samples, "n_neighbors": n_neighbors}
    return Clustering("umap_hdbscan", params, layers)


def fit_kmeans(Z: np.ndarray, ks: tuple = (50, 20, 8), seed: int = 0) -> Clustering:
    layers = [Layer(KMeans(k, n_init=4, random_state=seed).fit_predict(Z)) for k in ks]
    return Clustering("kmeans", {"ks": ks}, layers)


FITTERS = {"evoc": fit_evoc, "umap_hdbscan": fit_umap_hdbscan, "kmeans": fit_kmeans}


def attach_noise(Z: np.ndarray, labels: np.ndarray, k: int = 10) -> np.ndarray:
    """Give each noise point the majority label of its k nearest clustered neighbours (cosine)."""
    labels = labels.copy()
    core = labels >= 0
    if core.all() or not core.any():
        return labels
    nn = NearestNeighbors(n_neighbors=min(k, core.sum()), metric="cosine").fit(Z[core])
    _, idx = nn.kneighbors(Z[~core])
    votes = labels[core][idx]
    labels[~core] = [np.bincount(v).argmax() for v in votes]
    return labels


def stability(method: str, Z: np.ndarray, reference: Clustering, n_runs: int = 5,
              fraction: float = 0.8, seed: int = 0, **params) -> list[float]:
    """Per reference layer: mean ARI between the full-data clustering and clusterings of subsamples.

    Each reference layer is compared to the subsample layer with the closest number of
    clusters, on the points clustered (non-noise) in both.
    """
    rng = np.random.default_rng(seed)
    scores = [[] for _ in reference.layers]
    for run in range(n_runs):
        idx = np.sort(rng.choice(len(Z), size=int(fraction * len(Z)), replace=False))
        sub = FITTERS[method](Z[idx], seed=seed + run + 1, **params)
        for li, ref in enumerate(reference.layers):
            best = min(sub.layers, key=lambda l: abs(l.n_clusters - ref.n_clusters))
            a, b = ref.labels[idx], best.labels
            both = (a >= 0) & (b >= 0)
            scores[li].append(adjusted_rand_score(a[both], b[both]) if both.sum() > 1 else 0.0)
    return [float(np.mean(s)) for s in scores]


def parent_map(fine: np.ndarray, coarse: np.ndarray) -> dict[int, int]:
    """For each fine cluster, the coarse cluster containing most of its members."""
    out = {}
    for c in np.unique(fine[fine >= 0]):
        parents = coarse[(fine == c) & (coarse >= 0)]
        out[int(c)] = int(np.bincount(parents).argmax()) if len(parents) else -1
    return out
