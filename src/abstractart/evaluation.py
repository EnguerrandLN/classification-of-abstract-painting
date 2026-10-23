"""Evaluation harness.

There is no ground truth for "visual style", so a representation is scored with
proxies that should all go up when the representation gets better:

- artist retrieval: are the nearest neighbours of a painting by the same artist?
- cross-artist movement retrieval: once the query's own artist is removed from the
  gallery, are the neighbours from the same (proxy) movement? This is the closest
  proxy to "style that generalises across painters".
- clustering: NMI of a K-means partition with artists / movements, stability across
  seeds, cosine silhouette.

Every representation goes through the same `prepare` step so scores are comparable.
"""

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler, normalize


def prepare(X: np.ndarray, n_components: int = 256, seed: int = 0, standardize: bool = True) -> np.ndarray:
    """Standardise, reduce with PCA if wider than `n_components`, L2-normalise rows.

    `standardize=False` keeps the space as built (used for pre-weighted fusions).
    """
    X = np.nan_to_num(X.astype(np.float64))
    if not standardize:
        return normalize(X - X.mean(0))
    X = StandardScaler().fit_transform(X)
    if X.shape[1] > n_components:
        X = PCA(n_components, random_state=seed).fit_transform(X)
    return normalize(X)


def combine(blocks: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Concatenate prepared blocks so that each one weighs exactly `weights[i]` in squared cosine distance."""
    weights = weights or [1.0] * len(blocks)
    return np.hstack([np.sqrt(w) * prepare(B) for B, w in zip(blocks, weights)])


def retrieval_metrics(Z: np.ndarray, labels: np.ndarray, valid: np.ndarray,
                      exclude_groups: np.ndarray | None = None) -> dict:
    """Top-1, precision@10 and mAP of cosine retrieval among `valid` items.

    Queries are the valid items whose label appears at least twice in the gallery.
    If `exclude_groups` is given, gallery items sharing the query's group are removed.
    """
    Zv, lv = Z[valid], labels[valid]
    S = Zv @ Zv.T
    np.fill_diagonal(S, -np.inf)
    if exclude_groups is not None:
        gv = exclude_groups[valid]
        S[gv[:, None] == gv[None, :]] = -np.inf

    rel_all = (lv[:, None] == lv[None, :]) & np.isfinite(S)
    queries = rel_all.sum(1) > 0
    S, rel_all = S[queries], rel_all[queries]

    order = np.argsort(-S, axis=1)
    rel = np.take_along_axis(rel_all, order, axis=1)
    ranks = np.arange(1, rel.shape[1] + 1)
    ap = (np.cumsum(rel, axis=1) / ranks * rel).sum(1) / rel.sum(1)
    return {"top1": rel[:, 0].mean(), "p@10": rel[:, :10].mean(), "mAP": ap.mean(), "n_queries": int(queries.sum())}


def clustering_metrics(Z: np.ndarray, artists: np.ndarray, movements: np.ndarray, eval_mask: np.ndarray,
                       k: int = 20, seeds: tuple = (0, 1, 2)) -> dict:
    partitions = [KMeans(k, n_init=4, random_state=s).fit_predict(Z) for s in seeds]

    counts = Counter(artists[eval_mask])
    big = eval_mask & np.array([counts[a] >= 5 for a in artists])
    mv = eval_mask & (movements != "")
    return {
        "nmi_artist": np.mean([normalized_mutual_info_score(artists[big], p[big]) for p in partitions]),
        "nmi_movement": np.mean([normalized_mutual_info_score(movements[mv], p[mv]) for p in partitions]),
        "stability_ari": np.mean([adjusted_rand_score(a, b) for a, b in combinations(partitions, 2)]),
        "silhouette": silhouette_score(Z, partitions[0], metric="cosine"),
    }


def evaluate(X: np.ndarray, manifest: pd.DataFrame, movements: dict[str, str], k: int = 20,
             standardize: bool = True) -> dict:
    Z = prepare(X, standardize=standardize)
    artists = manifest["artist"].to_numpy()
    eval_mask = manifest["eval_artist"].to_numpy()
    mov = np.array([movements.get(a, "") for a in artists])
    has_mov = eval_mask & (mov != "")

    art = retrieval_metrics(Z, artists, eval_mask)
    xmv = retrieval_metrics(Z, mov, has_mov, exclude_groups=artists)
    out = {
        "artist_top1": art["top1"], "artist_p@10": art["p@10"], "artist_mAP": art["mAP"],
        "xartist_movement_top1": xmv["top1"], "xartist_movement_p@10": xmv["p@10"],
        "xartist_movement_mAP": xmv["mAP"],
    }
    out.update(clustering_metrics(Z, artists, mov, eval_mask, k=k))
    return out
