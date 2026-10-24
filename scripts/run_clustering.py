"""Phase 2: hierarchical density clustering of the default representation.

1. Grid over EVoC, UMAP+HDBSCAN and K-means; each configuration is scored WITHOUT labels
   by its stability under 80% subsampling (results/clustering_grid.csv). Layers with fewer
   than 4 clusters are dropped first: a 2-cluster split of the corpus is trivially stable and
   says nothing; configurations left with a single layer are not hierarchies and are skipped.
2. Per method, the most stable configuration whose layers cluster at least half of the
   paintings is kept, and each of its layers is scored with the labelled proxies, next to
   a K-means partition with the same number of clusters (results/clustering.md).
3. The exported hierarchy (outputs/clusters.csv) must reach a "family" level: among density
   configurations with at least 3 layers and a coarsest layer of at most 12 clusters, the
   most stable one is kept. It is made strictly nested (noise attached at the finest layer,
   coarser labels derived through the parent map) and gets one label column per layer
   (fine -> coarse) plus `core` flags; it is evaluated in that nested form. This is a product
   requirement, not a choice made by looking at the labels.

Duplicate copies of an artwork are left out of the clustering and inherit the label of
the kept copy.
"""

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from abstractart import OUTPUT_DIR, ROOT
from abstractart.clustering import FITTERS, attach_noise, nested_hierarchy, parent_map, stability
from abstractart.data import load_manifest, load_movements
from abstractart.embeddings import load_default_representation
from abstractart.evaluation import partition_metrics

RESULTS_DIR = ROOT / "results"
MIN_COVERAGE = 0.5
MIN_LAYER_CLUSTERS = 4
MIN_HIERARCHY_LAYERS = 3
MAX_COARSEST_CLUSTERS = 12
GRID = {
    "evoc": [{"noise_level": nl, "base_min_cluster_size": b, "n_neighbors": n}
             for nl in (0.3, 0.5, 0.7) for b in (5, 10) for n in (15, 30)],
    "umap_hdbscan": [{"n_neighbors": n, "min_samples": ms} for n in (15, 30) for ms in (1, 5)],
    "kmeans": [{"ks": (60, 20, 8)}],
}


def expand(values: np.ndarray, manifest: pd.DataFrame, keep: np.ndarray) -> np.ndarray:
    """Values computed on kept rows -> all rows; duplicate copies take their kept copy's value."""
    groups = manifest["dup_group"].to_numpy()
    kept_pos = {groups[i]: i for i in np.flatnonzero(keep) if groups[i] >= 0}
    full = np.empty(len(manifest), dtype=values.dtype)
    full[keep] = values
    for i in np.flatnonzero(~keep):
        full[i] = full[kept_pos[groups[i]]]
    return full


def main():
    manifest = load_manifest()
    movements = load_movements()
    keep = ~manifest["is_duplicate"].to_numpy()
    Z = load_default_representation(manifest["filename"].tolist())[keep]
    print(f"clustering {keep.sum()} paintings (duplicate copies excluded), dim {Z.shape[1]}")

    # 1. label-free grid
    fits, rows = {}, []
    for method, configs in GRID.items():
        for params in configs:
            cl = FITTERS[method](Z, **params)
            all_k = [l.n_clusters for l in cl.layers]
            cl.layers = [l for l in cl.layers if l.n_clusters >= MIN_LAYER_CLUSTERS]
            if len(cl.layers) < 2:
                print(f"{method:13s} {str(params):70s} k={all_k} skipped: fewer than 2 informative layers")
                continue
            stab = stability(method, Z, cl, **params)
            fits[(method, str(params))] = (cl, stab)
            for li, (layer, s) in enumerate(zip(cl.layers, stab)):
                rows.append({"method": method, "params": str(params), "layer": li,
                             "n_clusters": layer.n_clusters, "coverage": layer.coverage, "stability": s})
            print(f"{method:13s} {str(params):70s} k={[l.n_clusters for l in cl.layers]} "
                  f"stability={np.round(stab, 3).tolist()}")
    grid = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    grid.to_csv(RESULTS_DIR / "clustering_grid.csv", index=False)

    summary = grid.groupby(["method", "params"]).agg(
        layers=("n_clusters", lambda s: "/".join(map(str, s))), n_layers=("layer", "count"),
        coarsest=("n_clusters", "min"), coverage=("coverage", "mean"),
        stability=("stability", "mean")).reset_index()
    eligible = summary[summary["coverage"] >= MIN_COVERAGE]
    selected = eligible.loc[eligible.groupby("method")["stability"].idxmax()].assign(role="most stable")
    hierarchies = eligible[(eligible["method"] != "kmeans") & (eligible["n_layers"] >= MIN_HIERARCHY_LAYERS)
                           & (eligible["coarsest"] <= MAX_COARSEST_CLUSTERS)]
    exported = hierarchies.loc[hierarchies["stability"].idxmax()]
    selected = pd.concat([selected, exported.to_frame().T.assign(role="exported hierarchy")])
    selected = selected.drop_duplicates(["method", "params"], keep="last")

    # 2. labelled proxies for the selected configurations, vs K-means with the same k
    detail = []
    for _, sel in selected.iterrows():
        cl, stab = fits[(sel["method"], sel["params"])]
        if sel["role"] == "exported hierarchy":
            nested, core = nested_hierarchy(Z, cl.layers)
            layer_labels = [(np.where(c, n, -1), n) for n, c in zip(nested, core)]
        else:
            layer_labels = [(layer.labels, attach_noise(Z, layer.labels)) for layer in cl.layers]
        for li, ((raw, att), s) in enumerate(zip(layer_labels, stab)):
            labels, attached = expand(raw, manifest, keep), expand(att, manifest, keep)
            k = len(np.unique(att))
            m = partition_metrics(labels, attached, manifest, movements)
            km = expand(KMeans(k, n_init=4, random_state=0).fit_predict(Z), manifest, keep)
            km_same = partition_metrics(np.where(labels >= 0, km, -1), km, manifest, movements)
            detail.append({
                "method": sel["method"], "role": sel["role"], "layer": li, "k": k,
                "coverage": m["coverage"],
                "stability": s,
                "AMI artist (core)": m["ami_artist_core"], "AMI movement (core)": m["ami_movement_core"],
                "K-means same subset: AMI artist": km_same["ami_artist_core"],
                "K-means same subset: AMI movement": km_same["ami_movement_core"],
                "AMI movement (all, attached)": m["ami_movement_all"],
                "K-means all: AMI movement": km_same["ami_movement_all"],
                "movement purity": m["movement_purity"], "artist-dominated clusters": m["artist_dominated"],
            })
    detail = pd.DataFrame(detail)
    detail.to_csv(RESULTS_DIR / "clustering.csv", index=False)
    md = ["## Label-free selection (mean over layers)", "",
          summary.sort_values(["method", "stability"], ascending=[True, False]).to_markdown(index=False, floatfmt=".3f"),
          "", "## Selected configurations", "",
          selected[["role", "method", "params", "layers", "coverage", "stability"]].to_markdown(index=False, floatfmt=".3f"),
          "", "## Labelled proxies per layer", "", detail.to_markdown(index=False, floatfmt=".3f"), ""]
    (RESULTS_DIR / "clustering.md").write_text("\n".join(md), encoding="utf-8")
    print(detail.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # 3. export the nested hierarchy
    cl, _ = fits[(exported["method"], exported["params"])]
    nested, core = nested_hierarchy(Z, cl.layers)
    out = manifest[["filename", "artist"]].copy()
    for li, (lab, c) in enumerate(zip(nested, core)):
        out[f"l{li}"] = expand(lab, manifest, keep)
        out[f"l{li}_core"] = expand(c, manifest, keep)
    OUTPUT_DIR.mkdir(exist_ok=True)
    out.to_csv(OUTPUT_DIR / "clusters.csv", index=False)
    # parent of each cluster in the next coarser layer (exact, since the labels are nested)
    tree = {f"l{li}": parent_map(nested[li], nested[li + 1]) for li in range(len(nested) - 1)}
    (OUTPUT_DIR / "cluster_tree.json").write_text(json.dumps(tree, indent=1), encoding="utf-8")
    print(f"exported {exported['method']} {exported['params']}: {[len(np.unique(n)) for n in nested]} clusters "
          f"per layer -> {OUTPUT_DIR / 'clusters.csv'}, {OUTPUT_DIR / 'cluster_tree.json'}")


if __name__ == "__main__":
    main()
