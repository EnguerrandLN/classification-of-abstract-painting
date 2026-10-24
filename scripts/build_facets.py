"""Phase 4 (analysis): facet views, their validation, clusters and 2-D layouts.

1. Candidate views (colour, structure, composition) are built from SigLIP 2 text embeddings
   only, then checked against the measured descriptors, which play no part in building them:
   a view is kept if its nearest neighbours agree more on its own descriptors than on any
   other group (results/facets.md).
2. Each kept view is clustered with EVoC, as the overall hierarchy (plain HDBSCAN finds at most
   4 clusters in these 11-24 dimensional spaces). The layer is the most stable one under
   subsampling, without labels, among those readable on a map (4-20 clusters, >= 50% core).
   Clusters are named by the most distinctive terms of the view's facets.
3. 2-D UMAP layouts of the overall embedding and of every kept view (initialised from the
   overall layout, so that switching views moves the points smoothly), and the 8 nearest
   neighbours of every painting in each view's full space, not in 2-D.

Writes outputs/map_data.json, read by scripts/build_map.py. Duplicate copies are left out.
"""

import json

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from abstractart import OUTPUT_DIR, ROOT
from abstractart.clustering import attach_noise, fit_evoc, stability
from abstractart.data import load_manifest
from abstractart.embeddings import DEFAULT_MODEL, load_default_representation, load_embeddings
from abstractart.explain import MIN_TERM_Z, contrast
from abstractart.facets import CANDIDATE_VIEWS, DESCRIPTOR_GROUPS, facet_spaces, facet_specificity, view_passes
from abstractart.vocabulary import TERMS, grounding, image_text_similarity, term_prompts, ungrounded_terms

RESULTS_DIR = ROOT / "results"
MIN_CLUSTERS, MAX_CLUSTERS, MIN_COVERAGE = 4, 20, 0.5
EVOC_GRID = [{"noise_level": nl, "base_min_cluster_size": 10, "n_neighbors": 30} for nl in (0.3, 0.5)]
N_NEIGHBOURS = 8


def layout(X: np.ndarray, metric: str, init: np.ndarray | str = "spectral") -> np.ndarray:
    import umap
    xy = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric=metric, init=init,
                   random_state=0).fit_transform(X)
    xy = xy - xy.mean(0)
    return xy / np.abs(xy).max()


def neighbours(X: np.ndarray, metric: str) -> np.ndarray:
    _, idx = NearestNeighbors(n_neighbors=N_NEIGHBOURS + 1, metric=metric).fit(X).kneighbors(X)
    return idx[:, 1:]


def cluster_view(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Most stable EVoC layer among the readable ones (MIN_CLUSTERS..MAX_CLUSTERS, >= MIN_COVERAGE core)."""
    Z = Z.astype(np.float32)
    grid, best = [], None
    for params in EVOC_GRID:
        cl = fit_evoc(Z, **params)
        stab = stability("evoc", Z, cl, **params)
        for i, (layer, s) in enumerate(zip(cl.layers, stab)):
            grid.append({**params, "layer": i, "clusters": layer.n_clusters,
                         "coverage": round(layer.coverage, 3), "stability": round(s, 3)})
            readable = MIN_CLUSTERS <= layer.n_clusters <= MAX_CLUSTERS and layer.coverage >= MIN_COVERAGE
            if readable and (best is None or s > best[0]):
                best = (s, layer, params)
    if best is None:
        raise RuntimeError(f"no readable EVoC layer: {grid}")
    s, layer, params = best
    info = {"params": params, "clusters": layer.n_clusters, "coverage": layer.coverage, "stability": s, "grid": grid}
    return attach_noise(Z, layer.labels), layer.labels >= 0, info


def name_clusters(labels, core, term_sim, descriptors, facets, groups, excluded) -> dict[int, dict]:
    term_idx = [i for i, (f, t) in enumerate(TERMS) if f in facets and t not in excluded]
    desc_cols = [c for g in groups for c in DESCRIPTOR_GROUPS[g]]
    D = descriptors[desc_cols].to_numpy(dtype=float)
    everyone = np.ones(len(labels), bool)
    out = {}
    for c in np.unique(labels):
        rows = (labels == c) & core if ((labels == c) & core).sum() >= 3 else labels == c
        tz = contrast(term_sim[:, term_idx], rows, everyone)
        top = [(TERMS[term_idx[j]][1], float(tz[j])) for j in np.argsort(-tz)[:3] if tz[j] >= MIN_TERM_Z]
        dz = contrast(D, rows, everyone)
        meas = [(desc_cols[j], float(dz[j])) for j in np.argsort(-np.abs(dz))[:3] if abs(dz[j]) >= 0.4]
        out[int(c)] = {"title": ", ".join(t for t, _ in top[:2]) or "no distinctive term",
                       "terms": top, "measured": meas, "size": int((labels == c).sum()),
                       "core": int(((labels == c) & core).sum())}
    # Clusters sharing the same two top terms are told apart by their third term or main measurement
    titles = [c["title"] for c in out.values()]
    for c in out.values():
        if titles.count(c["title"]) > 1:
            extra = c["terms"][2][0] if len(c["terms"]) > 2 else (
                f"{c['measured'][0][0].replace('_', ' ')} {'high' if c['measured'][0][1] > 0 else 'low'}"
                if c["measured"] else "")
            c["title"] = f"{c['title']}, {extra}" if extra else c["title"]
    return out


def add_conformal_sets(overall: dict, files: list[str], n_layers: int) -> None:
    """Plausible groups of every painting (scripts/run_conformal.py), plus an "ambiguity" layer."""
    path = OUTPUT_DIR / "conformal_sets.json"
    if not path.exists():
        return
    conformal = json.loads(path.read_text(encoding="utf-8"))
    sets, meta = conformal["sets"], conformal["meta"]
    for layer, l in zip(overall["layers"], range(n_layers - 1, -1, -1)):
        layer["sets"] = [sets[f][f"l{l}"] for f in files]
        layer["sets_label"] = f"Plausible at {1 - meta['alpha']:.0%} confidence"
    n_fam = np.array([min(len(sets[f][f"l{n_layers - 1}"]), 3) for f in files])
    names = {1: "one plausible family", 2: "two plausible families", 3: "three or more plausible families"}
    overall["layers"].append({
        "name": f"Plausible families ({1 - meta['alpha']:.0%})", "labels": n_fam.tolist(),
        "core": (n_fam == 1).astype(int).tolist(), "core_names": ["ambiguous", "unambiguous"],
        "clusters": {str(k): {"title": t, "size": int((n_fam == k).sum())} for k, t in names.items()}})
    overall["about"] += (f" Each painting also gets its set of plausible groups, with {1 - meta['alpha']:.0%} "
                         "simultaneous confidence across the hierarchy (conformal clustering, Nath, Hur & Allen 2026, "
                         "adapted to EVoC and to nested groups).")


def main():
    manifest = load_manifest()
    keep = ~manifest["is_duplicate"].to_numpy()
    m = manifest[keep].reset_index(drop=True)
    files = m["filename"].tolist()
    descriptors = pd.read_csv(OUTPUT_DIR / "descriptors.csv").set_index("filename").loc[files].reset_index()
    global_desc = load_embeddings(DEFAULT_MODEL, files)["global"]
    default = load_default_representation(files)
    term_sim = image_text_similarity(global_desc, term_prompts())
    excluded = ungrounded_terms(grounding(term_sim, descriptors))  # same check as the phase 3 cards

    # 1. candidate views and their validation against the measured descriptors
    spaces = facet_spaces(global_desc, {v: c["facets"] for v, c in CANDIDATE_VIEWS.items()}, exclude=excluded)
    dz = {g: ((descriptors[c] - descriptors[c].mean()) / descriptors[c].std()).to_numpy()
          for g, c in DESCRIPTOR_GROUPS.items()}
    spec = facet_specificity({"default": default} | spaces, dz)
    kept = [v for v, c in CANDIDATE_VIEWS.items() if view_passes(spec[v], c["groups"])]
    print("neighbour agreement per descriptor group:\n", pd.DataFrame(spec).T.round(3).to_string())
    print("views kept:", kept)

    # 2-3. overall layout from the default embedding and the phase 2 hierarchy
    clusters = pd.read_csv(OUTPUT_DIR / "clusters.csv").set_index("filename").loc[files].reset_index()
    cards = json.loads((OUTPUT_DIR / "cluster_cards.json").read_text(encoding="utf-8"))
    n_layers = len([c for c in clusters.columns if c.startswith("l") and not c.endswith("_core")])
    overall_xy = layout(default, "cosine")
    views = {"overall": {
        "label": "Overall", "xy": overall_xy.round(4).tolist(), "neighbours": neighbours(default, "cosine").tolist(),
        "about": "Everything at once: SigLIP 2 global + Gram embedding, clustered by EVoC into a hierarchy "
                 "of families and groups.",
        "layers": [{"name": f"{len(np.unique(clusters[f'l{l}']))} {'families' if l == n_layers - 1 else 'groups'}",
                    "labels": clusters[f"l{l}"].tolist(), "core": clusters[f"l{l}_core"].astype(int).tolist(),
                    "clusters": {str(c["cluster"]): {"title": c["title"], "terms": c["terms"],
                                                    "measured": c["descriptors"], "size": c["size"], "core": c["core"],
                                                    "artists": c["artists"], "movement": c["movement"]}
                                 for c in cards if c["layer"] == l}}
                   for l in range(n_layers - 1, -1, -1)]}}
    add_conformal_sets(views["overall"], files, n_layers)

    report = ["# Facet views", "",
              "Each view is built from SigLIP 2 text embeddings of its vocabulary facets only (terms that",
              f"failed the grounding check excluded: {', '.join(sorted(excluded)) or 'none'}). Its nearest",
              "neighbours are then compared on the measured descriptors (0 = no better than random",
              "paintings, 1 = identical): a view is kept if it agrees more on its own descriptors than on",
              "any other group.", "",
              pd.DataFrame(spec).T.to_markdown(floatfmt=".3f"), "",
              f"Kept: {', '.join(kept)}. Dropped: {', '.join(v for v in CANDIDATE_VIEWS if v not in kept) or 'none'}.", ""]
    for v in kept:
        labels, core, info = cluster_view(spaces[v])
        named = name_clusters(labels, core, term_sim, descriptors, CANDIDATE_VIEWS[v]["facets"],
                              CANDIDATE_VIEWS[v]["groups"], excluded)
        metric = "euclidean"
        own = np.mean([spec[v][g] for g in CANDIDATE_VIEWS[v]["groups"]])
        other = max(a for g, a in spec[v].items() if g not in CANDIDATE_VIEWS[v]["groups"])
        views[v] = {"label": v.capitalize(), "xy": layout(spaces[v], metric, init=overall_xy).round(4).tolist(),
                    "neighbours": neighbours(spaces[v], metric).tolist(),
                    "about": f"Only the {v} vocabulary of SigLIP 2. Neighbours agree on the measured "
                             f"{' and '.join(CANDIDATE_VIEWS[v]['groups'])} descriptors ({own:.2f}) far more "
                             f"than on any other group ({other:.2f}; 0 = random, 1 = identical).",
                    "layers": [{"name": f"{info['clusters']} {v} groups", "labels": labels.tolist(),
                                "core": core.astype(int).tolist(), "clusters": {str(k): c for k, c in named.items()}}]}
        report += [f"## {v.capitalize()} view", "",
                   f"EVoC with {info['params']}. Most stable readable layer: {info['clusters']} clusters, "
                   f"{info['coverage']:.0%} core, stability {info['stability']:.2f}.", "",
                   pd.DataFrame(info["grid"]).to_markdown(index=False), "",
                   "| cluster | size (core) | most distinctive terms | measured |", "|--:|--:|:--|:--|"]
        for k, c in sorted(named.items(), key=lambda kv: -kv[1]["size"]):
            terms = ", ".join(f"{t} ({z:+.1f})" for t, z in c["terms"]) or "none"
            meas = ", ".join(f"{d.replace('_', ' ')} {z:+.1f}σ" for d, z in c["measured"]) or "none"
            report.append(f"| {k} | {c['size']} ({c['core']}) | {terms} | {meas} |")
        report.append("")
        print(f"{v}: {info}")

    data = {"paintings": [{"f": f, "a": a} for f, a in zip(files, m["artist"])], "views": views}
    (OUTPUT_DIR / "map_data.json").write_text(json.dumps(data), encoding="utf-8")
    (RESULTS_DIR / "facets.md").write_text("\n".join(report), encoding="utf-8")
    print(f"-> {OUTPUT_DIR / 'map_data.json'}, {RESULTS_DIR / 'facets.md'}")


if __name__ == "__main__":
    main()
