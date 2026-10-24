"""Conformal sets of plausible groups (src/abstractart/conformal.py) on the EVoC hierarchy.

1. Space: PCA of the default representation, smallest d whose anchored mixture puts >= 90% of the
   EVoC core paintings back in their own group (the soft clustering must agree where EVoC is sure).
2. Simulations with known latent labels, drawn from the mixture fitted on the real paintings (same
   anchors, variance and weights): Gaussian noise (mixture exactly right) and Student-t noise with
   3 degrees of freedom (heavy tails, mixture misspecified, as it is on real data).
3. Methods: split and weighted conformal clustering, each with the papers' classifier or with the
   anchored mixture as score model (first paper, Remark 1); baselines: hard labels and the
   uncalibrated cutoff.
4. Choice, fixed before looking at the real paintings: among the methods WITH a conformal guarantee,
   those reaching 1 - alpha - 0.01 coverage at every layer and simultaneously in both simulations;
   the one with the smallest sets (summed over layers) wins. A first version of this rule also let
   the baselines compete and picked the cutoff, which has no guarantee and is favoured by simulations
   whose noise model matches its own; see results/conformal.md.
5. Real paintings: set sizes, singletons, coverage of the EVoC label of core paintings, transitional vs
   core paintings; export of the chosen sets (outputs/conformal_sets.json).
"""

import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from abstractart import OUTPUT_DIR, ROOT
from abstractart.conformal import AnchoredMixture, cross_fitted_sets
from abstractart.data import load_manifest
from abstractart.embeddings import load_default_representation

RESULTS_DIR = ROOT / "results"
ALPHA = 0.1
DIMS = (8, 16, 24, 32, 48, 64)
N_SIM = 5
NOISES = {"gaussian": None, "student-t (3 df)": 3.0}
# name -> (method, score model, has a conformal guarantee)
METHODS = {
    "split, mixture score": ("split", "mixture", True),
    "weighted, mixture score": ("weighted", "mixture", True),
    "split, classifier": ("split", "classifier", True),
    "weighted, classifier": ("weighted", "classifier", True),
    "hard labels (baseline)": ("hard", "classifier", False),
    "cutoff (baseline)": ("cutoff", "mixture", False),
}


def hierarchy(clusters: pd.DataFrame):
    """Layer labels as contiguous indices, fine -> coarse, and parent index arrays."""
    n_layers = len([c for c in clusters.columns if c.startswith("l") and not c.endswith("_core")])
    ids = [np.unique(clusters[f"l{l}"]) for l in range(n_layers)]
    idx = [np.searchsorted(ids[l], clusters[f"l{l}"].to_numpy()) for l in range(n_layers)]
    core = [clusters[f"l{l}_core"].to_numpy() for l in range(n_layers)]
    parents = []
    for l in range(n_layers - 1):
        par = np.zeros(len(ids[l]), int)
        for k in range(len(ids[l])):
            par[k] = np.bincount(idx[l + 1][idx[l] == k]).argmax()
        parents.append(par)
    return ids, idx, core, parents


def choose_dimension(Z, idx0, core0):
    pca = PCA(max(DIMS), random_state=0).fit(Z)
    Xfull = pca.transform(Z)
    for d in DIMS:
        X = Xfull[:, :d]
        anchors = np.stack([X[(idx0 == k) & core0].mean(0) for k in range(idx0.max() + 1)])
        agree = (AnchoredMixture.fit(X, anchors).memberships(X).argmax(1) == idx0)[core0].mean()
        if agree >= 0.9:
            return d, X, anchors, agree
    return d, X, anchors, agree


def summarise(sets: list[np.ndarray], truth: list[np.ndarray]) -> dict:
    out = {"all_layers": float(np.all([S[np.arange(len(y)), y] for S, y in zip(sets, truth)], axis=0).mean())}
    for l, (S, y) in enumerate(zip(sets, truth)):
        out[f"l{l} coverage"] = float(S[np.arange(len(y)), y].mean())
        out[f"l{l} size"] = float(S.sum(1).mean())
    return out


def main():
    manifest = load_manifest()
    keep = ~manifest["is_duplicate"].to_numpy()
    files = manifest.loc[keep, "filename"].tolist()
    Z = load_default_representation(files)
    clusters = pd.read_csv(OUTPUT_DIR / "clusters.csv").set_index("filename").loc[files].reset_index()
    ids, idx, core, parents = hierarchy(clusters)
    sizes = [len(i) for i in ids]
    L = len(sizes)

    d, X, anchors, agree = choose_dimension(Z, idx[0], core[0])
    print(f"space: PCA d={d} (core agreement {agree:.2f}); layers {sizes}")
    mixture = AnchoredMixture.fit(X, anchors)

    # simulations with known latent labels
    sim_rows = []
    for noise, df in NOISES.items():
        for name, (method, score_model, guaranteed) in METHODS.items():
            runs = []
            for s in range(N_SIM):
                Xs, latent = mixture.sample(len(X), np.random.default_rng(100 + s), df=df)
                truth = [latent]
                for par in parents:
                    truth.append(par[truth[-1]])
                out = cross_fitted_sets(Xs, anchors, parents, sizes, ALPHA, method, score_model, seed=s)
                hier, marg = summarise(out["hierarchical"], truth), summarise(out["marginal"], truth)
                runs.append({"simultaneous coverage": hier["all_layers"],
                             **{f"l{l} coverage": marg[f"l{l} coverage"] for l in range(L)},
                             **{f"l{l} size": hier[f"l{l} size"] for l in range(L)}})
            sim_rows.append({"noise": noise, "method": name, "guarantee": guaranteed, **pd.DataFrame(runs).mean()})
            print(f"{noise:17s} {name:24s} simultaneous {sim_rows[-1]['simultaneous coverage']:.3f}")
    sim = pd.DataFrame(sim_rows)

    target = 1 - ALPHA - 0.01
    reaches = sim["simultaneous coverage"] >= target
    for l in range(L):
        reaches &= sim[f"l{l} coverage"] >= target
    sim["reaches target"] = reaches
    sim["total size"] = sim[[f"l{l} size" for l in range(L)]].sum(1)
    per_method = sim.groupby("method").agg(ok=("reaches target", "all"), size=("total size", "mean"),
                                           guarantee=("guarantee", "first"))
    candidates = per_method[per_method["ok"] & per_method["guarantee"]]
    chosen = candidates["size"].idxmin()
    first_rule = per_method[per_method["ok"]]["size"].idxmin()  # same rule without the guarantee filter
    print("chosen:", chosen, "| first rule would have picked:", first_rule)

    # real paintings
    real_rows, exported = [], None
    for name, (method, score_model, _) in METHODS.items():
        out = cross_fitted_sets(X, anchors, parents, sizes, ALPHA, method, score_model, seed=0)
        H = out["hierarchical"]
        for l in range(L):
            size = H[l].sum(1)
            real_rows.append({"method": name, "layer": l, "groups": sizes[l], "mean size": size.mean(),
                              "singletons": (size == 1).mean(),
                              "size, core": size[core[l]].mean(), "size, transitional": size[~core[l]].mean(),
                              "EVoC label covered (core)": H[l][core[l], idx[l][core[l]]].mean(),
                              "attached label covered (transitional)": H[l][~core[l], idx[l][~core[l]]].mean()})
        if name == chosen:
            exported = H
    real = pd.DataFrame(real_rows)
    print(real[real["method"] == chosen].round(3).to_string(index=False))

    sets = {f: {f"l{l}": [int(ids[l][k]) for k in np.flatnonzero(exported[l][i])] for l in range(L)}
            for i, f in enumerate(files)}
    meta = {"method": chosen, "alpha": ALPHA, "dimension": d, "layers": sizes,
            "guarantee": f"simultaneous over the {L} layers, level {1 - ALPHA:.0%}"}
    (OUTPUT_DIR / "conformal_sets.json").write_text(json.dumps({"meta": meta, "sets": sets}), encoding="utf-8")

    sim_table = sim[["noise", "method", "guarantee", "simultaneous coverage", *[f"l{l} coverage" for l in range(L)],
                     *[f"l{l} size" for l in range(L)], "reaches target"]]
    md = ["# Conformal sets of plausible groups", "",
          "Adapted from Nath, Hur & Allen (2026), *Inference for Clustering: Conformal Sets for Cluster Labels* and "
          "*Weighted Conformal Clustering*; adaptations and the hierarchical guarantee are documented in "
          f"`src/abstractart/conformal.py`. Space: PCA d={d} of the default representation (the anchored mixture "
          f"recovers {agree:.0%} of EVoC core paintings). Level alpha = {ALPHA}: marginal coverage per layer, and "
          f"hierarchical sets with alpha/{L} per layer restricted top-down (simultaneous guarantee).", "",
          f"## Simulations with known latent labels ({N_SIM} synthetic corpora per noise model)", "",
          "Coverage columns are per layer (marginal sets); sizes are those of the hierarchical sets.", "",
          sim_table.to_markdown(index=False, floatfmt=".3f"), "",
          f"**Chosen: {chosen}**, the smallest sets among the methods with a conformal guarantee that reach "
          f"{target:.2f} coverage at every layer and simultaneously, under both noise models.", "",
          f"Note on the rule: without the guarantee filter it would pick **{first_rule}** here. In a first run, "
          "before the mixture score model (Remark 1 of the first paper) was added, the unfiltered rule picked the "
          "uncalibrated cutoff, which has no coverage guarantee and is favoured by simulations drawn from the very "
          "model it trusts; the filter was added then, before any real painting was scored.", "",
          "## Real paintings (hierarchical sets)", "", real.to_markdown(index=False, floatfmt=".3f"), ""]
    (RESULTS_DIR / "conformal.md").write_text("\n".join(md), encoding="utf-8")
    print(f"-> {OUTPUT_DIR / 'conformal_sets.json'}, {RESULTS_DIR / 'conformal.md'}")


if __name__ == "__main__":
    main()
