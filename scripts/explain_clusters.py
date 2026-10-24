"""Phase 3: explain the clusters of the hierarchy exported by scripts/run_clustering.py.

1. Check that the vocabulary is grounded on this corpus (results/vocabulary_check.md):
   - label-free: Spearman correlation between each term and the descriptor that measures the
     same property (e.g. "black and white" vs chroma, "a pattern of dots" vs periodicity);
   - with the movement proxies: zero-shot recognition from one short prompt per movement.
2. One card per cluster, contrasted with its parent (families with the whole corpus):
   results/cluster_cards.md for the two coarsest layers, outputs/cluster_cards.json for all.

Requires outputs/descriptors.csv (scripts/compute_descriptors.py).
"""

import json
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from abstractart import OUTPUT_DIR, ROOT
from abstractart.data import load_manifest, load_movements
from abstractart.descriptors import NAMES as DESCRIPTOR_NAMES
from abstractart.embeddings import DEFAULT_MODEL, load_default_representation, load_embeddings
from abstractart.explain import make_card
from abstractart.vocabulary import (FACETS, MOVEMENT_PROMPTS, TEMPLATE, grounding, image_text_similarity,
                                    term_prompts, ungrounded_terms)

RESULTS_DIR = ROOT / "results"


def vocabulary_check(term_sim, descriptors, manifest, movements, global_desc) -> tuple[str, frozenset]:
    """Markdown report, and the terms whose correlation contradicts their measurement."""
    kept = ~manifest["is_duplicate"].to_numpy()
    corr = grounding(term_sim[kept], descriptors[kept])

    truth = manifest["artist"].map(movements)
    mask = manifest["eval_artist"].to_numpy() & truth.notna().to_numpy()
    names = list(MOVEMENT_PROMPTS)
    sim = image_text_similarity(global_desc[mask], [MOVEMENT_PROMPTS[n] for n in names])
    pred = np.array(names)[sim.argmax(1)]
    top3 = np.array(names)[np.argsort(-sim, axis=1)[:, :3]]
    y = truth[mask].to_numpy()
    zs = (f"Zero-shot movement recognition on {mask.sum()} paintings, {len(set(y))} movements: "
          f"balanced accuracy {balanced_accuracy_score(y, pred):.3f} (chance {1 / len(set(y)):.3f}), "
          f"top-3 accuracy {(top3 == y[:, None]).any(1).mean():.3f}.")
    agree = (corr["agrees"] == "yes").mean()
    failed = ungrounded_terms(corr)
    report = "\n".join([
        "## Label-free check: do the terms mean what they say?", "",
        f"{agree:.0%} of the expected term/descriptor correlations have the expected sign. "
        f"Terms that contradict their measurement are excluded from the cluster cards and the facet views: "
        f"{', '.join(sorted(failed)) or 'none'}.", "",
        corr.to_markdown(index=False, floatfmt=".3f"), "",
        "## Movement proxies", "", zs, "",
        "Caveat: SigLIP 2 was trained on web captions and may partly recognise famous artists by name,",
        "so this check measures how well the text side is grounded, not purely visual understanding.", ""])
    return report, failed


def render(card, heading: str) -> list[str]:
    facets = "; ".join(f"**{facet}**: " + ", ".join(f"{t} ({z:+.1f})" for t, z in card.terms[facet])
                       for facet in FACETS if card.terms[facet])
    desc = ", ".join(f"{d.replace('_', ' ')} {z:+.1f}σ" for d, z in card.descriptors)
    movement = f", movement proxy: {card.movement[0]} ({card.movement[1]:.0%})" if card.movement else ""
    return [f"{heading}: {card.title}", "",
            f"{card.size} paintings ({card.core} core){movement}", "",
            f"- Vocabulary: {facets or 'none'}",
            f"- Measured: {desc or 'none'}",
            "- Artists: " + ", ".join(f"{a} ({n})" for a, n in card.artists),
            "- Most central works: " + ", ".join(card.exemplars), ""]


def main():
    sys.stdout.reconfigure(errors="replace")  # Windows consoles cannot print every symbol (ρ, σ)
    manifest = load_manifest()
    movements = load_movements()
    files = manifest["filename"].tolist()
    clusters = pd.read_csv(OUTPUT_DIR / "clusters.csv")
    descriptors = pd.read_csv(OUTPUT_DIR / "descriptors.csv")
    assert (clusters["filename"] == manifest["filename"]).all() and (descriptors["filename"] == manifest["filename"]).all()
    descriptors = descriptors[DESCRIPTOR_NAMES]
    tree = json.loads((OUTPUT_DIR / "cluster_tree.json").read_text(encoding="utf-8"))

    global_desc = load_embeddings(DEFAULT_MODEL, files)["global"]
    term_sim = image_text_similarity(global_desc, term_prompts())
    embedding = load_default_representation(files)

    RESULTS_DIR.mkdir(exist_ok=True)
    check, failed_terms = vocabulary_check(term_sim, descriptors, manifest, movements, global_desc)
    (RESULTS_DIR / "vocabulary_check.md").write_text(check, encoding="utf-8")
    print(check)

    kept = ~manifest["is_duplicate"].to_numpy()
    n_layers = len([c for c in clusters.columns if c.startswith("l") and not c.endswith("_core")])
    coarsest = n_layers - 1
    cards = {}
    for layer in range(n_layers):
        labels = clusters[f"l{layer}"].to_numpy()
        core = clusters[f"l{layer}_core"].to_numpy() & kept
        parents = tree.get(f"l{layer}", {})
        for c in np.unique(labels):
            members = (labels == c) & kept
            parent = parents.get(str(c)) if layer < coarsest else None
            reference = kept if parent is None else (clusters[f"l{layer + 1}"].to_numpy() == parent) & kept
            cards[(layer, int(c))] = make_card(layer, int(c), parent, members, core, reference, term_sim,
                                               descriptors, manifest, movements, embedding, failed_terms)

    (OUTPUT_DIR / "cluster_cards.json").write_text(
        json.dumps([asdict(card) | {"title": card.title} for card in cards.values()], indent=1), encoding="utf-8")

    families = sorted((c for (l, _), c in cards.items() if l == coarsest), key=lambda c: -c.size)
    md = ["# Cluster cards", "",
          f"Families (layer {coarsest}) are contrasted with the whole corpus, sub-groups (layer {coarsest - 1}) "
          "with their family. Numbers are mean z-scores of the core paintings: vocabulary terms are SigLIP 2 "
          f'image-text similarities to "{TEMPLATE.format("...")}", descriptors are defined in '
          "`src/abstractart/descriptors.py`.", ""]
    for fam in families:
        md += render(fam, f"## Family {fam.cluster}")
        children = sorted((c for (l, _), c in cards.items() if l == coarsest - 1 and c.parent == fam.cluster),
                          key=lambda c: -c.size)
        if len(children) == 1:  # nested labels: a single child is the family itself
            md += [f"*Does not split further at layer {coarsest - 1}.*", ""]
            continue
        for child in children:
            md += render(child, f"### Sub-group {child.cluster}")
    (RESULTS_DIR / "cluster_cards.md").write_text("\n".join(md), encoding="utf-8")
    print(f"{len(cards)} cards -> {RESULTS_DIR / 'cluster_cards.md'}, {OUTPUT_DIR / 'cluster_cards.json'}")


if __name__ == "__main__":
    main()
