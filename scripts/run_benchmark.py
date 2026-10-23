"""Score every available representation with the same protocol.

Writes results/benchmark.csv and results/benchmark.md.
"""

import numpy as np
import pandas as pd

from abstractart import ROOT
from abstractart.data import load_manifest, load_movements
from abstractart.embeddings import load_embeddings
from abstractart.embeddings.foundation import ENCODERS
from abstractart.embeddings.legacy import load_legacy
from abstractart.evaluation import combine, evaluate

RESULTS_DIR = ROOT / "results"

COLUMNS = {
    "artist_top1": "Artist top-1",
    "artist_mAP": "Artist mAP",
    "xartist_movement_p@10": "Movement P@10 (cross-artist)",
    "xartist_movement_mAP": "Movement mAP (cross-artist)",
    "nmi_artist": "NMI artist",
    "nmi_movement": "NMI movement",
    "stability_ari": "Stability ARI",
    "silhouette": "Silhouette",
}


def candidates(filenames: list[str]) -> dict[str, tuple[np.ndarray, bool]]:
    """name -> (matrix, standardize). Pre-weighted spaces are evaluated as built."""
    rng = np.random.default_rng(0)
    out = {"random": (rng.normal(size=(len(filenames), 64)), True)}

    v1 = load_legacy(filenames)
    for name, X in v1.items():
        out[name] = (X, name != "v1/fusion_shipped")
    out["v1/gram+resnet+handcrafted_clean"] = (
        combine([v1["v1/gram_vgg16"], v1["v1/resnet50"], v1["v1/handcrafted_clean"]]), False)

    found = {}
    for model in sorted(ENCODERS):
        emb = load_embeddings(model, filenames)
        if emb is None:
            print(f"(no cached embeddings for {model}, run scripts/extract_embeddings.py --model {model})")
            continue
        found[model] = emb
        for kind, X in emb.items():
            out[f"{model}/{kind}"] = (X, True)
        out[f"{model}/global+gram"] = (combine([emb["global"], emb["gram"]]), False)
        out[f"{model}/global+stats+gram"] = (combine([emb["global"], emb["stats"], emb["gram"]]), False)
    if len(found) == 2:
        d, s = found["dinov2"], found["siglip2"]
        out["dinov2+siglip2/global"] = (combine([d["global"], s["global"]]), False)
        out["dinov2+siglip2/global+gram"] = (combine([d["global"], d["gram"], s["global"], s["gram"]]), False)
    return out


if __name__ == "__main__":
    manifest = load_manifest()
    movements = load_movements()
    rows = []
    for name, (X, standardize) in candidates(manifest["filename"].tolist()).items():
        scores = evaluate(X, manifest, movements, standardize=standardize)
        rows.append({"representation": name, "dim": X.shape[1], **scores})
        print(f"{name:40s} " + "  ".join(f"{k}={v:.3f}" for k, v in scores.items()))

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "benchmark.csv", index=False)
    table = df[["representation", "dim", *COLUMNS]].rename(columns=COLUMNS)
    (RESULTS_DIR / "benchmark.md").write_text(table.to_markdown(index=False, floatfmt=".3f"), encoding="utf-8")
    print(f"-> {RESULTS_DIR / 'benchmark.md'}")
