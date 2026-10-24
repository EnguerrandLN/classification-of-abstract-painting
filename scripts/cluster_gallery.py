"""Contact sheet of one clustering layer: one row per cluster, its most central core paintings.

    python scripts/cluster_gallery.py --layer 2

Writes outputs/galleries/layer{L}.jpg. Run scripts/run_clustering.py first.
"""

import argparse

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from abstractart import DATASET_DIR, OUTPUT_DIR
from abstractart.data import load_manifest, load_movements
from abstractart.embeddings import load_default_representation

THUMB = 120
LABEL_WIDTH = 260


def thumbnail(filename: str) -> Image.Image:
    img = Image.open(DATASET_DIR / filename)
    img.draft("RGB", (2 * THUMB, 2 * THUMB))
    img = img.convert("RGB")
    img.thumbnail((THUMB - 4, THUMB - 4))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--per-cluster", type=int, default=8)
    args = parser.parse_args()

    manifest = load_manifest()
    movements = load_movements()
    clusters = pd.read_csv(OUTPUT_DIR / "clusters.csv")
    assert (clusters["filename"] == manifest["filename"]).all()
    Z = load_default_representation(manifest["filename"].tolist())
    labels = clusters[f"l{args.layer}"].to_numpy()
    core = clusters[f"l{args.layer}_core"].to_numpy() & ~manifest["is_duplicate"].to_numpy()

    ids = pd.Series(labels).value_counts().index  # largest clusters first
    sheet = Image.new("RGB", (LABEL_WIDTH + args.per_cluster * THUMB, len(ids) * THUMB), "white")
    draw = ImageDraw.Draw(sheet)
    for row, c in enumerate(ids):
        members = np.flatnonzero((labels == c) & core)
        if len(members) == 0:
            members = np.flatnonzero(labels == c)
        centroid = Z[members].mean(0)
        central = members[np.argsort(-(Z[members] @ centroid))][: args.per_cluster]
        for col, i in enumerate(central):
            sheet.paste(thumbnail(manifest["filename"].iat[i]), (LABEL_WIDTH + col * THUMB, row * THUMB))

        in_cluster = manifest[labels == c]
        top_artists = in_cluster["artist"].value_counts().head(4)
        mov = in_cluster["artist"].map(movements).dropna().value_counts()
        text = [f"cluster {c}  ({len(in_cluster)} works, {int(((labels == c) & core).sum())} core)",
                *[f"  {a} ({n})" for a, n in top_artists.items()],
                f"  movement: {mov.index[0]} ({mov.iloc[0] / mov.sum():.0%})" if len(mov) else "  movement: -"]
        draw.multiline_text((6, row * THUMB + 6), "\n".join(text), fill="black", spacing=3)
        draw.line([(0, (row + 1) * THUMB - 1), (sheet.width, (row + 1) * THUMB - 1)], fill="#cccccc")

    out_dir = OUTPUT_DIR / "galleries"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"layer{args.layer}.jpg"
    sheet.save(path, quality=85)
    print(f"{len(ids)} clusters -> {path}")


if __name__ == "__main__":
    main()
