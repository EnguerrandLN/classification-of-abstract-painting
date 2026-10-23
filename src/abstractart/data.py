"""Dataset manifest: one row per image, keyed by filename.

Every feature table in the project is aligned on `manifest.filename`, never on
the implicit order of `os.listdir`, which differs between operating systems.
"""

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from . import DATA_DIR, DATASET_DIR

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MANIFEST_PATH = DATA_DIR / "manifest.csv"
DUPLICATES_PATH = DATA_DIR / "duplicates.csv"
MOVEMENTS_PATH = DATA_DIR / "movements.csv"

# "delaunay-r_12.jpg" -> "delaunay-r", "pollock-3.jpg" -> "pollock", "seligmann.jpg" -> "seligmann",
# "saura_1 (2).jpg" -> "saura" (Windows copy suffix)
_NAME_RE = re.compile(r"^(?P<artist>.+?)(?:[_-](?P<num>\d+))?(?:\s*\(\d+\))?$")

# Placeholder labels that do not identify a single author
UNKNOWN_ARTISTS = {"x"}

# Two files are the same artwork if the high-pass (detail) components of their centre-cropped
# 32x32 grayscale thumbnails correlate above this. Calibrated by eye on ~45 candidate pairs:
# the 16 true copies (other photo of the same work) score >= 0.92, the closest distinct pair
# (bergman_48 / rothko_53) 0.88. Plain thumbnails or dHash collide on monochromes, and without the
# crop, photographed frames make distinct works match. Correlation ignores brightness/colour shifts.
NEAR_DUP_CORRELATION = 0.9
BORDER_CROP = 0.1


def parse_artist(filename: str) -> str:
    stem = Path(filename).stem.lower()
    return _NAME_RE.match(stem).group("artist")


def list_images(dataset_dir: Path = DATASET_DIR) -> list[str]:
    return sorted(p.name for p in Path(dataset_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def _detail_thumbnail(img: Image.Image, size: int = 32) -> np.ndarray:
    img.draft("RGB", (8 * size, 8 * size))
    g = img.convert("L")
    w, h = g.size
    g = g.crop((int(w * BORDER_CROP), int(h * BORDER_CROP), int(w * (1 - BORDER_CROP)), int(h * (1 - BORDER_CROP))))
    g = g.resize((size, size), Image.BILINEAR)
    x = np.asarray(g, np.float32) - np.asarray(g.filter(ImageFilter.GaussianBlur(1.5)), np.float32)
    x = x.ravel()
    return (x - x.mean()) / (x.std() + 1e-6)


def _connected_groups(n: int, pairs: np.ndarray) -> np.ndarray:
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        parent[find(i)] = find(j)
    return np.array([find(i) for i in range(n)])


def build_manifest(dataset_dir: Path = DATASET_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (manifest, duplicate pairs)."""
    rows, thumbs = [], []
    for name in list_images(dataset_dir):
        path = Path(dataset_dir) / name
        with Image.open(path) as img:
            width, height = img.size
            thumbs.append(_detail_thumbnail(img))
        rows.append({
            "filename": name,
            "artist": parse_artist(name),
            "width": width,
            "height": height,
            "md5": hashlib.md5(path.read_bytes()).hexdigest(),
        })
    df = pd.DataFrame(rows)

    # Duplicates = byte-identical files or near-identical thumbnails
    T = np.stack(thumbs)
    corr = T @ T.T / T.shape[1]
    same_md5 = df["md5"].to_numpy()[:, None] == df["md5"].to_numpy()[None, :]
    i, j = np.where(np.triu((corr >= NEAR_DUP_CORRELATION) | same_md5, k=1))
    pairs = pd.DataFrame({
        "file_a": df["filename"].to_numpy()[i], "file_b": df["filename"].to_numpy()[j],
        "artist_a": df["artist"].to_numpy()[i], "artist_b": df["artist"].to_numpy()[j],
        "exact": same_md5[i, j], "detail_corr": corr[i, j].round(3),
    })

    groups = _connected_groups(len(df), np.stack([i, j], axis=1))
    df["dup_group"] = groups
    size = df.groupby("dup_group")["filename"].transform("size")
    df.loc[size == 1, "dup_group"] = -1
    # Within a group, the first file (alphabetically) is kept, the other copies are flagged
    df["is_duplicate"] = (df["dup_group"] >= 0) & df.duplicated("dup_group", keep="first")
    df["label_conflict"] = (df["dup_group"] >= 0) & (df.groupby("dup_group")["artist"].transform("nunique") > 1)

    # Images usable as ground truth for artist-based metrics
    df["eval_artist"] = ~df["is_duplicate"] & ~df["label_conflict"] & ~df["artist"].isin(UNKNOWN_ARTISTS)
    return df, pairs


def load_manifest(path: Path = MANIFEST_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_movements(path: Path = MOVEMENTS_PATH) -> dict[str, str]:
    """Artist -> movement proxy labels. Artists listed under several movements are dropped."""
    mv = pd.read_csv(path)
    counts = mv.groupby("artist")["movement"].nunique()
    unambiguous = counts[counts == 1].index
    mv = mv[mv["artist"].isin(unambiguous)].drop_duplicates("artist")
    return dict(zip(mv["artist"], mv["movement"]))
