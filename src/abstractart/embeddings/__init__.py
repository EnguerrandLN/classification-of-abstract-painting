"""Image representations: v2 foundation-model embeddings and v1 legacy baselines."""

from pathlib import Path

import numpy as np

from .. import OUTPUT_DIR

EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"


def save_embeddings(model: str, filenames: list[str], arrays: dict[str, np.ndarray]) -> Path:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = EMBEDDINGS_DIR / f"{model}.npz"
    np.savez_compressed(path, filename=np.array(filenames), **arrays)
    return path


def load_embeddings(model: str, filenames: list[str]) -> dict[str, np.ndarray] | None:
    """Load cached embeddings re-ordered like `filenames`, or None if not extracted yet."""
    path = EMBEDDINGS_DIR / f"{model}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    pos = {f: i for i, f in enumerate(data["filename"])}
    idx = [pos[f] for f in filenames]
    return {k: data[k][idx] for k in data.files if k != "filename"}
