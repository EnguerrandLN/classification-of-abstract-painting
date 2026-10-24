"""Image representations: v2 foundation-model embeddings and v1 legacy baselines."""

from pathlib import Path

import numpy as np

from .. import OUTPUT_DIR

EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"

# Chosen in phase 1 (results/benchmark.md): best single-model representation
DEFAULT_MODEL = "siglip2"
DEFAULT_DESCRIPTORS = ("global", "gram")


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


def load_default_representation(filenames: list[str]) -> np.ndarray:
    """SigLIP 2 global + Gram descriptors, each block standardised, reduced and equally weighted."""
    from ..evaluation import combine
    emb = load_embeddings(DEFAULT_MODEL, filenames)
    if emb is None:
        raise FileNotFoundError(f"run scripts/extract_embeddings.py --model {DEFAULT_MODEL} first")
    return combine([emb[d] for d in DEFAULT_DESCRIPTORS]).astype(np.float32)
