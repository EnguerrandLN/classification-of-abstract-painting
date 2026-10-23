"""Extract foundation-model embeddings for every image of the manifest.

    python scripts/extract_embeddings.py --model dinov2
    python scripts/extract_embeddings.py --model siglip2
"""

import argparse
import time

from abstractart import DATASET_DIR
from abstractart.data import load_manifest
from abstractart.embeddings import save_embeddings
from abstractart.embeddings.foundation import ENCODERS, extract

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(ENCODERS), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    filenames = load_manifest()["filename"].tolist()
    start = time.time()
    arrays = extract(args.model, [DATASET_DIR / f for f in filenames], device=args.device)
    path = save_embeddings(args.model, filenames, arrays)
    shapes = ", ".join(f"{k} {v.shape}" for k, v in arrays.items())
    print(f"{args.model}: {shapes} in {time.time() - start:.0f}s -> {path}")
