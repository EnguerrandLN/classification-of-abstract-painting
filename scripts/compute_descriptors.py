"""Compute the 20 interpretable descriptors of every painting -> outputs/descriptors.csv."""

from abstractart import DATASET_DIR, OUTPUT_DIR
from abstractart.data import load_manifest
from abstractart.descriptors import compute_all

if __name__ == "__main__":
    filenames = load_manifest()["filename"].tolist()
    df = compute_all([DATASET_DIR / f for f in filenames])
    df.insert(0, "filename", filenames)
    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_DIR / "descriptors.csv", index=False)
    print(df.describe().T[["mean", "std", "min", "max"]].round(3).to_string())
    print(f"missing values: {int(df.isna().sum().sum())} -> {OUTPUT_DIR / 'descriptors.csv'}")
