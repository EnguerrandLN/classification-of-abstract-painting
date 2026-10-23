"""Scan abstrait-v4/ and write data/manifest.csv and data/duplicates.csv."""

from abstractart.data import DUPLICATES_PATH, MANIFEST_PATH, build_manifest

if __name__ == "__main__":
    df, pairs = build_manifest()
    df.to_csv(MANIFEST_PATH, index=False)
    pairs.to_csv(DUPLICATES_PATH, index=False)

    counts = df.loc[~df["is_duplicate"], "artist"].value_counts()
    print(f"{len(df)} images, {df['artist'].nunique()} artists "
          f"({(counts == 1).sum()} with a single work, {(counts >= 20).sum()} with >= 20)")
    n_groups = df.loc[df["dup_group"] >= 0, "dup_group"].nunique()
    n_conflicts = df.loc[df["label_conflict"], "dup_group"].nunique()
    print(f"duplicate groups: {n_groups} ({pairs['exact'].sum()} exact pairs), "
          f"{df['is_duplicate'].sum()} extra copies, {n_conflicts} groups with conflicting artists")
    print(pairs.to_string(index=False))
    print(f"images usable for artist metrics: {df['eval_artist'].sum()} -> {MANIFEST_PATH}")
