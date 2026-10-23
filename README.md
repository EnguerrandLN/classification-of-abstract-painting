# Unsupervised Organisation of Abstract Art

How do you group abstract paintings when there is no object to recognise? This project builds a
label-free map of 1,700 abstract paintings (517 artists) from the visual signal alone, and measures
how well that map recovers authorship and art-historical movements it was never shown.

> **Status — v2 in progress.** The original school project (Télécom Paris, IM06, 2025–2026) is
> archived on the [`v1-projet-ecole`](../../tree/v1-projet-ecole) branch. v2 replaces its
> hand-weighted fusion of three feature families by a single vision foundation model, and adds a
> reproducible evaluation harness. Phase 1 (data cleaning, evaluation, embeddings) is done.

## Results (phase 1)

Every representation is scored with the same protocol (`src/abstractart/evaluation.py`):

- **Artist top-1 / mAP**: are a painting's nearest neighbours (cosine) by the same artist?
- **Movement P@10 (cross-artist)**: after removing the query's own artist from the gallery, are its
  10 nearest neighbours from the same movement? This is the closest proxy to "a style that
  generalises across painters". Movement labels are a coarse proxy written by the v1 team
  (`data/movements.csv`; artists listed under two movements are dropped: 14 movements, 926 images).
- **NMI movement**: agreement between a K-means partition (K = 20) and the movements.

| Representation | Artist top-1 | Artist mAP | Movement P@10 (cross-artist) | NMI movement |
|:--|--:|--:|--:|--:|
| random | 0.005 | 0.012 | 0.092 | 0.056 |
| v1 — fusion as shipped (handcrafted + VGG16 Gram + β-VAE/ResNet) | 0.259 | 0.123 | 0.135 | 0.148 |
| v1 — β-VAE alone | 0.067 | 0.030 | 0.094 | 0.083 |
| v1 — best re-weighted combination | 0.456 | 0.218 | 0.153 | 0.161 |
| DINOv2-B — global + Gram | 0.599 | 0.311 | 0.241 | 0.291 |
| SigLIP 2-B — global | 0.627 | 0.341 | 0.237 | 0.292 |
| **SigLIP 2-B — global + Gram (v2 default)** | **0.643** | **0.355** | **0.244** | **0.296** |
| DINOv2 + SigLIP 2 — global + Gram | 0.652 | 0.357 | 0.254 | 0.296 |

Full table, including every ablation: [`results/benchmark.md`](results/benchmark.md).

What the numbers say:

- **One frozen foundation model beats the whole v1 pipeline** (artist top-1 ×2.5, cross-artist
  movement precision ×1.8, NMI movement ×2), with no training and no hand-tuned weights.
- **The β-VAE was at chance level** and degraded every fusion it was part of. It is gone.
- **The v1 Gram-matrix idea survives**: Gram matrices of the transformer's patch tokens are as strong
  as the global embedding, and adding them to it helps for both backbones.
- **Combining two backbones adds ~1 point** for twice the compute: not worth it, so v2 uses a single
  model. SigLIP 2 is preferred over DINOv2 because it is also text-aligned, which the next phases use.
- **Cosine silhouettes stay between 0.03 and 0.12 for every representation**: the corpus is a
  continuum, not a set of well-separated blobs, and silhouette or stability do not track quality
  (the β-VAE and the handcrafted features are among the most "stable"). K-means is the wrong final
  tool (see roadmap).

## Pipeline

1. **Manifest** (`scripts/build_manifest.py`): one row per image, keyed by filename. Artist names
   are parsed robustly (`pollock-3.jpg`, `saura_1 (2).jpg`), and duplicates are detected
   (byte-identical files plus other photos of the same work, via a calibrated high-pass
   thumbnail correlation): 22 duplicate groups, 6 of them with conflicting artist labels, are
   excluded from the metrics. Every feature table is aligned on filenames, never on file order.
2. **Embeddings** (`scripts/extract_embeddings.py`): frozen DINOv2-B or SigLIP 2-B (NaFlex), each
   painting encoded **at its native aspect ratio** (no padding, no square resize, up to 1024
   patches). Three descriptors per image: `global` (pooled token + mean patch token), `stats`
   (per-channel mean/std of patch tokens at layers 3/6/9/12) and `gram` (Gram matrix of the same
   tokens projected on a 48-dim corpus PCA basis — the v1 VGG16 Gram signature on a modern backbone).
3. **Benchmark** (`scripts/run_benchmark.py`): v1 baselines (`legacy/v1_features`) and v2
   embeddings scored with the protocol above.

## Quick start

Requires Python ≥ 3.11 and, for extraction, a CUDA GPU (an 8 GB laptop GPU encodes the corpus in
about 4 minutes per model).

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -e .

python scripts/build_manifest.py
python scripts/extract_embeddings.py --model siglip2
python scripts/extract_embeddings.py --model dinov2
python scripts/run_benchmark.py
```

Model weights are downloaded from Hugging Face on first use (`google/siglip2-base-patch16-naflex`,
`facebook/dinov2-base`). Embeddings are cached in `outputs/` (git-ignored).

## Repository layout

```
abstrait-v4/            the 1,700 paintings (artist_n.jpg)
data/                   manifest.csv, duplicates.csv, movements.csv (proxy labels)
src/abstractart/        data.py, evaluation.py, embeddings/{foundation,legacy}.py
scripts/                build_manifest, extract_embeddings, run_benchmark
results/                benchmark.csv / benchmark.md
legacy/v1_features/     v1 feature caches, kept only as benchmark baselines
docs/                   references.md, v1 reports (EN/FR)
```

## Roadmap

- **Phase 2 — clustering that fits a continuum**: density-based, hierarchical clustering
  (HDBSCAN / EVōC) on the SigLIP 2 embedding, with a native "noise" label instead of the v1
  silhouette skimming, and a movement → sub-style hierarchy. Stability across seeds and
  bootstraps as a selection criterion.
- **Phase 3 — explain the groups**: a small set of interpretable descriptors (palette, entropy,
  symmetry, fractal dimension, line statistics) and formal-analysis descriptions (form, colour,
  texture, composition) to name and contrast clusters.
- **Phase 4 — facets and map**: facet-conditioned organisation using SigLIP 2's text alignment
  ("group by colour", "by gesture", "by geometry"), and a static, serverless interactive map.
- **Open question — dataset distribution**: the paintings are copyrighted; the repository should
  eventually ship a manifest and a download script rather than the images.

## v1 — school project (Télécom Paris IM06, 2025–2026)

Raphaël Legrand, Enguerrand Le Niliot, Eliott Dionnet, Arthur Evain — supervised by Henri Maître
and Loïc le Folgoc. Reports: [English](docs/v1/report_IM06_en.pdf) ·
[Français](docs/v1/rapport_IM06_fr.pdf). Code: [`v1-projet-ecole`](../../tree/v1-projet-ecole) branch.

v1 fused handcrafted descriptors (colour, texture, HOG, FFT…), VGG16 Gram matrices and a β-VAE via
weighted alpha-blending, then partitioned the space with K-means refined by silhouette-based
skimming, and showcased it in a WebGL map.

### My core contributions (v1)
While this was a collaborative academic project, I was personally responsible for the computer vision feature extraction, the algorithmic clustering refinement, and the data engineering. My specific contributions include:

* **CNN Feature Extraction:** Designed and implemented a custom pipeline using VGG16 to extract texture-based style signatures via Gram matrices, deliberately bypassing semantic object recognition.
* **Multimodal Fusion:** Engineered the weighted alpha-blending logic to normalize, standardize, and fuse the CNN, VAE, and geometric feature spaces into a cohesive mathematical topology.
* **Adaptive Skimming (Core Silhouette):** Developed a custom statistical filtering algorithm that purifies K-Means clusters by dynamically calculating exclusion thresholds based on the median Silhouette score of each cluster's "core."
* **Data Visualization & Pipeline Engineering:** Wrote the topological validation scripts (Inter-Artist cosine matrices, stylistic heatmaps) and built the Python pipelines to clean, format, and physically sort the dataset for the final WebGL application.

## References

See [`docs/references.md`](docs/references.md).
