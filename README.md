# Unsupervised Organisation of Abstract Art

How do you group abstract paintings when there is no object to recognise? This project builds a
label-free map of 1,700 abstract paintings (517 artists) from the visual signal alone, and measures
how well that map recovers authorship and art-historical movements it was never shown.

> **Status — v2 in progress.** The original school project (Télécom Paris, IM06, 2025–2026) is
> archived on the [`v1-projet-ecole`](../../tree/v1-projet-ecole) branch. v2 replaces its
> hand-weighted fusion of three feature families by a single vision foundation model, and adds a
> reproducible evaluation harness. Phases 1 (data cleaning, evaluation, embeddings), 2
> (hierarchical density clustering), 3 (cluster explanations) and 4 (facet views and a static
> interactive map) are done.

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
  tool (see phase 2).

## Clustering (phase 2)

[EVōC](https://evoc.readthedocs.io) (kNN graph → node embedding → density clustering) is run on
the SigLIP 2 global + Gram embedding, duplicate copies excluded. Hyperparameters are chosen
**without labels**, by stability under subsampling (ARI between the full clustering and
clusterings of 5 random 80% subsamples). The exported hierarchy must reach a "family" level
(≥ 3 layers, coarsest ≤ 12 clusters): **57 → 37 → 16 → 6 clusters**. Paintings outside any dense
region ("noise", about half) are attached at the finest layer to the cluster voted by their 10
nearest clustered neighbours and flagged as non-core; coarser labels are derived through the
tree, so the hierarchy is strictly nested for every painting.

AMI (adjusted mutual information) is used instead of NMI because NMI grows with the number of
clusters. Each layer is compared with K-means at the same K on **the same paintings**:

| Layer | Clusters | Core share | Stability | AMI movement, core: EVōC / K-means | AMI artist, core: EVōC / K-means | AMI movement, all (noise attached): EVōC / K-means |
|--:|--:|--:|--:|--:|--:|--:|
| 0 | 57 | 54% | 0.85 | **0.448** / 0.407 | **0.652** / 0.584 | 0.286 / 0.293 |
| 1 | 37 | 50% | 0.82 | **0.462** / 0.411 | **0.644** / 0.541 | 0.278 / 0.275 |
| 2 | 16 | 48% | 0.68 | **0.442** / 0.396 | **0.572** / 0.504 | 0.253 / 0.246 |
| 3 | 6 | 50% | 0.66 | **0.411** / 0.348 | **0.420** / 0.385 | 0.203 / 0.219 |

K-means stability is 0.32 / 0.43 / 0.51 at K = 60 / 20 / 8. Full grid and baselines:
[`results/clustering.md`](results/clustering.md).

What it says:

- **Density clusters are far more reproducible** than K-means (stability 0.66–0.85 vs 0.32–0.51).
- **On the dense core (about half of the corpus) they match movements and artists better than
  K-means on the very same paintings** (+0.04 to +0.06 AMI movement, +0.03 to +0.10 AMI artist).
- **The other half are transitional works**: once they are attached to their nearest cluster, the
  whole partition is no better than K-means (−0.02 to +0.01 AMI). No partition is convincing for them, so the map should show them as
  in-between rather than force a label — a principled version of the v1 silhouette skimming.
- **UMAP → HDBSCAN** clusters more paintings (72%) with a similar core quality, but its hierarchy
  collapses into 2 clusters as `min_cluster_size` grows; EVōC gives a usable multi-level tree
  (and fits in 0.1 s instead of ~16 s).
- **The 6 families are visually and historically coherent**: gestural abstraction (Mitchell,
  de Staël, Zao Wou-Ki, Leroy), minimal geometry and colour fields (Rothko, Albers, Kelly), orphism
  and cubo-futurism (Freundlich, Kandinsky, Delaunay, Kupka), matter and monochrome surfaces
  (Fontana, Soulages, Burri), op art and grids (Vasarely) and surrealist biomorphism (Toyen,
  af Klint, Ernst). The 16-cluster layer separates e.g. Rothko's colour fields, de Staël's slabs,
  calligraphic marks (Dubuffet, Twombly) or Zao Wou-Ki's lyrical landscapes. Phase 3 describes
  every cluster; galleries: `scripts/cluster_gallery.py`.

## Explaining the clusters (phase 3)

Each cluster gets a card ([`results/cluster_cards.md`](results/cluster_cards.md)) built from two
complementary, local sources:

- **20 interpretable descriptors** (`src/abstractart/descriptors.py`), measured at native aspect
  ratio: lightness, chroma, colourfulness, warm ratio, palette size, edge density, local entropy,
  flat areas, spectral slope, fractal dimension, rectilinearity, horizontal bias, orientation
  entropy, symmetry, centre focus, periodicity of repeated motifs… They describe clusters; they
  are not used to build them.
- **A 45-term formal-analysis vocabulary** in four facets (form, colour, texture, composition)
  scored zero-shot by SigLIP 2's text tower ("an abstract painting with thick impasto paint").

Every signal is z-scored and averaged over the cluster's core paintings. Families are contrasted
with the whole corpus, sub-groups with their family, so each card says what is specific at its
own level. For example, the Rothko sub-group reads *smooth gradients · soft luminous fields of
colour*, with measured spectral slope +1.2σ and symmetry +0.7σ, and the de Staël sub-group
*fragmented cubist planes · slabs of paint applied with a knife*.

The vocabulary is checked before it is trusted ([`results/vocabulary_check.md`](results/vocabulary_check.md)):

- **Label-free**: each term is correlated with the descriptor measuring the same property.
  13 of 16 correlations have the expected sign; colour and density terms are well grounded
  ("vivid saturated colours" ↔ colourfulness ρ = 0.60, "a dense crowded composition" ↔ local
  entropy 0.53, "black and white" ↔ chroma −0.49). The three terms that contradict their
  measurement ("a pattern of dots", "flat even paint", "a symmetrical composition") are excluded
  automatically. Geometric and compositional terms are weakly grounded, so for those facets the
  measured descriptors carry the explanation — consistent with recent findings that vision-language
  models handle formal analysis poorly (MMArt, 2026).
- **Movement proxies**: one prompt per movement recognises it zero-shot with a balanced accuracy
  of 0.20 (chance 0.07), top-3 accuracy 0.48.

Formal families are described crisply (gesture, geometry, matter, op art, colour fields). The
surrealist family is not: it is defined by imagery rather than form, and its best formal term
("biomorphic organic shapes", +0.30σ) sits at the reporting threshold.

## Facet views and interactive map (phase 4)

A painting can be close to another by its colours and far from it by its brushwork. Each **facet
view** keeps only what one part of the vocabulary can express: the SigLIP 2 text embeddings of
the facet's terms (centred, so the shared "an abstract painting with" cancels out, and without
the terms that failed the grounding check) span a small subspace, and the image embeddings are
projected on it. The views are built from text only, so they are checked against the measured
descriptors, which play no part in building them: are a painting's 10 nearest neighbours in the
view closer on each descriptor group than random paintings? (0 = random, 1 = identical;
[`results/facets.md`](results/facets.md))

| Space | colour | texture | form | composition |
|:--|--:|--:|--:|--:|
| Overall embedding | 0.30 | 0.47 | 0.39 | 0.18 |
| **Colour view** | **0.38** | 0.15 | 0.14 | 0.07 |
| **Structure view** (form + texture terms) | 0.14 | **0.37** | **0.39** | 0.13 |
| Composition view — dropped | 0.13 | 0.35 | 0.38 | 0.15 |

- **The colour view isolates colour**: its neighbours share palettes even better than in the
  overall embedding (0.38 vs 0.30) while ignoring brushwork (0.15).
- **Form and texture terms capture the same thing**, so they are merged into one **structure**
  view (gesture, geometry, matter), which in turn ignores colour (0.14).
- **The composition view fails** its own check (its neighbours agree on form, not on
  composition) and is dropped automatically, consistent with phase 3.

Each kept view is clustered with EVōC (most stable readable layer: 4–20 clusters, ≥ 50% core):
**5 colour groups** (polychrome and primary colours, cool blues, monochrome and black-and-white,
earthy ochres and golds, warm reds and oranges) and **18 structure groups** (impasto, knife slabs,
grids, smooth gradients, calligraphic signs, cubist planes…), named by their most distinctive
terms.

**The map** (`outputs/map/index.html`) is a single static page — no server, no library, no API:
open it from disk. Switch between the overall, colour and structure views (points glide between
layouts), pick a level of the hierarchy (6 families → 57 groups), search an artist, filter a
group from the legend, zoom until the dots become thumbnails, and click a painting to see its
groups in every view and its nearest paintings, computed in the full embedding rather than on
the 2-D map. Core paintings are filled dots, transitional ones are rings. The map is generated
locally and not committed, since its thumbnails derive from copyrighted works.

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
4. **Clustering** (`scripts/run_clustering.py`, ~10 min because of the UMAP baseline): label-free
   grid over EVōC, UMAP → HDBSCAN and K-means, evaluation, and export of the hierarchy to
   `outputs/clusters.csv` (one label + `core` flag per layer) and `outputs/cluster_tree.json`
   (parent of each cluster in the next coarser layer). `scripts/cluster_gallery.py --layer L`
   renders the most central paintings of every cluster of a layer.
5. **Explanations** (`scripts/compute_descriptors.py`, ~8 min, then `scripts/explain_clusters.py`):
   interpretable descriptors, vocabulary grounding check, and one card per cluster
   (`results/cluster_cards.md`, all layers in `outputs/cluster_cards.json`).
6. **Facets and map** (`scripts/build_facets.py`, then `scripts/build_map.py`): facet views, their
   validation and clusters, 2-D layouts and neighbours (`outputs/map_data.json`), then the static
   map with its thumbnails (`outputs/map/`).

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
python scripts/run_clustering.py
python scripts/cluster_gallery.py --layer 3
python scripts/compute_descriptors.py
python scripts/explain_clusters.py
python scripts/build_facets.py
python scripts/build_map.py       # then open outputs/map/index.html
```

Model weights are downloaded from Hugging Face on first use (`google/siglip2-base-patch16-naflex`,
`facebook/dinov2-base`). Embeddings are cached in `outputs/` (git-ignored).

## Repository layout

```
abstrait-v4/            the 1,700 paintings (artist_n.jpg)
data/                   manifest.csv, duplicates.csv, movements.csv (proxy labels)
src/abstractart/        data, evaluation, clustering, descriptors, vocabulary, explain, facets, embeddings/
scripts/                build_manifest, extract_embeddings, run_benchmark, run_clustering, cluster_gallery,
                        compute_descriptors, explain_clusters, build_facets, build_map
results/                benchmark.*, clustering.*, cluster_cards.md, vocabulary_check.md, facets.md
legacy/v1_features/     v1 feature caches, kept only as benchmark baselines
docs/                   references.md, v1 reports (EN/FR)
```

## Roadmap

- ~~Phase 2 — clustering that fits a continuum~~: done (see above).
- ~~Phase 3 — explain the groups~~: done (see above). Possible extension: free-text formal
  descriptions from a local vision-language model, only if they pass the same grounding check.
- ~~Phase 4 — facets and map~~: done (see above).
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
