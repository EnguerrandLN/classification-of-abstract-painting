# Unsupervised Organisation of Abstract Art (alternative version)

> This branch (`modernisation`) is an alternative version of the project, used to explore more
> recent methods. It is not the official project: the version built and submitted for the course
> (Télécom Paris, IM06, 2025-2026) is on [`main`](../../tree/main), and a frozen copy is kept on
> [`v1-projet-ecole`](../../tree/v1-projet-ecole). Here the same 1,700 paintings are revisited with
> vision foundation models (SigLIP 2, DINOv2), hierarchical density clustering (EVōC), a zero-shot
> formal vocabulary, facet views and conformal clustering. Each step is compared with the original
> pipeline using the same evaluation code.

The question stays the same: how can abstract paintings be grouped when there is no object to
recognise? The pipeline builds a map of 1,700 paintings by 517 artists from the images alone, then
measures how much of the artists and art-historical movements it recovers without ever seeing them.

## Phase 1: representations

All representations are scored with the same protocol (`src/abstractart/evaluation.py`):

- artist top-1 and mAP: are the nearest neighbours of a painting (cosine similarity) by the same
  artist?
- movement P@10 across artists: once the query's own artist is removed from the gallery, how many
  of its 10 nearest neighbours belong to the same movement? This is the closest proxy we have for a
  style shared by several painters. The movement labels come from the original team
  (`data/movements.csv`); artists listed under two movements are dropped, which leaves 14
  movements and 926 images.
- NMI movement: agreement between a K-means partition (K = 20) and the movements.

| Representation | Artist top-1 | Artist mAP | Movement P@10 (cross-artist) | NMI movement |
|:--|--:|--:|--:|--:|
| random | 0.005 | 0.012 | 0.092 | 0.056 |
| original fusion (handcrafted + VGG16 Gram + β-VAE/ResNet) | 0.259 | 0.123 | 0.135 | 0.148 |
| original β-VAE alone | 0.067 | 0.030 | 0.094 | 0.083 |
| original features, best re-weighting | 0.456 | 0.218 | 0.153 | 0.161 |
| DINOv2-B, global + Gram | 0.599 | 0.311 | 0.241 | 0.291 |
| SigLIP 2-B, global | 0.627 | 0.341 | 0.237 | 0.292 |
| **SigLIP 2-B, global + Gram (used on this branch)** | **0.643** | **0.355** | **0.244** | **0.296** |
| DINOv2 + SigLIP 2, global + Gram | 0.652 | 0.357 | 0.254 | 0.296 |

The full table with every ablation is in [`results/benchmark.md`](results/benchmark.md).

A single frozen model does better than the whole original pipeline: artist top-1 is 2.5 times
higher, cross-artist movement precision 1.8 times higher and NMI movement doubles, with no training
and no hand-tuned weights. The β-VAE turned out to be at chance level and made every fusion worse,
so it was dropped. The Gram matrix idea from the original project still works: Gram matrices of
the transformer's patch tokens are about as good as the global embedding, and combining the two
helps for both models. Using both models together adds about one point for twice the compute, so
the branch keeps a single model. SigLIP 2 is preferred to DINOv2 because it also embeds text, which
the later phases rely on.

Cosine silhouettes stay between 0.03 and 0.12 for every representation. The corpus looks like a
continuum rather than a set of separated clusters, and neither silhouette nor stability tracks
quality here (the β-VAE and the handcrafted features are among the most stable). K-means is
therefore a poor final step, which motivates phase 2.

## Phase 2: hierarchical density clustering

[EVōC](https://evoc.readthedocs.io) builds a nearest-neighbour graph, embeds it and runs a density
clustering on it. It is applied to the SigLIP 2 global + Gram embedding, without duplicate copies.
Hyperparameters are chosen without labels, by stability under subsampling (ARI between the full
clustering and clusterings of 5 random 80% subsamples). The exported hierarchy has to reach a
family level (at least 3 layers, coarsest layer with at most 12 clusters), which gives 57, 37, 16
and 6 clusters. About half of the paintings fall outside any dense region. They are attached at
the finest layer to the cluster chosen by a vote of their 10 nearest clustered neighbours and
marked as non-core, and the coarser labels are read from the tree, so the hierarchy is strictly
nested.

AMI (adjusted mutual information) is used instead of NMI, because NMI increases with the number of
clusters. Each layer is compared with K-means at the same K on the same paintings:

| Layer | Clusters | Core share | Stability | AMI movement, core: EVōC / K-means | AMI artist, core: EVōC / K-means | AMI movement, all paintings: EVōC / K-means |
|--:|--:|--:|--:|--:|--:|--:|
| 0 | 57 | 54% | 0.85 | **0.448** / 0.407 | **0.652** / 0.584 | 0.286 / 0.293 |
| 1 | 37 | 50% | 0.82 | **0.462** / 0.411 | **0.644** / 0.541 | 0.278 / 0.275 |
| 2 | 16 | 48% | 0.68 | **0.442** / 0.396 | **0.572** / 0.504 | 0.253 / 0.246 |
| 3 | 6 | 50% | 0.66 | **0.411** / 0.348 | **0.420** / 0.385 | 0.203 / 0.219 |

K-means stability is 0.32, 0.43 and 0.51 at K = 60, 20 and 8. The full grid and the baselines are
in [`results/clustering.md`](results/clustering.md).

The density clusters are much more stable than K-means (0.66 to 0.85 against 0.32 to 0.51). On the
dense core, about half of the corpus, they agree better with movements and artists than K-means on
the same paintings (+0.04 to +0.06 AMI for movements, +0.03 to +0.10 for artists). The other half
are transitional works: once they are attached to their nearest cluster, the full partition is no
better than K-means (-0.02 to +0.01 AMI). These paintings are better shown as in-between than
forced into one group, which is what the silhouette-based skimming of the original project was
trying to do.

UMAP followed by HDBSCAN clusters more paintings (72%) with a similar core quality, but its
hierarchy collapses into 2 clusters when `min_cluster_size` grows, while EVōC gives several usable
levels and runs in 0.1 s instead of about 16 s.

The 6 families are coherent visually and historically: gestural abstraction (Mitchell, de Staël,
Zao Wou-Ki, Leroy), minimal geometry and colour fields (Rothko, Albers, Kelly), orphism and
cubo-futurism (Freundlich, Kandinsky, Delaunay, Kupka), matter and monochrome surfaces (Fontana,
Soulages, Burri), op art and grids (Vasarely) and surrealist biomorphism (Toyen, af Klint, Ernst).
At 16 clusters, Rothko's colour fields, de Staël's slabs of paint, calligraphic marks (Dubuffet,
Twombly) and Zao Wou-Ki's landscapes each get their own group. `scripts/cluster_gallery.py` renders
the most central paintings of every cluster.

## Phase 3: describing the clusters

Each cluster gets a short description ([`results/cluster_cards.md`](results/cluster_cards.md))
built from two sources computed locally:

- 20 interpretable descriptors (`src/abstractart/descriptors.py`) measured at the native aspect
  ratio of each painting: lightness, chroma, colourfulness, share of warm colours, palette size,
  edge density, local entropy, flat areas, spectral slope, fractal dimension, rectilinearity,
  horizontal bias, orientation entropy, symmetry, centre focus, periodicity of repeated motifs, and
  a few others. They describe the clusters but are not used to build them.
- A vocabulary of 45 formal-analysis terms in four facets (form, colour, texture, composition),
  scored zero-shot with the text encoder of SigLIP 2 ("an abstract painting with thick impasto
  paint"). The terms were written by hand.

Each signal is standardised and averaged over the core paintings of the cluster. Families are
compared with the whole corpus and sub-groups with their family, so each description says what is
specific at its own level. The Rothko sub-group, for instance, is described as "smooth gradients,
soft luminous fields of colour", with a spectral slope of +1.2σ and a symmetry of +0.7σ, and the de
Staël sub-group as "fragmented cubist planes, slabs of paint applied with a knife".

The vocabulary is checked before being used ([`results/vocabulary_check.md`](results/vocabulary_check.md)).
Without labels, each term is correlated with the descriptor that measures the same property: 13
of 16 correlations have the expected sign, and colour and density terms are reliable ("vivid
saturated colours" against colourfulness, ρ = 0.60; "a dense crowded composition" against local
entropy, 0.53; "black and white" against chroma, -0.49). The three terms that contradict their
measurement ("a pattern of dots", "flat even paint", "a symmetrical composition") are removed
automatically. Geometric and compositional terms are weakly related to their measurements, so for
those facets the descriptors carry the explanation, in line with recent work showing that
vision-language models struggle with formal analysis (MMArt, 2026). With one prompt per movement,
SigLIP 2 recognises the movements with a balanced accuracy of 0.20 (chance is 0.07) and a top-3
accuracy of 0.48.

The formal families (gesture, geometry, matter, op art, colour fields) get clear descriptions. The
surrealist family does not: it is defined by its imagery more than by its form, and its best
formal term ("biomorphic organic shapes", +0.30σ) is right at the reporting threshold.

## Phase 4: facet views and interactive map

Two paintings can be close in colour and far apart in brushwork. A facet view keeps only what one
part of the vocabulary can express: the SigLIP 2 text embeddings of the facet's terms (centred to
remove the shared "an abstract painting with", and without the terms that failed the check) span
a small subspace, and the image embeddings are projected onto it. Since the views are built from
text only, they can be checked against the measured descriptors: are the 10 nearest neighbours of
a painting in the view closer on each group of descriptors than random paintings? (0 means random,
1 means identical; details in [`results/facets.md`](results/facets.md).)

| Space | colour | texture | form | composition |
|:--|--:|--:|--:|--:|
| Overall embedding | 0.30 | 0.47 | 0.39 | 0.18 |
| **Colour view** | **0.38** | 0.15 | 0.14 | 0.07 |
| **Structure view** (form and texture terms) | 0.14 | **0.37** | **0.39** | 0.13 |
| Composition view (dropped) | 0.13 | 0.35 | 0.38 | 0.15 |

The colour view isolates colour: neighbours share their palette more than in the overall
embedding (0.38 against 0.30) and ignore brushwork (0.15). Form and texture terms turned out to
capture the same thing, so they are merged into a single structure view (gesture, geometry,
matter), which ignores colour (0.14). The composition view fails its own check, since its
neighbours agree on form rather than on composition, and it is dropped automatically, which is
consistent with phase 3.

Each remaining view is clustered with EVōC, keeping the most stable layer that is readable on a map
(4 to 20 clusters, at least 50% core). This gives 5 colour groups (polychrome and primary colours,
cool blues, monochrome and black and white, ochres and golds, warm reds and oranges) and 18
structure groups (impasto, knife-applied paint, grids, smooth gradients, calligraphic signs, cubist
planes, and so on), named after their most distinctive terms.

The map (`outputs/map/index.html`) is a single static page that opens directly from disk, with no
server, library or external service. It switches between the overall, colour and structure views,
shows each level of the hierarchy (6 families down to 57 groups), filters by artist or by group,
displays thumbnails when zoomed in, and shows for each painting its groups in every view and its
nearest paintings, computed in the full embedding rather than on the 2-D layout. Core paintings are
drawn as filled dots and transitional ones as rings. The map is generated locally and not
committed, because its thumbnails come from copyrighted works.

## Plausible groups with a coverage guarantee

Since half of the paintings are transitional, attaching each of them to a single group is a
guess. This step re-implements and adapts two papers from 2026 by Nath, Hur and Allen, *Inference
for Clustering: Conformal Sets for Cluster Labels* and *Weighted Conformal Clustering*, to give each
painting a set of plausible groups with a coverage guarantee (`src/abstractart/conformal.py`,
[`results/conformal.md`](results/conformal.md)).

In the papers, a soft clustering is fitted first and the label of each point is sampled from its
soft memberships (taking the most likely label instead breaks the guarantee and leads to
under-coverage). The score of Romano, Sesia and Candès (2020) is then calibrated on a held-out
split, so that each set contains the latent group with probability at least 90%, up to terms that
vanish when the soft clustering is consistent and stable.

Three changes were needed for this project:

- EVōC gives hard labels, so its finest groups are used as fixed centres of a Gaussian mixture
  whose shared variance is fitted by EM, in the smallest PCA space (16 dimensions) where the mixture
  puts at least 90% of the EVōC core paintings back in their group. Because the centres are fixed,
  the label alignment step of the papers is not needed, and the mixture itself is used as the score
  model, as allowed by Remark 1 of the first paper.
- The papers only give sets for held-out points; here cross-fitting gives a set to every painting.
- The papers do not deal with hierarchies. Labels are sampled at the finest layer and propagated
  through the tree, each of the 4 layers is calibrated at α/4, and the sets are then restricted from
  the top down. With the union bound and the nesting of the labels, the four sets contain the
  painting's groups at the same time with probability at least 90%, and they are consistent with
  each other (the family of a plausible group is always plausible). The proof is in the module
  docstring.

The method was first tested on simulated corpora where the true groups are known, drawn from the
mixture fitted on the paintings with Gaussian or heavy-tailed noise. It reaches 92% to 97% coverage
at each layer and 94% to 96% on all layers at once, with sets of 1.0 to 1.1 groups. As the papers
predict, hard labels under-cover (87% at the finest layer). The weighted variant with a separate
classifier returns almost every group, because its likelihood ratios become very large where the
mixture is confident, and the uncalibrated cutoff over-covers without any guarantee. The rule used
to pick the method was fixed before the real paintings were scored, and how it was set is
explained in `results/conformal.md`.

On the real paintings, 81% have a single plausible family (mean set size 1.20). Core paintings have
smaller sets than transitional ones (1.10 against 1.30 for families, 1.26 against 1.97 at the
finest layer), and 99% of core paintings have their EVōC family in their set. The artists whose
works are most often ambiguous are those known for changing style: Kandinsky (29% of his works
have several plausible families), Kupka (27%), Hartung (24%) and Bergman (20%). Painters with a
very recognisable style are almost never ambiguous: de Staël, Mitchell and Zao Wou-Ki (0%),
Vasarely (3%) and Rothko (4%). The most frequent borders are between gesture and matter, between
matter and minimal geometry, and between gesture and minimal geometry. In the map, each painting
lists its plausible groups, and a "Plausible families" layer colours the corpus by ambiguity.

The guarantee holds on average over the paintings, not for each painting separately. It refers to
the groups of the mixture built on EVōC, not to an art-historical truth, and it relies on the
consistency and stability conditions of the papers.

## Pipeline

1. Manifest (`scripts/build_manifest.py`): one row per image, indexed by filename. Artist names are
   parsed from irregular filenames (`pollock-3.jpg`, `saura_1 (2).jpg`) and duplicates are
   detected, both identical files and other photographs of the same work (by correlating high-pass
   thumbnails, with a threshold checked by eye). The 22 duplicate groups, 6 of which have
   conflicting artist labels, are left out of the metrics. All feature tables are aligned on
   filenames rather than on file order.
2. Embeddings (`scripts/extract_embeddings.py`): frozen DINOv2-B or SigLIP 2-B (NaFlex). Each
   painting is encoded at its native aspect ratio, without padding or square resizing, with up to
   1024 patches. Three descriptors are kept per image: `global` (pooled token and mean patch
   token), `stats` (mean and standard deviation of the patch tokens at layers 3, 6, 9 and 12) and
   `gram` (Gram matrix of the same tokens projected on a 48-dimensional PCA basis of the corpus,
   which is the VGG16 Gram signature of the original project on a recent model).
3. Benchmark (`scripts/run_benchmark.py`): the original features (`legacy/v1_features`) and the
   embeddings of this branch, scored with the protocol above.
4. Clustering (`scripts/run_clustering.py`, about 10 minutes because of the UMAP baseline): grid over
   EVōC, UMAP with HDBSCAN and K-means, evaluation, and export of the hierarchy to
   `outputs/clusters.csv` (one label and one core flag per layer) and `outputs/cluster_tree.json`
   (parent of each cluster in the next layer). `scripts/cluster_gallery.py --layer L` renders the
   most central paintings of each cluster.
5. Descriptions (`scripts/compute_descriptors.py`, about 8 minutes, then
   `scripts/explain_clusters.py`): descriptors, vocabulary check and one description per cluster
   (`results/cluster_cards.md`, all layers in `outputs/cluster_cards.json`).
6. Conformal sets (`scripts/run_conformal.py`, about 3 minutes): simulations, choice of the method and
   plausible groups of every painting (`outputs/conformal_sets.json`).
7. Facets and map (`scripts/build_facets.py`, then `scripts/build_map.py`): facet views, their check
   and their clusters, 2-D layouts and neighbours (`outputs/map_data.json`, which includes the
   conformal sets when they exist), then the static map and its thumbnails (`outputs/map/`).

## Quick start

Python 3.11 or later is required, and a CUDA GPU for the extraction step (an 8 GB laptop GPU
encodes the corpus in about 4 minutes per model).

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
python scripts/run_conformal.py
python scripts/build_facets.py
python scripts/build_map.py       # then open outputs/map/index.html
```

Model weights are downloaded from Hugging Face the first time (`google/siglip2-base-patch16-naflex`,
`facebook/dinov2-base`). Embeddings are cached in `outputs/`, which is not tracked by git.

## Repository layout

```
abstrait-v4/            the 1,700 paintings (artist_n.jpg)
data/                   manifest.csv, duplicates.csv, movements.csv (proxy labels)
src/abstractart/        data, evaluation, clustering, descriptors, vocabulary, explain, facets, conformal, embeddings/
scripts/                build_manifest, extract_embeddings, run_benchmark, run_clustering, cluster_gallery,
                        compute_descriptors, explain_clusters, run_conformal, build_facets, build_map
results/                benchmark.*, clustering.*, cluster_cards.md, vocabulary_check.md, facets.md, conformal.md
legacy/v1_features/     features of the original project, kept as benchmark baselines
docs/                   references.md, reports of the original project (EN/FR)
```

## Possible extensions

- Free-text formal descriptions from a local vision-language model, kept only if they pass the
  same check as the vocabulary.
- The paintings are copyrighted: the repository should eventually ship a manifest and a download
  script instead of the images.

## Original project (Télécom Paris IM06, 2025-2026)

Raphaël Legrand, Enguerrand Le Niliot, Eliott Dionnet and Arthur Evain, supervised by Henri Maître
and Loïc le Folgoc. Reports: [English](docs/v1/report_IM06_en.pdf), [French](docs/v1/rapport_IM06_fr.pdf).
Code: [`main`](../../tree/main) branch (frozen copy on [`v1-projet-ecole`](../../tree/v1-projet-ecole)).

The original project fused handcrafted descriptors (colour, texture, HOG, FFT), VGG16 Gram matrices
and a β-VAE with weighted alpha-blending, partitioned the result with K-means refined by a
silhouette-based skimming, and presented it in a WebGL map.

### My core contributions (v1)
While this was a collaborative academic project, I was personally responsible for the computer vision feature extraction, the algorithmic clustering refinement, and the data engineering. My specific contributions include:

* **CNN Feature Extraction:** Designed and implemented a custom pipeline using VGG16 to extract texture-based style signatures via Gram matrices, deliberately bypassing semantic object recognition.
* **Multimodal Fusion:** Engineered the weighted alpha-blending logic to normalize, standardize, and fuse the CNN, VAE, and geometric feature spaces into a cohesive mathematical topology.
* **Adaptive Skimming (Core Silhouette):** Developed a custom statistical filtering algorithm that purifies K-Means clusters by dynamically calculating exclusion thresholds based on the median Silhouette score of each cluster's "core."
* **Data Visualization & Pipeline Engineering:** Wrote the topological validation scripts (Inter-Artist cosine matrices, stylistic heatmaps) and built the Python pipelines to clean, format, and physically sort the dataset for the final WebGL application.

## References

See [`docs/references.md`](docs/references.md).
