# Conformal sets of plausible groups

Adapted from Nath, Hur & Allen (2026), *Inference for Clustering: Conformal Sets for Cluster Labels* and *Weighted Conformal Clustering*; adaptations and the hierarchical guarantee are documented in `src/abstractart/conformal.py`. Space: PCA d=16 of the default representation (the anchored mixture recovers 92% of EVoC core paintings). Level alpha = 0.1: marginal coverage per layer, and hierarchical sets with alpha/4 per layer restricted top-down (simultaneous guarantee).

## Simulations with known latent labels (5 synthetic corpora per noise model)

Coverage columns are per layer (marginal sets); sizes are those of the hierarchical sets.

| noise            | method                  | guarantee   |   simultaneous coverage |   l0 coverage |   l1 coverage |   l2 coverage |   l3 coverage |   l0 size |   l1 size |   l2 size |   l3 size | reaches target   |
|:-----------------|:------------------------|:------------|------------------------:|--------------:|--------------:|--------------:|--------------:|----------:|----------:|----------:|----------:|:-----------------|
| gaussian         | split, mixture score    | True        |                   0.955 |         0.921 |         0.930 |         0.959 |         0.968 |     1.140 |     1.115 |     1.031 |     1.019 | True             |
| gaussian         | weighted, mixture score | True        |                   0.949 |         0.922 |         0.930 |         0.959 |         0.968 |     1.167 |     1.128 |     1.045 |     1.025 | True             |
| gaussian         | split, classifier       | True        |                   0.946 |         0.893 |         0.896 |         0.914 |         0.936 |     1.757 |     1.587 |     1.247 |     1.169 | True             |
| gaussian         | weighted, classifier    | True        |                   0.969 |         0.945 |         0.949 |         0.945 |         0.956 |    55.185 |    36.057 |    15.652 |     5.903 | True             |
| gaussian         | hard labels (baseline)  | False       |                   0.927 |         0.870 |         0.872 |         0.897 |         0.926 |     1.615 |     1.480 |     1.214 |     1.152 | False            |
| gaussian         | cutoff (baseline)       | False       |                   0.990 |         0.977 |         0.980 |         0.987 |         0.991 |     1.554 |     1.490 |     1.262 |     1.197 | True             |
| student-t (3 df) | split, mixture score    | True        |                   0.940 |         0.925 |         0.932 |         0.953 |         0.965 |     1.053 |     1.041 |     1.008 |     1.004 | True             |
| student-t (3 df) | weighted, mixture score | True        |                   0.940 |         0.925 |         0.932 |         0.953 |         0.965 |     3.064 |     2.648 |     1.964 |     1.504 | True             |
| student-t (3 df) | split, classifier       | True        |                   0.939 |         0.889 |         0.894 |         0.924 |         0.945 |     1.509 |     1.380 |     1.167 |     1.106 | False            |
| student-t (3 df) | weighted, classifier    | True        |                   0.960 |         0.944 |         0.950 |         0.951 |         0.964 |    55.383 |    36.138 |    15.692 |     5.903 | True             |
| student-t (3 df) | hard labels (baseline)  | False       |                   0.926 |         0.875 |         0.881 |         0.918 |         0.940 |     1.429 |     1.312 |     1.136 |     1.087 | False            |
| student-t (3 df) | cutoff (baseline)       | False       |                   0.969 |         0.960 |         0.965 |         0.975 |         0.982 |     1.386 |     1.340 |     1.180 |     1.132 | True             |

**Chosen: split, mixture score**, the smallest sets among the methods with a conformal guarantee that reach 0.89 coverage at every layer and simultaneously, under both noise models.

Note on the rule: without the guarantee filter it would pick **split, mixture score** here. In a first run, before the mixture score model (Remark 1 of the first paper) was added, the unfiltered rule picked the uncalibrated cutoff, which has no coverage guarantee and is favoured by simulations drawn from the very model it trusts; the filter was added then, before any real painting was scored.

## Real paintings (hierarchical sets)

| method                  |   layer |   groups |   mean size |   singletons |   size, core |   size, transitional |   EVoC label covered (core) |   attached label covered (transitional) |
|:------------------------|--------:|---------:|------------:|-------------:|-------------:|---------------------:|----------------------------:|----------------------------------------:|
| split, mixture score    |       0 |       57 |       1.583 |        0.624 |        1.264 |                1.966 |                       0.985 |                                   0.649 |
| split, mixture score    |       1 |       37 |       1.494 |        0.661 |        1.228 |                1.762 |                       0.986 |                                   0.714 |
| split, mixture score    |       2 |       16 |       1.291 |        0.761 |        1.116 |                1.453 |                       0.989 |                                   0.769 |
| split, mixture score    |       3 |        6 |       1.203 |        0.814 |        1.103 |                1.302 |                       0.987 |                                   0.848 |
| weighted, mixture score |       0 |       57 |       1.531 |        0.647 |        1.239 |                1.883 |                       0.986 |                                   0.639 |
| weighted, mixture score |       1 |       37 |       1.445 |        0.684 |        1.198 |                1.694 |                       0.986 |                                   0.702 |
| weighted, mixture score |       2 |       16 |       1.269 |        0.777 |        1.107 |                1.419 |                       0.989 |                                   0.764 |
| weighted, mixture score |       3 |        6 |       1.185 |        0.827 |        1.096 |                1.273 |                       0.986 |                                   0.842 |
| split, classifier       |       0 |       57 |       2.598 |        0.307 |        2.026 |                3.285 |                       0.986 |                                   0.777 |
| split, classifier       |       1 |       37 |       2.272 |        0.355 |        1.812 |                2.737 |                       0.989 |                                   0.818 |
| split, classifier       |       2 |       16 |       1.739 |        0.477 |        1.421 |                2.031 |                       0.995 |                                   0.850 |
| split, classifier       |       3 |        6 |       1.505 |        0.569 |        1.304 |                1.704 |                       0.995 |                                   0.914 |
| weighted, classifier    |       0 |       57 |      55.069 |        0.000 |       56.097 |               53.833 |                       0.996 |                                   0.934 |
| weighted, classifier    |       1 |       37 |      35.813 |        0.000 |       36.429 |               35.191 |                       0.995 |                                   0.942 |
| weighted, classifier    |       2 |       16 |      15.528 |        0.000 |       15.717 |               15.354 |                       0.998 |                                   0.950 |
| weighted, classifier    |       3 |        6 |       5.863 |        0.000 |        5.916 |                5.810 |                       0.998 |                                   0.982 |
| hard labels (baseline)  |       0 |       57 |       2.835 |        0.291 |        2.186 |                3.615 |                       0.965 |                                   0.737 |
| hard labels (baseline)  |       1 |       37 |       2.256 |        0.355 |        1.804 |                2.712 |                       0.979 |                                   0.783 |
| hard labels (baseline)  |       2 |       16 |       1.772 |        0.468 |        1.466 |                2.054 |                       0.994 |                                   0.826 |
| hard labels (baseline)  |       3 |        6 |       1.535 |        0.557 |        1.335 |                1.733 |                       0.987 |                                   0.913 |
| cutoff (baseline)       |       0 |       57 |       2.152 |        0.449 |        1.514 |                2.919 |                       0.995 |                                   0.767 |
| cutoff (baseline)       |       1 |       37 |       2.012 |        0.481 |        1.474 |                2.556 |                       0.996 |                                   0.828 |
| cutoff (baseline)       |       2 |       16 |       1.648 |        0.576 |        1.298 |                1.971 |                       0.999 |                                   0.867 |
| cutoff (baseline)       |       3 |        6 |       1.459 |        0.650 |        1.250 |                1.665 |                       0.995 |                                   0.930 |
