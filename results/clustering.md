## Label-free selection (mean over layers)

| method       | params                                                               | layers     |   n_layers |   coarsest |   coverage |   stability |
|:-------------|:---------------------------------------------------------------------|:-----------|-----------:|-----------:|-----------:|------------:|
| evoc         | {'noise_level': 0.7, 'base_min_cluster_size': 10, 'n_neighbors': 30} | 37/27/10   |          3 |         10 |      0.483 |       0.823 |
| evoc         | {'noise_level': 0.3, 'base_min_cluster_size': 10, 'n_neighbors': 30} | 42/19      |          2 |         19 |      0.551 |       0.807 |
| evoc         | {'noise_level': 0.3, 'base_min_cluster_size': 5, 'n_neighbors': 30}  | 61/29      |          2 |         29 |      0.573 |       0.802 |
| evoc         | {'noise_level': 0.7, 'base_min_cluster_size': 5, 'n_neighbors': 30}  | 57/27/10   |          3 |         10 |      0.495 |       0.788 |
| evoc         | {'noise_level': 0.5, 'base_min_cluster_size': 5, 'n_neighbors': 15}  | 63/42/18   |          3 |         18 |      0.597 |       0.772 |
| evoc         | {'noise_level': 0.5, 'base_min_cluster_size': 5, 'n_neighbors': 30}  | 57/37/16/6 |          4 |          6 |      0.531 |       0.749 |
| evoc         | {'noise_level': 0.7, 'base_min_cluster_size': 5, 'n_neighbors': 15}  | 67/45/9    |          3 |          9 |      0.586 |       0.748 |
| evoc         | {'noise_level': 0.7, 'base_min_cluster_size': 10, 'n_neighbors': 15} | 45/29/9    |          3 |          9 |      0.587 |       0.723 |
| evoc         | {'noise_level': 0.5, 'base_min_cluster_size': 10, 'n_neighbors': 30} | 40/16/6    |          3 |          6 |      0.530 |       0.719 |
| evoc         | {'noise_level': 0.5, 'base_min_cluster_size': 10, 'n_neighbors': 15} | 42/10      |          2 |         10 |      0.596 |       0.672 |
| evoc         | {'noise_level': 0.3, 'base_min_cluster_size': 5, 'n_neighbors': 15}  | 77/32/14/4 |          4 |          4 |      0.633 |       0.654 |
| evoc         | {'noise_level': 0.3, 'base_min_cluster_size': 10, 'n_neighbors': 15} | 45/36/14/4 |          4 |          4 |      0.650 |       0.649 |
| kmeans       | {'ks': (60, 20, 8)}                                                  | 60/20/8    |          3 |          8 |      1.000 |       0.419 |
| umap_hdbscan | {'n_neighbors': 15, 'min_samples': 5}                                | 76/46      |          2 |         46 |      0.720 |       0.771 |
| umap_hdbscan | {'n_neighbors': 15, 'min_samples': 1}                                | 129/58     |          2 |         58 |      0.827 |       0.701 |
| umap_hdbscan | {'n_neighbors': 30, 'min_samples': 1}                                | 122/54     |          2 |         54 |      0.791 |       0.684 |

## Selected configurations

| role               | method       | params                                                               | layers     |   coverage |   stability |
|:-------------------|:-------------|:---------------------------------------------------------------------|:-----------|-----------:|------------:|
| most stable        | evoc         | {'noise_level': 0.3, 'base_min_cluster_size': 10, 'n_neighbors': 30} | 42/19      |      0.551 |       0.807 |
| most stable        | kmeans       | {'ks': (60, 20, 8)}                                                  | 60/20/8    |      1.000 |       0.419 |
| most stable        | umap_hdbscan | {'n_neighbors': 15, 'min_samples': 5}                                | 76/46      |      0.720 |       0.771 |
| exported hierarchy | evoc         | {'noise_level': 0.5, 'base_min_cluster_size': 5, 'n_neighbors': 30}  | 57/37/16/6 |      0.531 |       0.749 |

## Labelled proxies per layer

| method       | role               |   layer |   k |   coverage |   stability |   AMI artist (core) |   AMI movement (core) |   K-means same subset: AMI artist |   K-means same subset: AMI movement |   AMI movement (all, attached) |   K-means all: AMI movement |   movement purity |   artist-dominated clusters |
|:-------------|:-------------------|--------:|----:|-----------:|------------:|--------------------:|----------------------:|----------------------------------:|------------------------------------:|-------------------------------:|----------------------------:|------------------:|----------------------------:|
| evoc         | most stable        |       0 |  42 |      0.566 |       0.814 |               0.645 |                 0.447 |                             0.574 |                               0.424 |                          0.296 |                       0.299 |             0.527 |                       0.190 |
| evoc         | most stable        |       1 |  19 |      0.532 |       0.799 |               0.570 |                 0.416 |                             0.464 |                               0.342 |                          0.260 |                       0.262 |             0.435 |                       0.053 |
| kmeans       | most stable        |       0 |  60 |      1.000 |       0.320 |               0.448 |                 0.299 |                             0.448 |                               0.299 |                          0.299 |                       0.299 |             0.538 |                       0.183 |
| kmeans       | most stable        |       1 |  20 |      1.000 |       0.428 |               0.381 |                 0.262 |                             0.381 |                               0.262 |                          0.262 |                       0.262 |             0.445 |                       0.050 |
| kmeans       | most stable        |       2 |   8 |      1.000 |       0.510 |               0.336 |                 0.246 |                             0.336 |                               0.246 |                          0.246 |                       0.246 |             0.386 |                       0.000 |
| umap_hdbscan | most stable        |       0 |  76 |      0.712 |       0.833 |               0.616 |                 0.411 |                             0.521 |                               0.350 |                          0.334 |                       0.283 |             0.622 |                       0.276 |
| umap_hdbscan | most stable        |       1 |  46 |      0.728 |       0.709 |               0.569 |                 0.405 |                             0.505 |                               0.341 |                          0.328 |                       0.287 |             0.579 |                       0.196 |
| evoc         | exported hierarchy |       0 |  57 |      0.544 |       0.848 |               0.652 |                 0.448 |                             0.584 |                               0.407 |                          0.286 |                       0.293 |             0.549 |                       0.211 |
| evoc         | exported hierarchy |       1 |  37 |      0.502 |       0.815 |               0.644 |                 0.462 |                             0.541 |                               0.411 |                          0.278 |                       0.275 |             0.510 |                       0.135 |
| evoc         | exported hierarchy |       2 |  16 |      0.478 |       0.675 |               0.572 |                 0.442 |                             0.504 |                               0.396 |                          0.253 |                       0.246 |             0.421 |                       0.125 |
| evoc         | exported hierarchy |       3 |   6 |      0.498 |       0.658 |               0.420 |                 0.411 |                             0.385 |                               0.348 |                          0.203 |                       0.219 |             0.324 |                       0.000 |
