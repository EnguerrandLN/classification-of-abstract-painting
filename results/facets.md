# Facet views

Each view is built from SigLIP 2 text embeddings of its vocabulary facets only (terms that
failed the grounding check excluded: a pattern of dots, a symmetrical composition, flat even paint). Its nearest
neighbours are then compared on the measured descriptors (0 = no better than random
paintings, 1 = identical): a view is kept if it agrees more on its own descriptors than on
any other group.

|             |   colour |   texture |   form |   composition |
|:------------|---------:|----------:|-------:|--------------:|
| default     |    0.300 |     0.470 |  0.389 |         0.182 |
| colour      |    0.376 |     0.145 |  0.138 |         0.068 |
| structure   |    0.143 |     0.373 |  0.386 |         0.128 |
| composition |    0.126 |     0.350 |  0.378 |         0.154 |

Kept: colour, structure. Dropped: composition.

## Colour view

EVoC {'noise_level': 0.5, 'base_min_cluster_size': 10, 'n_neighbors': 30} — most stable readable layer: 5 clusters, 64% core, stability 0.75.

|   noise_level |   base_min_cluster_size |   n_neighbors |   layer |   clusters |   coverage |   stability |
|--------------:|------------------------:|--------------:|--------:|-----------:|-----------:|------------:|
|           0.3 |                      10 |            30 |       0 |         37 |      0.533 |       0.778 |
|           0.3 |                      10 |            30 |       1 |         17 |      0.493 |       0.66  |
|           0.3 |                      10 |            30 |       2 |          5 |      0.74  |       0.683 |
|           0.5 |                      10 |            30 |       0 |         35 |      0.466 |       0.789 |
|           0.5 |                      10 |            30 |       1 |         14 |      0.515 |       0.718 |
|           0.5 |                      10 |            30 |       2 |          5 |      0.644 |       0.75  |

| cluster | size (core) | most distinctive terms | measured |
|--:|--:|:--|:--|
| 1 | 494 (317) | primary colours (+1.1), vivid saturated colours (+0.8), pastel colours (+0.4) | hue diversity +0.9σ, palette size +0.8σ, colourfulness +0.7σ |
| 0 | 414 (279) | cool blue tones (+1.0), soft luminous fields of colour (+0.3) | warm ratio -1.0σ |
| 3 | 307 (199) | a monochrome surface (+1.0), black and white (+1.0) | hue diversity -1.0σ, colourfulness -1.0σ, chromatic share -0.9σ |
| 4 | 253 (151) | earthy brown and ochre tones (+0.7), gold and metallic colours (+0.7) | warm ratio +0.7σ, hue diversity -0.6σ, chromatic share +0.4σ |
| 2 | 207 (132) | warm red and orange tones (+1.4), vivid saturated colours (+0.3) | chroma +1.3σ, chromatic share +0.9σ, colourfulness +0.9σ |

## Structure view

EVoC {'noise_level': 0.5, 'base_min_cluster_size': 10, 'n_neighbors': 30} — most stable readable layer: 18 clusters, 57% core, stability 0.87.

|   noise_level |   base_min_cluster_size |   n_neighbors |   layer |   clusters |   coverage |   stability |
|--------------:|------------------------:|--------------:|--------:|-----------:|-----------:|------------:|
|           0.3 |                      10 |            30 |       0 |         31 |      0.577 |       0.819 |
|           0.3 |                      10 |            30 |       1 |         24 |      0.589 |       0.835 |
|           0.3 |                      10 |            30 |       2 |          3 |      0.725 |       0.717 |
|           0.5 |                      10 |            30 |       0 |         31 |      0.552 |       0.828 |
|           0.5 |                      10 |            30 |       1 |         18 |      0.571 |       0.866 |
|           0.5 |                      10 |            30 |       2 |          4 |      0.685 |       0.61  |

| cluster | size (core) | most distinctive terms | measured |
|--:|--:|:--|:--|
| 14 | 200 (101) | watercolour washes (+1.0), dripped and splattered paint (+0.7), a landscape horizon (+0.6) | — |
| 0 | 166 (89) | a rough material surface (+0.8), thick impasto paint (+0.5), dripped and splattered paint (+0.4) | spectral slope -0.8σ, orientation entropy +0.5σ, rectilinearity -0.4σ |
| 13 | 143 (99) | fragmented cubist planes (+1.6), hard-edged geometric shapes (+1.5), figures (+1.1) | spectral slope +0.6σ |
| 3 | 127 (77) | a grid of squares (+1.2) | periodicity +1.7σ, spectral slope -0.6σ, rectilinearity +0.5σ |
| 2 | 104 (66) | smooth gradients (+1.4), a landscape horizon (+0.8), watercolour washes (+0.4) | orientation entropy -1.6σ, fractal dimension -1.5σ, flat ratio +1.4σ |
| 12 | 104 (61) | fragmented cubist planes (+1.3), hard-edged geometric shapes (+1.3), collage of paper and materials (+1.2) | local entropy -0.8σ, spectral slope +0.8σ, flat ratio +0.8σ |
| 10 | 88 (41) | figures (+1.7), biomorphic organic shapes (+1.4), calligraphic signs (+1.1) | orientation entropy +0.5σ, local entropy +0.5σ, flat ratio -0.5σ |
| 4 | 84 (43) | thin straight lines (+1.0), diagonal lines (+0.7), a slashed or punctured canvas (+0.6) | orientation entropy -1.5σ, rectilinearity +1.2σ, periodicity +0.8σ |
| 9 | 84 (59) | circles and concentric rings (+1.8), hard-edged geometric shapes (+0.9), one large simple shape (+0.6) | spectral slope +0.8σ, edge density -0.4σ |
| 11 | 81 (34) | fine detailed linework (+1.2), thin straight lines (+1.0), scribbles and drawn marks (+0.9) | spectral slope -0.7σ, fractal dimension +0.4σ |
| 1 | 79 (53) | slabs of paint applied with a knife (+1.5), thick impasto paint (+1.3), fragmented cubist planes (+1.1) | flat ratio -0.4σ |
| 15 | 72 (41) | calligraphic signs (+1.5), dripped and splattered paint (+1.2), scribbles and drawn marks (+0.8) | edge density +1.3σ, local entropy +1.0σ, fractal dimension +0.9σ |
| 16 | 70 (40) | loose gestural brushstrokes (+1.6), dripped and splattered paint (+1.3), thick impasto paint (+1.2) | local entropy +0.8σ, flat ratio -0.7σ, orientation entropy +0.6σ |
| 5 | 69 (35) | circles and concentric rings (+1.3), calligraphic signs (+0.7), biomorphic organic shapes (+0.7) | orientation entropy +0.6σ, rectilinearity -0.4σ, periodicity -0.4σ |
| 8 | 56 (30) | figures (+1.3), calligraphic signs (+1.0), scribbles and drawn marks (+0.9) | orientation entropy +0.6σ, flat ratio -0.5σ, rectilinearity -0.4σ |
| 6 | 53 (28) | thin straight lines (+0.8), diagonal lines (+0.6) | orientation entropy -2.4σ, rectilinearity +1.9σ, periodicity +1.9σ |
| 17 | 51 (32) | thick impasto paint (+2.0), a rough material surface (+1.6), slabs of paint applied with a knife (+1.5) | local entropy +1.1σ, fractal dimension +1.0σ, edge density +1.0σ |
| 7 | 44 (27) | a grid of squares (+1.5), hard-edged geometric shapes (+1.0), fragmented cubist planes (+0.7) | rectilinearity +1.3σ, orientation entropy -1.2σ, flat ratio +0.9σ |
