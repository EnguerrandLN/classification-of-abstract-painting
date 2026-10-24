## Label-free check: do the terms mean what they say?

81% of the expected term/descriptor correlations have the expected sign. Terms that contradict their measurement are excluded from the cluster cards: a pattern of dots, a symmetrical composition, flat even paint.

| term                         | descriptor          | expected sign   |   Spearman ρ | agrees   |
|:-----------------------------|:--------------------|:----------------|-------------:|:---------|
| black and white              | chroma              | −               |       -0.491 | yes      |
| vivid saturated colours      | colourfulness       | +               |        0.594 | yes      |
| warm red and orange tones    | warm_ratio          | +               |        0.359 | yes      |
| dark sombre colours          | lightness           | −               |       -0.357 | yes      |
| a monochrome surface         | palette_size        | −               |       -0.279 | yes      |
| a pattern of dots            | periodicity         | +               |       -0.160 | NO       |
| a repetitive optical pattern | periodicity         | +               |        0.200 | yes      |
| horizontal bands             | horizontal_bias     | +               |        0.057 | yes      |
| vertical stripes             | horizontal_bias     | −               |       -0.310 | yes      |
| flat even paint              | flat_ratio          | +               |       -0.050 | NO       |
| a sparse minimal composition | edge_density        | −               |       -0.423 | yes      |
| hard-edged geometric shapes  | rectilinearity      | +               |        0.066 | yes      |
| loose gestural brushstrokes  | orientation_entropy | +               |        0.319 | yes      |
| a symmetrical composition    | symmetry_lr         | +               |       -0.054 | NO       |
| a centred composition        | centre_focus        | +               |        0.025 | yes      |
| a dense crowded composition  | local_entropy       | +               |        0.535 | yes      |

## Movement proxies

Zero-shot movement recognition on 926 paintings, 14 movements: balanced accuracy 0.200 (chance 0.071), top-3 accuracy 0.479.

Caveat: SigLIP 2 was trained on web captions and may partly recognise famous artists by name,
so this check measures how well the text side is grounded, not purely visual understanding.
