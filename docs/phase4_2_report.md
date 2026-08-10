# Phase 4.2 - Mask-Aware Spatial JEPA Completion

## Decision

Mask-aware spatial JEPA improves over ordinary spatial JEPA in three of four
benchmark conditions and gives its largest gain on the previously weak
central-25% complex group. It also beats the CNN in central 25% and random 25%.
It does not improve central 50% and gives only a small gain on random 50%.

This is outcome A/B: the objective helps, especially for selected contiguous
and complex cases, but it is not a universal replacement for the CNN. The
structural result is useful; do not introduce EM conditioning yet.

Phase 3, Phase 4, and Phase 4.1 checkpoints and outputs were treated as frozen.
The Phase 2 forward EM CNN was not modified or used.

## Motivation and controlled hypothesis

Phase 4.1 used uniform JEPA loss over the full 8x8 latent map. Since the input
mask explicitly identifies what is missing, Phase 4.2 tests whether latent
prediction improves when the JEPA objective emphasizes hidden regions while
retaining a small known-region weight.

The only primary change is the JEPA weighting. The context encoder, target
encoder, predictor, decoder, latent size, EMA, optimizer, learning rate,
reconstruction coefficient, data split, and masks are unchanged from Phase
4.1.

## Soft mask transformation and objective

The binary `[B,1,16,16]` hidden mask is converted to `[B,1,8,8]` using 2x2
average pooling, not nearest-neighbor interpolation. Thus each latent cell
contains the fraction of hidden input pixels in its receptive block.

The primary weight map is:

`M8 = average_pool2d(M, kernel_size=2, stride=2)`

`W = 0.10 + 0.90 * M8`

Therefore known cells have weight 0.10, fully hidden cells have weight 1.0,
and boundary cells receive a soft intermediate weight. The loss is:

`L = L_masked-JEPA + 0.1 * L_masked-BCE`.

JEPA normalization is along the 64 feature channels at each 8x8 location;
the target representation is stop-gradient. Reconstruction BCE remains
unchanged and is computed only over missing pixels.

## Experiments and compute

All runs used the same 5k subset, seed 42, 4,000/500/500 split, mask rules,
AdamW settings, latent architecture, threshold, and complexity thresholds.

| Run | Mask | Best epoch | Epochs | Training seconds | Peak memory |
| --- | --- | ---: | ---: | ---: | ---: |
| 4.2A | central 25% | 11 | 21 | 18.525 | 52.0 MiB |
| 4.2B | central 50% | 15 | 25 | 22.780 | 52.0 MiB |
| 4.2C | random 25% | 13 | 23 | 22.331 | 52.0 MiB |
| 4.2D | random 50% | 11 | 21 | 20.690 | 52.0 MiB |

GPU was the RTX 3050 Laptop GPU with Python 3.14.3, PyTorch 2.10.0+cu126,
and CUDA 12.6.

Parameter counts are unchanged from Phase 4.1:

- CNN: 332,897
- Global JEPA: 726,113 total / 469,281 trainable
- Spatial JEPA: 397,793 total / 341,729 trainable
- Mask-aware spatial JEPA: 397,793 total / 341,729 trainable

## Four-way primary results

Masked IoU is the primary metric. Values are means over the same 500 test
geometries.

| Experiment | CNN | Global JEPA | Spatial JEPA | Mask-aware JEPA | M-aware - Spatial | M-aware - CNN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4.2A central 25% | 0.560517 | 0.441019 | 0.540335 | 0.569492 | +0.029158 | +0.008976 |
| 4.2B central 50% | 0.540114 | 0.496369 | 0.504855 | 0.503694 | -0.001161 | -0.036420 |
| 4.2C random 25% | 0.621486 | 0.527071 | 0.634318 | 0.650475 | +0.016157 | +0.028989 |
| 4.2D random 50% | 0.610473 | 0.506742 | 0.596194 | 0.599720 | +0.003527 | -0.010752 |

Other masked metrics:

| Metric | 4.2A CNN / Global / Spatial / Mask-aware | 4.2B CNN / Global / Spatial / Mask-aware | 4.2C CNN / Global / Spatial / Mask-aware | 4.2D CNN / Global / Spatial / Mask-aware |
| --- | ---: | ---: | ---: | ---: |
| BCE | 0.519495 / 0.499777 / 0.465511 / 0.464718 | 0.482765 / 0.492938 / 0.470181 / 0.470677 | 0.422844 / 0.452615 / 0.352842 / 0.347272 | 0.419563 / 0.445157 / 0.378554 / 0.378503 |
| Accuracy | 0.709375 / 0.716844 / 0.737594 / 0.730594 | 0.724281 / 0.718953 / 0.730391 / 0.729328 | 0.777031 / 0.734813 / 0.790938 / 0.793188 | 0.767000 / 0.742203 / 0.774875 / 0.775922 |
| Dice | 0.661990 / 0.480496 / 0.626674 / 0.660978 | 0.638769 / 0.578178 / 0.589241 / 0.588457 | 0.728588 / 0.626506 / 0.736175 / 0.755887 | 0.722620 / 0.598523 / 0.705066 / 0.709325 |

Full-image IoU was `0.840861 / 0.833518 / 0.850701 / 0.850603` in 4.2A,
`0.717688 / 0.700987 / 0.708804 / 0.707656` in 4.2B,
`0.879947 / 0.855916 / 0.888742 / 0.891153` in 4.2C, and
`0.771755 / 0.729352 / 0.771134 / 0.772831` in 4.2D, ordered CNN / global /
spatial / mask-aware. Full BCE, accuracy, Dice, and all per-sample values are
in each run's `metrics.json` and `per_sample_metrics.csv`.

Known-region error is exactly zero for all four models in every condition.

## Complexity-stratified analysis

The unchanged Phase 3 complexity thresholds are 3.0625 and 13.2917. Entries
are CNN / global JEPA / spatial JEPA / mask-aware JEPA.

| Experiment / group | Masked IoU | Masked Dice | Masked accuracy |
| --- | --- | --- | --- |
| 4.2A simple | 0.878777 / 0.890266 / 0.920163 / 0.908926 | 0.930870 / 0.935182 / 0.955020 / 0.949315 | 0.891766 / 0.901334 / 0.930214 / 0.920822 |
| 4.2A medium | 0.468131 / 0.355839 / 0.449428 / 0.443918 | 0.572889 / 0.395269 / 0.545157 / 0.531515 | 0.707056 / 0.700504 / 0.721069 / 0.722883 |
| 4.2A complex | 0.307040 / 0.041238 / 0.219863 / 0.324252 | 0.458096 / 0.074962 / 0.352360 / 0.473810 | 0.517122 / 0.535367 / 0.547624 / 0.534993 |
| 4.2B simple | 0.858392 / 0.847994 / 0.868128 / 0.865805 | 0.921576 / 0.914986 / 0.927585 / 0.926155 | 0.890449 / 0.878862 / 0.899008 / 0.896814 |
| 4.2B medium | 0.415812 / 0.366420 / 0.393496 / 0.392833 | 0.502112 / 0.419197 / 0.464090 / 0.462021 | 0.738609 / 0.730141 / 0.745917 / 0.744556 |
| 4.2B complex | 0.316242 / 0.242194 / 0.221012 / 0.220627 | 0.464170 / 0.366741 / 0.344769 / 0.345868 | 0.533870 / 0.538127 / 0.536256 / 0.536677 |
| 4.2C simple | 0.899746 / 0.823345 / 0.905368 / 0.903260 | 0.946168 / 0.899348 / 0.949266 / 0.947978 | 0.945839 / 0.895980 / 0.947683 / 0.947858 |
| 4.2C medium | 0.612026 / 0.421564 / 0.664269 / 0.671912 | 0.730695 / 0.503803 / 0.779606 / 0.787904 | 0.829032 / 0.763105 / 0.855343 / 0.857359 |
| 4.2C complex | 0.333679 / 0.309207 / 0.317615 / 0.361143 | 0.494721 / 0.449577 / 0.468738 / 0.521426 | 0.548840 / 0.536770 / 0.564091 / 0.568769 |
| 4.2D simple | 0.879549 / 0.830498 / 0.880748 / 0.881346 | 0.935121 / 0.905300 / 0.935810 / 0.936110 | 0.932409 / 0.902256 / 0.933506 / 0.934120 |
| 4.2D medium | 0.593457 / 0.393324 / 0.601394 / 0.591703 | 0.714291 / 0.463397 / 0.723175 / 0.712758 | 0.814012 / 0.772077 / 0.829486 / 0.829183 |
| 4.2D complex | 0.339465 / 0.266930 / 0.288070 / 0.306986 | 0.503853 / 0.396955 / 0.442315 / 0.464415 | 0.547062 / 0.543881 / 0.555109 / 0.557869 |

The strongest result is 4.2A complex: mask-aware JEPA reaches 0.324252,
beating ordinary spatial JEPA by 0.104389 and the CNN by 0.017212. Random-25%
complex also improves from 0.317615 to 0.361143 and beats the CNN.

## Boundary-focused analysis

Hidden boundary pixels are hidden pixels adjacent to a known pixel in the four
cardinal directions. Interior hidden pixels are the remaining hidden pixels.
Values below are CNN / spatial JEPA / mask-aware JEPA; global JEPA is retained
in `metrics.json` but omitted here for readability.

| Experiment / region | IoU | Dice | Accuracy |
| --- | --- | --- | --- |
| 4.2A boundary | 0.621494 / 0.607206 / 0.625766 | 0.709147 / 0.686105 / 0.707778 | 0.754214 / 0.780714 / 0.767000 |
| 4.2A interior | 0.504267 / 0.478189 / 0.515537 | 0.599501 / 0.547109 / 0.598352 | 0.674500 / 0.704056 / 0.702278 |
| 4.2B boundary | 0.610635 / 0.541576 / 0.545285 | 0.709356 / 0.620870 / 0.627383 | 0.755000 / 0.761250 / 0.757750 |
| 4.2B interior | 0.513538 / 0.489793 / 0.487743 | 0.602000 / 0.566122 / 0.564723 | 0.714042 / 0.720104 / 0.719854 |
| 4.2C boundary | 0.622049 / 0.634774 / 0.650697 | 0.729091 / 0.736605 / 0.756102 | 0.777111 / 0.790962 / 0.793072 |
| 4.2C interior* | 0.741309 / 0.748466 / 0.778119 | 0.745399 / 0.754601 / 0.782209 | 0.769939 / 0.784765 / 0.804703 |
| 4.2D boundary | 0.614504 / 0.601305 / 0.604180 | 0.726047 / 0.709699 / 0.713250 | 0.767274 / 0.776432 / 0.776903 |
| 4.2D interior | 0.564900 / 0.540779 / 0.550634 | 0.648103 / 0.616042 / 0.625707 | 0.764194 / 0.754153 / 0.763134 |

*Only 163 random-25% samples contained non-empty interior regions; all other
rows had 500 available samples. Mask-aware weighting improves boundary IoU in
4.2A and 4.2C, the two conditions where its overall gain is clearest.

## Paired test analysis

Differences are computed per identical test geometry. `M-S` is mask-aware minus
ordinary spatial JEPA; `M-C` is mask-aware minus CNN.

| Experiment | Comparison | Mean | Median | Std | P25 | P75 | Win rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4.2A | M-S | +0.029158 | 0.000000 | 0.096686 | -0.019376 | +0.075677 | 0.474 |
| 4.2A | M-C | +0.008976 | +0.000500 | 0.104743 | -0.038016 | +0.062314 | 0.502 |
| 4.2B | M-S | -0.001161 | 0.000000 | 0.045087 | -0.018767 | +0.017502 | 0.380 |
| 4.2B | M-C | -0.036420 | -0.017647 | 0.077117 | -0.071479 | +0.009747 | 0.334 |
| 4.2C | M-S | +0.016157 | +0.004263 | 0.066886 | -0.025145 | +0.052646 | 0.510 |
| 4.2C | M-C | +0.028989 | +0.012691 | 0.097860 | -0.028664 | +0.065128 | 0.532 |
| 4.2D | M-S | +0.003527 | 0.000000 | 0.044821 | -0.019490 | +0.019769 | 0.434 |
| 4.2D | M-C | -0.010752 | -0.003075 | 0.060843 | -0.039591 | +0.024587 | 0.438 |

## Mask and latent alignment diagnostics

The smoke test verified the exact mask mapping: a top-left 8x8 block becomes a
top-left 4x4 block of ones in M8, and one hidden pixel contributes 0.25 to its
corresponding latent cell. For the central-25% mask, mean M8 is 0.25 and mean
W is 0.325; for the central-50% mask, they are 0.50 and 0.55. The random masks
preserve the same mean missing ratio. Every per-sample mean/min/max is saved in
`mask_weight_statistics.csv`.

The smoke gradient check measured a high-weight to low-weight relative JEPA
gradient ratio of 9.75, consistent with the approximately 10:1 weight ratio.
The alignment images in `plots/mask_latent_alignment/alignment_examples.png`
show the original geometry, 16x16 mask, 8x8 soft mask, three latent norm maps,
and W. The predicted latent norm maps remain spatially structured while W
highlights the intended hidden regions.

## Collapse diagnostics

No main run approached the collapse threshold of mean latent standard deviation
`<1e-4`.

| Run | Context std | Target std | Predictor std |
| --- | ---: | ---: | ---: |
| 4.2A | 0.353144 | 0.367489 | 0.676103 |
| 4.2B | 0.388736 | 0.371283 | 0.572191 |
| 4.2C | 0.368235 | 0.381412 | 0.854434 |
| 4.2D | 0.347790 | 0.360752 | 0.741841 |

## Conclusion

Explicitly emphasizing missing regions helps spatial JEPA in the most relevant
cases, but not uniformly. The strongest evidence is central-25% complex
topology and random-25% medium/complex topology. Central-50% remains difficult,
and the model does not beat the CNN there.

The mask-aware objective is therefore a validated structural improvement over
uniform spatial JEPA, not a reason to introduce EM inputs yet. The next
experiment should investigate the remaining contiguous-50% bottleneck using
validation-selected structural or decoder improvements. Do not implement Phase
5 automatically.

## Files and artifacts

New files:

- `src/mask_aware_spatial_jepa_losses.py`
- `scripts/train_mask_aware_spatial_jepa.py`
- `scripts/smoke_test_mask_aware_spatial_jepa.py`
- `scripts/evaluate_mask_aware_spatial_jepa.py`
- `scripts/compare_phase4_2.py`
- `tests/test_mask_aware_spatial_jepa.py`
- `docs/phase4_2_report.md`

Each main run under `outputs/phase4_2/exp_4_2A` through `exp_4_2D` contains the
checkpoint, configuration, history, four-way metrics, per-sample CSV,
latent statistics, mask-weight statistics, comparison plots, and
`plots/mask_latent_alignment/` visualizations. Existing Phase 3, Phase 4, and
Phase 4.1 artifacts were not overwritten.
