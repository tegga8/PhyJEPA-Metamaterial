# Phase 4.1 - Spatial / Masked JEPA Completion

## Decision

Phase 4.1 supports the spatial-latent hypothesis, but only partially. The
spatial JEPA beats the previous global-latent JEPA in all four conditions and
beats the supervised CNN on random 25% masking. It remains slightly below the
CNN in the other three conditions, so this is outcome B: spatial structure
helps, but the JEPA objective/decoder is not yet a reliable replacement for
the supervised baseline.

Do not add electromagnetic conditioning yet. Phase 4.1 is complete and the
next experiment should remain structural and be selected using validation
performance.

Phase 3 and Phase 4 outputs were not overwritten. The Phase 2 forward EM CNN
was not modified or used.

## Motivation and hypothesis

Phase 4 compressed a complete 16x16 geometry into one global 128-dimensional
vector. The decoder therefore had to reconstruct position, boundaries, and
connectivity from a single vector. Phase 4.1 tests whether retaining spatial
coordinates in the representation addresses that bottleneck.

The hypothesis was that an aligned spatial latent map would preserve local
topology better than the global vector, especially for medium and complex
geometries.

## Architecture

The input remains exactly `[partial_geometry, mask]` with shape
`[B,2,16,16]`. The target encoder receives only complete geometry with shape
`[B,1,16,16]`.

| Component | Implementation | Output |
| --- | --- | --- |
| Context encoder | Conv 2->32, GroupNorm/ReLU, Conv 32->64, GroupNorm/ReLU, 2x downsample, Conv 64->64, GroupNorm/ReLU | `[B,64,8,8]` |
| Target encoder | Same geometry encoder with one input channel | `[B,64,8,8]` |
| Predictor | Conv 64->128, GroupNorm/GELU, Conv 128->128/GELU, Conv 128->64 | `[B,64,8,8]` |
| Decoder | Conv 64->64/ReLU, upsample 8x8->16x16, Conv 64->32/ReLU, Conv 32->1 | `[B,1,16,16]` |

No global pooling or flattening is applied before the JEPA objective. Each
latent location is compared to the aligned target location.

The target encoder is initialized from the context encoder. Its first
two-channel convolution is converted to one channel by averaging over the
input-channel dimension. After initialization, target parameters are excluded
from AdamW, receive no direct gradients, and are updated only by

`theta_target <- 0.996 * theta_target + 0.004 * theta_context`.

The main objective is

`L = L_JEPA + 0.1 * L_masked_BCE`,

where each spatial latent vector is normalized along its channel dimension and
the target representation is stop-gradient. Masked BCE is computed only over
missing geometry pixels.

Final geometry uses exact known-pixel compositing:

`G_final = (1-M) * G_partial + M * sigmoid(decoder(z_pred))`.

## Experiments and compute

The same 5k subset, seed 42, split, mask generation, threshold, and Phase 3
complexity descriptors were reused.

| Run | Mask | Best epoch | Epochs | Seconds | Peak memory |
| --- | --- | ---: | ---: | ---: | ---: |
| 4.1A | central 25% | 15 | 25 | 39.312 | 52.0 MiB |
| 4.1B | central 50% | 15 | 25 | 58.345 | 52.0 MiB |
| 4.1C | random 25% | 11 | 21 | 53.587 | 52.0 MiB |
| 4.1D | random 50% | 11 | 21 | 56.331 | 52.0 MiB |

Runs used AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 64,
maximum 75 epochs, patience 10, EMA decay 0.996, latent channels 64, and
predictor hidden channels 128. The environment was Python 3.14.3, PyTorch
2.10.0+cu126, CUDA 12.6, on the RTX 3050 Laptop GPU.

Parameter counts:

- Phase 3 CNN: 332,897
- Phase 4 global JEPA: 726,113 total / 469,281 trainable
- Phase 4.1 spatial JEPA: 397,793 total / 341,729 trainable

The spatial model is 1.20x the CNN's total parameters and 1.03x its trainable
parameters, while being substantially smaller than the global JEPA.

## Three-way masked completion results

Masked IoU is the primary metric. All values are means over the same 500 test
geometries.

| Experiment | CNN | Global JEPA | Spatial JEPA | Spatial-CNN | Spatial-Global |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4.1A central 25% | 0.560517 | 0.441019 | 0.540335 | -0.020182 | +0.099316 |
| 4.1B central 50% | 0.540114 | 0.496369 | 0.504855 | -0.035259 | +0.008486 |
| 4.1C random 25% | 0.621486 | 0.527071 | 0.634318 | +0.012832 | +0.107247 |
| 4.1D random 50% | 0.610473 | 0.506742 | 0.596194 | -0.014279 | +0.089451 |

Masked metrics:

| Metric | 4.1A CNN / Global / Spatial | 4.1B CNN / Global / Spatial | 4.1C CNN / Global / Spatial | 4.1D CNN / Global / Spatial |
| --- | ---: | ---: | ---: | ---: |
| BCE | 0.519495 / 0.499777 / 0.465511 | 0.482765 / 0.492938 / 0.470181 | 0.422844 / 0.452615 / 0.352842 | 0.419563 / 0.445157 / 0.378554 |
| Accuracy | 0.709375 / 0.716844 / 0.737594 | 0.724281 / 0.718953 / 0.730391 | 0.777031 / 0.734813 / 0.790938 | 0.767000 / 0.742203 / 0.774875 |
| IoU | 0.560517 / 0.441019 / 0.540335 | 0.540114 / 0.496369 / 0.504855 | 0.621486 / 0.527071 / 0.634318 | 0.610473 / 0.506742 / 0.596194 |
| Dice | 0.661990 / 0.480496 / 0.626674 | 0.638769 / 0.578178 / 0.589241 | 0.728588 / 0.626506 / 0.736175 | 0.722620 / 0.598523 / 0.705066 |

Full BCE, accuracy, IoU, and Dice are saved for all three models in each
experiment's `metrics.json`. Known-region error is exactly zero for CNN,
global JEPA, and spatial JEPA in every condition.

## Complexity-stratified analysis

The unchanged Phase 3 thresholds `3.0625` and `13.2917` define simple,
medium, and complex tertiles. Entries below are CNN / global JEPA / spatial
JEPA.

| Experiment / group | Masked IoU | Masked Dice | Masked accuracy |
| --- | --- | --- | --- |
| 4.1A simple | 0.878777 / 0.890266 / 0.920163 | 0.930870 / 0.935182 / 0.955020 | 0.891766 / 0.901334 / 0.930214 |
| 4.1A medium | 0.468131 / 0.355839 / 0.449428 | 0.572889 / 0.395269 / 0.545157 | 0.707056 / 0.700504 / 0.721069 |
| 4.1A complex | 0.307040 / 0.041238 / 0.219863 | 0.458096 / 0.074962 / 0.352360 | 0.517122 / 0.535367 / 0.547624 |
| 4.1B simple | 0.858392 / 0.847994 / 0.868128 | 0.921576 / 0.914986 / 0.927585 | 0.890449 / 0.878862 / 0.899008 |
| 4.1B medium | 0.415812 / 0.366420 / 0.393496 | 0.502112 / 0.419197 / 0.464090 | 0.738609 / 0.730141 / 0.745917 |
| 4.1B complex | 0.316242 / 0.242194 / 0.221012 | 0.464170 / 0.366741 / 0.344769 | 0.533870 / 0.538127 / 0.536256 |
| 4.1C simple | 0.899746 / 0.823345 / 0.905368 | 0.946168 / 0.899348 / 0.949266 | 0.945839 / 0.895980 / 0.947683 |
| 4.1C medium | 0.612026 / 0.421564 / 0.664269 | 0.730695 / 0.503803 / 0.779606 | 0.829032 / 0.763105 / 0.855343 |
| 4.1C complex | 0.333679 / 0.309207 / 0.317615 | 0.494721 / 0.449577 / 0.468738 | 0.548840 / 0.536770 / 0.564091 |
| 4.1D simple | 0.879549 / 0.830498 / 0.880748 | 0.935121 / 0.905300 / 0.935810 | 0.932409 / 0.902256 / 0.933506 |
| 4.1D medium | 0.593457 / 0.393324 / 0.601394 | 0.714291 / 0.463397 / 0.723175 | 0.814012 / 0.772077 / 0.829486 |
| 4.1D complex | 0.339465 / 0.266930 / 0.288070 | 0.503853 / 0.396955 / 0.442315 | 0.547062 / 0.543881 / 0.555109 |

Spatial JEPA closes the global-JEPA deficit particularly in medium topology:
it beats global JEPA by +0.094 IoU in 4.1A, +0.027 in 4.1B, +0.243 in 4.1C,
and +0.208 in 4.1D. Complex examples remain difficult and spatial JEPA is
still below the CNN in all four complex groups.

## Paired test analysis

Differences are computed per identical test geometry. `S-C` means spatial JEPA
minus CNN; `S-G` means spatial JEPA minus global JEPA.

| Experiment | Comparison | Mean | Median | Std | P25 | P75 | Wins |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4.1A | S-C | -0.020182 | 0.000000 | 0.119148 | -0.082213 | +0.032787 | 0.412 |
| 4.1A | S-G | +0.099316 | +0.070943 | 0.112019 | +0.012413 | +0.172789 | 0.774 |
| 4.1B | S-C | -0.035259 | -0.011561 | 0.078282 | -0.068951 | +0.011273 | 0.358 |
| 4.1B | S-G | +0.008486 | +0.015406 | 0.072194 | -0.021267 | +0.047619 | 0.568 |
| 4.1C | S-C | +0.012832 | +0.001974 | 0.101343 | -0.037032 | +0.052288 | 0.510 |
| 4.1C | S-G | +0.107247 | +0.073896 | 0.212819 | -0.019066 | +0.206867 | 0.696 |
| 4.1D | S-C | -0.014279 | -0.008393 | 0.060861 | -0.043248 | +0.020077 | 0.400 |
| 4.1D | S-G | +0.089451 | +0.057853 | 0.163535 | -0.012179 | +0.184829 | 0.710 |

Each `per_sample_metrics.csv` contains sample ID, mask metadata, all three
model metrics, both paired differences, and complexity descriptors. The two
paired histograms are saved as `plots/spatial_minus_cnn.png` and
`plots/spatial_minus_global.png`.

## Latent-collapse diagnostics and visualizations

The collapse threshold remains mean latent standard deviation `<1e-4`. No main
run approached it.

| Run | Context std mean [min,max] | Target std mean [min,max] | Predictor std mean [min,max] |
| --- | --- | --- | --- |
| 4.1A | 0.373680 [0.000000, 1.284813] | 0.396660 [0.000000, 1.704648] | 0.763266 [0.079007, 4.220674] |
| 4.1B | 0.392199 [0.000000, 1.934090] | 0.377054 [0.000000, 1.507171] | 0.567704 [0.057994, 3.603908] |
| 4.1C | 0.382217 [0.000000, 1.166561] | 0.376390 [0.000000, 1.517115] | 0.839935 [0.082899, 4.102627] |
| 4.1D | 0.353103 [0.000000, 1.181268] | 0.365975 [0.000000, 1.905736] | 0.801816 [0.091629, 5.027156] |

The required latent visualization is saved as
`plots/latent_norm_maps.png` in every experiment. It shows nonuniform 8x8
context, target, and predicted norm maps with corresponding spatial hotspots;
the predicted maps preserve coarse location patterns rather than becoming
constant maps. The latent summaries for every test sample are in
`latent_statistics.csv`.

## Controlled lambda=0 ablation

The optional central-25% lambda=0 run achieved a low latent loss but produced
a degenerate unsupervised decoder. Its thresholded masked IoU was `0.647156`,
equal to its masked accuracy, and it did not provide a meaningful trained
geometry decoder. This result is not used as evidence that pure JEPA improves
completion. It confirms that the fixed `0.1 * masked BCE` auxiliary term is
necessary for a useful decoder in this architecture.

## Smoke test and reproducibility

The CUDA smoke test used 500 train and 50 validation samples for two epochs. It
verified spatial shapes, finite JEPA/reconstruction losses, 192 context,
128 predictor, and 96 decoder finite nonzero gradient checks, target no-grad
behavior, EMA updates, finite latent statistics, exact compositing, and
checkpoint saving.

Every main run contains `best.pt`, `config.json`, `metrics.json`,
`training_history.csv`, `per_sample_metrics.csv`, `latent_statistics.csv`,
and `plots/`. The aggregate comparison is in
`outputs/phase4_1/comparison.md` and `comparison.csv`.

## Conclusion

Spatial latent prediction addresses the specific Phase 4 failure: it beats the
global JEPA in all four conditions, often by a large margin in medium
complexity, and reduces the gap to the CNN from the previous global model.
It is not yet a universal CNN replacement: the CNN remains better on central
25%, central 50%, and random 50%, and complex topology remains the primary
failure mode.

Therefore, spatial JEPA is a useful structural direction, but do not proceed
to physics conditioning yet. The next controlled experiment should investigate
the remaining objective/decoder or complex-topology bottleneck using validation
selection, without expanding the dataset or introducing EM inputs.

## Files

New Phase 4.1 implementation files:

- `src/spatial_jepa_completion_model.py`
- `src/spatial_jepa_completion_losses.py`
- `scripts/train_spatial_jepa_completion.py`
- `scripts/smoke_test_spatial_jepa.py`
- `scripts/evaluate_spatial_jepa_completion.py`
- `scripts/compare_phase4_1.py`
- `tests/test_spatial_jepa_completion.py`

Artifacts are under `outputs/phase4_1/`. Existing Phase 3 and Phase 4 results
remain unchanged.
