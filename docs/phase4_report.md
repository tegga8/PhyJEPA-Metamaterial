# Phase 4 - JEPA Partial-Structure Completion

## Decision

Phase 4 is scientifically complete, but the result is outcome C: the
supervised Phase 3 CNN remains the stronger completion model. JEPA training is
correct and reproducible, the representation did not collapse, and all four
masking conditions were evaluated with paired held-out test comparisons.
However, the JEPA + reconstruction model trails the CNN in masked IoU in all
four conditions. The evidence does not justify adding physics conditioning in
the next phase yet.

The forward EM surrogate was not modified or used. Phase 3 checkpoints and
outputs were reused unchanged.

## Research question and protocol

The experiment asks whether predicting a complete-geometry representation
with a JEPA-style context/target architecture improves topology completion over
direct supervised reconstruction.

The existing processed 5k subset was reused exactly: 4,000 train, 500
validation, and 500 test geometries. Inputs are `[partial_geometry, mask]` with
shape `[B, 2, 16, 16]`; the target encoder receives only the complete target
geometry `[B, 1, 16, 16]`. Mask generation, split offsets, complexity
descriptors, and the Phase 3 CNN checkpoints were not changed.

| Experiment | Mask | Hidden pixels | Ratio |
| --- | --- | ---: | ---: |
| 4A | centered 8x8 block | 64 | 25% |
| 4B | centered 8x16 block | 128 | 50% |
| 4C | random holes | 64 | 25% |
| 4D | random holes | 128 | 50% |

## JEPA implementation

The model contains:

1. a compact convolutional context encoder for partial geometry plus mask;
2. a separate one-channel target encoder for complete geometry;
3. a 128-to-256-to-256-to-128 predictor MLP; and
4. a compact latent-to-16x16 convolutional decoder.

The target encoder is initialized from the context encoder, with the first
two-channel convolution averaged into one geometry channel. It is excluded
from the optimizer, receives no direct gradients, and is updated only by

`theta_target <- 0.996 * theta_target + 0.004 * theta_context`.

The primary objective is normalized latent MSE with a stop-gradient target:

`L_JEPA = || normalize(z_pred) - stop_gradient(normalize(z_target)) ||^2`.

The main benchmark uses the configured auxiliary variant

`L = L_JEPA + 0.1 * L_masked_BCE`.

Masked BCE is computed only in the hidden region. The pure-JEPA variant was
also trained as a 4A ablation. It demonstrates that latent alignment alone
does not train the geometry decoder, so it is not used as the main CNN
comparison model.

All final completions use exact known-pixel compositing:

`G_final = (1 - M) * G_partial + M * sigmoid(decoder(z_pred))`.

Binary metrics threshold the missing-region prediction at 0.5. The maximum
known-region error was zero for both models in every benchmark run.

## Training and compute

All runs used AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 64,
seed 42, maximum 75 epochs, patience 10, latent dimension 128, predictor
hidden dimension 256, and EMA decay 0.996. The CUDA-enabled interpreter used
PyTorch `2.10.0+cu126`, CUDA `12.6`, Python `3.14.3`, and the RTX 3050 Laptop
GPU.

| Run | Best epoch | Completed epochs | Training seconds | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: |
| 4A | 12 | 22 | 45.310 | 52.56 MiB |
| 4B | 28 | 38 | 80.567 | 52.56 MiB |
| 4C | 22 | 32 | 76.357 | 52.56 MiB |
| 4D | 25 | 35 | 82.123 | 52.56 MiB |
| pure-JEPA 4A ablation | 4 | 14 | 25.053 | 57.08 MiB |

The Phase 3 CNN has 332,897 parameters. JEPA has 726,113 total parameters,
including the frozen EMA target encoder, and 469,281 trainable parameters.
Thus JEPA is 2.18x larger by total parameters and 1.41x larger by trainable
parameters; this is reported explicitly in the comparison rather than treated
as an equal-size model.

## CNN versus JEPA results

Metrics are means over the same 500 held-out test geometries. Masked IoU is
the primary metric.

| Metric | 4A central 25% | 4B central 50% | 4C random 25% | 4D random 50% |
| --- | ---: | ---: | ---: | ---: |
| CNN masked BCE | 0.519495 | 0.482765 | 0.422844 | 0.419563 |
| JEPA masked BCE | 0.499777 | 0.492938 | 0.452615 | 0.445157 |
| CNN masked accuracy | 0.709375 | 0.724281 | 0.777031 | 0.767000 |
| JEPA masked accuracy | 0.716844 | 0.718953 | 0.734813 | 0.742203 |
| CNN masked IoU | 0.560517 | 0.540114 | 0.621486 | 0.610473 |
| JEPA masked IoU | 0.441019 | 0.496369 | 0.527071 | 0.506742 |
| JEPA - CNN IoU | -0.119498 | -0.043745 | -0.094415 | -0.103730 |
| CNN masked Dice | 0.661990 | 0.638769 | 0.728588 | 0.722620 |
| JEPA masked Dice | 0.480496 | 0.578178 | 0.626506 | 0.598523 |

Full-image metrics are included for completeness. They are not the headline
metrics because known pixels are preserved by construction.

| Metric | 4A CNN / JEPA | 4B CNN / JEPA | 4C CNN / JEPA | 4D CNN / JEPA |
| --- | ---: | ---: | ---: | ---: |
| Full BCE | 0.255741 / 0.798451 | 0.319461 / 0.525531 | 0.275243 / 0.449171 | 0.335477 / 0.435543 |
| Full accuracy | 0.927344 / 0.929211 | 0.862141 / 0.859477 | 0.944258 / 0.933703 | 0.883500 / 0.871102 |
| Full IoU | 0.840861 / 0.833518 | 0.717688 / 0.700987 | 0.879947 / 0.855916 | 0.771755 / 0.729352 |
| Full Dice | 0.909463 / 0.904306 | 0.821626 / 0.808217 | 0.933353 / 0.919373 | 0.861272 / 0.830622 |

## Complexity-stratified results

The Phase 3 complexity score and tertile thresholds were reused (`3.0625` and
`13.2917`). Each entry is `CNN / JEPA / JEPA-CNN`.

### Masked IoU

| Group | 4A | 4B | 4C | 4D |
| --- | --- | --- | --- | --- |
| Simple | 0.878777 / 0.890266 / +0.011489 | 0.858392 / 0.847994 / -0.010398 | 0.899746 / 0.823345 / -0.076401 | 0.879549 / 0.830498 / -0.049052 |
| Medium | 0.468131 / 0.355839 / -0.112292 | 0.415812 / 0.366420 / -0.049392 | 0.612026 / 0.421564 / -0.190462 | 0.593457 / 0.393324 / -0.200133 |
| Complex | 0.307040 / 0.041238 / -0.265802 | 0.316242 / 0.242194 / -0.074048 | 0.333679 / 0.309207 / -0.024472 | 0.339465 / 0.266930 / -0.072535 |

### Masked Dice

| Group | 4A CNN / JEPA | 4B CNN / JEPA | 4C CNN / JEPA | 4D CNN / JEPA |
| --- | ---: | ---: | ---: | ---: |
| Simple | 0.930870 / 0.935182 | 0.921576 / 0.914986 | 0.946168 / 0.899348 | 0.935121 / 0.905300 |
| Medium | 0.572889 / 0.395269 | 0.502112 / 0.419197 | 0.730695 / 0.503803 | 0.714291 / 0.463397 |
| Complex | 0.458096 / 0.074962 | 0.464170 / 0.366741 | 0.494721 / 0.449577 | 0.503852 / 0.396955 |

### Masked accuracy

| Group | 4A CNN / JEPA | 4B CNN / JEPA | 4C CNN / JEPA | 4D CNN / JEPA |
| --- | ---: | ---: | ---: | ---: |
| Simple | 0.891766 / 0.901334 | 0.890449 / 0.878862 | 0.945839 / 0.895980 | 0.932409 / 0.902256 |
| Medium | 0.707056 / 0.700504 | 0.738609 / 0.730141 | 0.829032 / 0.763105 | 0.814012 / 0.772077 |
| Complex | 0.517122 / 0.535367 | 0.533870 / 0.538127 | 0.548840 / 0.536770 | 0.547062 / 0.543881 |

JEPA does not improve the difficult cases overall. Its largest deficit is in
the medium group for random masks and in the complex group for central 25%.
The small simple-group win in 4A is not consistent across mask regimes.

## Paired test analysis

The paired difference is computed per identical test geometry as
`JEPA_masked_IoU - CNN_masked_IoU`.

| Experiment | Mean | Median | Std. dev. | P25 | P75 | JEPA wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4A | -0.119498 | -0.090909 | 0.167700 | -0.233333 | 0.000000 | 0.226 |
| 4B | -0.043745 | -0.038935 | 0.070938 | -0.085007 | +0.003471 | 0.264 |
| 4C | -0.094415 | -0.078823 | 0.163136 | -0.185439 | +0.009296 | 0.278 |
| 4D | -0.103730 | -0.065976 | 0.139499 | -0.186442 | -0.008009 | 0.204 |

The complete paired rows, including `sample_id`, mask type, complexity
descriptors, and per-sample differences, are in each experiment's
`per_sample_metrics.csv`. Histograms and paired visual grids are in each
experiment's `plots/` directory.

## Collapse diagnostics and tests

The collapse monitor records mean per-dimension variance and mean standard
deviation for context, target, and predictor representations. No benchmark
run crossed the failure threshold of mean latent standard deviation `< 1e-4`.

| Run | Context std | Target std | Predictor std |
| --- | ---: | ---: | ---: |
| 4A | 0.227597 | 0.093097 | 0.864909 |
| 4B | 0.101889 | 0.041446 | 1.131671 |
| 4C | 0.057874 | 0.059679 | 1.048030 |
| 4D | 0.060084 | 0.062239 | 1.242471 |
| pure-JEPA 4A | 0.025762 | 0.011789 | 0.054363 |

The pure-JEPA ablation has much lower representation variation and its decoder
has no reconstruction training signal; this supports retaining the auxiliary
reconstruction loss for the benchmark rather than interpreting its low JEPA
loss as completion quality.

The CUDA smoke test used 500 train and 50 validation samples for two epochs.
It verified output shape `[50,1,16,16]`, 192 finite nonzero context gradient
checks, 128 finite nonzero predictor gradient checks, EMA target changes,
absence of target gradients, finite latent statistics, and successful exact
compositing.

## Visual analysis and artifacts

Each benchmark directory contains:

- `best.pt`, `config.json`, `training_history.csv`, and `metrics.json`;
- `per_sample_metrics.csv`; and
- `plots/paired_iou_histogram.png`, `random_comparisons.png`,
  `difficult_comparisons.png`, and `complex_comparisons.png`.

The visual comparisons agree with the aggregate metrics: both models preserve
the visible region exactly, while JEPA more often smooths or misplaces hidden
topology, especially in medium-complexity central completions. The aggregate
tables are also available in
`outputs/phase4_jepa/comparison.md` and `comparison.csv`.

## Conclusion and next-step decision

JEPA provides no evidence of improvement over the supervised CNN baseline in
this compact geometry-completion setup. It is more parameter-heavy and loses
on the primary masked IoU in every required condition, although it remains
stable and technically correct.

The likely bottleneck is the global 128-dimensional representation and the
decoder's need to recover spatial topology from it; EMA latent agreement alone
does not guarantee a useful spatial reconstruction. If Phase 4 is revisited,
the next experiment should be selected on validation data and should test a
spatial/multi-scale latent or a controlled decoder objective before any physics
conditioning is introduced.

Stop here. Do not implement Phase 5 automatically.

## Files

New implementation files:

- `src/jepa_completion_model.py`
- `src/jepa_completion_losses.py`
- `scripts/train_jepa_completion.py`
- `scripts/evaluate_jepa_completion.py`
- `scripts/compare_jepa_completion.py`
- `scripts/smoke_test_jepa.py`
- `tests/test_jepa_completion.py`

Phase 4 report and experiment artifacts are under `docs/phase4_report.md` and
`outputs/phase4_jepa/`. Existing Phase 3 files and outputs were not modified.
