# Phase 2.5 — Forward EM surrogate validation

## Decision

**The response-aware follow-up is accepted for resonance-sensitive use, but the
surrogate should not be scaled directly to 100k yet.** A modest residual CNN
with a resonance-weighted complex loss substantially improves resonance
localization on the preserved 500-ID holdout. Its global magnitude metrics are
mixed, so it is a targeted improvement rather than a universal replacement for
the baseline.

## Dataset and split methodology

The original processed 5k subset has 4,000/500/500 train/validation/test
structures. The new 30k subset has 24,000/3,000/3,000 structures, exactly
balanced across the four families in every split (6,000/750/750 from each of
PLG, PLR, PTN, and RDN).

The 30k set is a deterministic, **nested** extension of the 5k subset, seed
42: all original 5k IDs are retained and preserve their original split. The
remaining structures are selected without replacement by family. This gives a
leakage-free internal 30k experiment and a controlled 500-structure comparison:
the 30k checkpoint was evaluated on exactly the original 5k test IDs using its
own 30k train-only normalization statistics. No test responses participate in
normalization or training.

The verified channel convention is in [dataset_conventions.md](dataset_conventions.md).
In short, raw curve index 0 is `T_y`, the y-polarized cross-reflected complex
coefficient, and raw index 1 is `R_x`, the x-polarized co-reflected complex
coefficient, for x-polarized normal incidence. `T` does not mean transmission.
Frequencies span 2.00–12.00 GHz in 0.01 GHz steps.

## Model and training

Both runs use the unchanged `ForwardSurrogateCNN`: three GELU convolutional
stages with two max-pooling operations, 2×2 adaptive pooling, and a 512-unit
fully connected head predicting `[Re(T_y), Im(T_y), Re(R_x), Im(R_x)]` at 1001
frequencies. It has 2,208,932 trainable parameters.

The controlled 30k run used normalized MSE, AdamW (learning rate 0.001, weight
decay 1e-5), batch size 64, seed 42, a maximum of 75 epochs, and patience 15.
It stopped after 29 epochs; its best validation epoch was 14 (MSE 0.303738).
Training took 171.30 s on the CPU-only PyTorch build (PyTorch 2.12.0+cpu) on a
12th Gen Intel Core i5-12500H; no GPU was available. The completed historical
5k checkpoint did not contain a wall-clock record, so its time is explicitly
reported as unavailable rather than estimated.

## Metrics and resonance method

Component MAE/RMSE are calculated in raw complex-coefficient units. Magnitudes
are `hypot(real, imag)`. Pearson correlation is calculated for each individual
magnitude spectrum and then averaged (with medians saved in `metrics.json`),
not by flattening spectra together.

For resonance analysis, `scipy.signal.find_peaks` is applied independently to
each true magnitude spectrum and its negative, thereby detecting prominent
peaks and dips. Settings are absolute prominence 0.03 and minimum spacing 10
samples (0.10 GHz). A true feature is matched to the nearest predicted
prominent extremum of the same kind. Frequency-error summaries are conditional
on that match and the match rate is reported explicitly. Resonance-region MAE
is the magnitude MAE in the union of ±0.10 GHz windows around true features.
This avoids labeling small numerical ripples as resonances and handles spectra
with no qualifying feature without failure.

## Results

### 5k baseline (its held-out 500 test structures)

| Metric | Value |
| --- | ---: |
| normalized MSE | 0.323328 |
| complex MAE | 0.056750 |
| `T_y` magnitude MAE / RMSE | 0.063985 / 0.091752 |
| `R_x` magnitude MAE / RMSE | 0.043951 / 0.079691 |
| mean `T_y` / `R_x` correlation | 0.454141 / 0.747983 |
| median `T_y` / `R_x` correlation | 0.482697 / 0.824197 |
| resonance frequency error, mean / median / p90 / max (GHz) | 0.595004 / 0.140000 / 1.870000 / 5.840000 |
| resonance feature match rate | 94.58% (1291/1365) |
| resonance-region magnitude MAE | 0.248114 |
| inference time | 0.3258 s total; 0.6516 ms/sample |

The raw component MAEs are `T_y` real 0.073377, `T_y` imaginary 0.076256,
`R_x` real 0.039264, and `R_x` imaginary 0.038103. Corresponding RMSEs are
0.123559, 0.125377, 0.070972, and 0.071498.

### Final 30k model (its full 3,000-structure test set)

| Metric | Value |
| --- | ---: |
| normalized MSE | 0.311496 |
| complex MAE | 0.052415 |
| `T_y` magnitude MAE / RMSE | 0.050400 / 0.074016 |
| `R_x` magnitude MAE / RMSE | 0.043855 / 0.078385 |
| mean `T_y` / `R_x` correlation | 0.368827 / 0.753880 |
| resonance frequency error, mean / median / p90 / max (GHz) | 0.852919 / 0.410000 / 2.396000 / 6.740000 |
| resonance feature match rate | 82.43% (7085/8595) |
| resonance-region magnitude MAE | 0.237223 |
| inference time | 0.2612 s total; 0.0871 ms/sample |

### Controlled 5k vs 30k comparison on the same 500 IDs

| Metric | 5k | 30k |
| --- | ---: | ---: |
| normalized MSE | 0.323328 | 0.296555 |
| `T_y` magnitude MAE | 0.063985 | 0.049089 |
| `R_x` magnitude MAE | 0.043951 | 0.041775 |
| mean `T_y` correlation | 0.454141 | 0.352576 |
| mean `R_x` correlation | 0.747983 | 0.763447 |
| resonance frequency MAE (GHz) | 0.595004 | 0.843832 |
| resonance-region magnitude MAE | 0.248114 | 0.238875 |
| feature match rate | 94.58% | 82.20% |
| training time (s) | not recorded historically | 171.303 |
| inference time (ms/sample) | 0.6516 | 0.4748 |

Thus six times more data improves global fit and amplitude accuracy, but it
does not resolve—and on this exact shared holdout worsens—feature localization.
The full 30k test result shows the same issue, so this is not explained only by
the shared subset.

## Failure and geometry analysis

The 20 worst 5k examples, ranked *only* by normalized MSE, contain 18 RDN,
one PTN, and one PLG geometry. Visual inspection of
`plots/failures/worst_predictions.png` shows that the CNN generally follows
broad `T_y` trends but smooths or omits narrow dips; it commonly suppresses the
sharp, multi-peak `R_x` response. The dominant failure modes are missed or
shifted resonances and amplitude suppression, rather than a uniform baseline
offset. Random predictions, easiest/hardest tables, and all per-sample fields
are saved alongside the evaluation artifacts.

Geometry descriptors use occupied fill ratio, 4-connected occupied components,
and unlike horizontal/vertical adjacent-pixel transitions (excluding the outer
image border). Error has negligible fill-ratio association (Pearson r=0.049
for 5k and -0.001 for 30k normalized MSE) but strong associations with connected
components (r=0.661/0.688) and boundary transitions (r=0.663/0.702). The
analogous associations with `T_y` magnitude MAE are even stronger for components
(0.779/0.793) and boundary transitions (0.783/0.823). These are descriptive
correlations, not causal claims, but they align with the RDN-heavy failure set.

## Differentiability check

Using a continuous `[0,1]^(16×16)` relaxation and a normalized predicted-response
MSE to a held-out target, both checkpoints have finite, nonzero gradients over
all 256 pixels. For the final 30k full-test diagnostic: mean absolute gradient
0.004992, median 0.003552, p99 0.019131, and maximum 0.021332. The saved heatmap
is `plots/gradient_sanity.png`. The network is computationally differentiable
and gradients neither vanish identically nor explode in this check. This does
not establish useful inverse-design gradients for binary fabrication constraints.

## Recommendation and Phase 3

The response-aware model passes the resonance acceptance criterion and is the
preferred candidate for the next controlled experiment. Do **not** scale
directly to 100k or use it as the sole physics objective for physics-aware JEPA
yet: its y-cross magnitude MAE is slightly worse and its global normalized MSE
is higher than the baseline on the shared holdout. First repeat this candidate
with at least one additional seed and compare calibration and worst-case
failures; then decide whether 100k is justified. No JEPA, PINN, inverse design,
diffusion, or Maxwell-solver-in-the-loop code was introduced in this phase.

## Response-aware Phase 2.5 follow-up

The follow-up keeps the nested 30k split, seed 42, batch size 64, AdamW
learning rate 0.001, weight decay 1e-5, 75-epoch ceiling, and patience 15. The
candidate has shallow residual geometry blocks, retains a 4x4 spatial feature
map, and adds a small 1-D spectral refinement head. Its loss weights target
magnitude curvature while retaining normalized complex-component MSE. It has
1,384,120 parameters, compared with 2,208,932 for the baseline.

The run used the CUDA-enabled Python 3.14 environment (`torch 2.10.0+cu126`)
on the RTX 3050 Laptop GPU. It early-stopped at epoch 25; best validation loss
was at epoch 10. Training took 148.26 seconds. Artifacts are in
`outputs/phase2_forward_30k_response_aware_gpu/`.

### Controlled shared-500 test comparison

| Metric | 5k baseline | 30k baseline | Response-aware 30k |
| --- | ---: | ---: | ---: |
| normalized MSE | 0.323328 | 0.296555 | 0.312470 |
| y-cross magnitude MAE | 0.063985 | 0.049089 | 0.064300 |
| x-co magnitude MAE | 0.043951 | 0.041775 | 0.043339 |
| mean y-cross correlation | 0.454141 | 0.352576 | 0.524961 |
| mean x-co correlation | 0.747983 | 0.763447 | 0.656879 |
| resonance frequency MAE (GHz) | 0.595004 | 0.843832 | 0.316475 |
| resonance-region magnitude MAE | 0.248114 | 0.238875 | 0.227351 |
| feature match rate | 94.58% | 82.20% | 99.56% |

The response-aware model therefore improves the two resonance-focused
quantities over both existing runs, while broad spectral metrics remain a
trade-off. On the full 3,000-structure 30k test set, it reaches normalized MSE
0.325241, resonance frequency MAE 0.343248 GHz, resonance-region MAE 0.229442,
and feature match rate 99.49%. Its continuous-geometry gradient check is
finite and nonzero for all 256 pixels.
