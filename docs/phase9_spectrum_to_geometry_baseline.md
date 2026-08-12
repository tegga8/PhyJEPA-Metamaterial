# Phase 9 — Deterministic Spectrum → Geometry Baseline

## Hypothesis and contract

The first unconstrained inverse-design baseline is a deterministic mapping from
target EM response to a new geometry:

```text
S_target [B,4,1001] -> inverse model -> geometry
```

The inverse model may receive only the normalized target spectrum. The
implementation asserts and documents that it receives no partial geometry,
mask, source ID, original geometry, complexity metadata, or target geometry
latent as an input. Geometry and frozen geometry latents are supervised labels
and evaluation targets only.

## Fixed data and models

The experiment used the verified 5k subset with seed 42: 4,000 train, 500
validation, and 500 test samples. The Phase 8 complete-geometry autoencoder was
frozen. Its latent is `[B,64,8,8]`, and its decoder is used by the latent
predictor to decode predicted latents.

Three required baselines were compared:

1. EM nearest neighbor: the nearest response in the training set only, with
   its paired training geometry returned;
2. direct MLP: normalized spectrum directly to 16×16 geometry logits;
3. deterministic latent predictor: normalized spectrum to `[64,8,8]` latent,
   decoded by the frozen geometry decoder.

The latent objective was `latent MSE + 1.0 * geometry BCE`. The direct MLP used
geometry BCE. Binary geometry metrics use threshold 0.5. The selected Phase 7B
30k resonance-aware CNN was used only after generation for learned screening
response MSE; no surrogate loss was used during inverse training.

## Test results

| Method | latent MSE | geometry BCE | IoU | Dice | pixel accuracy | occupancy abs. difference | screening response MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EM nearest neighbor (train only) | 1.290497 | n/a | 0.479657 | 0.611589 | 0.678203 | 0.064328 | **0.259249** |
| direct MLP | n/a | 0.509034 | 0.477061 | 0.580257 | **0.715469** | 0.110313 | 0.483234 |
| latent predictor | **0.756675** | 0.552146 | **0.494825** | 0.610101 | 0.708016 | 0.076109 | 0.444132 |

The latent predictor improves IoU over the two baselines only modestly and
does not improve Dice over nearest neighbor. Most importantly, the nearest
neighbor is substantially better in screened response space. The deterministic
latent mapping therefore does not yet satisfy the relevant inverse-design
objective.

## Complexity-stratified response error

| Method | simple | medium | complex |
| --- | ---: | ---: | ---: |
| EM nearest neighbor | 0.034821 | 0.157808 | 0.592612 |
| direct MLP | 0.278783 | 0.433326 | 0.747474 |
| latent predictor | 0.144386 | 0.335738 | 0.864226 |

The latent predictor is competitive on simple geometries but degrades most on
the complex group. This is consistent with the forward-surrogate and geometry
autoencoder diagnostics: complexity is a real bottleneck, not merely a single
global average artifact.

## Failure diagnosis

The geometry autoencoder passed its representation gate, so the failure is not
an obviously unusable decoder. The inverse regression itself is difficult:
the response-to-latent mapping is ambiguous and the deterministic predictor
appears to average over geometry solutions. The train-only nearest neighbor
also provides strong evidence that response similarity can retrieve a better
screened response than the learned deterministic geometry output.

This result does not show that JEPA-derived geometry representations are
useless. It shows that a one-output deterministic predictor is not sufficient
for the many-to-one inverse relation represented by this dataset.

## Scientific decision

**Deterministic baseline: insufficient as the final inverse-design system.**

Proceeding to a one-to-many stochastic latent generator is justified as the
next controlled experiment, but only to test whether controlled diversity can
improve response satisfaction while preserving validity. Do not add a
differentiable surrogate physics loss yet. Candidate screening/ranking remains
the appropriate role for the selected forward surrogate.

## Reproducibility artifacts

- Inverse models: [spectrum_inverse_models.py](../src/spectrum_inverse_models.py)
- Training/evaluation: [train_phase9_inverse_baselines.py](../scripts/train_phase9_inverse_baselines.py)
- Tests: [test_spectrum_inverse_models.py](../tests/test_spectrum_inverse_models.py)
- Results: `../outputs/phase9_inverse_baselines/metrics.json`
- Per-sample diagnostics: `../outputs/phase9_inverse_baselines/*_per_sample.csv`

