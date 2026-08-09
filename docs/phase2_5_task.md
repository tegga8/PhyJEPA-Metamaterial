# Phase 2.5 task — Forward EM surrogate validation

## Objective

Determine whether the current forward CNN is accurate enough to serve as a
cheap differentiable electromagnetic surrogate before later physics-aware JEPA
work. The scope is strictly:

```text
16×16 geometry → complex electromagnetic reflection response
```

## In scope

- Verify and document the SUTD-PRCM response convention.
- Evaluate complex-component MAE/RMSE, magnitude MAE/RMSE, and per-spectrum
  Pearson correlation.
- Detect prominent resonance peaks and dips, measure feature-location error,
  and compare global and resonance-window error.
- Save aggregate and per-sample diagnostics, random predictions, and worst-case
  failures.
- Relate prediction error to fill ratio, 4-connected components, and boundary
  transitions.
- Build a deterministic, family-balanced 30k subset that preserves the original
  5k split assignments; train the unchanged baseline architecture on it.
- Compare 5k and 30k on the shared 500-ID holdout and report training/inference
  cost.
- Verify finite, nonzero gradients through a continuous geometry relaxation.
- Record reproducibility metadata, tests, and an evidence-based decision.

## Out of scope

- JEPA or masked completion
- PINNs, Maxwell residuals, FDTD, or FEM in training
- Inverse design
- Diffusion models, GANs, or topology completion

## Completion criteria

- [x] Channel convention documented in [dataset_conventions.md](dataset_conventions.md).
- [x] Forward metrics, resonance analysis, and geometry descriptors implemented.
- [x] 5k evaluation artifacts generated in `outputs/phase2_forward_evaluation/`.
- [x] Nested 30k subset generated with 24k/3k/3k family-balanced splits.
- [x] 30k baseline trained and evaluated without changing its architecture or
  optimizer settings.
- [x] Controlled 5k/30k comparison generated in
  `outputs/phase2_forward_scale_comparison/`.
- [x] Gradient sanity visualization generated.
- [x] Tests pass.
- [x] Evidence and recommendation recorded in
  [phase2_5_report.md](phase2_5_report.md).

## Decision

The unchanged 30k baseline improves some global amplitude metrics but does not
improve resonance localization. The controlled response-aware follow-up passes
the resonance criterion on the preserved 500-ID holdout: feature match rate
increases to 99.56% and resonance-frequency MAE falls to 0.316 GHz. Keep this
candidate for resonance-sensitive follow-up, but do not scale to 100k or use it
as the sole physics objective until a second seed and calibration/worst-case
checks confirm the trade-off in global metrics.
