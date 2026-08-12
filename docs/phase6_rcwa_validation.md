# Phase 6 — Independent Python RCWA Validation

## Outcome

**Classification: C — the Phase 2.5 surrogate is not yet trustworthy as an inverse-design physics objective.**

This phase uses the Python package `meent` with its PyTorch backend. It does not use CST, HFSS, COMSOL, or another desktop EM application. The results are independent RCWA calculations, not the original dataset ground truth.

## Environment and setup

- Python: `3.14.3`; PyTorch: `2.10.0+cu126`; CUDA: `12.6`; device: `NVIDIA GeForce RTX 3050 Laptop GPU`.
- RCWA package: `meent 0.12.0`, PyTorch backend, complex128.
- Geometry: 16×16 binary pixels mapped to a 20×20 raster with 0.5 mm cells, 1 mm air padding, and 10×10 mm period.
- Patch/backing thickness: 0.018/0.18 mm; substrate εr=2.65, loss tangent=0.003.
- Copper: σ=5.8×10⁷ S/m, μr=1. Meent’s passive exp(+iωt) convention uses negative imaginary ε internally; this is recorded in solver metadata.
- The project response representation remains four real channels. Calibration selected the explicit comparison mapping `p_to_ty_s_to_rx`: meent TM/p → stored channel 0 and TE/s → stored channel 2. This is an empirical mapping for comparison; it conflicts with the repository’s prose labels and must not be hidden.

## Frequency and convergence limits

The processed ML grid is 2–12 GHz at 1001 points. The completed RCWA validation used only `3` points: `2.0, 7.0, 12.0`. This is an exploratory compute-limited check, not a full-spectrum validation.

Orders 1, 3, 5, and 7 were tested on one representative geometry. The successive complex-response MSEs were `N1: n/a, N3: 0.000317928, N5: 0.00714379, N7: 0.0075401`. Orders 5 and 7 are closer than the lower-order jump, but no asymptotic convergence threshold was established. The validation artifacts therefore use order 1 only as a quick, reproducible exploratory setting; it is not presented as a converged production order.

## Sanity checks

Empty cell, uniform metal, and symmetric square tests pass the recorded one-percent passivity check. The largest reflected-plus-transmitted power was `0.999876`; ordinary transmission is numerically negligible because of the copper backing. The full suite passes: **52 tests**.

## Stored-data calibration

The quick calibration contains one held-out geometry per complexity group (3 total) and compares substrate candidates 0.15 and 0.20 mm. The best candidate is 0.15 mm by agreement only, not a claim about the original substrate.

| Metric | Best candidate |
|---|---:|
| Normalized overall MSE | 0.261084 |
| Re(Ty) normalized MSE | 0.996598 |
| Im(Ty) normalized MSE | 0.047739 |
| Re(Rx) normalized MSE | 6.65267e-07 |
| Im(Rx) normalized MSE | 2.76342e-07 |

The magnitude agreement is much better than the real-component agreement, but three frequencies and three structures are insufficient to establish reproduction of the stored solver data.

## CNN versus RCWA

On the same three structures and three frequencies, CNN-versus-RCWA normalized MSE is **0.273063**. RCWA-versus-stored MSE is **0.261084**, while CNN-versus-stored MSE is **0.001048**. The CNN and RCWA disagree mainly in the dominant mapped Ty real component.

## Generated geometry screen

Three targets × three candidate models (Phase 5A, Phase 5B small, Phase 5B medium) were screened at the three quick frequencies. Mean CNN-versus-RCWA MSE is **0.252293**. Spearman mean is `1.0`, pairwise agreement is `0.555556`, and top-1 overlap is `1.000000`. These statistics are exploratory only; the candidate count is too small for a ranking claim. The median-based low-CNN/high-RCWA screen found `4` rows; the worst recorded row is included in `generated/metrics.json`.

## Artifacts

The machine-readable summary is [metrics.json](../outputs/phase6_rcwa/metrics.json), with combined rows in [per_sample_metrics.csv](../outputs/phase6_rcwa/per_sample_metrics.csv). Plots are under [outputs/phase6_rcwa/plots](../outputs/phase6_rcwa/plots). The solver cache is configuration-keyed and preserves the exact metadata for each run.

## Recommendation

Do not advance to unconstrained inverse design based on this evidence. First restore a healthy compute path and run a modest but real calibration at the full 1001-point grid with a demonstrably converged Fourier order, then repeat generated ranking across substantially more candidates. If CNN–RCWA disagreement and exploitation remain high, recalibrate or retrain the forward surrogate before using its loss as a physics objective.
