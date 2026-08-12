# Phase 6.1 — Full-Spectrum RCWA Validation

## Executive result

**Final classification: C — not trustworthy as an inverse-design physics objective.**

This bounded run completed the software validation and a partial full-spectrum RCWA convergence experiment. It did not complete the required production gate. No training, retraining, inverse design, or unconstrained generation was run.

The stopping decision was compute-based: high-order meent solves were too slow for the requested campaign. The measured order-7 full-spectrum solve took approximately 30.9 minutes for one simple geometry on CPU. An order-9 CUDA benchmark did not finish three frequencies within 124 seconds; extrapolating that lower bound to 1001 frequencies already exceeds 11 hours for one geometry/order, before the remaining geometries and orders.

## Environment

- Python 3.14.3
- PyTorch 2.10.0+cu126, CUDA 12.6
- meent 0.12.0, PyTorch backend, complex128
- NVIDIA GeForce RTX 3050 Laptop GPU available
- Existing test suite: **56 passed, 0 failed**

## Full-spectrum protocol and geometry manifest

The authoritative grid is 2.00–12.00 GHz, 1001 points, with 0.01 GHz spacing. The frequency-vector and response-layout tests pass for this grid.

The required deterministic seed-42 manifest was generated at [validation_geometry_manifest.csv](../outputs/phase6_1/validation_geometry_manifest.csv). It contains nine source-ID-disjoint held-out geometries: three simple, three medium, and three complex. The manifest is selection metadata only; the nine-geometry downstream RCWA sweep was not started because no production Fourier order was established.

## Physical setup and mapping

The implementation uses a 10 × 10 mm period, 0.5 mm raster cells, 16 × 16 stored binary geometry pixels, and a two-cell (1 mm) air border. Patch copper is 0.018 mm thick, backing copper is 0.18 mm thick, substrate εr is 2.65 with loss tangent 0.003, and copper conductivity is 5.8 × 10⁷ S/m. Substrate thickness remains a calibration parameter; it was not established from the original CST source.

The physically documented mode mapping is `s_to_ty_p_to_rx`: meent zero-order s/TE is retained as the cross-polarized Ty coefficient and p/TM as the co-polarized Rx coefficient. The earlier exploratory Phase 6 comparison used `p_to_ty_s_to_rx` because it matched stored responses better at that time. These mappings are explicitly separate and remain unresolved; neither was silently relabeled as ground truth.

## Fourier-order convergence

The one completed representative geometry is simple, source ID `PLR/Data_003/014340`. Every listed spectrum below contains the full 1001-point grid. Successive packed-response MSE values were:

| transition | full-spectrum MSE | measured runtime |
|---|---:|---:|
| N=1 → N=3 | 0.00710550 | N=3: 43.5 s |
| N=3 → N=5 | 0.00615410 | N=5: 356.1 s |
| N=5 → N=7 | 0.00994316 | N=7: 1852.9 s |

The required criterion was `N=9 → N=11 <= 1e-4` for simple, medium, and complex representatives. N=9/N=11 and the medium/complex convergence runs were not completed. The available sequence is also non-monotonic and does not establish convergence. Therefore:

**Convergence status: NO CONVERGENCE ESTABLISHED.**

There is no selected production Fourier order. The available spectra are retained under [outputs/phase6_1/convergence](../outputs/phase6_1/convergence) as partial evidence only.

## Physical sanity checks

The existing three-point sanity run passed the one-percent passivity diagnostic for empty, uniform-metal, and symmetric-square cases. The reported quantity is the sum of meent diffraction efficiencies `de_ri + de_ti`; it is a diagnostic and does not impose an unqualified `|R|² + |T|² = 1` identity.

- Maximum reflected-plus-transmitted power: 0.999876
- Maximum ordinary transmitted power: approximately 4.84 × 10⁻¹¹⁴ in the recorded sanity cases
- All recorded sanity cases passed the ≤1% passivity bound

These checks support the implementation’s basic passive-power behavior but do not establish spectral or Fourier convergence.

## Dataset, CNN, and generated comparisons

The required nine-geometry full-spectrum comparison was not run because the convergence gate failed. Likewise, substrate-thickness calibration over 0.10–0.50 mm and the 20/20/20 generated-candidate validation were not run. Consequently, the following quantities are intentionally **not reported** for Phase 6.1:

- nine-geometry RCWA → stored-dataset MSE;
- nine-geometry CNN → dataset and CNN → RCWA MSE;
- full-spectrum target-error correlations and ranking agreement;
- full-spectrum exploitation-case count;
- independently evaluated Phase 5B physical Pareto frontier.

The prior Phase 6 exploratory artifacts remain available in [outputs/phase6_rcwa/metrics.json](../outputs/phase6_rcwa/metrics.json), but they used three frequencies, three structures, order 1 for the comparison, and three candidates per model. They must not be presented as Phase 6.1 full-spectrum evidence. That exploratory run reported a maximum passive power of 0.999876, CNN–RCWA MSE 0.273063, and four median-screened low-CNN/high-RCWA rows; those values are context only.

## Limitations and recommendation

The independent RCWA implementation is usable for bounded experiments and passes all 56 current tests, but this run cannot answer whether lower frozen-CNN target error implies lower independently evaluated RCWA target error. The unresolved polarization mapping, unestablished Fourier convergence, uncalibrated substrate thickness, and incomplete generated ranking test are decisive limitations.

Do not use the frozen CNN physics loss for unconstrained inverse design. Do not start training or a new research phase from this report. If Phase 6.1 is resumed later, the next step is a better-supported RCWA execution path or a substantially faster validated solver configuration, followed by N=9/N=11 convergence on all three complexity groups before any calibration or generated-candidate claims are made.

## Reproducibility artifacts

- Solver: [src/rcwa_solver.py](../src/rcwa_solver.py)
- Validation helpers: [src/rcwa_validation.py](../src/rcwa_validation.py)
- Convergence script: [scripts/analyze_rcwa_convergence.py](../scripts/analyze_rcwa_convergence.py)
- Partial full-spectrum spectra: [outputs/phase6_1/convergence](../outputs/phase6_1/convergence)
- Nine-geometry manifest: [outputs/phase6_1/validation_geometry_manifest.csv](../outputs/phase6_1/validation_geometry_manifest.csv)
- Earlier exploratory report: [phase6_rcwa_validation.md](phase6_rcwa_validation.md)
