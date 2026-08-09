# Phase 2.5 — Validate the Forward EM Surrogate

## Objective

Determine whether the current forward CNN is accurate enough to act as a useful differentiable EM surrogate before moving to masked completion and JEPA.

We currently have:

`G -> S(G)`

where `G` is a 16x16 binary metasurface and `S(G)` is its simulated electromagnetic response.

The existing 5k model is working, but global MSE/MAE may hide errors around narrow resonances. This phase therefore focuses on physical accuracy, failure analysis, and deciding whether to scale from 5k to 30k samples.

---

## 1. Verify the response convention

Before using results in a paper, verify the original SUTD-PRCM documentation for the meaning of:

- T
- R
- x-polarization
- y-polarization
- reflection/transmission

Document the verified convention in:

`docs/dataset_conventions.md`

Include:

- response channel definitions
- frequency range
- frequency spacing
- polarization convention
- reflection/transmission convention

Do not infer physical meaning only from variable names.

---

## 2. Add better evaluation metrics

The current normalized MSE is not enough.

Implement:

### Complex-response MAE

For each real/imaginary channel:

`MAE = mean(abs(predicted - true))`

Report:

- y-real MAE
- y-imaginary MAE
- x-real MAE
- x-imaginary MAE

### Magnitude MAE

For each polarization:

`|R| = sqrt(real(R)^2 + imag(R)^2)`

Report:

- y-polarized magnitude MAE
- x-polarized magnitude MAE

### RMSE

Calculate RMSE for:

- real/imaginary channels
- magnitude

### Correlation

Calculate Pearson correlation between predicted and true magnitude spectra for both polarizations.

---

## 3. Resonance-aware evaluation

This is the most important addition.

For each test spectrum:

1. Compute the true magnitude spectrum.
2. Detect prominent peaks/dips using `scipy.signal`.
3. Record their frequencies.
4. Find corresponding predicted peaks/dips.
5. Calculate resonance-frequency error.

For a true resonance at `f_true` and predicted resonance at `f_pred`:

`E_f = abs(f_pred - f_true)`

Report:

- mean resonance frequency error
- median error
- 90th percentile error
- maximum error

Also calculate local MAE around resonance regions.

Compare:

`global MAE vs resonance-region MAE`

---

## 4. Create `scripts/evaluate_forward.py`

The script should:

1. Load the best checkpoint.
2. Load the test split.
3. Generate predictions.
4. Convert predictions back to physical units.
5. Calculate all metrics.
6. Save a JSON summary.
7. Save per-sample metrics as CSV.

Suggested output:

```text
outputs/phase2_forward_evaluation/
├── metrics.json
├── per_sample_metrics.csv
├── resonance_metrics.json
└── plots/
```

---

## 5. Per-sample difficulty analysis

For every test structure calculate:

- source ID
- normalized MSE
- y magnitude MAE
- x magnitude MAE
- y correlation
- x correlation
- resonance error

Sort samples by error.

Save the 10 easiest and 10 hardest examples.

---

## 6. Visualize difficult examples

Create:

`scripts/plot_forward_failures.py`

For approximately 20 worst samples show:

```text
geometry
true y spectrum
predicted y spectrum
true x spectrum
predicted x spectrum
```

Save them under:

`outputs/phase2_forward_evaluation/plots/failures/`

Also create a random 20-sample visualization for comparison.

---

## 7. Geometry-complexity analysis

For every geometry calculate simple statistics:

### Fill ratio

`occupied_pixels / 256`

### Connected components

Count connected occupied regions.

### Boundary complexity

Count occupied/unoccupied neighboring-pixel transitions.

Optionally calculate:

- horizontal symmetry
- vertical symmetry
- rotational symmetry

Then investigate:

`geometry complexity -> prediction error`

This can reveal whether certain structures are systematically harder.

---

## 8. 5k vs 30k experiment

After the evaluation infrastructure works, create a balanced 30k subset:

- 24,000 train
- 3,000 validation
- 3,000 test

Maintain the same family balance as the 5k subset.

Keep fixed:

- architecture
- loss
- optimizer
- learning rate
- seed
- evaluation procedure

Save results separately under:

`outputs/phase2_forward_30k/`

Do not overwrite the 5k experiment.

---

## 9. Compare 5k and 30k

Create:

`scripts/compare_forward_scales.py`

Compare:

| Metric | 5k | 30k |
|---|---:|---:|
| normalized MSE | | |
| y magnitude MAE | | |
| x magnitude MAE | | |
| y correlation | | |
| x correlation | | |
| resonance frequency MAE | | |
| resonance-region MAE | | |
| training time | | |
| inference time | | |

The main question is:

> Does additional data substantially improve resonance prediction?

---

## 10. Decision rule

### If 30k substantially improves resonance accuracy

Scale later to 100k.

### If 30k improves only slightly

Investigate model capacity or response representation before scaling.

### If 30k does not improve

Investigate:

- model under-capacity
- loss function
- response representation
- dataset noise
- train/test distribution

Do not automatically scale to 100k.

---

## 11. Optional architecture experiment

Only if the current CNN appears capacity-limited.

Try **one** alternative:

- slightly larger CNN
- ResNet-style CNN
- small ConvNeXt-style model
- small ViT

Do not test many architectures.

The current CNN remains the baseline.

---

## 12. Optional response representation experiment

If resonance prediction remains poor, compare the current representation:

`[Re(Ry), Im(Ry), Re(Rx), Im(Rx)]`

against:

`[|Ry|, phase(Ry), |Rx|, phase(Rx)]`

Handle phase wrapping correctly.

Do not change representation unless evaluation shows a reason to do so.

---

## 13. Do NOT add PINN yet

Do not introduce:

- Maxwell PDE residuals
- PINN
- FDTD/FEM inside training
- JEPA
- inverse design

The current objective is only:

`G -> S(G)`

We first need a trustworthy, cheap forward model.

---

## 14. Gradient sanity check

Once the surrogate is validated, check whether it provides useful gradients.

Use a continuous relaxation:

`G in [0,1]^(16x16)`

Calculate:

`gradient = dL_physics / dG`

Verify:

- gradients are finite
- gradients are not identically zero
- gradients are not exploding

This is only a sanity check. Do not perform full inverse design yet.

---

## 15. Phase 2.5 acceptance criteria

Phase 2.5 is complete when:

- [ ] Response convention verified.
- [ ] Complex MAE implemented.
- [ ] Magnitude MAE implemented.
- [ ] RMSE implemented.
- [ ] Correlation implemented.
- [ ] Resonance detection implemented.
- [ ] Resonance-frequency error implemented.
- [ ] Resonance-region error implemented.
- [ ] Per-sample error table generated.
- [ ] Worst-case samples visualized.
- [ ] Random samples visualized.
- [ ] Geometry-complexity statistics calculated.
- [ ] 30k experiment completed.
- [ ] 5k vs 30k comparison generated.
- [ ] Decision made: scale data vs improve model.
- [ ] Gradient sanity check completed.
- [ ] Results saved reproducibly.
- [ ] `docs/phase2_5_report.md` written.

---

## 16. Expected outputs

```text
outputs/
└── phase2_forward_evaluation/
    ├── metrics.json
    ├── per_sample_metrics.csv
    ├── resonance_metrics.json
    └── plots/
        ├── random_predictions.png
        ├── worst_predictions.png
        ├── resonance_errors.png
        └── error_vs_geometry_complexity.png
```

Also create:

`docs/phase2_5_report.md`

The report should contain:

1. Current surrogate architecture.
2. Dataset size.
3. All evaluation metrics.
4. Resonance performance.
5. Failure analysis.
6. 5k vs 30k comparison.
7. Whether more data is necessary.
8. Whether architecture changes are necessary.
9. Whether the surrogate is suitable for differentiable optimization.
10. Recommendation for Phase 3.

---

## 17. Phase 3 preview

Only after Phase 2.5 succeeds:

`partial metasurface -> complete metasurface`

First build a simple supervised CNN completion baseline.

Then:

`JEPA completion`

Then:

`physics-aware JEPA`

Then:

`constraint-conditioned JEPA`

Each stage should answer a separate scientific question and provide a clean ablation baseline.
