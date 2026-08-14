# Physics-JEPA v3 Reproduction — Seed-123 Confirm Run

Label: `physics_jepa_v3_reproducibility`
Date: 2026-08-13
Status: **COMPLETE** — reproduction classified **REPRODUCED**.

## Purpose

The v3 experiment (`physics_jepa_v3_frequency_relational`) ran with seed 42.
This confirm run repeats the full v3 pipeline on the identical 5k-windowed
30k dataset with a fresh seed (123) plus a shuffled-pair control (seed 456),
to verify the central claim is not seed-specific: **the geometry-derived
predictor latent organizes by EM-response similarity (positive correct vs
shuffled separation) while staying geometry-decoupled and rank-stable.**

Everything is otherwise identical to the original v3 run: architecture
(`PhysicsJEPAFrequencyRelational`), optimizer (AdamW lr 1e-3 wd 1e-4, bs 64),
relational margin-term weight 0.1 / margin 0.2, variance 0.5 / covariance
0.05, EMA decay 0.996, centering off, 75 epochs / patience 10, 30k split.

## Reproduction protocol

| Run | Seed | Split | Config |
| --- | --- | --- | --- |
| correct | 123 | 24k / 3k / 3k, shuffle-seed 123 | same as v3 |
| shuffled control | 123 | same split, response/geometry pairings shuffles, shuffle-seed 456 | same as v3 |

Teaching signals (geometry ordering, family ordering, relational triplets) are
derived from the shuffled pairings in the control. Outputs:

* `outputs/physics_jepa_v3_repro/correct/` — trained model, config, history
* `outputs/physics_jepa_v3_repro/shuffled/` — shuffled control
* `outputs/physics_jepa_v3_repro/metrics.json`, `summary.csv`, `plots/`
* `outputs/physics_jepa_v3_repro/cached_latents/*/test/` — cached test latents

## Reproduction criteria (pre-registered)

The run is classified `REPRODUCED` iff all five criteria hold:

1. correct `z_pred` rho (latent-vs-EM-response distance, Spearman) > 0
2. correct rho > shuffled rho by a margin (separation > 0.05)
3. `|geometry rho|` < 0.2 (latent does not reduce to geometry)
4. no NaN in cached `z_target` / `z_pred` test latents
5. separation positive

## Results

| metric | original v3 (seed 42) | confirm (seed 123) |
| --- | --- | --- |
| EM-distance rho (z_pred) | +0.1495 | **+0.1280** |
| geometry-distance rho (z_pred) | -0.0092 | **-0.0286** |
| shuffled EM rho (z_pred) | -0.1445 | **-0.1633** |
| correct-vs-shuffled separation | +0.2940 | **+0.2913** |
| target rank-1 fraction | 0.0390 | 0.0399 |
| predictor rank-1 fraction | 0.7426 | 0.7084 |
| target effective rank | 31.58 | 31.53 |
| predictor effective rank | 1.77 | 1.94 |
| response probe R2 (z_pred) | 0.0598 | 0.0812 |
| response probe R2 (z_target) | 0.6562 | 0.6501 |
| resonance Ty MAE GHz (z_pred) | 3.264 | 3.211 |
| resonance Rx MAE GHz (z_pred) | 3.116 | 2.981 |
| feature-count MAE (z_pred) | 0.4694 | 0.4759 |
| within-family rho (z_pred) | - | -0.0785 |
| cross-family rho (z_pred) | - | 0.1989 |

Classification: `REPRODUCED` (all five criteria `True`).

## Notes and caveats

* The rank-1 / effective-rank profile transfers almost exactly (target
  effective rank 31.5, predictor ~1.8–1.9), confirming the predictor latent is
  intrinsically low-rank and symmetric-dominated in both runs.
* Within/cross-family Spearman (train-family vs test-family, fully exclusive)
  were not collected in the original run; seed-123 values are reported for
  future reference only. Their small magnitude is consistent with the global
  rho coming mostly from a coarse separation of EM-response neighborhoods.
* The positive separation is reproduced; the run does not attempt to explain
  the mechanism (probed in `physics_jepa_open_questions.md`).

## Artifacts

Trained with `scripts/train_physics_jepa_v3.py --mode full` (correct:
`--seed 123 --shuffle-seed 123`; shuffled initial control:
`--tag shuffled --shuffled-pairs --shuffle-seed 456 --seed 123`), evaluated
with `scripts/evaluate_physics_jepa_v3.py`, aggregated with
`scripts/evaluate_physics_jepa_v3_repro.py` (metrics/summary/plots and the
reproduction classification). Raw outputs under
`outputs/physics_jepa_v3_repro/{correct,shuffled}/`, plus `metrics.json`,
`summary.csv` and `outputs/physics_jepa_v3_repro/plots/*`.