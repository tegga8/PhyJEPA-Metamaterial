# Physics-JEPA Status (Branch B)

Current representation-gate research. **v1 and v2 are preserved as scientific
evidence. Do not modify their outputs.** The result of both versions is a
documented failure of the representation gate — this is the evidence base that
motivates the v3 frequency-aware + relational experiment.

## Representation gate (definition, frozen)

`docs/physics_representation_metrics.py` fixes the canonical metrics. The gate key metric is
`rho(D_z, D_S)` — Spearman correlation between latent Euclidean distance (L2-normalized) and
phase-2 normalized EM-response MSE distance. Supporting: `rho(D_z, D_G)` and correct-vs-shuffled
separation. Response/resonance linear probes and latent rank are secondary diagnostics.

## v1 — `physics_jepa` (seed42_32d)

Settings (from `outputs/physics_jepa/seed42_32d/config.json`):

| Setting | v1 |
| --- | --- |
| dataset / split | 30k; 24000 train / 3000 val / 3000 test, seed 42 |
| latent dims | 32 (and 64) |
| tokens / token dim | 12 / 64 |
| spectrum encoder | Conv1D 4->32->64, 12 tokens, self-attention, attention pool |
| EMA target | `ema_decay 0.996`, momentum spectrum encoder |
| loss | cross + 0.5*bootstrap + 0.1*variance (no covariance) |
| optimizer | AdamW lr 1e-3 wd 1e-4, batch 64, patience 10 |
| run | 42 epochs, best epoch 32, best val total 0.007684, CUDA, torch 2.10.0+cu126 |
| shuffled control | `outputs/physics_jepa/shuffled_32d` (config only, latents cached) |

Results (only surviving record is the comparison table in
`docs/phase13_physics_jepa_v2_collapse_fix.md`; there is **no dedicated v1 report doc**):

| Representation | raw rho(DS) | norm rho(DS) | rank-1 fraction |
| --- | ---: | ---: | ---: |
| z_target | -0.0238 | -0.0405 | 0.8911 |
| z_pred | +0.2939 | -0.1716 | 0.9759 |

- Response probe R2: z_pred 0.1585, z_target 0.2030.
- **Gate outcome: FAILED.** Severe low-rank collapse of the momentum spectrum target (rank-1 ≈ 0.89)
  and non-physical distance ordering of the geometry-derived predictor.

## v2 — `physics_jepa_v2_collapse_fix` (seed42_32d)

Settings (from `outputs/physics_jepa_v2/seed42_32d/config.json`):

| Setting | v2 |
| --- | --- |
| dataset / split | same 30k, same v1 split, seed 42 |
| latent dim | 32 |
| tokens / token dim | 12 / 64 |
| spectrum encoder | same tokenized Conv1D + attention |
| EMA target | `ema_decay 0.996` (same) |
| loss | cross + 0.5*bootstrap + 0.5*variance + 0.05*covariance |
| covariance on | `z_online` + `z_geometry` (online and geometry branches) |
| target centering | OFF (`target_centering=false`) |
| run | 65 epochs, best epoch 55, best val total 0.055665, CUDA |
| shuffled control | `outputs/physics_jepa_v2/shuffled_32d` (same architecture/settings) |
| evaluation | `outputs/physics_jepa_v2/evaluation_32d/evaluation.json` + plots + logs |

Results (from `docs/phase13_physics_jepa_v2_collapse_fix.md` and `evaluation.json`):

| Representation | v1 raw rho | v2 raw rho | v1 norm rho | v2 norm rho | v1 rank-1 | v2 rank-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| z_target | -0.0238 | +0.0112 | -0.0405 | -0.0169 | 0.8911 | 0.0378 |
| z_pred | +0.2939 | +0.0990 | -0.1716 | -0.1057 | 0.9759 | 0.7136 |

- Correct-vs-shuffled: correct z_pred norm rho **-0.1057**, shuffled +0.0050, separation **-0.1107**.
- Response probe R2: z_pred 0.0424 (down from 0.1585), z_target 0.3418 (up from 0.2030) — target improved but does not transfer to the predictor.
- Resonance frequency MAE (z_pred): 2.93/2.70 GHz; feature-count MAE 0.4311.
- Response-PCA feasibility reference (raw rho 0.95-0.999; normalized +0.21..0.43) confirms the metric is not the failure.
- **Gate outcome: FAILED.** Target collapse largely fixed, but the geometry-derived latent still does
  not organize according to EM-response similarity, and the predictor remains low-rank.

## Verdict

- `v1 representation gate: FAILED`
- `v2 representation gate: FAILED`
- The covariance fix halves target rank-1 collapse but does not create a physics-organized
  geometry->latent representation. This is the scientific starting point for v3.

Do **not** merge v1/v2 numbers into one "Physics-jEPA" result; keep them as separate labeled
experiments.