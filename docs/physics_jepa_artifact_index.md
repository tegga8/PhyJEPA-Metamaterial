# Physics-JEPA Artifact Index

Exact paths for Physics-JEPA v1/v2 evidence. Paths relative to repository root.
All `outputs/` content is git-ignored local artifact. **Do not modify
`outputs/physics_jepa/` (v1) or `outputs/physics_jepa_v2/` (v2).**

## Physics-JEPA v1 — `outputs/physics_jepa/`

| Kind | Path |
| --- | --- |
| 32-D checkpoint | `outputs/physics_jepa/seed42_32d/best.pt` |
| 32-D config | `outputs/physics_jepa/seed42_32d/config.json` |
| 32-D training history | `outputs/physics_jepa/seed42_32d/training_history.csv` |
| 32-D cached latents | `outputs/physics_jepa/seed42_32d/{train,val,test}_z_{geometry,online,target,pred}.npy` |
| 32-D cached inputs | `outputs/physics_jepa/seed42_32d/{train,val,test}_{geometry,response}.npy` + `{split}_source_ids.txt` |
| 32-D resonance targets | `outputs/physics_jepa/seed42_32d/{train,val,test}_resonance_targets.npy` |
| 64-D checkpoint | `outputs/physics_jepa/seed42_64d/best.pt` |
| 64-D config | `outputs/physics_jepa/seed42_64d/config.json` |
| 64-D latents | `outputs/physics_jepa/seed42_64d/{val,test}_z_*.npy` (val/test only) |
| shuffled control (32-D) | `outputs/physics_jepa/shuffled_32d/{best.pt,config.json,{val,test}_z_*.npy}` |
| smoke (overfit milestone) | `outputs/physics_jepa/smoke/smoke_report.json` |
| plots (32-D run) | `outputs/physics_jepa/seed42_32d/plots/*.png` |
| science report | **missing dedicated v1 doc** — v1 gate numbers only in `docs/phase13_physics_jepa_v2_collapse_fix.md` |
| training code | `scripts/train_physics_jepa.py` (default `--experiment-label physics_jepa_v1`) |
| tests | `tests/test_physics_jepa.py` |

## Physics-JEPA v2 — `outputs/physics_jepa_v2/`

| Kind | Path |
| --- | --- |
| correct-pair checkpoint | `outputs/physics_jepa_v2/seed42_32d/best.pt` |
| correct-pair config | `outputs/physics_jepa_v2/seed42_32d/config.json` |
| correct-pair training history | `outputs/physics_jepa_v2/seed42_32d/training_history.csv` |
| correct-pair cached latents | `outputs/physics_jepa_v2/seed42_32d/{val,test}_z_{geometry,online,target,pred}.npy` etc. |
| shuffled control | `outputs/physics_jepa_v2/shuffled_32d/{best.pt,config.json,{val,test}_z_*.npy}` |
| evaluation report | `outputs/physics_jepa_v2/evaluation_32d/evaluation.json` (large; includes baseline-mean arrays) |
| evaluation plots | `outputs/physics_jepa_v2/evaluation_32d/plots/*.png` |
| training logs | `outputs/physics_jepa_v2/logs/{seed42_32d,shuffled_32d,evaluation_32d}.{stdout,stderr}.log` |
| science report | `docs/phase13_physics_jepa_v2_collapse_fix.md` |
| gate numbers | `docs/physics_jepa_status.md` (v1 vs v2 table) |

## Shared evaluation utilities (ACTIVE_RESEARCH, reused by v3)

| Module | Role |
| --- | --- |
| `src/physics_representation_metrics.py` | canonical distances, `rho(D_z,D_S)`, `rho(D_z,D_G)`, probes, representative pairs |
| `src/physics_jepa_metrics_eval.py` | `evaluate_model_representation` battery, cached-latent loading |
| `src/physics_jepa_plots.py` | evaluation plotting |
| `scripts/evaluate_physics_representation.py` | gate runner (model-dir + compare-dir + size-compare-dir) |

## Status of required v1/v2 artifact set

| Required artifact | v1 | v2 |
| --- | --- | --- |
| checkpoint | exists | exists |
| config | exists | exists |
| training history | exists (32/64/shuffled) | exists (correct/shuffled) |
| evaluation | plots + numbers (no json in v1 dirs) | `evaluation.json` + plots |
| shuffled control | exists (32-D, latents cached) | exists (same settings) |
| report | **no dedicated report doc** (v1 numbers in v2 doc table) | `docs/phase13_physics_jepa_v2_collapse_fix.md` |

Explicitly stated: **v1 representation gate FAILED; v2 representation gate FAILED.**