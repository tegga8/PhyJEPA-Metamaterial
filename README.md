# PhyJEPA-Metamaterial

Physical latent representation and inverse design for SUTD-PRCM binary
metasurfaces (geometry `G` ~ 16x16 binary patterns, EM response `S` ~ complex
[4,1001] over 2-12 GHz).

> **Read `docs/research_state.md` first.** It is the concise, current snapshot
> for any new session. Full inventory: `docs/repository_inventory.json`;
> branch map: `docs/research_map.md`; review: `docs/repository_review.md`.

## Project objective

```text
S_target -> G_new
```

## What currently works

- **Dataset pipeline** (Phase 1): audited raw schema, seeded family-balanced
  processed subsets (`data/processed/`), `src/dataset.py`, `src/preprocess.py`.
- **Forward surrogate** (Phase 2/2.5/7B): learned CNN surrogate; frozen
  screening checkpoint `outputs/phase2_5/exp_C_30k_resonance/best.pt`.
- **Geometry autoencoder** (Phase 8, `G -> z_G -> G`): test IoU 0.965.
- **Stochastic inverse-design baseline** (Phase 10, `(S,eps) -> z_G -> G`):
  valid candidate generator, ~0.98 validity, 0.59 multi-solution success rate.
- **Candidate screening** (Phase 11/12): validity + forward-surrogate ranking,
  NN baseline, diversity/complexity metrics.

## What has failed

- Completion-oriented JEPA as the final objective (Phases 3-7).
- Physics-JEPA **v1**: severe low-rank collapse of the spectrum target.
- Physics-JEPA **v2**: target collapse largely fixed, but the geometry-derived
  predictor still does not organize by EM-response similarity (gate FAILED).
  See `docs/physics_jepa_status.md`.

## Current research hypothesis

```text
G <-> z_physics <-> S
```

A geometry-derived physical latent should organize by EM-response similarity.
Not yet established.

## Current next experiment

`physics_jepa_v3_frequency_relational` — frequency-aware spectrum encoder +
small relational physics term, same data/split/seed/settings as v2.
Prepared under `outputs/physics_jepa_v3/`. **Not trained.**

## Explicitly frozen

- VAE / inverse-design branch (Phases 8-12) — protected baseline, do not retrain.
- RCWA validation infrastructure (frozen; `meent==0.12.0`).
- Old JEPA / completion variants (Phases 3-7) — historical.
- Completed baselines (forward surrogate selection, geometry AE, inverse phase reports).
- Physics-JEPA v1/v2 outputs (`outputs/physics_jepa/`,
  `outputs/physics_jepa_v2/`) — keep as evidence, do not modify.

## Environment

`Python 3.14`, PyTorch `2.10.0+cu126`, CUDA 12.6; deps in `requirements.txt`
(pins `meent==0.12.0` for the frozen RCWA infra). Run from repo root:
`python -m pytest -q` (92 tests).