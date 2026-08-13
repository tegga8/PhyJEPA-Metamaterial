# Repository Review

Review of `PhyJEPA-Metamaterial` at organization checkpoint.

## Repository health

- Tests: **92 passed, 0 failed, 0 skipped** (Python 3.14.3, torch 2.10.0+cu126, CPU/GPU available).
  Pytest run: `python -m pytest -q` -> `92 passed, 3 warnings in 58.54s`.
- Imports: all 30 inspected modules import cleanly, including `src.physics_jepa`,
  `src.physics_jepa_losses`, `src.physics_jepa_training`, `src.physics_representation_metrics`,
  all inverse-design modules, and RCWA modules.
- Broken references: none found. All `docs/` cross-links verified during cleanup;
  the two root-level phase reports moved to `docs/` have zero inbound references.
- Duplicates: no duplicate scripts. `docs/phase2_5_task.md` vs `docs/phase2_5_report.md` are
  distinct (task spec vs results). Root `Phase_2_5_...` and `docs/phase2_5_report.md` are distinct too.
- Dead artifacts: `tests/te.py` (debug CUDA check) - archived. Two bare response-aware forward runs
  and one empty directory are recorded as output archive candidates (git-ignored).
- Large generated artifacts: `outputs/` is **~2.38 GB / 939 files**, all git-ignored.
  Largest dirs: `physics_jepa` 717 MB, `phase2_5` 317 MB, `phase2_forward_30k` 219 MB,
  `phase2_forward_30k_response_aware_gpu` 216 MB, `physics_jepa_v2` 210 MB.
  These are generated caches (latents, predictions, plots) and are regenerable.

## Research branches

- **Inverse-design baseline (Phases 8-12):** preserved as PROTECTED BASELINE. Checkpoints, configs,
  metrics, caches, and reports verified and indexed in `docs/inverse_design_artifact_index.md`.
  Decision status: Phase 12 executive decision E (generator valid but not competitive with
  train-only nearest-neighbor screening baseline). Frozen; not retrained.
- **Physics-JEPA (current):** v1 (rank-collapse failure) and v2 (collapse-fix; target fixed,
  predictor still fails) preserved and indexed in `docs/physics_jepa_artifact_index.md`.
  Both gates FAILED. This is the evidence base for v3.
- **Forward surrogate (Phases 2/2.5/7B):** active baseline; frozen screening surrogate chosen in 7B
  is `outputs/phase2_5/exp_C_30k_resonance/best.pt`.
- **Completion / old JEPA (Phases 3-7):** historical; not extended.
- **Validation infrastructure (RCWA, Phase 6/6.1):** frozen. `meent==0.12.0` pin retained.

## Protected artifacts (exact paths)

| Artifact | Path |
| --- | --- |
| Physics-JEPA v1 runs | `outputs/physics_jepa/` (seed42_32d, seed42_64d, shuffled_32d, smoke) |
| Physics-JEPA v2 runs | `outputs/physics_jepa_v2/` (seed42_32d, shuffled_32d, evaluation_32d, logs) |
| Phase 10 VAE checkpoint | `outputs/phase10_stochastic_inverse_design/generator/best.pt` |
| Phase 11 candidate metrics | `outputs/phase11_candidate_ranking/metrics.json` |
| Phase 12 metrics | `outputs/phase12/metrics.json` |
| Phase 8 geometry AE checkpoint | `outputs/phase8_geometry_autoencoder/best.pt` |
| Phase 9 inverse checkpoints | `outputs/phase9_inverse_baselines/{direct_mlp,latent_predictor}/best.pt` |
| Forward screening surrogate | `outputs/phase2_5/exp_C_30k_resonance/best.pt` |
| Dataset source/preprocessing code | `src/dataset.py`, `src/preprocess.py`, `scripts/build_subset.py` |
| RCWA validation | `src/rcwa_solver.py`, `src/rcwa_validation.py`, `outputs/phase6_rcwa/`, `outputs/phase6_1/` (FROZEN) |

## Archived artifacts

- `tests/te.py` -> moved to `archive/` (dead debug file).
- `Phase_1_SUTD_PRCM_Dataset_Infrastructure.md`, `Phase_2_5_Forward_EM_Surrogate_Validation.md`
  -> moved to `docs/` (historical reports, clean root).
- Deleted: **none**.

## Active files Codex may modify for future work

- Physics-JEPA source: `src/physics_jepa.py`, `src/physics_jepa_encoders.py`,
  `src/physics_jepa_losses.py`, `src/physics_jepa_training.py`,
  `src/physics_jepa_metrics_eval.py`, `src/physics_jepa_plots.py`,
  `src/physics_representation_metrics.py`, `src/spectral_masks.py`.
- Physics-JEPA scripts/tests: `scripts/train_physics_jepa.py`,
  `scripts/evaluate_physics_representation.py`, `tests/test_physics_jepa.py`.
- Canonical docs: `docs/research_state.md`, `docs/research_map.md`,
  `docs/physics_jepa_open_questions.md`, `README.md`.
- Protected/frozen files are NOT to be modified (see protected list above).

## Current research question

Does the geometry-derived physical latent organize according to EM-response
similarity (`G <-> z_physics <-> S`)? v1/v2 answer: no.

## Next experiment

Physics-JEPA v3: `physics_jepa_v3_frequency_relational` (frequency-aware spectrum
encoder + small relational physics term). Prepared under `outputs/physics_jepa_v3/`;
**training not started in this task.**

## Git

Review performed on commit `554a547` (branch `main`, clean tree). Archive moves and new docs
after this commit are listed in `docs/archive_candidates.md` and tracked by the inventory at the
next commit.