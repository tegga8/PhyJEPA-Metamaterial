# Inverse-Design Baseline (Branch A)

Preserved, reproducible baseline of the completed inverse-design research path.
**Do not retrain. Do not alter these experiments.** This branch is frozen as a
scientific baseline and remains available for the eventual return to inverse
design.

## Objective

```text
S_target --[learned candidate generation]--> G_new
```

The complete baseline path is a cascade of four phases (8-12) built on the
dataset (Phase 1), the forward surrogate (Phase 2/2.5), and the frozen learned
screening surrogate (Phase 7B selection).

## Branch layout

```text
DATASET (Phase 1)
  ├── FORWARD SURROGATE (Phase 2 / 2.5)
  │     └── learned screening surrogate (Phase 7B: outputs/phase2_5/exp_C_30k_resonance/best.pt)
  │           └── used only for candidate response screening in Phases 9-12
  └── INVERSE-DESIGN BASELINE
        ├── Phase 8  geometry autoencoder          G -> z_G -> G
        ├── Phase 9  deterministic inverse          S -> z_G -> G
        ├── Phase 10 stochastic latent VAE          (S, eps) -> z_G -> G
        ├── Phase 11 candidate screening           generation -> validity -> forward-surrogate ranking
        └── Phase 12 response-space generalization NN baseline, diversity, complexity
```

## Phase 8 - geometry autoencoder (`G -> z_G -> G`)

- Architecture: `SpatialGeometryEncoder` -> `z_G [B,64,8,8]` -> `SpatialGeometryDecoder`, full-image BCE.
- Report: `docs/phase8_geometry_latent_autoencoder.md`
- Artifacts: `outputs/phase8_geometry_autoencoder/` (best.pt, config.json, metrics.json, latents/, *_metrics.csv)
- Test metrics (from report): BCE 0.040649, IoU 0.965106, Dice 0.981470, pixel accuracy 0.982570.
- Decision: **gate supported**; latent usable as inverse-design basis.
- Gap: `training_history.csv` exists but is **0 bytes** (empty).

## Phase 9 - deterministic spectrum->geometry baseline (`S -> z_G -> G`)

- Models: EM nearest neighbor (train-only), direct MLP, deterministic latent predictor.
- Report: `docs/phase9_spectrum_to_geometry_baseline.md`
- Artifacts: `outputs/phase9_inverse_baselines/` (direct_mlp/best.pt, latent_predictor/best.pt, metrics.json, *_per_sample.csv)
- Test screening response MSE: NN 0.259249, direct MLP 0.483234, latent predictor 0.444132.
- Decision: deterministic mapping **not competitive** with train-only nearest neighbor.

## Phase 10 - stochastic latent VAE (`(S, eps) -> z_G -> G`)

- Model: `ConditionalLatentVAE` over the frozen Phase 8 latent space; posterior only at train time.
- Report: `docs/phase10_stochastic_inverse_design.md`
- Checkpoint: `outputs/phase10_stochastic_inverse_design/generator/best.pt` (**PROTECTED**)
- Metrics: `outputs/phase10_stochastic_inverse_design/metrics.json`, `candidate_metrics.csv`
- Test results: validity rate 0.981250, best screened response MSE 0.338217, multi-solution success 0.594, pairwise Hamming diversity 0.201257, K=8 candidates per target.
- Dataset/split: 5k subset, seed 42 (4000/500/500).

## Phase 11 - candidate generation, validity, forward-surrogate screening

- Report: `docs/phase11_candidate_ranking.md`
- Artifacts: `outputs/phase11_candidate_ranking/` (config.json, metrics.json, ranked_top_candidates.csv)
- Policies: all / valid / valid_novel. Ranking by screened response MSE against the frozen Phase 7B surrogate.
- Result: all-policy best screened response MSE 0.338217; complex targets remain the bottleneck (success fraction 0.293).

## Phase 12 - response-space generalization, NN baseline, diversity, complexity

- Report: `docs/phase12_response_generalization_and_bottleneck_diagnosis.md`
- Artifacts: `outputs/phase12/` (metrics.json, summary.csv, complexity/, diversity/, generation/, jepa_vs_ae/, nearest_neighbor/, targets/, plots/)
- Protocol: fixed Phase 10 generator (K=8), 180 interpolated stress targets, no leakage of geometry/latent into generator inputs.
- Overall interpolation best-of-8 MSE 0.3100; candidate validity 0.9583.
- **Executive decision E**: the stochastic generator is a valid target-conditioned candidate
  generator under the learned screening model, but is **not competitive with the fixed
  train-only EM nearest-neighbor baseline** on mean screening MSE. No new inverse
  architecture is introduced from these results.

## Reproducibility notes

- The frozen learned screening surrogate used in Phases 9-12 is
  `outputs/phase2_5/exp_C_30k_resonance/best.pt` (selected in
  `docs/phase7b_forward_screening_surrogate_selection.md`). It is a learned CNN, not Maxwell ground truth.
- Inverse-design source files are classified ACTIVE_BASELINE in
  `docs/repository_inventory.json` and must remain untouched by Physics-JEPA work.
- The phase10 training script imports shared helpers from
  `scripts/train_phase9_inverse_baselines.py`; do not split this cross-script import during cleanup.
