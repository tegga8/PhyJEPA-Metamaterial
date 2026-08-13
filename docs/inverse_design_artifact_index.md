# Inverse-Design Artifact Index

Exact paths for every reproducible artifact of the inverse-design baseline branch
(Branch A, Phases 8-12). Paths are relative to the repository root
(`C:\Users\tejas\Desktop\PhyJEPA-Metamaterial`). All `outputs/` content is
git-ignored local artifact.

## Phase 8 - geometry autoencoder (G -> z_G -> G)

| Kind | Path |
| --- | --- |
| checkpoint | `outputs/phase8_geometry_autoencoder/best.pt` |
| config | `outputs/phase8_geometry_autoencoder/config.json` |
| metrics (machine readable) | `outputs/phase8_geometry_autoencoder/metrics.json` |
| per-sample metrics | `outputs/phase8_geometry_autoencoder/train_metrics.csv`, `val_metrics.csv`, `test_metrics.csv` |
| training history | `outputs/phase8_geometry_autoencoder/training_history.csv` (**0 bytes / empty**) |
| latent cache | `outputs/phase8_geometry_autoencoder/latents/{train,val,test}.npy` + `{...,}_source_ids.txt` |
| report | `docs/phase8_geometry_latent_autoencoder.md` |
| training code | `scripts/train_geometry_autoencoder.py`, `src/geometry_autoencoder.py` |
| test | `tests/test_geometry_autoencoder.py` |

## Phase 9 - deterministic inverse (S -> z_G -> G)

| Kind | Path |
| --- | --- |
| direct MLP checkpoint | `outputs/phase9_inverse_baselines/direct_mlp/best.pt` |
| latent predictor checkpoint | `outputs/phase9_inverse_baselines/latent_predictor/best.pt` |
| config | `outputs/phase9_inverse_baselines/config.json` |
| metrics | `outputs/phase9_inverse_baselines/metrics.json` |
| per-sample metrics | `outputs/phase9_inverse_baselines/{direct_mlp,latent_predictor,nearest_neighbor}_per_sample.csv` |
| report | `docs/phase9_spectrum_to_geometry_baseline.md` |
| training code | `scripts/train_phase9_inverse_baselines.py`, `src/spectrum_inverse_models.py` |
| test | `tests/test_spectrum_inverse_models.py` |

## Phase 10 - stochastic latent VAE ((S, eps) -> z_G -> G)

| Kind | Path |
| --- | --- |
| **VAE checkpoint (PROTECTED)** | `outputs/phase10_stochastic_inverse_design/generator/best.pt` |
| config | `outputs/phase10_stochastic_inverse_design/config.json` |
| metrics | `outputs/phase10_stochastic_inverse_design/metrics.json` |
| candidate metrics | `outputs/phase10_stochastic_inverse_design/candidate_metrics.csv` |
| validity limits | `outputs/phase10_stochastic_inverse_design/validity_limits.json` |
| latent standardization | `outputs/phase10_stochastic_inverse_design/latent_standardization.npz` |
| generated candidate cache | `outputs/phase10_stochastic_inverse_design/test_candidates_binary.npy`, `test_candidate_responses_normalized.npy` |
| report | `docs/phase10_stochastic_inverse_design.md` |
| training code | `scripts/train_phase10_stochastic_inverse_design.py`, `src/conditional_latent_vae.py` |
| test | `tests/test_conditional_latent_vae.py` |

## Phase 11 - candidate screening

| Kind | Path |
| --- | --- |
| metrics | `outputs/phase11_candidate_ranking/metrics.json` |
| config | `outputs/phase11_candidate_ranking/config.json` |
| ranked candidates | `outputs/phase11_candidate_ranking/ranked_top_candidates.csv` |
| report | `docs/phase11_candidate_ranking.md` |
| code | `scripts/rank_phase10_candidates.py` |
| test | `tests/test_phase11_candidate_ranking.py` |

## Phase 12 - response-space metrics

| Kind | Path |
| --- | --- |
| summary + metrics | `outputs/phase12/metrics.json`, `outputs/phase12/summary.csv` |
| config | `outputs/phase12/config.json` |
| complexity | `outputs/phase12/complexity/complexity_metrics.csv` |
| diversity | `outputs/phase12/diversity/{diversity_metrics,stochasticity_classification,useful_diversity}.csv` |
| generation cache | `outputs/phase12/generation/{candidates,candidate_metrics,interpolated_candidate_metrics}.csv` |
| jepa-vs-AE comparison | `outputs/phase12/jepa_vs_ae/comparison.csv` |
| nearest neighbor baseline | `outputs/phase12/nearest_neighbor/metrics.csv` |
| interpolated targets | `outputs/phase12/targets/{interpolated_targets,target_manifest}.csv` |
| plots | `outputs/phase12/plots/*.png` |
| report | `docs/phase12_response_generalization_and_bottleneck_diagnosis.md` |
| code | `scripts/evaluate_phase12_response_generalization.py` |
| test | `tests/test_phase12_response_generalization.py` |

## Shared frozen surrogate used by Phases 9-12

| Kind | Path |
| --- | --- |
| learned screening surrogate (**PROTECTED**) | `outputs/phase2_5/exp_C_30k_resonance/best.pt` |
| selection report | `docs/phase7b_forward_screening_surrogate_selection.md` |
| selection metrics | `outputs/phase2_5/surrogate_selection/{metrics.json,selection_table.csv,selection_report.md}` |
| selector | `scripts/select_forward_screening_surrogate.py` |
| test | `tests/test_forward_screening_selection.py` |