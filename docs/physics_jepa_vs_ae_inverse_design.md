# Physics-JEPA vs Geometry-AE Representation — Inverse Design

Label: `physics_jepa_vs_ae_inverse_design`
Date: 2026-08-13
Status: **COMPLETE** — paired representation ablation, AE branch wins.

## Question

Does replacing the ordinary Phase 8 geometry-AE latent (the Phase 10 inverse
contract's target space) with the Physics-JEPA v3 geometry-derived latent
improve stochastic inverse design — same generator, same noise, same
screening, only the representation changes?

## Protocol (controlled ablation)

Both branches use the identical Phase 10 stochastic inverse pipeline:

* Generator: `ConditionalLatentVAE` (256-D), trained 30 epochs / patience 6,
  beta-KL 1e-4, prior weight 0.5, batch 64, seed 42, identical data loaders.
* Branch targets: reconstruction of the 64x8x8 latent from the **normalized
  target response** (conditioning input). Only the latent being reconstructed
  differs.
* Representation A (`ae`): the Phase 8 geometry-AE latent `z_AE(G)`.
* Representation B (`physics_jepa`): frozen seed-123 v3 predictor latent
  `z_PJ(G)` (32-D) mapped to the AE latent space by a single **frozen linear
  least-squares adapter** `A: R^32 -> R^4096`, fit on the train split only
  (adapter R2 on validation: 0.2155). The adapter defines the branch-B
  training target; it is never a test-time conditioning input.
* Paired sampling: identical `torch.Generator(seed=1042)` noise sequence for
  both branches, K = 8 candidates per target (Phase 10 contract).
* Screening: frozen `exp_C_30k_resonance` surrogate; learned empirical success
  threshold 0.30 (a screening heuristic, not a physical law).
* Data: 5k subset, Phase 10 split 4000/500/500 (seed 42), test = 500 targets.
* Baselines: train-only EM nearest neighbor; NN-relative improvement
  `I = 1 - best_of_8_MSE / NN_MSE`.

Dataset: `data/processed/sutd_prcm_5k`; latents
`outputs/phase8_geometry_autoencoder/latents`; autoencoder
`outputs/phase8_geometry_autoencoder/best.pt`; v3 frozen
`outputs/physics_jepa_v3_repro/correct/best.pt`.

## Results

| metric | ae | physics_jepa (frozen adapter) | nn baseline |
| --- | --- | --- | --- |
| best-of-8 MSE mean | **0.3592** | 0.5008 | 0.2592 |
| best-of-8 MSE median | **0.1037** | 0.2206 | 0.0423 |
| median-of-8 MSE mean | **0.5478** | 0.6671 | - |
| validity rate | **0.9125** | 0.8960 | - |
| useful diversity / target | **3.562** | 3.096 | - |
| nearest-train hamming (novelty) | 0.1568 | 0.1341 | - |
| improvement vs NN (mean) | -2.354 | -6.107 | - |
| generator beats NN (fraction) | 36.2% | 23.6% | - |

Paired (500 targets): **AE wins 73.8%**, Physics-JEPA wins 26.0%, ties 0.2%.
Physics-JEPA loses more and wins less on every aggregate. The advantage is
structural, not noise: it persists at the median (0.104 vs 0.221) and across
all complexity groups (simple 0.130 | 0.353, medium 0.325 | 0.428, complex
0.636 | 0.725).

Interpolation stress test (paired target-pairs, alpha 0.25/0.50/0.75): both
branches near parity — best-of-8 MSE 0.3519 vs 0.3528; Physics-JEPA has
slightly higher validity (0.899 vs 0.860) and its best alpha is 0.50.
Interpolation does not change the ranking.

NN beats both generators on most targets (64/76% of targets). Generative
sampling adds diversity over the NN even where per-target best-of-8 worse
(novelty 0.157 / 0.134 in pixel-Hamming).

## Interpretation

The v3 `z_pred` representation — deliberately rank-collapsed and symmetry
dominated (rank-1 fraction 0.69, effective rank ~2.05) for the *geometry-side
objective* — maps poorly into the 4096-D spatial latent space through a
linear adapter (adapter R2 0.216). The generator then tries to emit that
spatial latent from the response and recovers a worse screening distribution
than the AE's own latent, where the generator already had 4000 train targets
of well-calibrated signal.

This is a fair but **high-bar** failure for Physics-JEPA: the adapter is the
minimal bridge demanded by the design contract, and meaningful gains would
require a genuinely non-linear map or changing the generator's latent contract
to operate directly in `z_PJ` space. None of the pass-through / original
geometry / mask / source-id channels were available in either branch, so the
comparison isolates the representation variable (anti-leakage table in
`metrics.json`).

## Artifacts

`scripts/run_physics_jepa_inverse_comparison.py` (end-to-end),
`outputs/physics_jepa_inverse_comparison/`:
`metrics.json`, `summary.csv`, `comparison.csv` (per-target, 500 rows),
`{ae,physics_jepa}/candidate_metrics.csv`,
`paired_candidates/*_candidates_binary.npy`,
`interpolated/*_candidate_metrics.csv`, `plots/*` (best-of-8, median,
NN-improvement scatter, complexity/validity/diversity bars, plus copied
v3 reproduction plots).