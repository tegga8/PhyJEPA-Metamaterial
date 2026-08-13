# Research State

**Read this first.** Concise, current snapshot. Companion files:
`research_map.md`, `physics_jepa_status.md`, `physics_jepa_open_questions.md`,
`repository_review.md`, `repository_inventory.json`.

## CURRENT OBJECTIVE

`G <-> z_physics <-> S` — learn a physical latent `z_physics` from geometry such
that geometry-derived latents organize by EM-response similarity. Long-term goal
remains `S_target -> G_new` (inverse design).

## ACTIVE BRANCH

Physics-JEPA representation gate. Previous inverse-design branch (Phases 8-12)
is frozen as a protected baseline. Old completion/JEPA work (3-7) is historical.

## PROTECTED BASELINE

- Phase 8 geometry AE ckpt: `outputs/phase8_geometry_autoencoder/best.pt`
- Phase 9 inverse ckpts: `outputs/phase9_inverse_baselines/{direct_mlp,latent_predictor}/best.pt`
- Phase 10 VAE ckpt: `outputs/phase10_stochastic_inverse_design/generator/best.pt`
- Phase 11 metrics: `outputs/phase11_candidate_ranking/metrics.json`
- Phase 12 metrics: `outputs/phase12/metrics.json`
- Forward screening surrogate: `outputs/phase2_5/exp_C_30k_resonance/best.pt`
- RCWA validation: frozen, not developed further.

## CURRENT FAILURE

Representation gate FAILED for both Physics-JEPA v1 and v2:

- v1: severe low-rank collapse of the momentum spectrum target (rank-1 0.89), predictor norm-rho(Dz,DS) negative.
- v2: covariance fix reduced target rank-1 to 0.04, but geometry-derived predictor still fails physics
  organization (z_pred norm-rho -0.1057; correct-vs-shuffled separation -0.1107).

## CURRENT HYPOTHESIS

The geometry->physics mapping fails because (i) the spectrum target representation does not preserve
frequency-local structure sufficiently (4x1001 -> small token set before attention, no explicit
frequency-position encoding) and (ii) pointwise JEPA matching does not require latent distances to
reflect EM-response similarity.

## NEXT EXPERIMENT

`physics_jepa_v3_frequency_relational` — frequency-aware spectrum encoder + small relational
physics term, on the SAME 30k data / v2 split / seed / latent 32 / optimizer / EMA /
variance+covariance settings. Shuffled-pair control included. **Not yet trained.**

## STOP CONDITIONS

- Treat as success only on the canonical metrics (`rho(D_z,D_S)` primary, plus `rho(D_z,D_G)`,
  correct-vs-shuffled separation, response/resonance probes, latent rank), never on JEPA loss alone.
- If v3 establishes a useful physical latent -> resume inverse design on top of it.
- If v3 fails like v2 -> stop forcing JEPA; do not start another sequence of loosely justified phases.