# Research Map

Repository branches and how data flows through them.

```text
DATASET  (Phase 1 pipeline; src/dataset.py, src/preprocess.py, data/processed/)
  │
  ├──> FORWARD SURROGATE  (Phase 2 / 2.5 / 7B; src/models.py, src/metrics.py)
  │        └──> screening/evaluator
  │              └── frozen learned screening surrogate:
  │                  outputs/phase2_5/exp_C_30k_resonance/best.pt
  │
  ├──> COMPLETION / OLD JEPA  (Phases 3, 4, 4.1, 4.2, 5a, 5b, 6, 7)   [HISTORICAL]
  │        ├── Phase 3   supervised CNN completion
  │        ├── Phase 4   global JEPA completion (CNN wins)
  │        ├── Phase 4.1 spatial JEPA completion
  │        ├── Phase 4.2 mask-aware spatial JEPA
  │        ├── Phase 5a  physics-conditioned spatial JEPA
  │        ├── Phase 5b  physics-consistent completion
  │        └── Phase 6/7 counterfactual + RCWA validation
  │
  ├──> PHYSICS-CONDITIONED COMPLETION  (historical)
  │
  ├──> INVERSE-DESIGN BASELINE  (Phases 8-12)   [FROZEN BASELINE]
  │        ├── geometry AE                 (Phase 8:  G -> z_G -> G)
  │        ├── deterministic inverse       (Phase 9:  S -> z_G -> G)
  │        ├── stochastic VAE              (Phase 10: (S,eps) -> z_G -> G)
  │        └── candidate screening         (Phase 11/12)
  │
  └──> CURRENT RESEARCH  (Physics-JEPA representation gate)
          ├── Physics-JEPA v1 ❌  (outputs/physics_jepa; target rank-1 collapse 0.89)
          ├── Physics-JEPA v2 ❌  (outputs/physics_jepa_v2; target fixed, predictor still fails)
          └── NEXT: frequency-aware + relational Physics-JEPA  (physics_jepa_v3_frequency_relational)
```

## Branch separation

| Branch | Phase span | Status | Guard rails |
| --- | --- | --- | --- |
| dataset | 1 | ACTIVE_BASELINE | never touched |
| forward surrogate | 2, 2.5, 7B | ACTIVE_BASELINE | surrogate ckpt frozen |
| completion / old JEPA | 3-7 | HISTORICAL | do not extend |
| physics-conditioned completion | 5a, 5b | HISTORICAL | do not extend |
| validation (RCWA) | 6, 6.1 | VALIDATION_INFRASTRUCTURE / FROZEN | do not develop during cleanup |
| inverse-design baseline | 8-12 | PROTECTED BASELINE | do not retrain; source classified ACTIVE_BASELINE |
| Physics-JEPA | current | ACTIVE_RESEARCH | v1/v2 outputs immutable; only this branch is extended by v3 |

## Data contracts that keep branches honest

- Responses: `float32 [4, 1001]` normalized (train stats only); channels y-cross Re/Im then x-co Re/Im.
- Geometry: binary `[B,1,16,16]`.
- Screen surrogate: learned CNN only — **not** Maxwell ground truth; independent validation is RCWA (frozen).
- Representation gate metric: `rho(D_z, D_S)` with L2-normalized latent Euclidean distance
  (fixed in `src/physics_representation_metrics.py`).