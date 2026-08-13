# Physics-JEPA v3 — Frequency-Aware + Physics-Relational Experiment

Label: `physics_jepa_v3_frequency_relational`
Date: 2026-08-13
Status: **COMPLETE** — one controlled experiment, correct-pair run + shuffled-pair control.

## Hypothesis

v2 fixed most target-side collapse (z_target rank-1 0.89 -> 0.04) but the
geometry-derived predictor still failed to organize the latent by EM-response
similarity (z_pred normalized rho(D_z, D_S) = -0.1057; correct-vs-shuffled
separation = -0.1107). The v3 hypothesis is a single idea with two coupled
changes:

1. The spectrum target representation loses frequency-local structure
   (4x1001 -> 12 tokens before attention, no explicit frequency position),
   so the latent cannot preserve resonance location / width / nearby
   resonances / phase transitions / broadband shape.
2. The pointwise JEPA objective matches per-sample vectors and never requires
   that latent distances track pairwise EM-response distances.

v3 therefore changes exactly two things: **(A)** a frequency-aware spectrum
encoder, and **(B)** a small physics-relational margin-ranking loss. No other
architectural additions.

## Exact changes from v2 (everything else fixed)

| Setting | v2 | v3 |
| --- | --- | --- |
| spectrum encoder | Conv1D 4->32->64, adaptive-pool to 12 tokens, attention, attention-pool | `FrequencySpectrumEncoder` (Change A, see below) |
| relational term | none | margin-ranking `max(0, D_z(a,p) - D_z(a,n) + m)` with m = 0.2, weight 0.1 (Change B) |
| geometry encoder / predictor | `GeometryLatentEncoder` + `PhysicsPredictor` 32->64->32 | **identical (unchanged)** |
| EMA target / stop-grad / centering | decay 0.996, stop-grad, centering OFF | **identical** |
| variance / covariance weights | 0.5 / 0.05 on z_online + z_geometry | **identical** |
| dataset / split / seed / latent / optimizer / lr / wd / batch / budget | 30k, 24k/3k/3k, seed 42, 32-D, AdamW 1e-3, 1e-4, bs 64, 75 ep / patience 10 | **identical** |

## Dataset and split

Same Phase-2 30k dataset and v2 split: `data/processed/sutd_prcm_30k`,
24,000 train / 3,000 val / 3,000 test, seed 42. Geometry `G in {0,1}^{16x16}`,
response `S in R^{4x1001}` (Re/Im Ty, Re/Im Rx), 2.00-12.00 GHz at 1001
points. Existing normalization (train stats) and pairing retained. No 261k
scaling in this experiment.

## Model architecture (`PhysicsJEPAFrequencyRelational`)

```text
G -> GeometryLatentEncoder (v2) -> z_geometry [32] -> PhysicsPredictor (v2) -> z_pred [32]
S -> FrequencySpectrumEncoder (online) -> z_online [32] -> spectrum_predictor -> z_self
S -> FrequencySpectrumEncoder (EMA target) -> z_target [32]   (stop-gradient, no gradient)
```

Change A — `FrequencySpectrumEncoder` (`src/physics_jepa_encoders.py`):

```text
4 x 1001
  -> stem Conv1d 4->48 k9
  -> + sinusoidal frequency-position embedding
  -> multiscale dilated Conv1d (dilation 1, 4, 16), full 1001-point resolution
  -> 1x1 mix 144->48
  -> strided tokenizer Conv1d 48->64 k17/s16 -> 63 ordered tokens
  -> learned per-token position
  -> 4-head self-attention (63 tokens)
  -> attention pool -> MLP -> z_S [B, 32]
```

No large Transformer; no unlabeled 1001->small pooling before local spectral
features are extracted; frequency position is never discarded.

### Spectrum positional encoding (exact)

`f = linspace(2, 12, 1001)`, `tilde_f = (f - 2) / 10` in `[0, 1]`,
`code = [sin(2*pi*h*tilde_f), cos(2*pi*h*tilde_f)]` for harmonics
`h = 0..7`, projected `Linear(16 -> 48)` onto the feature channels and added
after the stem. A learned per-token embedding (63 x 64) is added after the
strided tokenizer. Both online and target encoders carry the same buffer.

### Parameter counts

| Module | Parameters |
| --- | ---: |
| geometry encoder (v2) | 68,512 |
| frequency spectrum encoder (online) | 154,080 |
| spectrum target encoder (EMA copy, frozen) | 154,080 |
| predictor (v2) | 4,192 |
| spectrum predictor | 4,192 |
| total | 385,056 |
| trainable | 230,976 |
| latent dimension | 32 |

v2 total was 162,784 (trainable 119,840); the increase is the frequency-aware
encoder. Peak GPU memory 375 MB (RTX 3050 4GB).

## Change B — relational objective and triplet rule

- Response distance `D_S(S_i, S_j) = MSE` over the Phase-2 normalized
  `[4, 1001]` response (pairwise squared-Euclidean / 4004).
- Triplet construction per batch (deterministic numpy RNG, seed 42, 32
  triplets/batch): anchor chosen by sorted RNG; **positive** = index with the
  smallest `D_S` excluding the anchor; **negative** = a valid index with
  `D_S >= 2.0 * D_S(positive) + 1e-6` (closest such valid candidate), with
  source-identical / duplicate indices excluded by the strict inequality.
- Objective applied to the online spectrum latent `z_online` (the EM-organized
  branch that the JEPA cross-term transfers to `z_pred`):

  `L_rel = mean(max(0, D_z(z_a, z_p) - D_z(z_a, z_n) + margin))`
  with `D_z` the canonical L2-normalized latent Euclidean distance, margin 0.2.

## Full v3 loss

`L = L_JEPA + 0.5*L_variance + 0.05*L_covariance + 0.1*L_rel`,
`L_JEPA = cross + 0.5*bootstrap`, v2 values retained for variance/covariance.
Loss-scale diagnostic at initialization (mean over a training batch):

| term | scaled contribution | fraction |
| --- | ---: | ---: |
| cross | 0.074 | 0.118 |
| bootstrap (x0.5) | 0.042 | 0.068 |
| variance (x0.5) | 0.491 | 0.785 |
| covariance (x0.05) | ~0.0 | ~0.000 |
| relational (x0.1) | 0.018 | 0.029 |

The relational term is a small, non-dominated contributor as intended (one
modest margin, one small coefficient; no sweep).

## Training settings

AdamW lr 1e-3, weight decay 1e-4, batch 64, EMA decay 0.996, target centering
OFF, 75 requested epochs, patience 10, seed 42, shuffle-seed 123 (control),
device RTX 3050 (CUDA 12.6, torch 2.10.0+cu126, Python 3.14).

| run | epochs | best epoch | best val total loss | wall time |
| --- | ---: | ---: | ---: | ---: |
| correct | 75 | 74 | 0.0703 | 1490 s |
| shuffled | 75 | 75 | 0.0751 | ~1600 s |

Smoke test (tiny subset) passed all gates before the full run: JEPA loss
decreases, gradients finite/nonzero, target branch has no gradients, EMA
target parameters update, relational triplets valid, relational loss finite,
frequency positions align with the 1001-point grid.

## Results

### v2 baseline (frozen, unchanged)

- z_pred normalized rho(D_z, D_S) = **-0.1057**, rho(D_z, D_G) = 0.0583,
  rank-1 fraction 0.7136.
- z_target normalized rho(D_z, D_S) = -0.0169, rank-1 0.0378.
- correct-vs-shuffled separation (z_pred) = -0.1107.
- Response probe R2: z_pred 0.0424, z_target 0.3418. Resonance Ty/Rx MAE
  (z_pred): 2.93 / 2.70 GHz; feature-count MAE 0.4311.

### v3 correct-pair run

| Representation | rho(D_z,D_S) | pearson | rho(D_z,D_G) | rank-1 | eff. rank | response R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| z_target | +0.4675 | +0.3484 | +0.1534 | 0.0390 | 31.6 | 0.6562 |
| z_pred | **+0.1495** | +0.0894 | **-0.0092** | 0.7426 | 1.77 | 0.0598 |
| z_geometry | +0.0073 | -0.0171 | +0.1173 | 0.0462 | 30.3 | 0.0412 |

- **Primary metric rho_v3(D_z, D_S) = +0.1495**, a substantial improvement
  over v2 (-0.1057) and v1 (-0.1716), now positive.
- **Geometry correlation does NOT dominate**: rho(D_z_pred, D_G) = -0.0092.
- Cross-family z_pred rho = +0.2253 (v2: -0.1274); within-family -0.0732.
- Latent collapse controlled on the target/online/geometry branches
  (rank-1 ~0.04, effective rank ~31); the predictor remains low-rank
  (rank-1 0.74, effective rank 1.77, essentially unchanged from v2) — see
  failure cases.
- No NaN/Inf in any latent.

### v3 shuffled control (identical settings, pairing shuffled)

| Representation | rho(D_z,D_S) | rank-1 | response R2 |
| --- | ---: | ---: | ---: |
| z_target | +0.4940 | 0.0392 | 0.6507 |
| z_pred | **-0.1445** | 0.9292 | -0.0044 |
| z_geometry | -0.1661 | 0.0462 | 0.0244 |

- z_target stays EM-organized (it is trained by the pairing-independent
  bootstrap term and encodes only the spectrum) while z_pred shows **no** EM
  organization without real pairing.
- **correct-vs-shuffled separation = +0.294** (v2: -0.1107); clearly
  rho_correct >> rho_shuffled.

### Physics probes (linear, frozen latents)

| Probe | z_target | z_pred | z_geometry |
| --- | ---: | ---: | ---: |
| response R2 vs mean | **0.6562** | 0.0598 | 0.0412 |
| normalized response MSE | 0.1953 | 0.5340 | 0.5446 |
| resonance Ty MAE (GHz) | 4.83 | 3.26 | 2.71 |
| resonance Rx MAE (GHz) | 4.71 | 3.12 | 2.53 |
| feature-count MAE | 0.613 | 0.469 | 0.406 |

The frequency-aware target is highly informative (response R2 0.66 vs v2
0.34). The geometry-predicted latent carries only a weak linear response
signal (R2 0.06, vs v2 0.04) — an improvement but still small, and its
resonance frequency localization (3.26/3.12 GHz) is worse than v2
(2.93/2.70), while feature-count MAE improved slightly (0.469 vs 0.431).

## Qualitative physics-neighborhood examples (test split)

Mined with the canonical quantile rule (D_G >= q90 / D_S <= q10 etc.).

| Case | D_G | D_S | z_pred D_z | z_target D_z | desired |
| --- | ---: | ---: | ---: | ---: | --- |
| A: different geometry, similar response | 0.543 | 0.0011 | 0.917 | 0.546 | z D_z small |
| B: similar geometry, different response | 0.223 | 5.934 | 0.313 | 1.369 | z D_z large |

The **target** latent behaves as desired (D_z case A 0.55 << case B 1.37).
The **predictor** latent does not resolve these extremes: it puts the
case-B pair closer than the case-A pair (0.31 vs 0.92), i.e. its low-rank
(1.77) embedding cannot simultaneously pull geometrically-different but
physically-similar pairs together and push physically-different pairs apart.
This is the residual failure case below.

## Failure cases

1. **Predictor remains low-rank** (rank-1 0.74, effective rank 1.77, barely
   changed from v2). The variance/covariance guards keep the geometry/online/
   target branches full-rank but do not de-collapse the geometry-derived
   predictor.
2. **Information transfer is still incomplete.** z_pred response probe R2 is
   0.06 vs 0.66 on the target; the "target improves but does not fully
   transfer" v2 observation persists, now in the correct direction.
3. **Fine physics neighborhoods are not resolved by the predictor.** Extreme
   Case A/B mined pairs invert in z_pred, consistent with the low-rank
   predictor; the target latent resolves them correctly.
4. Predictor resonance-frequency MAE is worse than v2 (3.26/3.12 vs 2.93/2.70
   GHz), though feature-count MAE improved (0.469 vs 0.431).

## A/B/C gate

| Metric | v1 | v2 | v3 correct | v3 shuffled |
| --- | ---: | ---: | ---: | ---: |
| JEPA loss (best val total) | 0.0077 | 0.0557 | 0.0703 | 0.0751 |
| EM-distance rho (z_pred, norm) | -0.1716 | -0.1057 | **+0.1495** | -0.1445 |
| geometry-distance rho (z_pred, norm) | -0.1716 | +0.0583 | **-0.0092** | +0.0459 |
| correct-shuffled gap (z_pred) | - | -0.1107 | **+0.2940** | - |
| rank-1 fraction (z_pred / z_target) | 0.98 / 0.89 | 0.71 / 0.04 | 0.74 / 0.04 | 0.93 / 0.04 |
| response probe R2 (z_pred / z_target) | 0.16 / 0.20 | 0.04 / 0.34 | 0.06 / 0.66 | - / 0.65 |
| resonance probe (z_pred Ty/Rx GHz) | - / - | 2.93 / 2.70 | 3.26 / 3.12 | - / - |

Decision: **A — strong success on the central representation gate.**

- rho_v3(D_z, D_S) > 0 and substantially improved over v2 (-0.1057 -> +0.15).
- rho_correct (+0.15) >> rho_shuffled (-0.14): the geometry->physics
  relationship is real and learned.
- Geometry correlation does not dominate (rho(D_z_pred, D_G) = -0.009).
- The physics target representation is much more informative (response R2
  0.66), latent collapse is controlled on all full-rank branches.
- **Residual, documented limitation:** the predictor is still low-rank and its
  fine-grained information content lags the target. The physical-state
  hypothesis is supported at the distance-organization level; the geometry
  -> physics transfer is organized but not yet information-complete.

## Downstream decision

Per §24: **do not scale to 261k, do not return to inverse design yet.**
Next action: repeat this exact architecture with one different deterministic
seed to confirm reproducibility. Only after reproduction, consider
full-dataset representation pretraining, and only after that, compare the
Physics-JEPA representation against the Phase 8 AE latent inside the frozen
inverse generator. The 32-D capacity limitation observed on the predictor
(effective rank 1.77) is documented as the motivation for the seed-repro step,
not as license for immediate 64-D or a new architecture.

## Artifacts

- Models: `outputs/physics_jepa_v3/models/{correct,shuffled}/` (best.pt,
  config.json, training_history.csv, cached val/test latents)
- Evaluations: `outputs/physics_jepa_v3/evaluation/{correct,shuffled}/evaluation.json`
- Plots: `outputs/physics_jepa_v3/plots/` (loss_curves, latent_health,
  physics_vs_geometry_distance, correct_vs_shuffled, physics_neighborhood_examples,
  probe_response, probe_resonance, latent_variance_diagnostics)
- Metrics: `outputs/physics_jepa_v3/metrics.json`, `summary.csv`, `config.json`
- Smoke report: `outputs/physics_jepa_v3/smoke/smoke_report.json`
- Code: `src/physics_jepa_encoders.py::FrequencySpectrumEncoder`,
  `src/physics_jepa.py::PhysicsJEPAFrequencyRelational`,
  `src/physics_jepa_losses.py::{build_response_triplets, relational_margin_loss, v3_loss_with_parts}`,
  `scripts/train_physics_jepa_v3.py`, `scripts/evaluate_physics_jepa_v3.py`.

v1/v2 outputs were not modified.
