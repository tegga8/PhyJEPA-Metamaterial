# Physics-JEPA Open Questions

Open hypotheses after v1/v2. These are **hypotheses, not proven root causes**. The
v3 frequency-aware + relational experiment is designed to probe them jointly.

## 1. Frequency-local information may be lost

The current spectrum encoder
(`src/physics_jepa_encoders.py::PhysicsSpectrumEncoder`) maps `[B,4,1001]` through
`Conv1d(4->32->64)` with kernel 9 and then `AdaptiveAvgPool1d(num_tokens=12)` **before**
attention. Consequences to test:

- narrow resonances (a few GHz points wide) can be mixed away by 1001->12 adaptive pooling;
- resonance location / bandwidth / phase-change / nearby-resonance structure is not explicitly
  encoded;
- no explicit frequency-position embedding is attached to the tokens.
- Retaining local frequency resolution at token scale is expected to be necessary to represent the
  physics the residual probe needs (narrow resonances, resonance location, bandwidth, phase changes,
  nearby resonances).

## 2. The geometry -> physics mapping remains weak

- In both v1 and v2 the predictor (`z_pred`) shows weak or negative normalized correlation with
  EM-response distance even when the target branch improves (v2 response-probe R2 rose to 0.34 on
  `z_target` but fell to 0.04 on `z_pred`).
- Question: is the bottleneck the target representation (hypothesis 1), the objective
  (hypothesis 3), or fundamentally the ambiguity of geometry->physics (multiple geometries map to
  similar responses)?

## 3. Pointwise JEPA loss may not enforce EM-similarity organization

The loss is a normalized latent MSE between predicted and target latents. It matches per-sample
vectors; it never requires that pairwise latent distances track pairwise EM-response distances.
- Test: add a small relational term so that for triplets with
  `D_S(S_i,S_j) < D_S(S_i,S_k)`, the latents satisfy `D_z(z_i,z_j) < D_z(z_i,z_k)` (soft, weighted).

## 4. The predictor remains partially low-rank after v2

v2 `z_pred` rank-1 fraction is still 0.71 despite the covariance term being applied to the online
and geometry branches. The target is now full-rank, but the geometry/predictor branch is not.
- Questions: does the predictor need its own collapse guard? Or does fixing the target content
  (hypotheses 1/3) remove the cause?

## 5. Target representation improved but the predictor did not

v2 fixed target collapse (rank-1 0.89 -> 0.04; response R2 0.20 -> 0.34) but `z_pred` did not
improve. This decoupling is the single most important observation for v3.

## 6. Hypotheses remain one idea, not three separate changes

The v3 modification is a single hypothesis: a spectrum target that preserves frequency-local
structure + an objective that explicitly requires EM-similarity ordering. Do not decompose this
into three unrelated architectural changes.