# Research Continuation Report: Phases 6-11

## From physics-conditioned completion to spectrum-to-geometry inverse design

**Project:** PhyJEPA-Metamaterial  
**Scope:** Phases 6, 6.1, 7, 7B, 8, 9, 10, and 11  
**Primary objective:**

```text
target electromagnetic response S_target -> new geometry G_new
```

**Current status:** The project now has a controlled learned candidate-
generation and screening baseline. It does not yet have a physically validated
inverse-design system. The independent RCWA path passes software and basic
passivity checks, but full-spectrum Fourier-order convergence and the required
generated-candidate calibration campaign remain incomplete.

---

## 1. Executive summary

Phases 6-11 converted the project from a physics-conditioned completion study
into a staged inverse-design investigation. The work tested whether a learned
forward surrogate could support physical optimization, whether target spectra
direct completion in the intended direction, whether complete geometries have
a usable latent representation, and whether one-to-many generation improves
the spectrum-to-geometry mapping.

The principal findings are:

- The independent RCWA implementation is operational for bounded experiments
  and passes software and basic passivity tests, but is too slow in the current
  configuration to establish the required full-spectrum convergence gate.
- The Phase 5A completion model is target-sensitive, but counterfactual target
  responses do not reliably direct completions toward the requested response.
- The 30k resonance-aware forward CNN is the strongest available learned
  screening surrogate, with shared-holdout normalized MSE `0.300002` and the
  strongest recorded local perturbation sign agreement (`0.944`). It remains a
  surrogate, not Maxwell ground truth.
- The geometry autoencoder passes its representation gate with test IoU
  `0.965106`, Dice `0.981470`, and pixel accuracy `0.982570`.
- Deterministic spectrum-to-geometry predictors are insufficient: direct MLP
  screened response MSE is `0.483234`, and the latent predictor is `0.444132`.
  Train-only response nearest-neighbor retrieval is stronger at `0.259249`.
- The stochastic latent generator improves over both learned deterministic
  baselines with best-of-8 screened response MSE `0.338217`, while producing
  `98.125%` valid candidates and nontrivial diversity.
- Validity-first ranking preserves all 500 test targets and has essentially the
  same screening error as unrestricted ranking. A hard novelty filter raises
  novelty but reduces coverage to `72.8%` and worsens mean best screened
  response MSE to `0.451252`.

The defensible conclusion is:

> The project has a reproducible learned baseline for generating and screening
> candidates under `S_target -> G_new`. It does not yet have evidence for
> Maxwell-valid inverse design, calibrated surrogate optimization, or reliable
> counterfactual physical direction.

---

## 2. Research question and stage contract

The project separates several related but non-equivalent tasks:

```text
completion:
    (G_partial, M) -> G

physics-conditioned completion:
    (G_partial, M, S_target) -> G

physics-consistent completion:
    (G_partial, M, S_target) -> G -> F_surrogate(G)

inverse design:
    S_target -> G_new
```

Phases 8-11 focus on the final task. A new geometry is not required to contain
a known partial geometry, and the inverse model must not receive information
that identifies the paired original geometry.

The anti-leakage contract was:

- input: normalized target response with shape `[B, 4, 1001]`;
- allowed stochastic input: independent noise at generation time;
- forbidden inputs: partial geometry, mask, source ID, original geometry,
  complexity metadata, and target geometry latent;
- geometry and frozen geometry latents: labels or evaluation targets only;
- forward CNN: post-generation screening only, never a training loss;
- independent RCWA: separate validation evidence, not a hidden training signal.

This contract matters because one response can correspond to multiple
geometries. A model that receives the original geometry, partial geometry, or a
paired latent could achieve a low numerical loss while failing the actual
inverse-design problem.

---

## 3. Shared protocol

### 3.1 Geometry, physics, and response representation

The stored geometry is a binary `16 x 16` pattern. For RCWA, it is mapped to a
`20 x 20` raster using `0.5 mm` cells and a `1 mm` air border. The period is
`10 x 10 mm`.

The response representation is four real channels on the authoritative grid:

```text
2.00 GHz to 12.00 GHz
1001 points
0.01 GHz spacing
[Re(T_y), Im(T_y), Re(R_x), Im(R_x)]
```

The documented physical setup contains patch copper thickness `0.018 mm`,
backing copper thickness `0.18 mm`, substrate relative permittivity `2.65`,
substrate loss tangent `0.003`, copper conductivity `5.8 x 10^7 S/m`, and
relative permeability `1`. Substrate thickness was not established from the
original CST source.

### 3.2 Dataset split

The learned inverse-design experiments use the verified 5k subset and seed-42
split:

| Split | Samples |
| --- | ---: |
| Train | 4,000 |
| Validation | 500 |
| Test | 500 |

The test set is source-ID disjoint from training. The same test targets are
used for the Phase 8-11 learned experiments. Phase 7 evaluates 500 held-out
A-to-B pairs per condition.

### 3.3 Metric families

- **Response MSE:** distance between predicted or screened and target response
  channels. In Phases 7B-11 this is learned-surrogate MSE unless marked RCWA.
- **IoU and Dice:** binary geometry overlap after thresholding at `0.5`.
- **Pixel accuracy:** fraction of equal binary pixels.
- **Occupancy difference:** absolute difference in occupied-pixel fraction.
- **Validity:** finite, non-empty, non-full geometry within occupancy and simple
  topology limits.
- **Pixel novelty:** nearest-training-set pixel Hamming distance.
- **Latent novelty:** nearest-training-set MSE in the frozen Phase 8 latent.
- **Multi-solution success:** at least two candidates for one target with MSE at
  or below `0.30`.
- **Complexity stratification:** simple, medium, and complex groups are always
  retained because global means hide a large difficulty gradient.

The `0.30` response threshold is near the Phase 9 train-only nearest-neighbor
screening MSE (`0.259249`). It is a learned screening gate, not a physical
accuracy threshold.

---

## 4. Phase 6: surrogate calibration and independent RCWA

Phase 6 contains three related records. They are reported separately so that
historical surrogate-only diagnostics are not confused with later exploratory
RCWA or the stricter full-spectrum gate.

### 4.1 Historical surrogate-only audit

The initial Phase 6 audit asked whether the frozen Phase 2.5 CNN could be used
as a physics objective. No compatible independent solver interface was then
available. The audit measured only:

```text
G -> frozen CNN -> S_surrogate
```

against stored responses for held-out dataset geometries. It could not measure
the required generated-geometry comparison:

```text
G_generated -> independent solver -> S_real
```

The held-out stored-reference diagnostic gave overall normalized MSE `0.340066`.
Errors were concentrated in `R_x`:

| Component | Normalized MSE |
| --- | ---: |
| Re(Ty) | 0.059497 |
| Im(Ty) | 0.040873 |
| Re(Rx) | 0.609494 |
| Im(Rx) | 0.650400 |

Magnitude MAE was `0.061682` for `|Ty|` and `0.043540` for `|Rx|`. A
resonance-sensitive diagnostic matched `88/95` reference features (`92.63%`),
with mean resonance-frequency error `0.360 GHz` and mean resonance-region
magnitude MAE `0.247681`.

The complexity trend was substantial:

| Group | Samples | Mean stored-reference normalized MSE |
| --- | ---: | ---: |
| Simple | 12 | 0.044324 |
| Medium | 12 | 0.268695 |
| Complex | 12 | 0.707179 |

Phase 5A/5B generated geometries sometimes had lower surrogate-to-target error,
but this measured only the learned objective. It could not test surrogate
exploitation because generated structures had no independent `S_real`. The
classification was C and unconstrained inverse design was stopped.

### 4.2 Exploratory independent RCWA

The next run integrated `meent 0.12.0` with its PyTorch backend and complex128.
It used only three frequencies (`2.0`, `7.0`, and `12.0 GHz`) and orders `1, 3,
5, 7` on one representative geometry.

The exploratory comparison mapping was `p_to_ty_s_to_rx`: meent TM/p was
compared with stored channel 0 and TE/s with stored channel 2. The stricter
validation path documents `s_to_ty_p_to_rx`, so this mapping remains unresolved
and was not silently relabeled as physical ground truth.

A quick calibration selected substrate thickness `0.15 mm` among tested
options, but used only three structures and three frequencies. Its best overall
normalized MSE was `0.261084`:

| Component | Normalized MSE |
| --- | ---: |
| Re(Ty) | 0.996598 |
| Im(Ty) | 0.047739 |
| Re(Rx) | 6.65267e-07 |
| Im(Rx) | 2.76342e-07 |

CNN-versus-RCWA MSE on the same small set was `0.273063`. The generated screen
used only three targets and three candidate models. Its mean CNN-versus-RCWA
MSE was `0.252293`, with Spearman mean `1.0`, pairwise agreement `0.555556`,
and top-1 overlap `1.0`; the sample was too small for a ranking claim.

The implementation passed `52` tests and the recorded maximum reflected-plus-
transmitted power was `0.999876`, supporting basic passive-power behavior but
not convergence.

### 4.3 Full-spectrum Phase 6.1 gate

Phase 6.1 tightened the protocol to the full `2-12 GHz`, `1001`-point grid and
created a deterministic nine-geometry manifest: three simple, three medium,
and three complex held-out geometries, all source-ID disjoint.

Only one simple representative completed the convergence experiment:

| Transition | Full-spectrum MSE | Runtime |
| --- | ---: | ---: |
| N=1 to N=3 | 0.00710550 | 43.5 s for N=3 |
| N=3 to N=5 | 0.00615410 | 356.1 s for N=5 |
| N=5 to N=7 | 0.00994316 | 1852.9 s for N=7 |

The required `N=9` to `N=11` difference of at most `1e-4` was not reached,
and the available sequence was non-monotonic. Order 7 took about 30.9 minutes
for one simple geometry on CPU. An order-9 CUDA benchmark did not finish three
frequencies within 124 seconds; extrapolation to 1001 points would exceed 11
hours for one geometry and order.

Therefore:

```text
Convergence status: NO CONVERGENCE ESTABLISHED
Selected production order: none
Final classification: C
```

Basic sanity behavior remained positive. Maximum reflected-plus-transmitted
power was `0.999876`, and ordinary transmitted power was approximately
`4.84 x 10^-114` in the recorded cases. The test suite at this stage had `56`
passing tests.

The following were intentionally not reported: nine-geometry full-spectrum
MSE, CNN-to-RCWA full-spectrum ranking, generated exploitation counts, a
physical Phase 5B Pareto frontier, and the planned `20/20/20` generated-
candidate validation.

### 4.4 Phase 6 decision

The RCWA code is usable for bounded engineering experiments, but not yet as a
production physical objective. The unresolved mapping, uncalibrated substrate
thickness, missing Fourier convergence, and incomplete generated validation
are decisive limitations. Later phases therefore use the learned CNN only as
a labeled screening surrogate.

---

## 5. Phase 7: counterfactual target direction

### 5.1 Question and controls

Phase 5A showed that changing the supplied EM response changes hidden pixels.
Phase 7 tested whether changing the target from `S_A` to `S_B` moves completion
toward B's geometry and response.

For each held-out geometry A, a deterministic derangement supplied a different
held-out geometry B:

```text
partial(A), S_A -> correct completion
partial(A), S_B -> counterfactual completion
partial(A), no response -> matched control
```

The partial input and mask were identical across all calls. The complete target
geometry was used only for evaluation bookkeeping. Four Phase 5A conditions
were evaluated on the seed-42 5k test split, with 500 A-to-B pairs per
condition.

### 5.2 Directional results

| Condition | Geometry gain vs correct | Response gain to B vs correct | Response gain to B vs control | Geometry gains > 0 | Response gains > 0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Central 25% (5aA) | +0.006719 | -0.056301 | -0.054806 | 0.508 | 0.398 |
| Central 50% (5aB) | +0.016586 | -0.036353 | -0.007904 | 0.538 | 0.516 |
| Random 25% (5aC) | +0.000953 | +0.002075 | -0.011595 | 0.382 | 0.442 |
| Random 50% (5aD) | +0.005383 | +0.006444 | -0.001476 | 0.546 | 0.528 |

The two central-block conditions had negative mean response gain versus the
correct completion. The random-hole conditions were near zero. No condition
had positive mean gain versus both correct completion and control.

| Condition | Correct to A | Counterfactual to A | Correct to B | Counterfactual to B |
| --- | ---: | ---: | ---: | ---: |
| 5aA | 0.358360 | 0.502548 | 0.840440 | 0.896740 |
| 5aB | 0.417229 | 0.555927 | 0.845420 | 0.881773 |
| 5aC | 0.357265 | 0.359988 | 0.855561 | 0.853485 |
| 5aD | 0.356128 | 0.396968 | 0.827900 | 0.821456 |

### 5.3 Decision

The model is response-sensitive but not reliably target-directed. It moves away
from A more strongly than it moves toward B in screened response space.

**Decision: B - target-sensitive but weakly directed.**

This stopped the direct use of the Phase 5A completion model as evidence for
inverse design and justified evaluating a separate unconstrained pipeline.

---

## 6. Phase 7B: forward screening surrogate selection

### 6.1 Objective and selection rule

Before inverse modeling, existing forward checkpoints were compared as cheap
candidate screeners. No checkpoint was retrained and no new inverse model was
optimized against the comparison.

Candidates were a 5k normalized-MSE baseline, a 5k resonance-aware CNN, a 30k
resonance-aware CNN, and a 30k response-aware CNN. The lexicographic priority
was broad normalized response MSE, resonance-frequency error, then
resonance-region magnitude MAE. Complexity and gradient diagnostics were
retained as safety checks.

### 6.2 Results

| Candidate | Normalized MSE | Resonance error (GHz) | Resonance MAE | Feature match | Inference ms/sample | Gradient sign agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5k MSE | 0.331021 | 0.525456 | 0.241849 | 0.964103 | 0.373897 | 0.832 |
| 5k resonance | 0.361289 | 0.103201 | 0.234538 | 1.000000 | 0.388122 | 0.848 |
| 30k resonance | **0.300002** | 0.243154 | **0.226388** | 0.994139 | 0.384390 | **0.944** |
| 30k response-aware | 0.312470 | 0.316475 | 0.227351 | 0.995604 | 0.667629 | not recorded |

The selected checkpoint is:

```text
outputs/phase2_5/exp_C_30k_resonance/best.pt
```

Its shared-holdout normalized MSE means were `0.0365` simple, `0.1800`
medium, and `0.6923` complex. The complex group remains a major screening
risk. The checkpoint is suitable for ranking and filtering only; its gradients
are not calibrated physical gradients.

---

## 7. Phase 8: complete-geometry latent autoencoder

### 7.1 Method

Phase 8 trained a standalone autoencoder on complete geometries only:

```text
G [B,1,16,16] -> encoder -> z_G [B,64,8,8]
                         -> decoder -> geometry logits [B,1,16,16]
```

It used no masks, responses, JEPA loss, completion predictor, or physics
loss. Training used AdamW, learning rate `1e-3`, weight decay `1e-4`, batch
size 64, a 75-epoch ceiling, patience 10, and binary threshold `0.5`.

### 7.2 Results and decision

| Split | BCE | IoU | Dice | Pixel accuracy | Occupancy difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 0.0358 | 0.9728 | 0.9858 | 0.9865 | 0.0034 |
| Validation | 0.0413 | 0.9640 | 0.9809 | 0.9828 | 0.0043 |
| Test | **0.040649** | **0.965106** | **0.981470** | **0.982570** | **0.004086** |

Test latents, decoder logits, and probabilities were finite. Latent caches were
stored in deterministic manifest order with matching source-ID manifests.

**Decision: Supported.** The autoencoder provides a usable geometry latent and
decoder for controlled inverse baselines. Latent distance is not automatically
a physical design distance and is retained as a representation diagnostic.

---

## 8. Phase 9: deterministic spectrum-to-geometry baselines

### 8.1 Contract and models

The first true unconstrained inverse-design mapping was:

```text
S_target [B,4,1001] -> inverse model -> G_new
```

The inverse input contained only normalized target response. The three models
were train-only response nearest neighbor, direct MLP to geometry logits, and a
deterministic latent predictor to `[64,8,8]` followed by the frozen decoder.

The latent objective was latent MSE plus geometry BCE; the direct model used
geometry BCE. The selected forward CNN was applied after generation for
screening only.

### 8.2 Test results

| Method | Latent MSE | Geometry BCE | IoU | Dice | Pixel accuracy | Occupancy difference | Screening response MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EM nearest neighbor | 1.290497 | n/a | 0.479657 | 0.611589 | 0.678203 | 0.064328 | **0.259249** |
| Direct MLP | n/a | 0.509034 | 0.477061 | 0.580257 | **0.715469** | 0.110313 | 0.483234 |
| Latent predictor | **0.756675** | 0.552146 | **0.494825** | 0.610101 | 0.708016 | 0.076109 | 0.444132 |

### 8.3 Complexity and diagnosis

| Method | Simple | Medium | Complex |
| --- | ---: | ---: | ---: |
| EM nearest neighbor | 0.034821 | 0.157808 | 0.592612 |
| Direct MLP | 0.278783 | 0.433326 | 0.747474 |
| Latent predictor | 0.144386 | 0.335738 | 0.864226 |

The latent predictor slightly improved geometry IoU but did not solve response
matching. The direct and latent models are consistent with averaging over a
many-to-one inverse relation. The decoder passed its gate, so the main failure
is the inverse mapping rather than an unusable geometry representation.

**Decision:** deterministic inverse design is insufficient as the final system;
a one-to-many stochastic generator is justified.

---

## 9. Phase 10: stochastic latent inverse design

### 9.1 Method and leakage controls

Phase 10 tested:

```text
S_target + noise -> conditional latent generator -> multiple geometries
```

The conditional latent VAE predicted a response-conditioned prior and used a
posterior only during training. Sampled latent codes were decoded by the frozen
Phase 8 decoder. The generator used hidden dimension 256, latent shape
`[64,8,8]`, KL weight `1e-4`, prior weight `0.5`, AdamW, and early stopping;
the best checkpoint was epoch 12.

At inference it received only normalized target response and independent noise.
It did not receive partial geometry, mask, source ID, original geometry,
complexity metadata, or target geometry latent. The forward CNN was screening
only.

### 9.2 Candidate results

Eight candidates were generated for each of 500 test spectra, giving 4,000
candidates. Validity required finite, non-empty, non-full geometries within
training-derived occupancy and simple topology limits.

| Metric | Value |
| --- | ---: |
| Candidate count | 4,000 |
| Validity rate | 0.981250 |
| Valid candidates per target | 7.850000 |
| Best screened response MSE | **0.338217** |
| Best valid screened response MSE | 0.338230 |
| Multi-solution success fraction | 0.594000 |
| Pairwise candidate Hamming diversity | 0.201257 |
| Nearest-train pixel novelty | 0.158440 |
| Nearest-train latent novelty MSE | 0.710898 |
| Duplicate candidates per target | 0.008000 |

The stochastic model improved over the direct MLP by `0.145017` MSE and over
the deterministic latent predictor by `0.105915`, but remained `0.078968` above
train-only nearest-neighbor retrieval.

### 9.3 Complexity results

| Group | Targets | Validity | Best screened response MSE | Multi-solution success | Pixel novelty | Latent novelty MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Simple | 178 | 1.000000 | 0.124743 | 0.859551 | 0.043635 | 0.213514 |
| Medium | 155 | 0.949194 | 0.284115 | 0.677419 | 0.143665 | 0.720161 |
| Complex | 167 | 0.991018 | 0.615966 | 0.233533 | 0.294521 | 1.232447 |

Complex targets remain the bottleneck: novelty and validity are high, but
response satisfaction is low. The next useful improvement is ranking or
refinement, not simply more validity regularization.

**Decision:** supported as a controlled stochastic baseline, not a physically
validated inverse solver.

---

## 10. Phase 11: surrogate ranking

### 10.1 Scope and policies

Phase 11 evaluated the saved 4,000 Phase 10 candidates without retraining. It
grouped candidates by target, filtered them, deduplicated by exact binary
geometry hash, and ranked them by learned-surrogate response MSE.

Policies:

- `all`: all candidates;
- `valid`: Phase 10 deterministic-valid candidates only;
- `valid_novel`: valid candidates with pixel novelty at least `0.05` and
  latent novelty MSE at least `0.25`.

### 10.2 Aggregate results

| Policy | Target coverage | Best screened response MSE | Top-1 success | Selected validity | Pixel novelty | Latent novelty MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 1.000 | 0.338217 | 0.634 | 0.984 | 0.160977 | 0.723211 |
| valid | 1.000 | 0.338230 | 0.634 | 1.000 | 0.161492 | 0.723560 |
| valid_novel | 0.728 | 0.451252 | 0.508 | 1.000 | 0.212665 | 0.943453 |

The validity filter changes mean best screening MSE by only `0.000013` while
raising selected validity to 1.0 and preserving all 500 targets. The novelty
filter raises both novelty measures but loses 136 targets and worsens mean best
screened MSE by `0.113021`. Only four exact duplicate geometries were removed
from the per-target candidate sets.

### 10.3 Complexity results

| Policy / group | Coverage | Best screened response MSE | Success | Pixel novelty | Latent novelty MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| all / simple | 178/178 | 0.124743 | 0.888 | 0.045931 | 0.225409 |
| all / medium | 155/155 | 0.284115 | 0.710 | 0.144657 | 0.732305 |
| all / complex | 167/167 | 0.615966 | 0.293 | 0.298746 | 1.245362 |
| valid_novel / simple | 74/178 | 0.235344 | 0.784 | 0.072424 | 0.341435 |
| valid_novel / medium | 124/155 | 0.354590 | 0.637 | 0.178144 | 0.888460 |
| valid_novel / complex | 166/167 | 0.619705 | 0.289 | 0.300970 | 1.252903 |

Simple targets are easiest to match but frequently fail a novelty requirement.
Complex targets are novel enough to survive, but still have poor response
matching. Novelty is therefore not a substitute for physical or spectral
quality.

**Decision:** supported as a learned-surrogate ranking stage. The default
artifact-level policy is `valid`; `valid_novel` is optional when novelty is a
hard requirement.

---

## 11. Consolidated comparison

| Method | New geometry? | IoU | Dice | Pixel accuracy | Screened response MSE | Role |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Train-only EM nearest neighbor | No, retrieves train geometry | 0.479657 | 0.611589 | 0.678203 | **0.259249** | Retrieval reference |
| Direct deterministic MLP | Yes | 0.477061 | 0.580257 | **0.715469** | 0.483234 | Deterministic baseline |
| Deterministic latent predictor | Yes | **0.494825** | 0.610101 | 0.708016 | 0.444132 | Latent baseline |
| Stochastic generator, best of 8 | Yes, multiple | n/a | n/a | n/a | **0.338217** | Candidate generator |
| Phase 11 valid top-1 | Yes, selected | n/a | n/a | n/a | 0.338230 | Validity-first screen |

Geometry reconstruction and response matching answer different questions. The
stochastic and ranked models are evaluated by target response, validity,
diversity, and novelty rather than exact recovery of a paired geometry. The
nearest-neighbor result is a strong reference but is retrieval, not generation.

---

## 12. Established findings and unsupported claims

### 12.1 Established findings

1. Basic RCWA software behavior and passivity are supported by bounded tests.
2. Full-spectrum production convergence is not established.
3. The selected forward CNN is the best available screening artifact, with
   complexity-dependent error.
4. Response conditioning changes completion outputs without reliable
   counterfactual direction.
5. Complete geometry has a stable enough latent for controlled baselines.
6. A deterministic inverse output is inadequate for the many-to-one mapping.
7. Stochastic sampling improves learned screening response over deterministic
   learned baselines.
8. Validity-first ranking preserves coverage at negligible screening cost.
9. Novelty and response satisfaction currently trade off, especially for simple
   targets.

### 12.2 Unsupported claims

The results do not establish that:

- generated geometries are Maxwell-valid;
- lower CNN target error implies lower RCWA target error;
- the CNN is a calibrated differentiable physics objective;
- counterfactual targets produce physically correct direction;
- the stochastic generator solves arbitrary spectra;
- pixel or autoencoder-latent novelty is physical novelty;
- the small exploratory RCWA ranking result generalizes to 1001 points;
- a converged production RCWA Fourier order has been selected.

---

## 13. Recommended operating procedure

For current artifact-level use:

1. Provide a normalized target spectrum on the verified `[4,1001]` grid.
2. Generate multiple candidates with the Phase 10 conditional generator.
3. Apply deterministic validity checks.
4. Rank valid candidates using the selected Phase 7B CNN.
5. Retain the `valid` top candidate or a small top-k set.
6. Report complexity, surrogate response error, validity, diversity, novelty,
   and coverage together.
7. Treat all output as learned-screening evidence until independent RCWA is
   completed.

Use `valid_novel` only when being away from the training distribution is an
explicit hard requirement, and report its coverage penalty.

---

## 14. Required next physical validation gate

The next meaningful step is a small independent RCWA campaign after making the
solver computationally practical and physically unambiguous. It should:

1. resolve polarization/channel mapping;
2. calibrate substrate thickness and remaining physical parameters;
3. establish a Fourier order using simple, medium, and complex representatives;
4. run the exact 1001-point frequency grid;
5. evaluate held-out dataset geometries and selected generated geometries;
6. compare surrogate and RCWA errors per channel and frequency;
7. measure Spearman, pairwise, and top-k ranking agreement;
8. count low-surrogate/high-RCWA exploitation cases;
9. compare `all`, `valid`, and `valid_novel` under independent RCWA;
10. only then decide whether surrogate-guided refinement or a physics loss is
    justified.

The planned `20/20/20` generated-candidate validation is a reasonable minimum
starting point after convergence is established. Until this gate passes, do not
claim physically validated inverse design, Maxwell-consistent geometries, or
calibrated surrogate optimization.

---

## 15. Reproducibility and artifact index

### Phase reports

- [Phase 6 historical surrogate calibration](phase6_surrogate_calibration.md)
- [Phase 6 exploratory RCWA validation](phase6_rcwa_validation.md)
- [Phase 6.1 full-spectrum RCWA validation](phase6_1_full_spectrum_rcwa_validation.md)
- [Phase 7 counterfactual target direction](phase7_counterfactual_target_direction.md)
- [Phase 7B forward screening surrogate selection](phase7b_forward_screening_surrogate_selection.md)
- [Phase 8 geometry latent autoencoder](phase8_geometry_latent_autoencoder.md)
- [Phase 9 deterministic inverse baselines](phase9_spectrum_to_geometry_baseline.md)
- [Phase 10 stochastic inverse design](phase10_stochastic_inverse_design.md)
- [Phase 11 candidate ranking](phase11_candidate_ranking.md)

### Main implementation and output locations

- RCWA: `src/rcwa_solver.py`, `src/rcwa_validation.py`,
  `scripts/analyze_rcwa_convergence.py`;
- Phase 7: `scripts/evaluate_phase7_counterfactual.py`,
  `outputs/phase7_counterfactual/`;
- Phase 7B: `scripts/select_forward_screening_surrogate.py`,
  `outputs/phase2_5/surrogate_selection/`;
- Phase 8: `src/geometry_autoencoder.py`,
  `outputs/phase8_geometry_autoencoder/`;
- Phase 9: `src/spectrum_inverse_models.py`,
  `outputs/phase9_inverse_baselines/`;
- Phase 10: `src/conditional_latent_vae.py`,
  `outputs/phase10_stochastic_inverse_design/`;
- Phase 11: `scripts/rank_phase10_candidates.py`,
  `outputs/phase11_candidate_ranking/`.

The repository test suite currently reports:

```text
72 passed, 3 warnings
```

The warnings are NumPy deprecation warnings inside the installed `meent`
dependency during RCWA tests and did not cause failures.

---

## 16. Final scientific conclusion

The evidence supports the following progression:

```text
uncalibrated surrogate
    -> bounded RCWA evidence, but no convergence gate
    -> target sensitivity without reliable counterfactual direction
    -> selected forward screening surrogate
    -> validated geometry latent
    -> deterministic inverse baselines
    -> stochastic candidate generator
    -> validity and novelty-aware surrogate ranking
```

The strongest current achievement is a reproducible candidate-generation and
screening framework with explicit failure boundaries. It generates mostly valid
and diverse geometries, improves over learned deterministic baselines in
surrogate response space, preserves full coverage under validity-first ranking,
and exposes the complex-target bottleneck.

The decisive next evidence must come from an independently converged,
full-spectrum RCWA evaluation of selected generated geometries. Until then,
Phase 11 `valid` ranking is the recommended learned artifact-level policy, and
the forward CNN must remain a screening surrogate rather than a physical
objective.
