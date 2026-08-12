# Phase 12 — Response-Space Generalization, Multimodality, and Bottleneck Diagnosis

## Executive decision: E

The fixed Phase 10 stochastic generator is a valid target-conditioned candidate
generator under the learned screening model, but it is **not competitive with
the fixed train-only EM nearest-neighbor baseline on mean screening MSE**. No
new inverse architecture should be introduced from these results. The immediate
work is to diagnose response-space coverage and the learned screening model's
complex-case limitations before scaling the generator.

The screening CNN is a learned forward surrogate. It is not Maxwell ground
truth, an exact physical solver, or independent EM validation.

## Protocol and anti-leakage contract

- Fixed generator: `outputs/phase10_stochastic_inverse_design/generator/best.pt`
  (Phase 10 checkpoint; K=8).
- Held-out evaluation reuses the Phase 10 candidate cache; it does not retrain
  or silently resample the baseline.
- Interpolation stress test: 10 deterministic source pairs for each of the six
  complexity-pair types, at alpha = 0.25, 0.50, and 0.75: **180 targets**.
- Generator inputs are only normalized target response and random noise. Source
  IDs, geometries, masks, complexity, geometry latents, and split labels are
  evaluation metadata and are not passed to `generator.sample`.
- `screening MSE < 0.30` is an empirical learned-screening threshold, not a
  physical law.

Interpolated spectra are deliberately treated as response-space stress targets;
they are not asserted to be physically realizable.

## Table 1 — Interpolated target performance

Each row aggregates 10 targets. “Median MSE” is the mean of each target's
within-K median screening MSE.

| Complexity pair | Alpha | Best-of-8 MSE | Median MSE | Validity |
| --- | ---: | ---: | ---: | ---: |
| simple-simple | .25 | .7228 | 1.0734 | 1.000 |
| simple-simple | .50 | .1787 | .4718 | 1.000 |
| simple-simple | .75 | .0638 | .1080 | 1.000 |
| medium-medium | .25 | .0588 | .1246 | .900 |
| medium-medium | .50 | .0939 | .2473 | .875 |
| medium-medium | .75 | .0730 | .2418 | .750 |
| complex-complex | .25 | .5107 | 1.3030 | 1.000 |
| complex-complex | .50 | .4193 | .9490 | 1.000 |
| complex-complex | .75 | .4451 | 1.2834 | 1.000 |
| simple-medium | .25 | .1115 | .3576 | .975 |
| simple-medium | .50 | .1105 | .2291 | .925 |
| simple-medium | .75 | .0849 | .2397 | .938 |
| simple-complex | .25 | .3323 | .7658 | .963 |
| simple-complex | .50 | .3028 | .6952 | 1.000 |
| simple-complex | .75 | .5540 | 1.0021 | .950 |
| medium-complex | .25 | .6739 | 1.1467 | 1.000 |
| medium-complex | .50 | .5427 | .7548 | .988 |
| medium-complex | .75 | .3016 | .6457 | .988 |

Overall interpolation best-of-8 MSE is **0.3100** and candidate validity is
**0.9583**. Some blended targets screen well, including several cross-complexity
cases, but the result is uneven and is not evidence that arbitrary blended
spectra are physically feasible.

## Table 2 — Generator versus train-only nearest-neighbor retrieval

| Complexity | NN MSE | Generator MSE | Relative improvement | Generator-win fraction |
| --- | ---: | ---: | ---: | ---: |
| simple | .0348 | .1247 | -2.3320 | .393 |
| medium | .1578 | .2841 | -3.3527 | .484 |
| complex | .5926 | .6160 | -.0452 | .527 |
| all | **.2592** | .3382 | -1.8847 mean | .466 |

Median NN and generator MSE are .0423 and .1068. NN wins on 53.4% of targets;
the generator wins on 46.6%; no ties occurred within `1e-6`. The relative
improvement mean is sensitive to very small NN errors, so its median (-.0449)
is the more stable summary. The generated candidates can still be novel, but
that is a novelty/diversity result—not response superiority.

## Table 3 — Useful diversity

Per-target detail is in `outputs/phase12/diversity/useful_diversity.csv`.

| Summary | Value |
| --- | ---: |
| useful-diversity rate (at least two good candidates) | .594 |
| average good-candidate count | 4.180 |
| mean pairwise geometry Hamming distance | .2013 |
| Phase 10 candidate validity | .9813 |

The rate is encouraging, but it cannot by itself establish physics-preserving
inverse diversity because the response quantity is from the learned screening
surrogate. Candidate-level plots explicitly compare distance from the best
candidate with the change in screening error.

## VAE stochasticity

“High” and “low” use empirical medians over held-out targets: pairwise geometry
Hamming .1071 and pairwise surrogate-response MSE .0616.

| Type | Fraction | Interpretation |
| --- | ---: | --- |
| Type 1 | .112 | high geometry diversity, low surrogate-response variation |
| Type 2 | .388 | high diversity, high response variation |
| Type 3 | .388 | low diversity, low response variation |
| Type 4 | .112 | low diversity, high response variation |

Useful multimodality (Type 1) is present but is a minority. Type 2 and Type 4
show that a substantial fraction of the apparent diversity is accompanied by
response degradation or instability.

## Table 4 — JEPA versus autoencoder

| Representation | Reconstruction IoU | Best-of-8 response MSE | Validity | Diversity | Novelty | Complex MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 8 geometry AE | .9649 | .3382 | .9813 | .2013 | .1584 | .6160 |
| validated spatial JEPA completion | .6454 | not applicable | not applicable | not applicable | not applicable | not applicable |

This is a negative but important compatibility finding. The validated spatial
JEPA checkpoint is a masked-geometry completion model: its inference path
requires geometry context. Supplying that context in response-only inverse
design would violate the Phase 12 anti-leakage contract. Consequently a fair
JEPA-vs-AE response-only inverse comparison has **not been established**, and
JEPA must not be claimed as the critical representation for this generator.

## Table 5 — Bottleneck diagnosis

| Hypothesis | Evidence | Status |
| --- | --- | --- |
| D1 EM representation | Current conditioning is Flatten `[4,1001]` → 256-D MLP (1,091,072 parameters). It retains all 1001 bins and has no global pooling, but has no frequency-local inductive bias. | plausible; not isolated |
| D2 geometry latent | Phase 8 reconstruction IoU: simple 1.0000, medium .9901, complex .9047. | contributes on complex data; not primary proof |
| D3 inverse multimodality | Complex targets have 53.37 response-local train neighbors on average, with mean pairwise geometry Hamming .4533 under the documented adaptive threshold `max(.01, 1.25×nearest response distance)`. | strongly supported |
| D4 forward-surrogate limitation | Held-out screening-surrogate MSE is .0365 simple, .1800 medium, and .6923 complex. | strongly supported; complex candidate screening is unreliable |

The most immediate bottleneck is the combination of strong complex-response
multimodality and the screening surrogate's weak complex-geometry fidelity.
The AE degrades for complex geometry, while the MLP conditioning lacks a
frequency-local bias, but this phase does not isolate either as the sole cause.

## Architecture decision

**Other — preserve the current generator and do not scale it yet.** Nearest
neighbor remains substantially better in mean screening error, while the
complex screening surrogate error (.6923) is larger than the generator's
complex candidate error (.6160). First evaluate response-space coverage and
calibrate/improve the forward screening model on complex geometries. Only then
test a small frequency-aware EM encoder if conditioning diagnostics show it is
the limiting factor. A conditional latent flow is a justified later test if
multimodality remains the limiting factor after reliable screening.

## Claim discipline

Proven:

- The fixed system produces target-conditioned candidates from response plus
  noise only.
- Train-only nearest-neighbor retrieval has lower mean learned-screening MSE.

Supported but not proven:

- The sampled generator exhibits geometry and surrogate-response variation.
- Complex targets are markedly harder under the learned screening model.

Not established:

- Generated geometries reproduce the targets under Maxwell equations.
- JEPA-specific representation learning improves response-only inverse design.
- Interpolated targets are physically realizable.

## Reproducibility artifacts

- Evaluator: `scripts/evaluate_phase12_response_generalization.py`
- Machine-readable metrics: `outputs/phase12/metrics.json`
- Candidate, retrieval, diversity, complexity, JEPA/AE, and plot artifacts:
  `outputs/phase12/`

No RCWA, FEM, FDTD, CST/HFSS, PINN, or solver development was performed in
this phase.
