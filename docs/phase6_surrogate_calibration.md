# Phase 6 - Forward-Surrogate Calibration and Physics-Objective Validation

> Historical note: this document records the surrogate-only audit that ran
> before the independent Python RCWA work. Its statements that no solver was
> available describe that earlier audit. The current RCWA evidence is in
> [phase6_rcwa_validation.md](phase6_rcwa_validation.md), and remains
> exploratory/compute-limited with a conservative C classification.

## Decision

**Classification: C - not reliable for optimization yet because independent real-solver calibration is unavailable.**

Phase 6 does not modify Phase 5A or Phase 5B. It audits whether the frozen Phase 2.5 `ForwardSurrogateCNN` can be trusted beyond stored held-out data. The repository contains no compatible FEM, FDTD, Maxwell, COMSOL, HFSS, CST, openEMS, MEEP, or other solver interface. Therefore no real EM responses were created, no solver results were invented, and the critical generated-geometry calibration question remains unanswered.

The frozen model retains useful stored-reference diagnostics on held-out dataset geometries, but these cannot establish that low surrogate target error on a generated geometry means low real EM error. Do not use Phase 5B's surrogate term as the sole objective for unconstrained inverse design.

## Objective and protocol

The required future problem remains distinct from the completed phases:

```text
completion:                  (G_partial, M) -> G
physics-conditioned:         (G_partial, M, S_target) -> G
physics-consistent:          (G_partial, M, S_target) -> G -> F_surrogate(G)
future inverse design:       S_target -> G_new
```

Phase 6 is validation only. It adds no diffusion, VAE/GAN, PINN, differentiable solver, arbitrary-target optimization, or generator redesign.

The intended calibration comparison is:

```text
G -> frozen Phase 2.5 surrogate -> S_surrogate
G -> independent real EM solver  -> S_real
```

The second branch is unavailable. Stored test responses are used only as paired held-out reference curves for *dataset geometries*. They are never represented as an independent new solver result for a Phase 5A/5B-generated geometry.

All response fields preserve `[Re(T_y), Im(T_y), Re(R_x), Im(R_x)]` over 2.00--12.00 GHz at 1001 points. Normalized errors use the Phase 2 train statistics; magnitude metrics use denormalized complex components.

## Real solver availability

| Item | Result |
| --- | --- |
| Compatible solver in repository | **No** |
| Solver configuration / executable | None found |
| Fresh solver spectra | None generated |
| Solver-vs-surrogate metrics, ranking, exploitation detection | Not available |

`outputs/phase6/real_solver_predictions/UNAVAILABLE.txt` records this explicitly. A solver must be supplied or integrated before an A or B trust classification is possible.

## Validation geometry manifest

The deterministic seed-42 manifest contains 12 held-out test structures per fixed Phase 4.2 complexity group: 36 dataset geometries. The same 36 source-ID-disjoint test partials were completed under all four Phase 5A/5B mask conditions.

| Category | Records |
| --- | ---: |
| Dataset geometries | 36 |
| Phase 5A generated geometries | 144 |
| Phase 5B generated geometries (small + medium) | 288 |
| Total manifest records | 468 |

The manifest is [validation_geometry_manifest.csv](../outputs/phase6/validation_geometry_manifest.csv). It records source ID, condition, and complexity provenance. There was no separate substantially off-distribution collection in the repository; generated completions are the available generated-geometry audit set.

## Held-out stored-reference diagnostic

This table measures `MSE(S_surrogate, S_stored_reference)` only for the 36 original held-out test geometries. It is an internal reference diagnostic, not the requested fresh solver calibration.

| Metric | Value |
| --- | ---: |
| Overall normalized MSE | 0.340066 |
| Re(`T_y`) normalized MSE | 0.059497 |
| Im(`T_y`) normalized MSE | 0.040873 |
| Re(`R_x`) normalized MSE | 0.609494 |
| Im(`R_x`) normalized MSE | 0.650400 |
| `|T_y|` MAE | 0.061682 |
| `|R_x|` MAE | 0.043540 |

The error remains concentrated in the two `R_x` components. This matches the established forward-surrogate limitation and makes it particularly unsafe to infer real generated-geometry performance from a global response MSE alone.

Frequency-wise error is saved in `metrics.json` and plotted in [frequency_wise_error.png](../outputs/phase6/plots/frequency_wise_error.png). The curve is not flat: the surrogate's error grows substantially in structured spectral regions, so aggregate MSE hides relevant frequency-local failures.

## Resonance-sensitive diagnostic

For the held-out dataset-reference comparison only, the existing peak/dip procedure (`prominence=0.03`, minimum spacing `0.10 GHz`, matching extrema by type) produced:

| Metric | Value |
| --- | ---: |
| Matched / reference spectral features | 88 / 95 (92.63%) |
| Resonance-frequency error, mean / median / p90 | 0.360 / 0.125 / 1.061 GHz |
| Resonance-region magnitude MAE, mean | 0.247681 |

These figures describe agreement with stored held-out references, not fresh solver behavior. Bandwidth and phase-feature measurements are not reported: there is no independent solver curve, and automatic feature correspondence would not be reliable enough to support those claims.

## Dataset versus generated diagnostics

Generated rows measure only `MSE(F_surrogate(G_generated), S_target)`. They do not measure `MSE(F_surrogate(G_generated), S_real(G_generated))` or `MSE(S_real(G_generated), S_target)`.

| Geometry source | Samples | Mean surrogate-target MSE | Mean masked IoU |
| --- | ---: | ---: | ---: |
| Phase 5A | 144 | 0.387739 | 0.610715 |
| Phase 5B small | 144 | 0.366409 | 0.582484 |
| Phase 5B medium | 144 | 0.359789 | 0.584910 |

The lower Phase 5B surrogate objective is consistent with the Phase 5B training effect. It is **not evidence of generated-geometry calibration**, because `S_real` is unknown for each generated pattern. Representative curves are saved as [dataset](../outputs/phase6/plots/surrogate_vs_stored_reference_dataset.png), [Phase 5A generated](../outputs/phase6/plots/surrogate_vs_target_phase5a_generated.png), and [Phase 5B generated](../outputs/phase6/plots/surrogate_vs_target_phase5b_generated.png); generated plots are explicitly surrogate-versus-target, not surrogate-versus-real.

## Complexity analysis

The held-out dataset-reference MSE rises sharply with target geometry complexity.

| Group | Samples | Mean stored-reference normalized MSE |
| --- | ---: | ---: |
| Simple | 12 | 0.044324 |
| Medium | 12 | 0.268695 |
| Complex | 12 | 0.707179 |

This is a material warning: complex geometries are the exact region in which Phase 5A/5B completion has shown its largest trade-offs, while the frozen surrogate's stored-reference error is also greatest. Without solver runs, it is unknown whether generated structures amplify this failure further.

## Ranking and surrogate exploitation

Ranking quality cannot be calculated because the same candidate geometries have no `e_real = D(S_real, S_target)` values.

| Metric | Result |
| --- | --- |
| Spearman correlation | N/A |
| Pairwise ordering agreement | N/A |
| Top-k overlap | N/A |
| Low-surrogate / high-real exploitation cases | N/A, not testable |

The corresponding plots deliberately state this limitation rather than fabricating a relationship: [ranking](../outputs/phase6/plots/surrogate_ranking_vs_real_ranking.png), [target-error scatter](../outputs/phase6/plots/surrogate_error_vs_real_solver_error.png), and [exploitation examples](../outputs/phase6/plots/surrogate_exploitation_examples.png).

## Phase 5B Pareto analysis

The Pareto plot reuses the complete 500-sample evaluation of every mask condition, with geometry error `1 - masked IoU` and frozen-surrogate target MSE. It is available at [phase5a_5b_pareto.png](../outputs/phase6/plots/phase5a_5b_pareto.png) and in [pareto_summary.csv](../outputs/phase6/pareto_summary.csv).

| Condition | Phase 5A (geometry error / physics MSE) | 5B small | 5B medium |
| --- | --- | --- | --- |
| Central 25% | 0.434590 / 0.358306 | 0.490071 / 0.378729 | 0.492692 / 0.350482 |
| Central 50% | 0.463549 / 0.417442 | 0.495003 / 0.403173 | 0.484884 / 0.391451 |
| Random 25% | 0.347701 / 0.357226 | 0.370011 / 0.329652 | 0.394181 / 0.317499 |
| Random 50% | 0.417913 / 0.356095 | 0.429111 / 0.334554 | 0.406147 / 0.335062 |

There is no universal scalar weight: the medium term gives the lowest surrogate score for the first three conditions, while random 50% medium improves both criteria over Phase 5A. This is an internal learned-objective trade-off, not a physical Pareto frontier until solver calibration exists.

## Reproducibility

Artifacts are under `outputs/phase6/`:

```text
config.json
validation_geometry_manifest.csv
surrogate_predictions/
real_solver_predictions/UNAVAILABLE.txt
metrics.json
per_sample_metrics.csv
pareto_summary.csv
plots/
```

The run used seed 42, binary completion threshold 0.5, Python 3.14.3, PyTorch `2.10.0+cu126`, CUDA 12.6, and the NVIDIA GeForce RTX 3050 Laptop GPU. Runtime was 10.037 seconds. The Phase 2.5 checkpoint remains unchanged at `outputs/phase2_5/exp_A_5k_mse/best.pt`.

## Recommendation

Stop here. Do not start unconstrained generation, diffusion, VAE/GAN, arbitrary-target optimization, PINN, or differentiable FEM/FDTD work.

The next stage is to make a small, solver-only calibration set: run the same approximately 20--50 independent dataset and Phase 5A/5B generated geometries through a compatible EM solver with the exact 1001-point representation. Then calculate absolute/per-channel/magnitude/frequency-wise agreement, ranking fidelity, and low-surrogate/high-real exploitation. If those results remain poor, improve or recalibrate the forward model before treating its loss as a physics objective.
