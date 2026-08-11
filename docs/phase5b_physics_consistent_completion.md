# Phase 5B - Physics-Consistent Conditioned Completion

## Decision

**Classification: B - the surrogate physics loss works, but it trades against geometry completion.**

Phase 5B successfully adds a differentiable frozen-surrogate physics-consistency term to the Phase 5A conditioned completion model. The loss sends finite gradients through the continuous completed geometry into the decoder, predictor, FiLM path, and EM encoder while keeping the Phase 2.5 forward surrogate frozen.

Held-out results show the intended response effect: at least one Phase 5B weight improves frozen-surrogate response MSE in all four mask conditions. The geometry result is not yet robust. Masked IoU drops in six of eight nonzero-weight comparisons and improves only for random 50% with the medium weight. This supports physics-consistent regularization against the learned surrogate, not arbitrary inverse design or solver-validated physical correctness.

No FEM solver, Maxwell residual, PINN residual, or unconstrained target optimization was used.

## Objective

Phase 5B preserves the Phase 5A data contract:

```text
(partial geometry, hidden mask, paired target EM response) -> complete geometry
```

It adds a frozen Phase 2.5 `ForwardSurrogateCNN` loss on the continuous completion:

```text
partial geometry + mask + paired normalized target response
  -> Phase 5A conditioned completion logits
  -> sigmoid probabilities, composited with observed pixels
  -> frozen Phase 2.5 forward surrogate
  -> normalized response MSE to paired target
```

The training loss is:

```text
L = L_mask-aware-JEPA + 0.1 L_masked-BCE + lambda_p L_surrogate-MSE
```

The surrogate input is `input*(1-mask) + sigmoid(logits)*mask`. No thresholding, rounding, detaching, NumPy conversion, or surrogate-weight update occurs on this path. Binary thresholding at 0.5 is evaluation-only.

## Loss weights and environment

A one-epoch diagnostic on central 25% measured initial `L_JEPA=0.024901`, `L_BCE=0.672724` (weighted contribution `0.067272`), and `L_physics=0.497300`. The tested nonzero weights were:

| Run | lambda_p | Initial physics contribution |
| --- | ---: | ---: |
| Phase 5A reference | 0.00 | 0 |
| Phase 5B small | 0.05 | about 0.025 |
| Phase 5B medium | 0.15 | about 0.075 |

Completed CUDA runs used Python 3.14.3, PyTorch `2.10.0+cu126`, CUDA 12.6, and the NVIDIA GeForce RTX 3050 Laptop GPU.

| Run | Mask | lambda_p | Epochs / best | Seconds | Peak GPU memory |
| --- | --- | ---: | --- | ---: | ---: |
| 5bA small | central 25% | 0.05 | 30 / 20 | 96.811 | 276,851,200 B |
| 5bA medium | central 25% | 0.15 | 37 / 27 | 122.391 | 276,851,200 B |
| 5bB small | central 50% | 0.05 | 34 / 24 | 96.721 | 276,851,200 B |
| 5bB medium | central 50% | 0.15 | 37 / 27 | 343.156 | 276,851,200 B |
| 5bC small | random 25% | 0.05 | 22 / 12 | 251.872 | 276,851,200 B |
| 5bC medium | random 25% | 0.15 | 22 / 12 | 245.635 | 276,851,200 B |
| 5bD small | random 50% | 0.05 | 22 / 12 | 49.639 | 276,851,200 B |
| 5bD medium | random 50% | 0.15 | 37 / 27 | 142.119 | 276,851,200 B |

## Held-out results

All results use 500 held-out source-ID-disjoint test structures. The binary composited completion is passed to the frozen Phase 2.5 5k MSE surrogate for response evaluation. Lower response MSE is better.

| Run | Model | Masked IoU | IoU delta vs 5A | Response MSE | MSE delta vs 5A |
| --- | --- | ---: | ---: | ---: | ---: |
| central 25% | Phase 5A | 0.565410 | - | 0.358306 | - |
| central 25% | 5B small | 0.509929 | -0.055481 | 0.378729 | +0.020423 |
| central 25% | 5B medium | 0.507308 | -0.058101 | 0.350482 | -0.007824 |
| central 50% | Phase 5A | 0.536451 | - | 0.417442 | - |
| central 50% | 5B small | 0.504997 | -0.031454 | 0.403173 | -0.014269 |
| central 50% | 5B medium | 0.515116 | -0.021335 | 0.391451 | -0.025991 |
| random 25% | Phase 5A | 0.652299 | - | 0.357226 | - |
| random 25% | 5B small | 0.629989 | -0.022309 | 0.329652 | -0.027574 |
| random 25% | 5B medium | 0.605819 | -0.046480 | 0.317499 | -0.039727 |
| random 50% | Phase 5A | 0.582087 | - | 0.356095 | - |
| random 50% | 5B small | 0.570889 | -0.011198 | 0.334554 | -0.021541 |
| random 50% | 5B medium | 0.593853 | +0.011766 | 0.335062 | -0.021033 |

The strongest response gain is random 25% medium (`-0.039727` normalized MSE). The only geometry IoU gain is random 50% medium (`+0.011766`). Central masks gain response consistency but lose geometry IoU, which means the surrogate loss is steering completions in a physically measurable direction while also changing the geometry prior.

## Diagnostics

The medium checkpoints pass the continuous-gradient check in every condition: finite physics gradients reach the decoder and predictor, FiLM and EM encoder gradients are nonzero, all surrogate parameters have `requires_grad=False`, and no surrogate parameter gradients are produced.

| Run | Gradient path pass | Finite-difference sign agreement | Different-target hidden-pixel diff, small / medium | Permuted minus correct response MSE, small / medium |
| --- | --- | ---: | ---: | ---: |
| central 25% | yes | 0.875 | 0.281250 / 0.252604 | 0.130538 / 0.154173 |
| central 50% | yes | 0.625 | 0.223958 / 0.307292 | 0.268404 / 0.338453 |
| random 25% | yes | 0.875 | 0.059896 / 0.067708 | 0.027880 / 0.048995 |
| random 50% | yes | 0.500 | 0.092448 / 0.135417 | 0.089684 / 0.172837 |

Conditioning remains target-sensitive after adding the physics loss: changing the paired response while holding the partial geometry and mask fixed changes hidden pixels in all four conditions. Test-only response permutation also increases frozen-surrogate response error in all four conditions.

## Artifact index

```text
outputs/phase5b/diagnostic_5bA/
outputs/phase5b/physics_5bA_small ... physics_5bD_medium/
outputs/phase5b/evaluation_5bA ... evaluation_5bD/
outputs/phase5b/comparison.csv
outputs/phase5b/comparison.md
outputs/phase5b/logs/
```

Every training directory contains `config.json`, `best.pt`, `training_history.csv`, and mask manifests. Evaluation directories contain `metrics.json`, `per_sample_metrics.csv`, `conditioning_sensitivity.csv`, and plots.

## Recommendation

Do not call Phase 5B a solver-validated inverse-design method. The correct conclusion is narrower and useful: a frozen learned physics surrogate can regularize Phase 5A toward lower predicted response error, but the current scalar loss weight often sacrifices masked geometry IoU. The next step should be calibration against a small real-solver subset and then either adaptive loss weighting or a Pareto-style selection criterion, not a larger architecture jump.
