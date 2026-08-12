# Phase 7 — Counterfactual Target Direction

## Hypothesis

Phase 5A established that changing the supplied EM response changes hidden
pixels. This experiment tests the stronger claim that changing the target from
`S_A` to a counterfactual `S_B` moves the completion toward the geometry and
response associated with B.

The response-space measurements below use the frozen Phase 2.5 CNN only as a
learned forward screening surrogate. They are not independent Maxwell or RCWA
validation, especially given the Phase 6.1 convergence result.

## Method and leakage controls

For each held-out test geometry A, a deterministic derangement supplies a
different held-out geometry B:

```text
partial(A), S_A -> correct completion
partial(A), S_B -> counterfactual completion
partial(A), no response -> matched Phase 4.2 control
```

The fixed partial geometry and mask are identical across all three calls. The
counterfactual model receives only the partial geometry, mask, and response;
the complete target geometry is passed to the model API only for evaluation
bookkeeping and is not used by the forward path. Source IDs are retained only
to audit the A/B pairing.

The experiment used all four existing matched Phase 5A conditions, each on the
seed-42 5k test split: 4,000 train / 500 validation / 500 test, 500 A→B pairs
per condition. The binary threshold was 0.5. Runs used the existing checkpoints
and the CPU-only runtime available in this environment; no training or prior
artifact was modified.

## Directional metrics

For response metrics, a positive gain means the counterfactual has lower
surrogate MSE to `S_B` than the comparison. For geometry metrics, a positive
gain means lower full-geometry pixel MSE to `G_B` than the correct-target
completion. The full-geometry comparison is conservative because the known
region remains fixed to A by protocol.

| condition | geometry gain vs correct | response gain to B vs correct | response gain to B vs control | fraction geometry gains >0 | fraction response gains >0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| central 25% (5aA) | +0.006719 | -0.056301 | -0.054806 | 0.508 | 0.398 |
| central 50% (5aB) | +0.016586 | -0.036353 | -0.007904 | 0.538 | 0.516 |
| random 25% (5aC) | +0.000953 | +0.002075 | -0.011595 | 0.382 | 0.442 |
| random 50% (5aD) | +0.005383 | +0.006444 | -0.001476 | 0.546 | 0.528 |

The target-direction response result is not stable across mask conditions:
the two central-block conditions have negative mean gain versus the correct
completion, while the two random-hole conditions are near zero. No condition
shows a positive mean gain versus both the correct completion and the matched
control.

For context, the mean surrogate response MSEs for the correct and
counterfactual outputs were:

| condition | correct → A | counterfactual → A | correct → B | counterfactual → B |
| --- | ---: | ---: | ---: | ---: |
| 5aA | 0.358360 | 0.502548 | 0.840440 | 0.896740 |
| 5aB | 0.417229 | 0.555927 | 0.845420 | 0.881773 |
| 5aC | 0.357265 | 0.359988 | 0.855561 | 0.853485 |
| 5aD | 0.356128 | 0.396968 | 0.827900 | 0.821456 |

The counterfactual does produce a distinct geometry, but it generally moves
away from A more strongly than it moves toward B in screened response space.
This is target sensitivity without reliable target direction.

## Failure cases and limitations

- The learned forward CNN is not an independent physical oracle. The result
  cannot establish counterfactual direction under RCWA.
- A/B pairs are a deterministic index derangement, not nearest or matched
  response pairs. This is intentional for the first broad directional test,
  but response distance between A and B varies.
- Full geometry distance includes the unchanged known region. The comparison
  remains valid as a paired diagnostic, but hidden-region directional analysis
  would be a useful follow-up only if the stage is revisited.
- The counterfactual completion has substantially worse surrogate error to A
  in all four conditions, confirming that the response input affects the
  output; that alone is not evidence of useful inverse direction.

## Scientific decision

**Decision: B — target-sensitive but weakly directed.**

The Phase 5A model should not be treated as evidence that a requested EM
response directs completion toward a corresponding new geometry. The Stage A
stop condition is therefore triggered. In accordance with the continuation
plan, do not proceed directly to stochastic generation, latent inverse design,
or additional physics-loss training on the basis of this model.

## Reproducibility artifacts

- Evaluator: [evaluate_phase7_counterfactual.py](../scripts/evaluate_phase7_counterfactual.py)
- Unit tests: [test_phase7_counterfactual.py](../tests/test_phase7_counterfactual.py)
- Per-condition metrics and pair manifests: `../outputs/phase7_counterfactual/5aA/` through `5aD/`
- Frozen screening checkpoint: `../outputs/phase2_5/exp_A_5k_mse/best.pt`

