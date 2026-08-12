# Phase 7B — Forward Screening Surrogate Selection

## Objective

Select the best existing learned forward checkpoint for cheap candidate
screening before inverse-design experiments. This is an artifact-only audit:
no checkpoint was retrained and no new predictions were generated.

The comparison uses the shared original 5k test IDs (500 structures) wherever
possible. Each checkpoint is evaluated with the train-only normalization
statistics belonging to its own training subset. The comparison is therefore
controlled for held-out structures while preserving each model's valid input
scaling.

## Candidates

- 5k normalized-MSE baseline: `outputs/phase2_5/exp_A_5k_mse/best.pt`
- 5k resonance-aware CNN: `outputs/phase2_5/exp_B_5k_resonance/best.pt`
- 30k resonance-aware CNN: `outputs/phase2_5/exp_C_30k_resonance/best.pt`
- 30k response-aware CNN: `outputs/phase2_forward_30k_response_aware_gpu/best.pt`

## Selection priorities

The explicit lexicographic priority was:

1. lowest shared normalized response MSE;
2. lowest shared resonance-frequency error;
3. lowest shared resonance-region magnitude MAE.

Complexity-stratified metrics and gradient diagnostics are retained as safety
checks, not hidden optimization terms.

## Shared-holdout results

| Candidate | normalized MSE | resonance frequency error (GHz) | resonance-region MAE | feature match | inference ms/sample | local gradient sign agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5k MSE | 0.331021 | 0.525456 | 0.241849 | 0.964103 | 0.373897 | 0.832 |
| 5k resonance | 0.361289 | 0.103201 | 0.234538 | 1.000000 | 0.388122 | 0.848 |
| 30k resonance | **0.300002** | 0.243154 | **0.226388** | 0.994139 | 0.384390 | **0.944** |
| 30k response-aware | 0.312470 | 0.316475 | 0.227351 | 0.995604 | 0.667629 | not recorded |

The 5k resonance-aware model has the lowest mean resonance-frequency error,
but its broad normalized MSE is materially worse. The 30k response-aware model
is competitive on feature matching but is slower and has slightly worse broad
and resonance-region errors than the 30k resonance-aware model.

## Decision

**Selected screening checkpoint:**
`outputs/phase2_5/exp_C_30k_resonance/best.pt`

It has the best broad response fidelity and the best resonance-region error on
the shared holdout, with finite nonzero gradients and the strongest recorded
local perturbation sign agreement among the complete diagnostics.

This checkpoint is named the **learned forward screening surrogate**. It is
not Maxwell ground truth. Use it for candidate ranking/filtering only; do not
use it as an unconstrained differentiable physics objective or claim that its
gradients are physically calibrated.

## Complexity diagnostics

The selected 30k resonance-aware model's shared-holdout normalized MSE means
were 0.0365 (simple), 0.1800 (medium), and 0.6923 (complex). Its corresponding
resonance-frequency errors were 0.2071, 0.2368, and 0.2409 GHz. The complex
group remains substantially harder, so later candidate screening must report
complexity-stratified results and should not hide this failure mode behind a
single average.

## Reproducibility artifacts

- Selector: [select_forward_screening_surrogate.py](../scripts/select_forward_screening_surrogate.py)
- Machine-readable result: `../outputs/phase2_5/surrogate_selection/metrics.json`
- Comparison table: `../outputs/phase2_5/surrogate_selection/selection_table.csv`
- Report: `../outputs/phase2_5/surrogate_selection/selection_report.md`

