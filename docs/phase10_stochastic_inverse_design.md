# Phase 10 - Stochastic Spectrum to Geometry Inverse Design

## Hypothesis and contract

Phase 9 showed that a single deterministic output is not enough for inverse
design: the learned models produced plausible-looking geometries, but their
screened response error lagged behind the train-only nearest-neighbor baseline.
The next controlled test is therefore a one-to-many generator:

```text
S_target [B,4,1001] + noise -> stochastic inverse generator -> candidate geometries
```

At inference time the generator receives only the normalized target response
and optional noise. It does not receive partial geometry, masks, source IDs,
paired geometry, target geometry latent labels, or complexity metadata.

## Method

The model is a conditional latent VAE over the frozen Phase 8 geometry latent
space. It predicts a response-conditioned latent prior and uses a posterior
only during training:

```text
normalized S_target -> prior(z_G | S_target)
normalized S_target + z_G label -> posterior(z_G | S_target, z_G)
sampled z_G -> frozen Phase 8 decoder -> geometry logits
```

Training used the fixed 5k subset and seed-42 split: 4,000 train, 500
validation, and 500 test samples. Geometry latents came from the validated
Phase 8 cache. The frozen Phase 8 decoder converted sampled latents into
binary 16x16 geometries. The selected Phase 7B 30k resonance-aware CNN was
used only after generation to screen candidate responses; it was not used as a
training loss and is not treated as Maxwell ground truth.

The generator was trained with AdamW, hidden dimension 256, latent shape
`[64,8,8]`, KL weight `1e-4`, prior weight `0.5`, and early stopping. The best
checkpoint was epoch 12.

## Candidate evaluation

For each of the 500 test spectra, the generator produced 8 candidates, for
4,000 total screened geometries. A candidate was considered valid if it was
finite, non-empty, non-full, and stayed within the training-set occupancy and
simple topology limits.

| Metric | Value |
| --- | ---: |
| validity rate | 0.981250 |
| valid candidates per target | 7.850000 |
| best screened response MSE | 0.338217 |
| best valid screened response MSE | 0.338230 |
| multi-solution success fraction | 0.594000 |
| pairwise candidate Hamming diversity | 0.201257 |
| nearest train pixel Hamming novelty | 0.158440 |
| nearest train latent MSE novelty | 0.710898 |
| duplicate candidates per target | 0.008000 |

The multi-solution criterion was deliberately fixed before interpretation:
at least two valid candidates for the same target with screened response MSE
`<= 0.30`. This threshold is close to the Phase 9 train-only nearest-neighbor
screened response MSE of 0.259249, so it is a useful but not physical gate.

## Comparison to deterministic baselines

| Method | screened response MSE | notes |
| --- | ---: | --- |
| EM nearest neighbor (train only) | **0.259249** | strongest response match, but retrieval not generation |
| direct MLP | 0.483234 | single geometry output |
| deterministic latent predictor | 0.444132 | best deterministic geometry IoU, weak response match |
| stochastic generator, best of 8 | 0.338217 | generated candidates, high validity, diverse samples |

The stochastic generator improves substantially over the learned deterministic
baselines in screened response space, but it does not beat the train-only
nearest-neighbor response baseline. This is still a useful result: the model is
now producing multiple mostly valid generated candidates instead of collapsing
to one averaged geometry.

## Complexity-stratified results

| Target group | targets | validity | best screened response MSE | multi-solution success | nearest train pixel Hamming | nearest train latent MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| simple | 178 | 1.000000 | 0.124743 | 0.859551 | 0.043635 | 0.213514 |
| medium | 155 | 0.949194 | 0.284115 | 0.677419 | 0.143665 | 0.720161 |
| complex | 167 | 0.991018 | 0.615966 | 0.233533 | 0.294521 | 1.232447 |

The complex group remains the bottleneck. Its candidates are novel and mostly
valid, but screened response satisfaction falls sharply. This mirrors the
Phase 9 complexity trend and suggests that the next improvement should target
response satisfaction/ranking, not basic geometry validity.

## Scientific decision

**Supported as a controlled stochastic inverse-design baseline, but not yet a
final inverse-design system.**

The generator passes the basic one-to-many gate: it produces valid, diverse,
non-duplicate candidates conditioned only on target spectra, and the best of 8
improves over deterministic learned baselines in screened response space. The
remaining gap to train-only nearest-neighbor retrieval and the weak complex
case performance mean that Phase 10 should be treated as a candidate generator
for downstream ranking/optimization, not as a physically validated inverse
solver.

Do not claim Maxwell-valid inverse design from these results. The forward CNN
is still a learned screening surrogate. The next controlled stage should rank
or refine generated candidates under the selected surrogate while preserving
the anti-leakage contract, then reserve any physical validation for a separate
RCWA/Maxwell check if the required solver support exists.

## Reproducibility artifacts

- Model: [conditional_latent_vae.py](../src/conditional_latent_vae.py)
- Training/evaluation: [train_phase10_stochastic_inverse_design.py](../scripts/train_phase10_stochastic_inverse_design.py)
- Tests: [test_conditional_latent_vae.py](../tests/test_conditional_latent_vae.py)
- Checkpoint: `../outputs/phase10_stochastic_inverse_design/generator/best.pt`
- Metrics: `../outputs/phase10_stochastic_inverse_design/metrics.json`
- Candidate metrics: `../outputs/phase10_stochastic_inverse_design/candidate_metrics.csv`
- Candidate geometries: `../outputs/phase10_stochastic_inverse_design/test_candidates_binary.npy`
- Screened candidate responses: `../outputs/phase10_stochastic_inverse_design/test_candidate_responses_normalized_30k.npy`
