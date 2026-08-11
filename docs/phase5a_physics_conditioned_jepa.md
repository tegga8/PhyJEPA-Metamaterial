# Phase 5A — Physics-Conditioned Spatial JEPA

## Decision

**Classification: B — conditioning works, but geometry supervision still dominates.**

The model demonstrably uses the paired EM target: changing a response while holding the same partial geometry and mask fixed changes the predicted hidden pixels in all four conditions, and test-only response permutation changes completions. Correct pairing improves the frozen-surrogate response metric in three of four conditions and improves geometry completion for both central masks. It does not improve geometry completion consistently: both random-mask settings lose masked IoU, with the largest loss on complex random geometries. This is evidence for target-sensitive conditioning, not yet evidence for robust physics-consistent generation or arbitrary inverse design.

No physics loss, forward-surrogate loss, PINN term, Maxwell residual, solver, or arbitrary-target training was used.

## Objective and data contract

Phase 5A tests:

```text
(partial geometry, hidden-mask, paired target EM response) -> complete geometry
```

The EM condition is the audited processed response of the same complete geometry:

```text
S_i = [Re(T_y), Im(T_y), Re(R_x), Im(R_x)]  in R^[4,1001]
```

The channels remain complex-component channels; no magnitude-only conversion or phase removal occurs. Frequencies are the existing 2.00–12.00 GHz, 1001-point, 0.01 GHz grid. `PhysicsCompletionDataset` obtains the raw response and geometry from the same `source_dataset.indices[index]` position, then applies only the saved Phase 2 train statistics (`src/physics_conditioned_dataset.py:14-45`). It returns both normalized `response` for the model and `response_raw` for evaluation. Geometry and response are never independently shuffled during training.

This extends the Phase 4.2 `CompletionDataset` mask protocol rather than regenerating splits. Train/validation/test remain 4,000/500/500 source-ID-disjoint structures. The four conditions use the existing deterministic masks and split seed offsets:

| Run | Mask | Hidden pixels | Ratio |
| --- | --- | ---: | ---: |
| 5aA | centered block | 64 | 25% |
| 5aB | centered block | 128 | 50% |
| 5aC | random holes | 64 | 25% |
| 5aD | random holes | 128 | 50% |

Mask manifests for train/validation/test are saved beside every control and Phase 5A checkpoint under `outputs/phase5a/`.

## Architecture

### Preserved Phase 4.2 path

The control is a fresh training run of the exact Phase 4.2 architecture and objective:

```text
[partial_geometry, mask] [B,2,16,16]
    -> SpatialGeometryEncoder(2,64)              z_context [B,64,8,8]
    -> SpatialPredictor(64,128)                  z_pred [B,64,8,8]
    -> SpatialGeometryDecoder(64)                logits [B,1,16,16]

complete geometry [B,1,16,16]
    -> frozen EMA SpatialGeometryEncoder(1,64)   z_target [B,64,8,8]
```

The context encoder, geometry-only target encoder, predictor network, decoder, EMA update, 64-channel 8x8 latent, mask construction, optimizer, learning rate, batch size, seed, threshold, and Phase 4.2 losses are preserved from `src/spatial_jepa_completion_model.py:10-137`, `src/mask_aware_spatial_jepa_losses.py:9-48`, and `scripts/train_mask_aware_spatial_jepa.py:90-180`.

The fresh controls reproduce the **protocol** (not a bitwise re-run of the historical checkpoints): same subset, masks, seed 42, AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 64, maximum 75 epochs, patience 10, EMA 0.996, `alpha=0.10`, `gamma=1.0`, and `lambda_recon=0.1`. Their held-out scores differ modestly from the earlier saved Phase 4.2 checkpoints, so all claims below compare each Phase 5A run with its freshly trained matched control rather than mixing historical and fresh checkpoints.

### EM encoder and FiLM addition

The only primary additions are implemented in `src/physics_conditioned_spatial_jepa.py:11-123`:

```text
normalized S_target [B,4,1001]
    -> Conv1d 4->32, GroupNorm, GELU
    -> Conv1d 32->64, GroupNorm, GELU
    -> Conv1d 64->128, GroupNorm, GELU
    -> adaptive global frequency pooling
    -> Linear 128->128
    = z_phys [B,128]

z_phys -> Linear 128->128 -> GELU -> Linear 128->128
       -> gamma [B,64], beta [B,64]

h_conditioned = gamma * z_context + beta
z_pred = unchanged SpatialPredictor(h_conditioned)
```

The final FiLM linear layer is initialized with zero weights and zero bias. Therefore at initialization `gamma=1` and `beta=0`, making `h_conditioned` exactly equal to the unconditioned Phase 4.2 context map. The target encoder remains geometry-only; EM is not passed to it. The code validates these facts and the target no-gradient condition in `tests/test_physics_conditioned_phase5a.py`.

Parameter counts:

| Model | Total | Trainable | Added EM encoder | Added FiLM |
| --- | ---: | ---: | ---: | ---: |
| Phase 4.2 control | 397,793 | 341,729 | — | — |
| Phase 5A | 520,577 | 464,513 | 89,760 | 33,024 |

The increase is 122,784 parameters; no raw 4,004-value spectrum is concatenated with an image or spatial feature map.

## EM-embedding sanity check

Before integration, `scripts/pretrain_em_encoder.py:44-161` trained the 128-D encoder with a linear `128 -> 4*1001` readout on correctly paired, train-normalized spectra. This is an embedding validation only; the decoder is not used by the completion model.

| Held-out diagnostic | Value |
| --- | ---: |
| normalized response MSE | 0.155751 |
| zero-embedding normalized MSE | 0.985736 |
| best validation MSE | 0.147830 (epoch 48) |
| Re(`T_y`) raw MAE | 0.061615 |
| Im(`T_y`) raw MAE | 0.065975 |
| Re(`R_x`) raw MAE | 0.025333 |
| Im(`R_x`) raw MAE | 0.025571 |

The embedding substantially beats the zero baseline, so the Phase 5A integration proceeded. It used the best encoder checkpoint as initialization and continued to train the encoder jointly with the completion model.

## Training objective and compute

The Phase 5A loss is unchanged from Phase 4.2:

```text
L = mask_aware_spatial_JEPA(z_pred, stop_gradient(z_target), W)
    + 0.1 * masked_BCE(logits, complete_geometry, mask)

W = 0.10 + 0.90 * average_pool_2x2(mask)
```

`scripts/train_physics_conditioned_spatial_jepa.py:52-197` passes the normalized paired response only to the EM encoder/FiLM path and otherwise applies the original loss. The EMA target update remains unchanged. Completed CUDA runs used Python 3.14.3, PyTorch `2.10.0+cu126`, CUDA 12.6, and the RTX 3050 Laptop GPU.

| Run | Control epochs / best | Control sec | Phase 5A epochs / best | Phase 5A sec | Phase 5A peak GPU memory |
| --- | --- | ---: | --- | ---: | ---: |
| 5aA central 25% | 25 / 15 | 58.075 | 24 / 14 | 90.372 | 259,990,016 B |
| 5aB central 50% | 25 / 15 | 58.379 | 25 / 15 | 90.309 | 259,990,016 B |
| 5aC random 25% | 22 / 12 | 60.303 | 20 / 10 | 69.184 | 259,990,016 B |
| 5aD random 50% | 21 / 11 | 56.015 | 21 / 11 | 45.910 | 259,990,016 B |

The EM embedding sanity run took 78.770 seconds and peaked at 239,020,032 B. The conditioned validation FiLM deviation became nonzero in every run, so the FiLM pathway did not remain at its identity initialization.

## Held-out geometry and frozen-surrogate physics results

For every test completion, the binary composited geometry is passed to the frozen Phase 2.5 5k MSE `ForwardSurrogateCNN` checkpoint (`outputs/phase2_5/exp_A_5k_mse/best.pt`). The evaluator computes normalized response MSE over `[4,1001]`, the four separate channel MSEs, and raw `|T_y|`/`|R_x|` magnitude MAE. This is evaluation only; no forward-surrogate gradient or loss reaches Phase 5A training (`scripts/evaluate_phase5a.py:61-285`).

| Run | Control masked IoU | Phase 5A masked IoU | Delta | Control EM MSE | Phase 5A EM MSE | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| central 25% | 0.540550 | 0.565410 | **+0.024860** | 0.373365 | 0.358306 | **-0.015059** |
| central 50% | 0.533219 | 0.536451 | **+0.003232** | 0.468157 | 0.417442 | **-0.050715** |
| random 25% | 0.683701 | 0.652299 | **-0.031402** | 0.354647 | 0.357226 | **+0.002579** |
| random 50% | 0.603049 | 0.582087 | **-0.020962** | 0.370669 | 0.356095 | **-0.014574** |

Negative response-MSE delta is better. The complete comparison is saved in `outputs/phase5a/comparison.csv` and `outputs/phase5a/comparison.md`.

Phase 5A per-channel normalized response MSE:

| Run | Re(`T_y`) | Im(`T_y`) | Re(`R_x`) | Im(`R_x`) |
| --- | ---: | ---: | ---: | ---: |
| central 25% | 0.083658 | 0.056301 | 0.633549 | 0.659716 |
| central 50% | 0.106879 | 0.070814 | 0.732944 | 0.759129 |
| random 25% | 0.075612 | 0.051275 | 0.638823 | 0.663194 |
| random 50% | 0.087150 | 0.058257 | 0.629283 | 0.649689 |

Known-region error is exactly 0.0 for both methods in all four held-out evaluations, confirming compositing preserves observed pixels.

### Interpretation of the two scientific questions

**Q1 — does EM conditioning improve geometry completion?** Yes for central 25% and marginally for central 50%; no for random 25% and random 50%. It is therefore not a universal geometry-completion improvement.

**Q2 — does EM conditioning improve response matching?** Yes for central 25%, central 50%, and random 50%; no for random 25%. These are frozen-surrogate measurements, not solver validation.

The response benefit is most decisive for central 50% (MSE reduction 0.050715), where geometry IoU changes only +0.003232. This is why pixel IoU alone would miss an important effect, but it is also why the result is not sufficient to call the method an inverse-design system.

## Conditioning sensitivity and permutation test

For three deterministic test partials per condition, the evaluator fixes the partial geometry and mask then supplies its own response plus two responses from different complete geometries. It records hidden-pixel difference, prediction-to-prediction masked IoU, and latent MSE; the mandatory visualizations are `plots/conditioning_sensitivity.png` in every `outputs/phase5a/evaluation_5a*/` directory.

| Run | Different-target hidden-pixel difference | Test-only permuted hidden-pixel difference | Correct minus permuted geometry IoU | Permuted minus correct EM MSE |
| --- | ---: | ---: | ---: | ---: |
| central 25% | 0.213542 | 0.226375 | +0.080402 | +0.137925 |
| central 50% | 0.225260 | 0.214547 | +0.058306 | +0.135680 |
| random 25% | 0.039063 | 0.085469 | -0.009586 | +0.002476 |
| random 50% | 0.063802 | 0.099719 | -0.005293 | +0.042098 |

Sensitivity passes all four conditions because changing `S_target` produces measurable hidden-pixel changes (all means exceed 0.01). The permutation diagnostic also passes the predeclared output-change and EM-error criteria in all four conditions: mismatched responses change predictions and increase mean response error relative to the original paired target. The response signal has a much stronger effect for central masks; random 25% has the smallest alternative-target change.

The geometry-IoU part of the permutation result is favorable only for the two central masks. This agrees with the main geometry result: conditioning is useful there, while random masks already expose enough local information that this simple conditioning mechanism can compete with rather than improve the geometry prior.

## Complexity analysis

Complexity tertiles reuse the established score `connected_components_4 + boundary_transitions_4 / 32` with thresholds 3.0625 and 13.2917. Values below are Phase 5A minus matched control; lower response MSE is better.

| Run | Simple IoU delta / EM MSE delta | Medium IoU delta / EM MSE delta | Complex IoU delta / EM MSE delta |
| --- | --- | --- | --- |
| central 25% | +0.00545 / -0.01281 | +0.03037 / -0.04427 | +0.04044 / +0.00966 |
| central 50% | +0.01482 / -0.02554 | +0.03783 / -0.04600 | -0.04124 / -0.08193 |
| random 25% | +0.00216 / +0.00721 | -0.01753 / +0.00845 | -0.08004 / -0.00780 |
| random 50% | +0.00015 / -0.01537 | +0.00333 / -0.01334 | -0.06601 / -0.01487 |

The central 25% model gains geometry IoU at every complexity level, including complex structures, although complex response matching is slightly worse. Central 50% improves response matching in every group but loses complex-geometry IoU. Both random conditions expose the main failure mode: complex geometry completion degrades materially even when surrogate response matching improves.

## Qualitative artifacts and failure cases

Each evaluation directory contains:

- `per_sample_metrics.csv` with geometry, response, permutation, and complexity fields;
- `conditioning_sensitivity.csv` with same-partial/different-response diagnostics;
- `plots/representative_geometry_and_em.png`, showing complete target, partial geometry, control completion, Phase 5A completion, target response, and both frozen-surrogate responses;
- `plots/conditioning_sensitivity.png`, showing at least three same-partial/different-target response examples with resulting geometries and their forward-surrogate spectra;
- `metrics.json` with aggregates, paired deltas, complexity, parameter counts, and pass flags.

The relevant failure cases are not silent EM-path collapse: the embedding and FiLM modulation are active and outputs change. Instead, the current 128-D global spectral embedding can over-steer geometry completion for highly visible random masks, particularly complex examples. The response target is global while the completion error is local, so a small FiLM modulation before the predictor does not yet resolve that trade-off consistently.

## Reproducibility artifact index

```text
outputs/phase5a/em_embedding/
outputs/phase5a/control_4_2A ... control_4_2D/
outputs/phase5a/physics_5aA ... physics_5aD/
outputs/phase5a/evaluation_5aA ... evaluation_5aD/
outputs/phase5a/comparison.csv
outputs/phase5a/comparison.md
```

Every training directory contains `config.json`, `best.pt`, `training_history.csv`, and mask manifests. The Phase 5A configurations additionally record the EM-normalization-statistics path, EM encoder checkpoint/configuration, embedding dimension, FiLM method and identity initialization, baseline reference, seed, parameters, training time, and peak GPU memory. Evaluation artifacts contain the required metrics, per-sample rows, plots, and diagnostic results.

## Recommendation

Do not automatically add a physics loss or proceed to Phase 5B. The first transition—geometry completion to physics-conditioned completion—is established: correct paired EM targets affect the generated hidden geometry and can improve frozen-surrogate response matching. The evidence is not yet stable across mask regimes or complex geometries.

The justified next decision is a small robustness study, not a broader model family: repeat the paired control/conditioning comparison with another seed and assess forward-surrogate calibration against a small real-solver subset before deciding whether a physics-consistency loss is warranted.

