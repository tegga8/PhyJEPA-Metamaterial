# Phase 3 - Supervised Partial-Structure Completion

## Decision

Phase 3 succeeds. A small supervised CNN learns to reconstruct hidden binary
geometry pixels from the visible geometry plus an explicit mask and
substantially beats zero-fill on every experiment. This is sufficient evidence
to justify beginning Phase 4 JEPA completion. The forward EM surrogate was not
modified or used in training.

The main limitation is topology complexity: completion quality is high for
simple geometries and much lower for high-boundary, multi-component geometries.
Future JEPA work should retain the explicit mask channel and known-pixel
compositing, and should include hard/complex examples in its evaluation.

## Dataset and masking

The existing processed 5k dataset was reused without regenerating geometry
splits: 4,000 train, 500 validation, and 500 test samples. Each target is the
original complete binary `[1, 16, 16]` geometry. No complete test geometry is
used during training.

The model input is `[partial_geometry, mask]` with shape `[2, 16, 16]`:

- channel 0: visible geometry, with hidden pixels set to zero
- channel 1: binary mask, where 1 means hidden
- target: complete binary geometry

Mask details:

| Experiment | Mask | Hidden pixels | Ratio |
| --- | --- | ---: | ---: |
| 3A | centered 8x8 block | 64 | 25% |
| 3B | centered 8x16 block | 128 | 50% |
| 3C | random holes | 64 | 25% |
| 3D | random holes | 128 | 50% |

Central masks are fixed structured masks. Random masks use exact pixel counts
and per-sample deterministic seeds. Train, validation, and test use different
split seed offsets, so random test masks are not repeated training masks. Mask
manifests and their seed rules are saved in each experiment directory.

## Model and training

All four experiments use the same `CompletionCNN` with 332,897 parameters:

1. Conv 2->32, GroupNorm, ReLU
2. Conv 32->64, ReLU, 2x downsample
3. Conv 64->128, ReLU, 2x downsample
4. 128-channel bottleneck convolution
5. two bilinear upsampling stages with 128->64->32 convolutions
6. 32->1 output logits

Training uses BCEWithLogitsLoss, AdamW, learning rate `1e-3`, weight decay
`1e-4`, batch size 64, seed 42, maximum 75 epochs, and early stopping
patience 10. The final completed geometry is always composited as

`(1 - mask) * partial + mask * prediction`

so observed pixels are preserved exactly. Evaluation thresholds probabilities
at 0.5. All runs used CUDA-enabled Python 3.14 (`torch 2.10.0+cu126`) on the
RTX 3050 Laptop GPU.

## Results

All values are means over the 500 held-out test geometries. Masked metrics are
computed only over hidden pixels.

| Metric | 3A central 25% | 3B central 50% | 3C random 25% | 3D random 50% |
| --- | ---: | ---: | ---: | ---: |
| full BCE | 0.255741 | 0.319461 | 0.275243 | 0.335477 |
| masked BCE | 0.519493 | 0.482765 | 0.422845 | 0.419564 |
| full accuracy | 0.927344 | 0.862133 | 0.944258 | 0.883508 |
| masked accuracy | 0.709375 | 0.724266 | 0.777031 | 0.767016 |
| full IoU | 0.840861 | 0.717681 | 0.879947 | 0.771762 |
| masked IoU | 0.560517 | 0.540106 | 0.621486 | 0.610476 |
| full Dice | 0.909463 | 0.821620 | 0.933353 | 0.861278 |
| masked Dice | 0.661990 | 0.638761 | 0.728588 | 0.722626 |
| known-region error | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| training seconds | 134.303 | 77.394 | 84.261 | 50.702 |
| best epoch | 63 | 37 | 37 | 22 |

The random-mask model generalizes to independently seeded test masks and
performs slightly better than the fixed central-mask model at the same nominal
missing ratios. Increasing central missingness from 25% to 50% decreases masked
IoU from 0.561 to 0.540. Random masking decreases from 0.621 to 0.610.

## Negative controls

The zero-fill baseline predicts zero for every hidden pixel. The occupancy-prior
baseline uses the training occupancy prior (`0.472839`), which is below the 0.5
threshold and therefore produces the same binary prediction as zero-fill in
these experiments.

| Experiment | CNN masked IoU | Zero-fill masked IoU | CNN masked Dice | Zero-fill masked Dice |
| --- | ---: | ---: | ---: | ---: |
| 3A | 0.560517 | 0.000000 | 0.661990 | 0.000000 |
| 3B | 0.540106 | 0.000000 | 0.638761 | 0.000000 |
| 3C | 0.621486 | 0.000000 | 0.728588 | 0.000000 |
| 3D | 0.610476 | 0.000000 | 0.722626 | 0.000000 |

The CNN therefore learns meaningful topology rather than receiving credit only
for copying known pixels.

## Geometry-complexity analysis

The Phase 2.5 descriptors were reused: connected components plus boundary
transitions divided by 32. Test samples were split into simple, medium, and
complex tertiles using thresholds `3.0625` and `13.2917`.

Mean masked IoU by group:

| Group | 3A | 3B | 3C | 3D |
| --- | ---: | ---: | ---: | ---: |
| simple | 0.878777 | 0.858392 | 0.899746 | 0.879549 |
| medium | 0.468131 | 0.415812 | 0.612026 | 0.593457 |
| complex | 0.307040 | 0.316218 | 0.333679 | 0.339476 |

Complex examples are the dominant failure mode. Their hidden topology is less
recoverable from local context, especially under central masks where the
missing region can remove an entire connection or component.

## Visual analysis

Each experiment saves deterministic random prediction grids and the ten worst
test examples ranked by masked IoU:

- `outputs/phase3_completion/exp_3A/plots/`
- `outputs/phase3_completion/exp_3B/plots/`
- `outputs/phase3_completion/exp_3C/plots/`
- `outputs/phase3_completion/exp_3D/plots/`

The visual pattern agrees with the metrics: the network fills broad structures
well, but smooths or omits fine disconnected components and boundary details.

## Reproducibility and tests

Every experiment contains `best.pt`, `config.json`, `training_history.csv`,
`metrics.json`, per-sample metrics, mask manifests, and plots. The implementation
is separated into:

- `src/completion_dataset.py`
- `src/completion_model.py`
- `src/completion_losses.py`
- `scripts/create_completion_data.py`
- `scripts/train_completion.py`
- `scripts/evaluate_completion.py`
- `scripts/compare_completion.py`

Phase 3-specific tests cover mask ratios, binary masks, partial/target
consistency, known-pixel compositing, model shape, probability range, and mask
channel usage. The complete test suite passes.

## Conclusion

Partial-structure completion is sufficiently learnable to justify Phase 4 JEPA
completion. Begin Phase 4 with the supervised CNN as the baseline to beat, and
retain the following safeguards:

- explicit mask channel
- independent random test masks
- exact known-pixel compositing
- masked-region IoU/Dice as primary metrics
- complexity-stratified evaluation

Do not add physics conditioning, the forward surrogate, PINNs, EM losses, or
inverse-design optimization to Phase 4's initial baseline. Those are separate
experiments for later phases.
