# Phase 8 — Complete-Geometry Geometry Latent Autoencoder

## Hypothesis

A clean complete-geometry autoencoder can reconstruct the binary 16×16
patterns with a reliable spatial latent. If this gate fails, a spectrum-to-
geometry model should not be built on top of the latent.

## Method

The model is:

```text
G [B,1,16,16]
  -> SpatialGeometryEncoder
  -> z_G [B,64,8,8]
  -> SpatialGeometryDecoder
  -> geometry logits [B,1,16,16]
```

This is a new standalone `GeometryAutoencoder`. It uses only complete
geometry and full-image BCE. It does not use masks, EM responses, JEPA loss,
the completion predictor, or any physics/surrogate loss.

The fixed 5k subset and seed-42 split were used: 4,000 train, 500 validation,
and 500 test. Training used AdamW, learning rate 1e-3, weight decay 1e-4,
batch size 64, a 75-epoch ceiling, patience 10, and a 0.5 binary evaluation
threshold. The runtime environment was CPU-only PyTorch 2.12.0; the model and
split were unchanged from the requested protocol.

## Results

| Split | BCE | IoU | Dice | Pixel accuracy | absolute occupancy difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 0.0358 | 0.9728 | 0.9858 | 0.9865 | 0.0034 |
| validation | 0.0413 | 0.9640 | 0.9809 | 0.9828 | 0.0043 |
| test | **0.040649** | **0.965106** | **0.981470** | **0.982570** | **0.004086** |

The exact machine-readable values are in
`outputs/phase8_geometry_autoencoder/metrics.json`.

## Representation and finite-output checks

- Input shape: `[B,1,16,16]`
- Latent shape: `[B,64,8,8]`
- Decoder output shape: `[B,1,16,16]`
- Test latent finite fraction: 1.0
- Test logits finite fraction: 1.0
- Test probabilities finite fraction: 1.0
- Train/validation/test latent caches are saved in deterministic manifest order.
- Each cache has a matching source-ID text manifest to prevent geometry/latent
  pairing mistakes in Stage D.

## Interpretation

The autoencoder reconstructs the held-out geometry distribution well enough to
establish a usable geometry latent baseline. The small occupancy bias and the
gap between train and test IoU should still be retained as diagnostics; exact
geometry reconstruction is not the final inverse-design objective.

## Scientific decision

**Supported. Proceed to Stage D: deterministic spectrum → geometry latent.**

The next model must receive only normalized target EM response `[B,4,1001]`
and, for the deterministic baseline, no noise. It must not receive partial
geometry, source ID, original geometry, mask, complexity, or target geometry
latent. The frozen geometry autoencoder decoder and the cached target latents
will be used only to construct the supervised latent/regression target and
geometry reconstruction objective.

## Reproducibility artifacts

- Model: [geometry_autoencoder.py](../src/geometry_autoencoder.py)
- Training/evaluation: [train_geometry_autoencoder.py](../scripts/train_geometry_autoencoder.py)
- Tests: [test_geometry_autoencoder.py](../tests/test_geometry_autoencoder.py)
- Checkpoint: `../outputs/phase8_geometry_autoencoder/best.pt`
- Metrics: `../outputs/phase8_geometry_autoencoder/metrics.json`
- Latent caches and manifests: `../outputs/phase8_geometry_autoencoder/latents/`

