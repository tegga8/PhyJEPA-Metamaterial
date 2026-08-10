# PhyJEPA Metamaterial — Phase 1

This repository prepares the supplied SUTD PRT metasurface data for later
masked-structure work.  The raw release remains in its supplied directories
(`PLGDATASET`, `PLRDATASET`, `PTNDATASET`, `RDNDATASET`) and is never modified.

## Verified raw schema

- 261,000 geometry/response pairs across PLG, PLR, PTN, and RDN families.
- Geometry: binary, flattened `256` values representing a `16×16` pattern.
- Raw response: complex `complex128 [2, 1001]`, with y-cross-polarized (`T`)
  then x-co-polarized (`R`) reflection coefficients under x-polarized normal
  incidence. `T` is an upstream coefficient label, not transmission.
- Frequency: 2–12 GHz, sampled at 1,001 points (0.01 GHz spacing), as defined
  in the accompanying SUTD-PRCM paper.

## Commands

Run from the repository root:

```powershell
python scripts/inspect_dataset.py --raw-root . --check-duplicates
python scripts/build_subset.py --raw-root . --output-root data/processed/sutd_prcm_5k
python scripts/visualize_samples.py --subset-root data/processed/sutd_prcm_5k
python scripts/verify_phase1.py --subset-root data/processed/sutd_prcm_5k
python -m pytest -q
```

The subset is fixed by seed `42`, balanced across the four design families,
and split into 4,000 train, 500 validation, and 500 test structures.  It
stores responses as `float32 [4, 1001]` in this order: y-cross reflection
real/imaginary, then x-co reflection real/imaginary.
Normalization statistics are calculated from training responses only.

## Phase 2: forward EM surrogate

Train the compact CNN baseline after Phase 1 verification:

```powershell
python scripts/overfit_forward.py --subset-root data/processed/sutd_prcm_5k
python scripts/train_forward.py --subset-root data/processed/sutd_prcm_5k
python scripts/verify_phase2.py --subset-root data/processed/sutd_prcm_5k --run-dir outputs/phase2_forward_75ep
```

The model predicts the normalized `[4, 1001]` response. Training saves its
best checkpoint, learning curve, held-out prediction plots, and physical-unit
magnitude metrics under `outputs/phase2_forward/`.

The completed 5k baseline is stored in `outputs/phase2_forward_75ep/`. It
achieves a held-out normalized MSE of `0.3233`, with magnitude MAE of `0.0640`
for y-polarized reflection and `0.0440` for x-polarized reflection. It is a
working baseline that learns broad spectral behavior; narrow resonances remain
underfit and should motivate the planned 30k data-scale experiment.

## Phase 2.5: surrogate validation

The validated channel convention and reproducible diagnostics are documented in
[`docs/dataset_conventions.md`](docs/dataset_conventions.md). Run the 5k
evaluation without altering the baseline checkpoint:

```powershell
python scripts/evaluate_forward.py --subset-root data/processed/sutd_prcm_5k --checkpoint outputs/phase2_forward_75ep/best.pt
python scripts/build_subset.py --raw-root . --output-root data/processed/sutd_prcm_30k --size 30000 --seed 42 --base-subset-root data/processed/sutd_prcm_5k
python scripts/train_forward.py --subset-root data/processed/sutd_prcm_30k --output-dir outputs/phase2_forward_30k --epochs 75 --patience 15
python scripts/evaluate_forward.py --subset-root data/processed/sutd_prcm_30k --checkpoint outputs/phase2_forward_30k/best.pt --output-dir outputs/phase2_forward_30k/evaluation
python scripts/compare_forward_scales.py
```

The response-aware follow-up keeps the nested 30k split and optimizer settings
but targets resonance localization with a smaller residual CNN and a
resonance-weighted complex loss. Run it with the CUDA-enabled interpreter when
available:

```powershell
& C:\Users\tejas\AppData\Local\Programs\Python\Python314\python.exe scripts/train_forward.py --subset-root data/processed/sutd_prcm_30k --output-dir outputs/phase2_forward_30k_response_aware_gpu --model ResponseAwareSurrogateCNN --loss resonance_weighted_complex --epochs 75 --patience 15 --device cuda
& C:\Users\tejas\AppData\Local\Programs\Python\Python314\python.exe scripts/evaluate_forward.py --subset-root data/processed/sutd_prcm_30k --checkpoint outputs/phase2_forward_30k_response_aware_gpu/best.pt --output-dir outputs/phase2_forward_30k_response_aware_gpu/evaluation --device cuda
```

The final controlled validation uses the unchanged baseline CNN for all three
experiments and writes the required artifacts under `outputs/phase2_5/`:

```powershell
& C:\Users\tejas\AppData\Local\Programs\Python\Python314\python.exe scripts/train_forward.py --subset-root data/processed/sutd_prcm_5k --output-dir outputs/phase2_5/exp_A_5k_mse --model ForwardSurrogateCNN --loss normalized_mse --epochs 75 --patience 15 --device cuda
& C:\Users\tejas\AppData\Local\Programs\Python\Python314\python.exe scripts/train_forward.py --subset-root data/processed/sutd_prcm_5k --output-dir outputs/phase2_5/exp_B_5k_resonance --model ForwardSurrogateCNN --loss resonance_weighted_complex --resonance-weight 4 --magnitude-weight 0.15 --epochs 75 --patience 15 --device cuda
& C:\Users\tejas\AppData\Local\Programs\Python\Python314\python.exe scripts/train_forward.py --subset-root data/processed/sutd_prcm_30k --output-dir outputs/phase2_5/exp_C_30k_resonance --model ForwardSurrogateCNN --loss resonance_weighted_complex --resonance-weight 4 --magnitude-weight 0.15 --epochs 75 --patience 15 --device cuda
& C:\Users\tejas\AppData\Local\Programs\Python\Python314\python.exe scripts/compare_forward_experiments.py
```

## Phase 3: supervised partial-structure completion

Phase 3 synthesizes partial examples from the complete 5k geometries. The
completion model receives `[partial_geometry, mask]`, predicts missing pixels,
and composites the result with known pixels preserved exactly. The completed
experiments and report are in [`docs/phase3_report.md`](docs/phase3_report.md).

```powershell
& C:\Users\tejas\AppData\Local\Programs\Python\Python314\python.exe scripts/train_completion.py --subset-root data/processed/sutd_prcm_5k --output-dir outputs/phase3_completion/exp_3A --mask-type central_block --missing-ratio 0.25 --device cuda
& C:\Users\tejas\AppData\Local\Programs\Python\Python314\python.exe scripts/evaluate_completion.py --checkpoint outputs/phase3_completion/exp_3A/best.pt --output-dir outputs/phase3_completion/exp_3A --device cuda
& C:\Users\tejas\AppData\Local\Programs\Python\Python314\python.exe scripts/compare_completion.py
```

The four completed runs are stored under `outputs/phase3_completion/exp_3A`
through `exp_3D`; use their saved `config.json` files for the exact mask and
training settings.

## Phase 4: JEPA partial-structure completion

Phase 4 evaluates a compact JEPA context/target encoder against the unchanged
Phase 3 supervised CNN. It keeps the explicit mask channel and exact
known-pixel compositing. The completed runs and scientific conclusion are in
[`docs/phase4_report.md`](docs/phase4_report.md).

Run the smoke test first:

```powershell
& C:\Users\tejas\AppData\Local\Programs\Python\Python314\python.exe scripts/smoke_test_jepa.py --subset-root data/processed/sutd_prcm_5k --output-dir outputs/phase4_jepa/smoke_test --device cuda --epochs 2
```

The full benchmark uses the same 5k subset, seed, split, and masks as Phase 3:

```powershell
& C:\Users\tejas\AppData\Local\Programs\Python\Python314\python.exe scripts/train_jepa_completion.py --subset-root data/processed/sutd_prcm_5k --output-dir outputs/phase4_jepa/exp_4A --mask-type central_block --missing-ratio 0.25 --variant jepa_reconstruction --lambda-recon 0.1 --device cuda
& C:\Users\tejas\AppData\Local\Programs\Python\Python314\python.exe scripts/evaluate_jepa_completion.py --checkpoint outputs/phase4_jepa/exp_4A/best.pt --output-dir outputs/phase4_jepa/exp_4A --device cuda
& C:\Users\tejas\AppData\Local\Programs\Python\Python314\python.exe scripts/compare_jepa_completion.py
```

The completed experiment directories are `outputs/phase4_jepa/exp_4A` through
`exp_4D`; their `config.json` files contain the exact settings. The pure-JEPA
variant is available through `--variant pure_jepa` and was recorded as a 4A
ablation. Phase 4 concludes that the supervised CNN wins the primary masked
IoU comparison, so physics-conditioned completion is intentionally deferred.

## Phase 4.1: spatial / masked JEPA completion

Phase 4.1 replaces the global JEPA vector with a coordinate-preserving
`[B,64,8,8]` latent map while keeping the same task, masks, split, EMA target
encoder, and evaluation. The completed three-way comparison is documented in
[`docs/phase4_1_report.md`](docs/phase4_1_report.md).

Run the required smoke test:

```powershell
& C:\Users\tejas\AppData\Local\Programs\Python\Python314\python.exe scripts/smoke_test_spatial_jepa.py --subset-root data/processed/sutd_prcm_5k --output-dir outputs/phase4_1/spatial_smoke --device cuda --epochs 2
```

The completed benchmark runs are in `outputs/phase4_1/exp_4_1A` through
`exp_4_1D`. They compare CNN, global JEPA, and spatial JEPA without modifying
the earlier outputs. Spatial JEPA beats global JEPA in all four conditions and
beats the CNN on random-25%, but physics conditioning remains deferred.

## Phase 4.2: mask-aware spatial JEPA completion

Phase 4.2 keeps the Phase 4.1 architecture fixed and changes only the JEPA
objective weighting. The 16x16 hidden mask is average-pooled to 8x8 and used as
`W = 0.10 + 0.90*M8`. The completed four-way comparison and boundary analysis
are documented in [`docs/phase4_2_report.md`](docs/phase4_2_report.md).

Run the CUDA smoke test:

```powershell
& C:\Users\tejas\AppData\Local\Programs\Python\Python314\python.exe scripts/smoke_test_mask_aware_spatial_jepa.py --subset-root data/processed/sutd_prcm_5k --output-dir outputs/phase4_2/mask_aware_smoke --device cuda --epochs 2 --alpha 0.1
```

The completed alpha=0.10 runs are under `outputs/phase4_2/exp_4_2A` through
`exp_4_2D`. They compare CNN, global JEPA, ordinary spatial JEPA, and
mask-aware spatial JEPA. No EM conditioning is introduced.
