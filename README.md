# PhyJEPA Metamaterial — Phase 1

This repository prepares the supplied SUTD PRT metasurface data for later
masked-structure work.  The raw release remains in its supplied directories
(`PLGDATASET`, `PLRDATASET`, `PTNDATASET`, `RDNDATASET`) and is never modified.

## Verified raw schema

- 261,000 geometry/response pairs across PLG, PLR, PTN, and RDN families.
- Geometry: binary, flattened `256` values representing a `16×16` pattern.
- Raw response: complex `complex128 [2, 1001]`, with y-polarized (`T`) then
  x-polarized (`R`) reflection coefficients.
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
stores responses as `float32 [4, 1001]` in this order: y-reflection real,
y-reflection imaginary, x-reflection real, x-reflection imaginary.
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
