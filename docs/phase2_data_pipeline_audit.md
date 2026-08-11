# Phase 2 Data-Pipeline Audit: SUTD-PRCM Geometry to EM Response

**Audit date:** 2026-08-10  
**Scope:** the raw SUTD-PRCM files, the repository preprocessing code, the processed 5k/30k subsets, the Phase 2 forward surrogate, and EM usage in Phases 3, 4, 4.1, and 4.2.  
**Action taken:** read-only inspection and diagnostics only. No dataset, model, checkpoint, or training output was modified, and no training was run.

## Executive conclusion

The active Phase 2 pipeline is internally consistent for the fixed 5k subset and the nested 30k subset:

- Raw geometry and response rows are paired by the same shard and offset.
- The processed 5k rows exactly reconstruct from their raw source IDs.
- The processed 5k and nested 30k manifests have disjoint train/validation/test splits, complete coverage, and no exact duplicate geometry or response rows.
- The raw release itself contains 261,000 rows in 122 paired shards and 56 exact duplicate geometries. All 56 duplicate pairs cross families; 41 pairs have different stored EM responses. This is not a failure of the fixed 5k subset, but it is a material risk for future full-dataset or independently sampled splits.
- Phase 2 predicts four normalized real-valued channels from geometry only. The EM response is the supervised target; it is not an input, and no Maxwell solver or solver-in-the-loop path is present.
- Phases 3, 4, 4.1, and 4.2 are geometry-completion experiments. Their shared dataset wrapper opens and validates `responses.npy`, but their samples, models, and losses use only geometry and masks. No EM value enters those phases operationally.

The main unresolved provenance issue is semantic rather than numerical: raw array positions are mapped to the repository's documented `T_y`/`R_x` reflected coefficients, but the raw `.npy` files do not carry standalone per-channel metadata. The mapping is explicit in repository code and documentation and agrees with the cited source paper; it should remain encoded in machine-readable metadata to prevent `T` from being misread as transmission.

## 1. Evidence classification

This report separates three kinds of evidence:

1. **Byte-level/runtime checks:** shapes, dtypes, finite values, ranges, exact raw-to-processed equality, split overlaps, and duplicate hashes measured from the files in this working copy.
2. **Code-defined behavior:** the exact pairing, channel extraction, frequency-vector creation, normalization, loader, model, and loss behavior in `src/` and `scripts/`.
3. **Repository/source-paper semantics:** the physical interpretation of the two response positions. The local raw directories contain NumPy arrays but no standalone release README. The repository convention is documented in `docs/dataset_conventions.md`; the cited source is the [SUTD-PRCM paper](https://arxiv.org/abs/2203.00002).

Where semantics are documentation-supported rather than encoded in the raw array, that limitation is stated explicitly below.

## 2. Raw dataset: actual files and schema

### 2.1 On-disk layout

The working copy contains these raw family roots:

```text
PLGDATASET/PLGDATASET/full_data_list/Data_*/
PLRDATASET/PLRDATASET/full_data_list/Data_*/
PTNDATASET/PTNDATASET/full_data_list/Data_*/
RDNDATASET/RDNDATASET/full_data_list/Data_*/
```

Each `Data_*` shard contains exactly one `Integrate_image_*.npy` file and one `Integrate_curve_*.npy` file. `src/preprocess.py:30-36` resolves both the direct and nested family-root forms. `src/preprocess.py:39-59` then:

- discovers the four families in the fixed order `PLG`, `PLR`, `PTN`, `RDN`;
- sorts shard directories numerically;
- requires exactly one image and one curve file per shard;
- memory-maps both arrays;
- requires image shape `[N, 256]`;
- requires curve shape `[N, 2, 1001]`;
- requires equal image/curve row counts.

The actual raw scan found:

| Family | Shards | Rows |
| --- | ---: | ---: |
| PLG | 4 | 30,000 |
| PLR | 3 | 60,000 |
| PTN | 4 | 60,000 |
| RDN | 111 | 111,000 |
| **Total** | **122** | **261,000** |

### 2.2 Geometry array

The raw image row is a flattened 256-element binary pattern. Runtime inspection found:

- shape per sample: `[256]`, reshaped by preprocessing to `[16, 16]`;
- values: only `0` and `1` across all 261,000 rows;
- dtypes: `uint8` in most families and `int64` in at least the PTN shards;
- invalid geometry values: `0`;
- physical interpretation documented by the repository/source paper: 1 means a present 0.5 mm square copper patch and 0 means absent.

The dtype difference is harmless for the current pipeline because the processed output is explicitly allocated as `uint8` and the PyTorch dataset converts it to `float32` on access.

### 2.3 Raw EM response array

Every curve shard is `complex128` with shape `[N, 2, 1001]`. The two positions are not two real/imaginary components; each position is a complete complex spectral curve:

| Raw position | Repository/source-paper symbol | Physical meaning used by this project | Processed channels |
| ---: | --- | --- | --- |
| `curve[offset, 0]` | `T` / `T_y` | y-polarized cross-polarized **reflected** coefficient | channel 0 `Re(T_y)`, channel 1 `Im(T_y)` |
| `curve[offset, 1]` | `R` / `R_x` | x-polarized co-polarized **reflected** coefficient | channel 2 `Re(R_x)`, channel 3 `Im(R_x)` |

The setup described by the repository convention is x-polarized normal incidence on a pure reflective, copper-backed metasurface. In particular, `T` is the upstream dataset's coefficient name for the y-cross reflected component; it is **not transmission**. The source-paper description also states that the samples contain x- and y-polarized reflection and that the pure-reflective setting omits transmission for this release.

This semantic mapping is explicit in `src/preprocess.py:204-208`, `src/dataset.py:3-7`, and `src/metrics.py:1-5`. The raw arrays themselves do not contain a channel-name field, so a consumer that bypasses repository code could still mislabel position 0.

### 2.4 Frequency axis

The raw curve files contain 1001 spectral samples but no frequency vector is read from them. Preprocessing creates the axis at `src/preprocess.py:211` as:

```python
np.linspace(2.0, 12.0, 1001, dtype=np.float32)
```

The processed `frequency_ghz.npy` was checked against this exact vector. Its endpoints are 2.0 and 12.0 GHz, with nominal spacing 0.01 GHz (floating-point adjacent differences vary only at approximately 1e-6 GHz due to `float32` representation). The local code and artifact therefore define the operational axis as:

```text
frequency_ghz: [2.00, 2.01, ..., 12.00], length 1001
```

The source-paper HTML contains both a broad description saying 2–10 GHz and a later figure description saying 2–12 GHz. The repository's generated artifact and all runtime checks consistently use 2–12 GHz; that local array should be treated as authoritative for this project.

## 3. Raw-to-processed transformation

The complete transformation is:

```text
Integrate_image_*.npy [N,256] uint8/int64
        + same-shard Integrate_curve_*.npy [N,2,1001] complex128
        + row offset
                     |
                     v
source ID = FAMILY/Data_shard/offset
geometry = image[offset].reshape(16,16)
response = [Re(curve[offset,0]), Im(curve[offset,0]),
            Re(curve[offset,1]), Im(curve[offset,1])]
                     |
                     v
geometries.npy [N,1,16,16] uint8
responses.npy  [N,4,1001] float32
frequency_ghz.npy [1001] float32, generated 2–12 GHz
source_ids.txt, splits/*.txt, train_response_stats.npz, metadata.json
```

The materialization code is `src/preprocess.py:188-227`:

1. Allocate `geometries.npy` as `[size, 1, 16, 16]`, `uint8`.
2. Allocate `responses.npy` as `[size, 4, 1001]`, `float32`.
3. Open each source shard once as a memory map.
4. Copy the selected flattened image into the `[16,16]` geometry plane.
5. Extract real and imaginary parts in the fixed four-channel order above.
6. Write the frequency vector and source-ID manifest.
7. Compute train-only response statistics over both the train sample axis and frequency axis.
8. Write split manifests and metadata.

The exact raw-to-processed equality check was run for all 5,000 5k rows: **0 geometry mismatches and 0 response mismatches** at the stored `float32` representation.

## 4. Processed subset and normalization

`src/dataset.py:24-99` defines `SUTDPRCMDataset`. It loads:

| File | Runtime content |
| --- | --- |
| `geometries.npy` | memory-mapped `[N,1,16,16]` binary geometry |
| `responses.npy` | memory-mapped `[N,4,1001]` float32 EM target |
| `frequency_ghz.npy` | `[1001]` frequency axis |
| `source_ids.txt` | raw family/shard/offset manifest |
| `splits/{train,val,test}.txt` | IDs for each split |
| `train_response_stats.npz` | `mean` and `std`, each `[4,1]` |
| `metadata.json` | schema, family, frequency, selection, and split metadata |

`_validate_layout()` at `src/dataset.py:75-83` enforces the geometry, response, frequency, and manifest lengths. The dataset converts each geometry and response to `float32` in `src/dataset.py:88-94`. Geometry values are not standardized. When normalization is enabled, response normalization is:

```text
y_normalized[c,f] = (y_float32[c,f] - mean[c,0]) / std[c,0]
```

The statistics are one scalar mean and one scalar standard deviation per channel, pooled over all train structures and all 1001 frequencies. They are not frequency-dependent statistics. `src/preprocess.py:219-226` computes them; `src/dataset.py:68-71, 91-94` applies them. The saved 5k statistics exactly match a fresh recomputation from the train manifest (maximum absolute mean/std difference: 0.0).

`build_dataloaders()` at `src/dataset.py:101-122` creates train/validation/test loaders with batch size configurable, `shuffle=True` only for train, `num_workers=0`, and `pin_memory=False` by default. Batches are initially CPU tensors; the training script explicitly moves both geometry and target to the selected PyTorch device.

## 5. Subset selection, pairing, and splits

### 5.1 5k subset

`src/preprocess.py:66-81` selects `size / 4` rows per family using a seeded NumPy generator and `replace=False`, then shuffles the combined references. `src/preprocess.py:104-127` requires the subset size to be divisible by four. For the stored 5k subset:

| Split | Total | PLG | PLR | PTN | RDN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 4,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| Validation | 500 | 125 | 125 | 125 | 125 |
| Test | 500 | 125 | 125 | 125 | 125 |

`src/preprocess.py:84-101` splits independently within each family using 80/10/10 integer counts, then shuffles each split. The actual manifests satisfy:

- 5,000 unique source IDs;
- each split internally unique;
- zero train/validation, train/test, or validation/test ID overlap;
- split union exactly equals `source_ids.txt`.

This is a row-level split, not a shard-level split. Each family and many shards can occur in all three splits. This is valid for the current source-ID check, but it is why geometry-level duplicate detection is also necessary.

### 5.2 Nested 30k subset

`src/preprocess.py:113-186` supports a nested extension. The stored `data/processed/sutd_prcm_30k` keeps every 5k source ID in its original split and samples additional rows without replacement by family. The actual check confirms that all 4,000 train, 500 validation, and 500 test IDs from the 5k subset are preserved in their corresponding 30k splits. The nested 30k split is 24,000/3,000/3,000 and has no exact duplicate geometry or response rows.

The separate `data/processed/sutd_prcm_30k_independent` artifact is not nested: it overlaps the 5k source manifest in 1,679 IDs. It has two exact duplicate-geometry groups:

```text
PLG/Data_002/017380  (train)  ==  RDN/Data_081/000851  (test)
PLG/Data_002/000049  (train)  ==  PLR/Data_003/023132  (train)
```

The second pair also has an exact duplicate processed response. This independent artifact should not be used for a leakage-sensitive comparison without a geometry-grouped split or an explicit decision about duplicate handling.

### 5.3 Duplicate audit of the full raw release

The read-only `scripts/inspect_dataset.py --check-duplicates` scan found:

```text
raw rows:              261,000
unique geometry rows: 260,944
duplicate extra rows:       56
duplicate groups:            56
```

Every duplicate group is a pair crossing families: 15 PLG/PLR pairs and 41 PLG/RDN pairs. Comparing the associated raw complex curves found 15 pairs with identical responses and 41 pairs with different responses. The code does not currently resolve why identical geometry rows can have different EM curves; the audit does not assume that this is either corruption or a legitimate hidden simulation condition.

## 6. Phase 2 model input, output, and loss

### 6.1 Training path

The Phase 2 training path is `scripts/train_forward.py:129-211`:

```text
SUTDPRCMDataset(normalize_response=True)
        -> DataLoader batch
        -> geometry [B,1,16,16], normalized target [B,4,1001]
        -> .to(device)
        -> model(geometry)
        -> prediction [B,4,1001]
        -> loss(prediction, target)
```

The stored Phase 2.5 Experiment A metadata confirms the actual baseline used `ForwardSurrogateCNN`, 2,208,932 parameters, AdamW, learning rate `1e-3`, weight decay `1e-5`, batch size 64, seed 42, and `device: cuda` under Python 3.14 / PyTorch `2.10.0+cu126`. This confirms that the CNN can and did run on the CUDA-enabled environment for that completed run.

### 6.2 Baseline CNN

`src/models.py:9-37` defines `ForwardSurrogateCNN`:

```text
[B,1,16,16]
  -> Conv 1->16, GELU, MaxPool       [B,16,8,8]
  -> Conv 16->32, GELU, MaxPool       [B,32,4,4]
  -> Conv 32->64, GELU, AdaptivePool  [B,64,2,2]
  -> Flatten -> Linear 256->512
  -> GELU -> Linear 512->4004
  -> reshape                          [B,4,1001]
```

The model does not receive the frequency vector, family label, shard ID, material parameters, incidence parameters, or any EM response channel as input. It maps geometry to a complete spectral response.

The separate `ResponseAwareSurrogateCNN` in `src/models.py:40-95` is a later Phase 2.5 architecture variant with residual geometry blocks, a retained 4x4 feature map, and a shallow 1-D spectral refinement head. It does not change the preprocessing schema or add EM inputs; it remains a geometry-to-response surrogate.

### 6.3 Baseline loss and evaluation

For `loss: normalized_mse`, `scripts/train_forward.py:34-47` returns `nn.MSELoss()`. The training step at `scripts/train_forward.py:172-177` minimizes the elementwise MSE between the predicted and train-normalized four-channel target over all channels and frequencies.

The model therefore learns the two complex coefficients indirectly as four real regression channels. It does not use a complex-valued layer or complex loss. Evaluation unnormalizes with the train statistics (`scripts/train_forward.py:80-92`) and computes magnitudes using `hypot` over each real/imaginary pair (`src/metrics.py:29-33`).

The optional `resonance_weighted_complex_loss` in `src/losses.py:13-59` still starts from normalized component MSE. It unnormalizes both tensors, computes target magnitude curvature, applies the same feature weights to each real/imaginary pair, and optionally adds a magnitude loss. It is response-aware, but it is not a Maxwell residual and does not call an EM solver. It also does not receive `frequency_ghz`; it relies on the uniformly sampled point index.

## 7. EM usage in later completion phases

There are two different meanings of “uses the EM dataset” here:

1. **Incidental loading/validation:** `CompletionDataset` constructs `SUTDPRCMDataset` at `src/completion_dataset.py:68-90`. The shared dataset constructor opens `responses.npy` and validates its shape even though completion samples do not need it.
2. **Operational EM conditioning or supervision:** `CompletionDataset.__getitem__()` at `src/completion_dataset.py:97-104` reads only the complete geometry, creates a mask, forms the partial geometry, and returns `input`, `target`, `mask`, and `sample_id`. No response value is returned.

The phases therefore behave as follows:

| Phase | Input to model | Target/loss | EM response operationally used? |
| --- | --- | --- | --- |
| 3 | `[partial_geometry, mask]`, `[B,2,16,16]` | complete geometry; full BCE training and masked BCE evaluation | **No**; only incidental shared-array load/validation |
| 4 | same geometry/mask input | global JEPA latent alignment plus optional masked reconstruction BCE | **No** |
| 4.1 | same geometry/mask input | spatial JEPA loss over `[B,64,8,8]` plus masked reconstruction BCE | **No** |
| 4.2 | same geometry/mask input | mask-weighted spatial JEPA loss plus masked reconstruction BCE | **No** |

Evidence for the completion path is `src/completion_dataset.py:97-104`, `scripts/train_completion.py:96-101`, `scripts/train_jepa_completion.py:126-135`, `scripts/train_spatial_jepa_completion.py:129-140`, and `scripts/train_mask_aware_spatial_jepa.py:140-150`. The later model implementations also have geometry-only encoders: `src/completion_model.py:10-45`, `src/jepa_completion_model.py:70-125`, and `src/spatial_jepa_completion_model.py:83-138`.

Consequently, the current Phase 3–4.2 results establish structural completion behavior, not EM-aware completion or physics-conditioned JEPA behavior. The existing reports correctly defer EM conditioning.

## 8. Information retained, transformed, or discarded

| Information | Raw | Processed/Phase 2 status | Audit assessment |
| --- | --- | --- | --- |
| 16x16 binary geometry | flattened `[256]` | `[1,16,16]` `uint8`, then `float32` in PyTorch | retained exactly; PTN integer dtype is canonicalized |
| Two complex curves | `[2,1001]` `complex128` | four `[4,1001]` `float32` channels | complex values retained mathematically as Re/Im, with `float32` precision reduction |
| Channel semantics | not stored as field in `.npy` | documented/fixed by channel order and metadata | retained by convention; vulnerable if consumers ignore metadata |
| Frequency samples | implicit 1001 positions | generated `[1001]` 2–12 GHz vector | explicit in processed artifact; not read from raw files |
| Absolute response scale | raw coefficient units | train-only per-channel affine normalization for model input/target | invertible using saved stats |
| EM magnitude | derivable from Re/Im | computed only for metrics/loss diagnostics | not stored as a separate target |
| EM phase | derivable from Re/Im | not explicitly stored or unwrapped | no information loss beyond Re/Im float precision, but no phase-specific objective |
| Material/solver/incidence metadata | described externally, not per row in these arrays | absent from model inputs and processed sample records | discarded/not available for conditioning |
| Family/shard/offset provenance | raw path and row position | `source_ids.txt` | retained |
| Split membership | generated during subset creation | `splits/*.txt` | retained and checked |
| EM response in Phases 3–4.2 | available in processed file | not returned by completion samples or consumed by completion losses | operationally discarded for those experiments |

The most consequential current information loss for future physics work is not the Re/Im representation itself; it is the absence of per-sample EM-setting metadata and the fact that completion loaders do not expose EM targets. A future EM-conditioned completion objective would need an explicit response-return path and a documented alignment between partial geometry, complete geometry, and response.

## 9. Findings and recommended guardrails

### Confirmed correct

- Raw image/curve pairing is enforced by shard and row count.
- The four-channel extraction order is deterministic and verified against every 5k source row.
- Train-only statistics are used for normalization; validation/test data do not contribute to them.
- The fixed 5k and nested 30k manifests are source-ID disjoint across splits and duplicate-free at the exact geometry/response level.
- Phase 2 CUDA execution is recorded in the completed Experiment A metadata.
- Later completion phases do not silently use EM as a target or input.

### Risks to address before scaling or physics conditioning

1. Add a machine-readable `response_semantics` block to every processed subset, including raw positions, polarization, reflection/co-reflection meaning, incidence, and the explicit statement that `T` is not transmission.
2. Make frequency provenance explicit. The current operational axis is correct for this working copy, but it is generated rather than read from a raw-file field, and the source paper contains an inconsistent broad frequency statement.
3. Use geometry-grouped duplicate detection before any full-data or independent split. Do not automatically delete the 56 raw duplicates: 41 have different responses and require provenance investigation first.
4. Preserve the current 5k/nested-30k split policy for controlled comparisons, or record a deliberate group-level policy for future independent subsets.
5. If Phase 5 or later work makes EM part of the objective, add an explicit EM-aware dataset contract rather than relying on the current completion wrapper, which only returns geometry and masks.

## 10. Audit commands and results

The following read-only checks were run from the repository root:

```text
python scripts/inspect_dataset.py --raw-root . --check-duplicates
```

Result: 261,000 rows, 122 paired shards, binary geometry, finite complex responses, and 56 duplicate geometry rows beyond unique rows.

Additional in-memory diagnostics checked:

- exact raw-to-processed reconstruction for all 5,000 5k rows;
- source-ID uniqueness and split union/overlap;
- family balance per split;
- exact geometry and response duplicate hashes in 5k, nested 30k, and independent 30k subsets;
- recomputation of train response statistics;
- preservation of all 5k IDs in their original split in nested 30k.

No command in this audit wrote files or ran training.

## Source index

- `src/preprocess.py:13-59` — family discovery, shard validation, raw shapes.
- `src/preprocess.py:62-101` — source IDs, balanced selection, split construction.
- `src/preprocess.py:104-186` — subset and nested-extension policy.
- `src/preprocess.py:188-246` — processed arrays, Re/Im extraction, frequency axis, statistics, metadata.
- `src/dataset.py:24-94` — processed dataset loading, validation, normalization, returned tensor shapes.
- `src/dataset.py:101-122` — Phase 2 DataLoader construction.
- `scripts/train_forward.py:129-211` — device selection, model, optimizer, target movement, loss, checkpoint metadata.
- `src/models.py:9-37` — baseline CNN input/output and architecture.
- `src/losses.py:13-59` — optional resonance-weighted response loss.
- `src/metrics.py:15-33,47-76` — channel names, unnormalization, magnitudes, component and spectral metrics.
- `src/completion_dataset.py:68-125` — geometry/mask-only completion samples and loaders.
- `scripts/train_completion.py:96-130` — Phase 3 geometry-only training.
- `scripts/train_jepa_completion.py:126-168` — Phase 4 JEPA geometry-only training.
- `scripts/train_spatial_jepa_completion.py:129-173` — Phase 4.1 spatial JEPA geometry-only training.
- `scripts/train_mask_aware_spatial_jepa.py:140-179` — Phase 4.2 mask-aware geometry-only training.
- `data/processed/sutd_prcm_5k/metadata.json` — stored 5k schema and split metadata.
- `data/processed/sutd_prcm_30k/metadata.json` — stored nested 30k schema and split metadata.
- `docs/dataset_conventions.md` — repository-level physical response convention.
- `outputs/phase2_5/exp_A_5k_mse/training_metadata.json` — actual CUDA baseline run metadata.

