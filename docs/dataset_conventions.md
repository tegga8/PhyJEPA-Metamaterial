# SUTD-PRCM response convention

This repository uses the SUTD polarized reflection of complex metasurfaces
(SUTD-PRCM) data release. The source paper is Zhang *et al.*, “SUTD-PRCM
Dataset and Neural Architecture Search Approach for Complex Metasurface
Design,” [arXiv:2203.00002](https://arxiv.org/abs/2203.00002), Section 2.1.
The local raw files corroborate the storage layout: paired
`Integrate_image_*.npy` arrays have shape `[N, 256]`, and
`Integrate_curve_*.npy` arrays have complex shape `[N, 2, 1001]`.

## Physical setup

The source applies an **x-polarized plane wave at normal incidence** to a pure
reflective metasurface (copper-backed substrate). There are no transmission
spectra in this release. Each 16×16 binary input indicates absence (0) or
presence (1) of a 0.5 mm square copper patch.

The paper denotes the two complex reflected-field coefficients by `T` and `R`:

| Raw index | Paper symbol | Meaning | Processed channels |
| ---: | --- | --- | --- |
| 0 | `T` | y-polarized, cross-polarized reflected coefficient | 0: `Re(T_y)`, 1: `Im(T_y)` |
| 1 | `R` | x-polarized, co-polarized reflected coefficient | 2: `Re(R_x)`, 3: `Im(R_x)` |

`T` must **not** be interpreted as transmission: it is the source dataset's
name for the y-polarized reflected component. The processed float32 response is
therefore `[Re(T_y), Im(T_y), Re(R_x), Im(R_x)]`, each with 1001 samples.
Magnitude metrics use `|T_y| = hypot(Re(T_y), Im(T_y))` and equivalently for
`|R_x|`.

## Frequency axis and units

The spectra run from **2.00 to 12.00 GHz**, inclusive, at **1001** uniformly
spaced points. `frequency_ghz.npy` is generated as `linspace(2.0, 12.0, 1001)`,
so the spacing is **0.01 GHz** (10 MHz). Magnitudes are dimensionless field
coefficient magnitudes, as supplied by the simulated complex coefficients.

## Provenance check

The raw dataset folders in this working copy contain arrays but no standalone
release README. This convention was verified against the source paper's explicit
simulation description and its stated `[T, R]` x/y reflected responses, then
mapped directly to the local `src/preprocess.py` extraction order. It supersedes
the earlier shorthand “y then x reflection,” which omitted the essential
cross/co-polarization and `T`-is-not-transmission distinction.
