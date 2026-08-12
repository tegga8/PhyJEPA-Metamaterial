"""Contiguous frequency-interval masking for the spectral JEPA variant.

The spectrum is masked over the 1001 frequency points with a small number of
contiguous keep blocks (gaps are the masked regions).  These masks are used
only by the spectral-masking variant, never inside the core geometry->physics
training.
"""

from __future__ import annotations

import numpy as np


def random_contiguous_masks(
    frequency_points: int,
    num_masks: int,
    keep_fraction: float,
    num_intervals: int = 2,
    seed: int | None = None,
) -> np.ndarray:
    """Generate ``[num_masks, frequency_points]`` float masks with keep=1.0.

    Each mask keeps roughly ``keep_fraction`` of the frequency axis as
    ``num_intervals`` contiguous blocks separated by masked gaps.
    """
    if frequency_points <= 0 or num_masks <= 0:
        raise ValueError("frequency_points and num_masks must be positive")
    if not 0.0 < keep_fraction < 1.0:
        raise ValueError(f"keep_fraction must be strictly inside (0, 1), got {keep_fraction}")
    if num_intervals <= 0:
        raise ValueError("num_intervals must be positive")
    rng = np.random.default_rng(seed)
    masks = np.zeros((num_masks, frequency_points), dtype=np.float32)
    keep_total = int(round(keep_fraction * frequency_points))
    for mask_index in range(num_masks):
        mask = masks[mask_index]
        lengths = _split_into_intervals(rng, keep_total, min(num_intervals, keep_total))
        for length in lengths:
            if length <= 0:
                continue
            start = _place_block(mask, length, rng)
            mask[start:start + length] = 1.0
    return masks


def _split_into_intervals(rng: np.random.Generator, keep_total: int, intervals: int) -> list[int]:
    if intervals <= 1:
        return [keep_total]
    points = np.sort(rng.choice(np.arange(1, keep_total), size=intervals - 1, replace=False))
    bounds = np.concatenate([[0], points, [keep_total]])
    return [int(end - start) for start, end in zip(bounds[:-1], bounds[1:])]


def _place_block(mask: np.ndarray, length: int, rng: np.random.Generator) -> int:
    """Place a keep block of ``length`` pixels leaving a 1-pixel gap from existing blocks."""
    maximum = mask.shape[0] - length
    attempts = list(range(maximum + 1))
    rng.shuffle(attempts)
    for start in attempts:
        low = max(0, start - 1)
        high = min(mask.shape[0], start + length + 1)
        if not mask[low:high].any():
            return start
    raise RuntimeError("Could not place non-overlapping contiguous mask block")


def apply_mask(response: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out masked frequency points of a ``[B, 4, F]`` response array."""
    response = np.asarray(response)
    mask = np.asarray(mask)
    if response.ndim != 3 or response.shape[2] != mask.shape[-1]:
        raise ValueError(f"Expected response [B, 4, F] and mask [.., F], got {tuple(response.shape)} and {tuple(mask.shape)}")
    broadcast = mask.reshape((1, 1, *mask.shape)) if mask.ndim == 1 else mask[:, None, None, :]
    return response * broadcast


def validate_spectral_mask(mask: np.ndarray, max_intervals: int = 4, min_coverage: float = 0.05, max_coverage: float = 0.95) -> dict[str, object]:
    """Validate a single contiguous-interval keep mask and describe it."""
    mask = np.asarray(mask)
    checks: dict[str, object] = {"shape": mask.shape, "dtype": str(mask.dtype)}
    checks["is_1d"] = bool(mask.ndim == 1)
    if not checks["is_1d"]:
        return checks
    checks["finite"] = bool(np.isfinite(mask).all())
    checks["values_binary"] = bool(np.isin(mask, (0.0, 1.0)).all())
    coverage = float(mask.mean())
    checks["coverage"] = coverage
    checks["coverage_in_range"] = bool(min_coverage <= coverage <= max_coverage)
    runs = np.diff(np.concatenate([[0], (mask > 0).astype(np.int64), [0]]))
    starts = np.where(runs == 1)[0]
    ends = np.where(runs == -1)[0]
    intervals = int(len(starts))
    checks["interval_count"] = intervals
    contiguous = all(int(end) > int(start) for start, end in zip(starts, ends))
    checks["intervals_contiguous"] = bool(contiguous and intervals >= 1)
    checks["interval_count_within_limit"] = bool(1 <= intervals <= max_intervals)
    checks["valid"] = bool(
        checks["finite"]
        and checks["values_binary"]
        and checks["coverage_in_range"]
        and checks["intervals_contiguous"]
        and checks["interval_count_within_limit"]
    )
    return checks
