"""Reusable Phase 2.5 resonance and geometry analysis helpers."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from scipy.signal import find_peaks


def geometry_complexity(geometry: np.ndarray) -> dict[str, float]:
    """Describe a binary 16x16 pattern using 4-connected occupied components.

    Boundary complexity counts unlike horizontal and vertical adjacent pixel
    pairs.  It deliberately excludes the exterior image border, so the measure
    is invariant to the definition of the surrounding simulation cell.
    """
    image = np.asarray(geometry).squeeze()
    if image.shape != (16, 16):
        raise ValueError(f"Expected a 16x16 geometry, got {image.shape}")
    if not np.isin(image, (0, 1)).all():
        raise ValueError("Geometry complexity requires a binary geometry")
    occupied = image.astype(bool)
    seen = np.zeros_like(occupied, dtype=bool)
    components = 0
    for row, col in zip(*np.nonzero(occupied & ~seen)):
        if seen[row, col]:
            continue
        components += 1
        queue: deque[tuple[int, int]] = deque([(int(row), int(col))])
        seen[row, col] = True
        while queue:
            current_row, current_col = queue.popleft()
            for next_row, next_col in (
                (current_row - 1, current_col), (current_row + 1, current_col),
                (current_row, current_col - 1), (current_row, current_col + 1),
            ):
                if 0 <= next_row < 16 and 0 <= next_col < 16 and occupied[next_row, next_col] and not seen[next_row, next_col]:
                    seen[next_row, next_col] = True
                    queue.append((next_row, next_col))
    return {
        "fill_ratio": float(occupied.mean()),
        "connected_components_4": float(components),
        "boundary_transitions_4": float(np.count_nonzero(occupied[:, 1:] != occupied[:, :-1]) + np.count_nonzero(occupied[1:, :] != occupied[:-1, :])),
        "horizontal_symmetry": float(np.array_equal(occupied, np.fliplr(occupied))),
        "vertical_symmetry": float(np.array_equal(occupied, np.flipud(occupied))),
        "rotational_symmetry_180": float(np.array_equal(occupied, np.rot90(occupied, 2))),
    }


def detect_resonance_features(
    magnitude: np.ndarray,
    frequency_ghz: np.ndarray,
    prominence: float = 0.03,
    distance_points: int = 10,
) -> list[dict[str, Any]]:
    """Find prominent magnitude peaks and dips, without treating ripples as features.

    ``prominence`` is an absolute reflection-magnitude threshold; ``distance``
    is in 0.01-GHz samples for this dataset.  The two passes identify maxima
    and minima independently because either may be an important resonance.
    """
    magnitude = np.asarray(magnitude, dtype=float)
    frequency_ghz = np.asarray(frequency_ghz, dtype=float)
    if magnitude.ndim != 1 or frequency_ghz.shape != magnitude.shape:
        raise ValueError("Magnitude and frequency must be aligned one-dimensional arrays")
    if not (np.isfinite(magnitude).all() and np.isfinite(frequency_ghz).all()):
        raise ValueError("Resonance detection requires finite arrays")
    peaks, peak_properties = find_peaks(magnitude, prominence=prominence, distance=distance_points)
    dips, dip_properties = find_peaks(-magnitude, prominence=prominence, distance=distance_points)
    features: list[dict[str, Any]] = []
    for index, feature_prominence in zip(peaks, peak_properties["prominences"]):
        features.append({"kind": "peak", "index": int(index), "frequency_ghz": float(frequency_ghz[index]), "prominence": float(feature_prominence)})
    for index, feature_prominence in zip(dips, dip_properties["prominences"]):
        features.append({"kind": "dip", "index": int(index), "frequency_ghz": float(frequency_ghz[index]), "prominence": float(feature_prominence)})
    return sorted(features, key=lambda feature: (feature["index"], feature["kind"]))


def resonance_errors(
    true_magnitude: np.ndarray,
    predicted_magnitude: np.ndarray,
    frequency_ghz: np.ndarray,
    prominence: float = 0.03,
    distance_points: int = 10,
    window_ghz: float = 0.10,
) -> dict[str, Any]:
    """Match true extrema to predicted extrema of the same type by frequency."""
    true_features = detect_resonance_features(true_magnitude, frequency_ghz, prominence, distance_points)
    predicted_features = detect_resonance_features(predicted_magnitude, frequency_ghz, prominence, distance_points)
    errors: list[float] = []
    for feature in true_features:
        candidates = [candidate for candidate in predicted_features if candidate["kind"] == feature["kind"]]
        if candidates:
            errors.append(min(abs(feature["frequency_ghz"] - candidate["frequency_ghz"]) for candidate in candidates))
    mask = np.zeros(len(frequency_ghz), dtype=bool)
    for feature in true_features:
        mask |= np.abs(frequency_ghz - feature["frequency_ghz"]) <= window_ghz
    local_mae = float(np.mean(np.abs(true_magnitude[mask] - predicted_magnitude[mask]))) if mask.any() else float("nan")
    return {
        "true_feature_count": len(true_features),
        "matched_feature_count": len(errors),
        "frequency_errors_ghz": errors,
        "resonance_region_magnitude_mae": local_mae,
    }


def finite_summary(values: list[float]) -> dict[str, float | int | None]:
    """Summarize finite values while making an empty feature set explicit."""
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if not len(data):
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(len(data)), "mean": float(data.mean()), "median": float(np.median(data)),
        "p90": float(np.percentile(data, 90)), "max": float(data.max()),
    }
