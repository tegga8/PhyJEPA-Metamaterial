"""Shared, cache-aware Phase 6 RCWA validation helpers."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.rcwa_solver import RCWAConfig, RCWASolveResult, geometry_to_physical_pattern, pack_response, solve_geometry


CHANNELS = ("Re(Ty)", "Im(Ty)", "Re(Rx)", "Im(Rx)")
PHYSICAL_MAPPING = "s_to_ty_p_to_rx"
MAPPING_CANDIDATES = (
    "s_to_ty_p_to_rx",
    "p_to_ty_s_to_rx",
    "conj_s_to_ty_p_to_rx",
    "conj_p_to_ty_s_to_rx",
)
PHASE42_COMPLEXITY_THRESHOLDS = (3.0625, 13.291666666666664)


def json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(json_ready(data), indent=2) + "\n", encoding="utf-8")


def cache_key(geometry: np.ndarray, frequencies_ghz: Iterable[float], config: RCWAConfig) -> str:
    digest = hashlib.sha256()
    # Bump when numerical semantics change so previous spectra are never reused.
    digest.update(b"phase6-rcwa-cache-v3-passive-loss-complex128-auto")
    digest.update(geometry_to_physical_pattern(geometry).tobytes())
    digest.update(np.asarray(frequencies_ghz, dtype=np.float64).tobytes())
    digest.update(json.dumps(asdict(config), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def cached_solve(
    geometry: np.ndarray,
    frequencies_ghz: Iterable[float],
    config: RCWAConfig,
    cache_dir: str | Path,
) -> RCWASolveResult:
    """Reuse an exact geometry/config/frequency run; never silently recompute it."""
    cache_root = Path(cache_dir); cache_root.mkdir(parents=True, exist_ok=True)
    key = cache_key(geometry, frequencies_ghz, config)
    path = cache_root / f"{key}.npz"
    meta_path = cache_root / f"{key}.json"
    if path.exists() and meta_path.exists():
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(meta_path.read_text(encoding="utf-8")); metadata["cache_hit"] = True
        return RCWASolveResult(data["ty"], data["rx"], data["reflected_power"], data["transmitted_power"], metadata)
    result = solve_geometry(geometry, frequencies_ghz, config=config)
    metadata = {**result.metadata, "cache_key": key, "cache_hit": False}
    np.savez_compressed(path, ty=result.ty, rx=result.rx, reflected_power=result.reflected_power, transmitted_power=result.transmitted_power)
    save_json(meta_path, metadata)
    result.metadata = metadata
    return result


def normalize(response_raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(response_raw) - np.asarray(mean)) / np.asarray(std)


def response_metrics(prediction_raw: np.ndarray, target_raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict[str, float]:
    prediction, target = normalize(prediction_raw, mean, std), normalize(target_raw, mean, std)
    delta = prediction - target
    output: dict[str, float] = {"normalized_mse": float(np.square(delta).mean())}
    for index, channel in enumerate(CHANNELS):
        output[f"{channel}_normalized_mse"] = float(np.square(delta[index]).mean())
        output[f"{channel}_raw_mae"] = float(np.abs(prediction_raw[index] - target_raw[index]).mean())
    for name, start in (("Ty", 0), ("Rx", 2)):
        pred_magnitude = np.hypot(prediction_raw[start], prediction_raw[start + 1])
        target_magnitude = np.hypot(target_raw[start], target_raw[start + 1])
        output[f"{name}_magnitude_mae"] = float(np.abs(pred_magnitude - target_magnitude).mean())
    return output


def frequency_wise_normalized_mse(
    prediction_raw: np.ndarray, target_raw: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Return the four-channel normalized MSE at every simulated frequency."""
    prediction, target = normalize(prediction_raw, mean, std), normalize(target_raw, mean, std)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[0] != 4:
        raise ValueError("Expected matching raw responses with shape [4, frequency]")
    return np.square(prediction - target).mean(axis=0)


def channel_frequency_wise_normalized_mse(
    prediction_raw: np.ndarray, target_raw: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Return normalized squared error with shape [4, frequency]."""
    prediction, target = normalize(prediction_raw, mean, std), normalize(target_raw, mean, std)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[0] != 4:
        raise ValueError("Expected matching raw responses with shape [4, frequency]")
    return np.square(prediction - target)


def aggregate_metric_rows(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean every numeric metric, rejecting an accidental empty calibration."""
    if not rows:
        raise ValueError("Cannot aggregate zero metric rows")
    numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float, np.number))]
    return {key: float(np.mean([row[key] for row in rows])) for key in numeric}


def complexity_groups(geometries: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Reproduce Phase 4.2's components-plus-boundaries tercile grouping."""
    from src.forward_analysis import geometry_complexity

    scores = np.asarray([
        geometry_complexity(np.asarray(geometry).squeeze())["connected_components_4"]
        + geometry_complexity(np.asarray(geometry).squeeze())["boundary_transitions_4"] / 32.0
        for geometry in geometries
    ])
    first, second = np.quantile(scores, (1 / 3, 2 / 3))
    labels = ["simple" if score <= first else "medium" if score <= second else "complex" for score in scores]
    return scores, labels


def phase42_complexity_group(score: float) -> str:
    """The fixed Phase 4.2 complexity bins, not new quantiles for this phase."""
    low, high = PHASE42_COMPLEXITY_THRESHOLDS
    return "simple" if score <= low else "medium" if score <= high else "complex"


def phase42_complexity_groups(geometries: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Score geometries with the Phase 4.2 descriptor and fixed thresholds."""
    from src.forward_analysis import geometry_complexity

    scores = np.asarray([
        geometry_complexity(np.asarray(geometry).squeeze())["connected_components_4"]
        + geometry_complexity(np.asarray(geometry).squeeze())["boundary_transitions_4"] / 32.0
        for geometry in geometries
    ], dtype=np.float64)
    return scores, [phase42_complexity_group(float(score)) for score in scores]


def select_phase42_balanced_indices(geometries: np.ndarray, per_complexity: int, seed: int = 42) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Choose a deterministic, balanced held-out sample under the Phase 4.2 bins."""
    if per_complexity < 1:
        raise ValueError("per_complexity must be positive")
    scores, labels = phase42_complexity_groups(geometries)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    label_array = np.asarray(labels)
    for label in ("simple", "medium", "complex"):
        matches = np.flatnonzero(label_array == label)
        if len(matches) < per_complexity:
            raise ValueError(f"Only {len(matches)} {label} geometries available")
        selected.extend(sorted(rng.choice(matches, size=per_complexity, replace=False).tolist()))
    return np.asarray(selected, dtype=np.int64), labels, scores


def select_balanced_indices(geometries: np.ndarray, per_complexity: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    scores, labels = complexity_groups(geometries)
    selected: list[int] = []
    for label in ("simple", "medium", "complex"):
        matches = np.flatnonzero(np.asarray(labels) == label)
        if len(matches) < per_complexity:
            raise ValueError(f"Only {len(matches)} {label} geometries available")
        selected.extend(matches[:per_complexity].tolist())
    return np.asarray(selected, dtype=np.int64), labels, scores


def packed(result: RCWASolveResult) -> np.ndarray:
    return pack_response(result.ty, result.rx)


def pack_modes(result: RCWASolveResult, mapping: str = "s_to_ty_p_to_rx") -> np.ndarray:
    """Apply an explicit dataset-channel mapping to meent's s/p amplitudes.

    At normal incidence the s/p basis has a convention-dependent x/y naming.
    Calibration evaluates these candidates rather than silently assuming one.
    """
    candidates = {
        "s_to_ty_p_to_rx": (result.ty, result.rx),
        "p_to_ty_s_to_rx": (result.rx, result.ty),
        "conj_s_to_ty_p_to_rx": (np.conj(result.ty), np.conj(result.rx)),
        "conj_p_to_ty_s_to_rx": (np.conj(result.rx), np.conj(result.ty)),
    }
    try:
        ty, rx = candidates[mapping]
    except KeyError as exc:
        raise ValueError(f"Unknown channel mapping {mapping!r}; choose from {sorted(candidates)}") from exc
    return pack_response(ty, rx)


def convergence_row(current: np.ndarray, previous: np.ndarray | None) -> dict[str, float | None]:
    """Full-spectrum successive-order diagnostics for the packed response."""
    current_array = np.asarray(current)
    if current_array.shape[0] != 4:
        raise ValueError("Expected packed response [4, frequency]")
    if previous is None:
        return {"complex_response_mse": None, **{f"{channel}_mse": None for channel in CHANNELS}}
    previous_array = np.asarray(previous)
    if previous_array.shape != current_array.shape:
        raise ValueError("Successive packed responses must have matching shapes")
    delta = current_array - previous_array
    return {
        "complex_response_mse": float(np.square(delta).mean()),
        **{f"{channel}_mse": float(np.square(delta[index]).mean()) for index, channel in enumerate(CHANNELS)},
    }


def ranking_statistics(cnn_errors: Iterable[float], rcwa_errors: Iterable[float], top_ks: Sequence[int] = (1, 3, 5)) -> dict[str, float | int | None]:
    """Ranking fidelity for one shared-target candidate set."""
    from scipy.stats import kendalltau, pearsonr, spearmanr

    cnn = np.asarray(list(cnn_errors), dtype=float)
    rcwa = np.asarray(list(rcwa_errors), dtype=float)
    if cnn.ndim != 1 or rcwa.shape != cnn.shape or len(cnn) < 2:
        raise ValueError("Ranking statistics need matching one-dimensional arrays with at least two candidates")
    pearson = pearsonr(cnn, rcwa).statistic
    spearman = spearmanr(cnn, rcwa).statistic
    kendall = kendalltau(cnn, rcwa).statistic
    pairwise = [
        (cnn[left] - cnn[right]) * (rcwa[left] - rcwa[right]) > 0
        for left in range(len(cnn)) for right in range(left + 1, len(cnn))
        if cnn[left] != cnn[right] and rcwa[left] != rcwa[right]
    ]
    result: dict[str, float | int | None] = {
        "candidate_count": int(len(cnn)),
        "pearson": float(pearson) if np.isfinite(pearson) else None,
        "spearman": float(spearman) if np.isfinite(spearman) else None,
        "kendall_tau": float(kendall) if np.isfinite(kendall) else None,
        "pairwise_ordering_agreement": float(np.mean(pairwise)) if pairwise else None,
        "mean_absolute_error_difference": float(np.mean(np.abs(cnn - rcwa))),
        "median_absolute_error_difference": float(np.median(np.abs(cnn - rcwa))),
    }
    cnn_rank, rcwa_rank = np.argsort(cnn), np.argsort(rcwa)
    for top_k in top_ks:
        result[f"top_{top_k}_overlap"] = (
            float(len(set(cnn_rank[:top_k]).intersection(rcwa_rank[:top_k])) / top_k)
            if len(cnn) >= top_k else None
        )
    return result
