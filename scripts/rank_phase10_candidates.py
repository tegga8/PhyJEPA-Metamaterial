"""Rank Phase 10 generated candidates under deterministic screening rules."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


RANKING_MODES = ("all", "valid", "valid_novel")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def geometry_hash(geometry: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(geometry).tobytes()).hexdigest()


def load_candidate_rows(phase10_dir: Path) -> list[dict[str, Any]]:
    path = phase10_dir / "candidate_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["target_index"] = int(row["target_index"])
        row["candidate_index"] = int(row["candidate_index"])
        row["valid"] = parse_bool(row["valid"])
        for key in (
            "response_mse",
            "nearest_train_pixel_hamming",
            "nearest_train_latent_mse",
            "occupancy",
            "connected_components_4",
            "boundary_transitions_4",
            "iou",
            "dice",
            "pixel_hamming",
            "occupancy_abs_difference",
            "target_complexity_score",
        ):
            row[key] = float(row[key])
    return rows


def attach_geometry_hashes(rows: list[dict[str, Any]], candidates: np.ndarray) -> None:
    expected = {(int(row["target_index"]), int(row["candidate_index"])) for row in rows}
    if candidates.ndim != 5 or candidates.shape[2:] != (1, 16, 16):
        raise ValueError(f"Expected candidates [targets,K,1,16,16], got {candidates.shape}")
    available = {(target_index, candidate_index) for target_index in range(candidates.shape[0]) for candidate_index in range(candidates.shape[1])}
    if expected != available:
        raise AssertionError("Candidate CSV rows do not match candidate array indexes")
    for row in rows:
        geometry = candidates[row["target_index"], row["candidate_index"], 0]
        row["geometry_hash"] = geometry_hash(geometry)


def passes_mode(row: dict[str, Any], mode: str, min_pixel_novelty: float, min_latent_novelty: float) -> bool:
    if mode == "all":
        return True
    if not row["valid"]:
        return False
    if mode == "valid":
        return True
    if mode == "valid_novel":
        return row["nearest_train_pixel_hamming"] >= min_pixel_novelty and row["nearest_train_latent_mse"] >= min_latent_novelty
    raise ValueError(f"Unknown ranking mode: {mode}")


def deduplicate_by_geometry(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_hash: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["geometry_hash"])
        if key not in best_by_hash or row["response_mse"] < best_by_hash[key]["response_mse"]:
            best_by_hash[key] = row
    return list(best_by_hash.values())


def finite_mean(values: list[float]) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else None


def finite_percentile(values: list[float], percentile: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if finite.size else None


def summarize_selected(selected: list[dict[str, Any]], total_targets: int, response_threshold: float) -> dict[str, Any]:
    response_values = [row["response_mse"] for row in selected]
    return {
        "target_coverage_fraction": len(selected) / total_targets if total_targets else None,
        "targets_covered": len(selected),
        "targets_missing": total_targets - len(selected) if total_targets else 0,
        "best_response_mse_mean": finite_mean(response_values),
        "best_response_mse_median": finite_percentile(response_values, 50),
        "best_response_mse_p90": finite_percentile(response_values, 90),
        "top1_success_fraction": float(np.mean([row["response_mse"] <= response_threshold for row in selected])) if selected else None,
        "selected_validity_rate": float(np.mean([row["valid"] for row in selected])) if selected else None,
        "nearest_train_pixel_hamming_mean": finite_mean([row["nearest_train_pixel_hamming"] for row in selected]),
        "nearest_train_latent_mse_mean": finite_mean([row["nearest_train_latent_mse"] for row in selected]),
        "candidate_target_pixel_hamming_mean": finite_mean([row["pixel_hamming"] for row in selected]),
        "occupancy_abs_difference_mean": finite_mean([row["occupancy_abs_difference"] for row in selected]),
    }


def complexity_summary(selected: list[dict[str, Any]], total_by_group: dict[str, int], response_threshold: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in ("simple", "medium", "complex"):
        rows = [row for row in selected if row["complexity_group"] == group]
        result[group] = summarize_selected(rows, total_by_group[group], response_threshold)
    return result


def rank_candidates(rows: list[dict[str, Any]], min_pixel_novelty: float, min_latent_novelty: float, response_threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["target_index"]].append(row)
    target_indexes = sorted(grouped)
    total_by_group = {
        group: len({row["target_index"] for row in rows if row["complexity_group"] == group})
        for group in ("simple", "medium", "complex")
    }
    selected_rows: list[dict[str, Any]] = []
    mode_metrics: dict[str, Any] = {}
    for mode in RANKING_MODES:
        selected_for_mode: list[dict[str, Any]] = []
        candidates_after_filter: list[int] = []
        successful_targets = 0
        duplicates_removed = 0
        for target_index in target_indexes:
            original = grouped[target_index]
            filtered = [row for row in original if passes_mode(row, mode, min_pixel_novelty, min_latent_novelty)]
            deduplicated = deduplicate_by_geometry(filtered)
            duplicates_removed += len(filtered) - len(deduplicated)
            candidates_after_filter.append(len(deduplicated))
            successful_targets += int(sum(row["response_mse"] <= response_threshold for row in deduplicated) >= 2)
            if deduplicated:
                best = min(deduplicated, key=lambda row: row["response_mse"])
                selected_for_mode.append(best)
                selected_rows.append({"ranking_mode": mode, **best})
        mode_metrics[mode] = {
            **summarize_selected(selected_for_mode, len(target_indexes), response_threshold),
            "multi_solution_success_fraction": successful_targets / len(target_indexes),
            "candidates_after_filter_mean": float(np.mean(candidates_after_filter)),
            "duplicates_removed": duplicates_removed,
            "complexity_summaries": complexity_summary(selected_for_mode, total_by_group, response_threshold),
        }
    report = {
        "phase": "11_candidate_generation_and_surrogate_ranking",
        "ranking_modes": {
            "all": "rank all generated candidates by learned-screening response MSE",
            "valid": "rank only deterministic-valid candidates",
            "valid_novel": "rank valid candidates that also satisfy minimum pixel and latent novelty",
        },
        "constraints": {
            "min_pixel_novelty": min_pixel_novelty,
            "min_latent_novelty_mse": min_latent_novelty,
            "response_threshold": response_threshold,
            "deduplicate_per_target_by_exact_binary_geometry": True,
        },
        "targets": len(target_indexes),
        "metrics": mode_metrics,
        "scientific_decision": "ranking is an artifact-level screen over Phase 10 candidates; it is not independent physical validation",
    }
    return report, selected_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase10-dir", type=Path, default=Path("outputs/phase10_stochastic_inverse_design"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase11_candidate_ranking"))
    parser.add_argument("--min-pixel-novelty", type=float, default=0.05)
    parser.add_argument("--min-latent-novelty", type=float, default=0.25)
    parser.add_argument("--response-threshold", type=float, default=0.30)
    args = parser.parse_args()

    rows = load_candidate_rows(args.phase10_dir)
    candidates = np.load(args.phase10_dir / "test_candidates_binary.npy")
    attach_geometry_hashes(rows, candidates)
    report, selected_rows = rank_candidates(rows, args.min_pixel_novelty, args.min_latent_novelty, args.response_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "ranked_top_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": report["metrics"], "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
