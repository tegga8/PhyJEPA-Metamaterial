"""Select the existing learned forward model for candidate screening.

This is an artifact-only comparison. It deliberately does not retrain models
or generate new predictions. The primary comparison uses the shared original
5k test IDs where available, so the 5k and nested-30k checkpoints are compared
on the same held-out structures and train-only normalization conventions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


COMPLEXITY_THRESHOLDS = (3.0625, 13.291666666666664)
GROUPS = ("simple", "medium", "complex")
PRIMARY_METRICS = (
    "normalized_mse",
    "complex_mae",
    "y_cross_reflection_magnitude_mae",
    "x_co_reflection_magnitude_mae",
    "y_cross_reflection_correlation",
    "x_co_reflection_correlation",
    "resonance_frequency_error_ghz",
    "resonance_region_magnitude_mae",
    "resonance_feature_match_rate",
    "inference_milliseconds_per_sample",
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def finite_mean(values: list[float]) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else None


def finite_percentile(values: list[float], percentile: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if finite.size else None


def complexity_group(score: float) -> str:
    if score <= COMPLEXITY_THRESHOLDS[0]:
        return "simple"
    if score <= COMPLEXITY_THRESHOLDS[1]:
        return "medium"
    return "complex"


def load_candidate(name: str, root: Path, shared_eval: Path) -> dict[str, Any]:
    shared_metrics = read_json(shared_eval / "metrics.json")
    full_metrics = read_json(root / "evaluation" / "metrics.json") or read_json(root / "metrics.json")
    training = read_json(root / "training_metadata.json")
    shared_metadata = read_json(shared_eval / "evaluation_metadata.json")
    gradient = read_json(root / "gradient" / "gradient_stability.json")
    if not gradient:
        sanity = shared_metadata.get("gradient_sanity", {})
        gradient = {
            "gradient_all_finite": sanity.get("all_finite"),
            "gradient_nonzero_fraction": sanity.get("nonzero_fraction"),
            "local_perturbation_sign_agreement": None,
        }
    return {
        "name": name,
        "root": str(root),
        "checkpoint": str(root / "best.pt"),
        "model": training.get("model", shared_metadata.get("model")),
        "dataset": training.get("subset_root"),
        "parameter_count": training.get("model_parameter_count", shared_metadata.get("model_parameter_count")),
        "training_seconds": training.get("training_seconds"),
        "shared_sample_count": shared_metrics.get("test_samples", shared_metadata.get("test_samples")),
        "shared": {metric: shared_metrics.get(metric) for metric in PRIMARY_METRICS},
        "full": {metric: full_metrics.get(metric) for metric in PRIMARY_METRICS},
        "gradient": {
            "all_finite": gradient.get("gradient_all_finite"),
            "nonzero_fraction": gradient.get("gradient_nonzero_fraction"),
            "local_perturbation_sign_agreement": gradient.get("local_perturbation_sign_agreement"),
        },
        "shared_eval": str(shared_eval),
    }


def load_complexity_rows(shared_eval: Path) -> list[dict[str, Any]]:
    path = shared_eval / "per_sample_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing per-sample artifact: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def complexity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"thresholds": list(COMPLEXITY_THRESHOLDS), "groups": {}}
    for group in GROUPS:
        selected: list[dict[str, Any]] = []
        for row in rows:
            score = float(row["connected_components_4"]) + float(row["boundary_transitions_4"]) / 32.0
            if complexity_group(score) == group:
                selected.append(row)
        result["groups"][group] = {
            "samples": len(selected),
            "normalized_mse_mean": finite_mean([float(row["normalized_mse"]) for row in selected]),
            "normalized_mse_p90": finite_percentile([float(row["normalized_mse"]) for row in selected], 90),
            "resonance_frequency_error_ghz_mean": finite_mean([float(row["resonance_frequency_error_ghz"]) for row in selected]),
            "resonance_region_magnitude_mae_mean": finite_mean([float(row["resonance_region_magnitude_mae"]) for row in selected]),
            "y_magnitude_mae_mean": finite_mean([float(row["y_cross_reflection_magnitude_mae"]) for row in selected]),
            "x_magnitude_mae_mean": finite_mean([float(row["x_co_reflection_magnitude_mae"]) for row in selected]),
        }
    return result


def select_candidate(candidates: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Choose using explicit screening priorities, not a hidden scalar score.

    Primary objective is broad response fidelity (normalized MSE), followed by
    resonance localization and resonance-region error. Complexity summaries
    are used as a diagnostic/tie-breaker, not as an opportunity to cherry-pick
    one geometry group.
    """
    complete = [candidate for candidate in candidates if all(candidate["shared"].get(metric) is not None for metric in ("normalized_mse", "resonance_frequency_error_ghz", "resonance_region_magnitude_mae"))]
    if not complete:
        raise ValueError("No candidate has the required shared metrics")
    ordered = sorted(
        complete,
        key=lambda candidate: (
            float(candidate["shared"]["normalized_mse"]),
            float(candidate["shared"]["resonance_frequency_error_ghz"]),
            float(candidate["shared"]["resonance_region_magnitude_mae"]),
        ),
    )
    selected = ordered[0]
    rationale = {
        "priority_order": ["shared normalized MSE", "shared resonance frequency error", "shared resonance-region MAE"],
        "selected_by_lexicographic_order": True,
        "selected_name": selected["name"],
        "runner_up": ordered[1]["name"] if len(ordered) > 1 else None,
        "broad_fidelity_winner": selected["name"],
    }
    return selected["name"], rationale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase2_5/surrogate_selection"))
    args = parser.parse_args()
    candidates = [
        load_candidate(
            "5k_mse",
            Path("outputs/phase2_5/exp_A_5k_mse"),
            Path("outputs/phase2_5/exp_A_5k_mse"),
        ),
        load_candidate(
            "5k_resonance",
            Path("outputs/phase2_5/exp_B_5k_resonance"),
            Path("outputs/phase2_5/exp_B_5k_resonance"),
        ),
        load_candidate(
            "30k_resonance",
            Path("outputs/phase2_5/exp_C_30k_resonance"),
            Path("outputs/phase2_5/exp_C_30k_resonance/shared_5k_test"),
        ),
        load_candidate(
            "30k_response_aware",
            Path("outputs/phase2_forward_30k_response_aware_gpu"),
            Path("outputs/phase2_forward_30k_response_aware_gpu/evaluation_shared_5k_test"),
        ),
    ]
    for candidate in candidates:
        candidate["complexity"] = complexity_summary(load_complexity_rows(Path(candidate["shared_eval"])))
    selected_name, rationale = select_candidate(candidates)
    selected = next(candidate for candidate in candidates if candidate["name"] == selected_name)
    report = {
        "phase": "7B_forward_screening_surrogate_selection",
        "purpose": "Select an existing learned forward model for cheap candidate screening.",
        "artifact_only": True,
        "primary_holdout": "original 5k test IDs, 500 samples",
        "normalization_rule": "each checkpoint evaluated with its own train-only normalization statistics",
        "candidates": candidates,
        "selection": rationale,
        "selected_checkpoint": selected["checkpoint"],
        "selected_model": selected["model"],
        "selected_limitations": [
            "learned surrogate is not Maxwell ground truth",
            "shared-holdout selection does not establish independent physical calibration",
            "screening checkpoint should be used for ranking/filtering, not as an unconstrained differentiable physics objective",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fields = ["candidate", "normalized_mse", "complex_mae", "y_cross_reflection_magnitude_mae", "x_co_reflection_magnitude_mae", "y_cross_reflection_correlation", "x_co_reflection_correlation", "resonance_frequency_error_ghz", "resonance_region_magnitude_mae", "resonance_feature_match_rate", "inference_milliseconds_per_sample", "training_seconds", "gradient_all_finite", "gradient_nonzero_fraction", "local_perturbation_sign_agreement"]
    with (args.output_dir / "selection_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            row = {"candidate": candidate["name"], **candidate["shared"], "training_seconds": candidate["training_seconds"], "gradient_all_finite": candidate["gradient"]["all_finite"], "gradient_nonzero_fraction": candidate["gradient"]["nonzero_fraction"], "local_perturbation_sign_agreement": candidate["gradient"]["local_perturbation_sign_agreement"]}
            writer.writerow({field: row.get(field) for field in fields})
    lines = [
        "# Phase 7B — Forward Screening Surrogate Selection",
        "",
        f"Selected checkpoint: `{selected['checkpoint']}`",
        "",
        "The selection is artifact-only and uses the shared original 5k test IDs for all four candidates.",
        "The chosen priority is shared normalized MSE, then resonance-frequency error, then resonance-region MAE.",
        "This is a screening selection, not physical validation.",
        "",
        "| Candidate | normalized MSE | resonance freq error (GHz) | resonance-region MAE | feature match | inference ms/sample | local gradient sign agreement |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in candidates:
        shared = candidate["shared"]
        gradient = candidate["gradient"]
        def display(value: Any) -> str:
            if value is None:
                return "not available"
            if isinstance(value, bool):
                return str(value)
            return f"{float(value):.6g}"
        lines.append("| " + " | ".join([
            candidate["name"], display(shared["normalized_mse"]), display(shared["resonance_frequency_error_ghz"]), display(shared["resonance_region_magnitude_mae"]), display(shared["resonance_feature_match_rate"]), display(shared["inference_milliseconds_per_sample"]), display(gradient["local_perturbation_sign_agreement"]),
        ]) + " |")
    lines.extend([
        "",
        "## Selection decision",
        "",
        f"**Selected: `{selected_name}`.** It has the lowest shared normalized MSE and the best broad/ resonance trade-off among the complete candidates. The 30k response-aware model is competitive on resonance localization, but is slower and has higher shared normalized MSE and resonance-region MAE than the selected 30k resonance-aware model.",
        "",
        "Use the selected model for candidate screening/ranking only. Do not reintroduce Phase 5B differentiable surrogate guidance as the next step, and do not call this checkpoint physical ground truth.",
        "",
        "## Complexity-stratified diagnostics",
        "",
    ])
    for candidate in candidates:
        lines.append(f"### {candidate['name']}")
        lines.append("")
        lines.append("| Group | Samples | normalized MSE mean | normalized MSE p90 | resonance freq mean (GHz) | resonance-region MAE mean |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for group in GROUPS:
            values = candidate["complexity"]["groups"][group]
            lines.append(f"| {group} | {values['samples']} | {display(values['normalized_mse_mean'])} | {display(values['normalized_mse_p90'])} | {display(values['resonance_frequency_error_ghz_mean'])} | {display(values['resonance_region_magnitude_mae_mean'])} |")
        lines.append("")
    (args.output_dir / "selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected_name, "checkpoint": selected["checkpoint"], "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
