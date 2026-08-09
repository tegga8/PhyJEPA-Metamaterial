"""Compare completed 5k and 30k forward-surrogate evaluations reproducibly."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    "normalized_mse", "y_cross_reflection_magnitude_mae", "x_co_reflection_magnitude_mae",
    "y_cross_reflection_correlation", "x_co_reflection_correlation", "resonance_frequency_error_ghz",
    "resonance_region_magnitude_mae", "training_seconds", "inference_seconds", "test_samples",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def evaluate_run(path: Path) -> dict[str, Any]:
    """Read available data and retain None for unavailable optional metrics."""
    metrics = read_json(path / "metrics.json")
    training = read_json(path / "training_metadata.json")
    # Training output predates Phase 2.5 metric names; retain its old values if
    # a user compares pre-existing results without rerunning evaluation.
    old_metrics = read_json(path / "test_metrics.json")
    metrics.setdefault("y_cross_reflection_magnitude_mae", old_metrics.get("y_reflection_magnitude_mae"))
    metrics.setdefault("x_co_reflection_magnitude_mae", old_metrics.get("x_reflection_magnitude_mae"))
    metrics.setdefault("normalized_mse", old_metrics.get("normalized_mse"))
    metrics.setdefault("training_seconds", training.get("training_seconds"))
    return {metric: metrics.get(metric) for metric in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-5k", type=Path, default=Path("outputs/phase2_forward_evaluation"))
    parser.add_argument("--evaluation-30k", type=Path, default=Path("outputs/phase2_forward_30k/evaluation_shared_5k_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase2_forward_scale_comparison"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = {"5k": evaluate_run(args.evaluation_5k), "30k": evaluate_run(args.evaluation_30k)}
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "5k", "30k"])
        writer.writeheader()
        writer.writerows({"metric": metric, "5k": runs["5k"][metric], "30k": runs["30k"][metric]} for metric in METRICS)
    report_lines = ["# Forward surrogate scale comparison", "", "| Metric | 5k | 30k |", "| --- | ---: | ---: |"]
    for metric in METRICS:
        def display(value: Any) -> str:
            return "not available" if value is None else f"{float(value):.6g}"
        report_lines.append(f"| {metric} | {display(runs['5k'][metric])} | {display(runs['30k'][metric])} |")
    (args.output_dir / "comparison.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Error distributions are optional: missing per-sample files should not
    # prevent a numeric comparison.
    distributions: dict[str, np.ndarray] = {}
    for label, directory in (("5k", args.evaluation_5k), ("30k", args.evaluation_30k)):
        path = directory / "per_sample_metrics.csv"
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                distributions[label] = np.asarray([float(row["normalized_mse"]) for row in csv.DictReader(handle)])
    if distributions:
        figure, axis = plt.subplots(figsize=(6, 3.8))
        for label, values in distributions.items():
            axis.hist(values, bins=35, alpha=0.5, density=True, label=label)
        axis.set_xlabel("per-sample normalized MSE")
        axis.set_ylabel("density")
        axis.legend()
        figure.tight_layout()
        figure.savefig(args.output_dir / "normalized_mse_distribution.png", dpi=180)
        plt.close(figure)
    print(f"Saved {args.output_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
