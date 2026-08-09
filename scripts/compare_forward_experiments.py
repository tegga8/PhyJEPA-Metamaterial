"""Compare the controlled Phase 2.5 A/B/C experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = (
    "normalized_mse",
    "complex_mae",
    "y_cross_reflection_magnitude_mae",
    "x_co_reflection_magnitude_mae",
    "y_cross_reflection_correlation",
    "x_co_reflection_correlation",
    "resonance_frequency_error_ghz",
    "resonance_region_magnitude_mae",
    "resonance_feature_match_rate",
    "training_seconds",
    "inference_milliseconds_per_sample",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_experiment(path: Path) -> dict[str, Any]:
    metrics = read_json(path / "metrics.json")
    training = read_json(path / "training_metadata.json")
    metrics.setdefault("training_seconds", training.get("training_seconds"))
    return {metric: metrics.get(metric) for metric in METRICS}


def display(value: Any) -> str:
    return "not available" if value is None else f"{float(value):.6g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-a", type=Path, default=Path("outputs/phase2_5/exp_A_5k_mse"))
    parser.add_argument("--exp-b", type=Path, default=Path("outputs/phase2_5/exp_B_5k_resonance"))
    parser.add_argument("--exp-c", type=Path, default=Path("outputs/phase2_5/exp_C_30k_resonance"))
    parser.add_argument("--exp-c-shared", type=Path, default=Path("outputs/phase2_5/exp_C_30k_resonance/shared_5k_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase2_5"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = {"A_5k_MSE": load_experiment(args.exp_a), "B_5k_resonance": load_experiment(args.exp_b), "C_30k_resonance": load_experiment(args.exp_c)}

    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", *runs])
        writer.writeheader()
        for metric in METRICS:
            writer.writerow({"metric": metric, **{name: runs[name][metric] for name in runs}})

    lines = ["# Phase 2.5 controlled experiment comparison", "", "| Metric | A: 5k MSE | B: 5k resonance | C: 30k resonance |", "| --- | ---: | ---: | ---: |"]
    for metric in METRICS:
        lines.append(f"| {metric} | {display(runs['A_5k_MSE'][metric])} | {display(runs['B_5k_resonance'][metric])} | {display(runs['C_30k_resonance'][metric])} |")
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shared_runs = {"A_5k_MSE": load_experiment(args.exp_a), "B_5k_resonance": load_experiment(args.exp_b), "C_30k_resonance_shared_500": load_experiment(args.exp_c_shared)}
    with (args.output_dir / "comparison_shared_500.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", *shared_runs])
        writer.writeheader()
        for metric in METRICS:
            writer.writerow({"metric": metric, **{name: shared_runs[name][metric] for name in shared_runs}})
    shared_lines = ["# Phase 2.5 controlled experiment comparison on the shared 500-ID test set", "", "| Metric | A: 5k MSE | B: 5k resonance | C: 30k resonance |", "| --- | ---: | ---: | ---: |"]
    for metric in METRICS:
        shared_lines.append(f"| {metric} | {display(shared_runs['A_5k_MSE'][metric])} | {display(shared_runs['B_5k_resonance'][metric])} | {display(shared_runs['C_30k_resonance_shared_500'][metric])} |")
    (args.output_dir / "comparison_shared_500.md").write_text("\n".join(shared_lines) + "\n", encoding="utf-8")
    print(f"Saved {args.output_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
