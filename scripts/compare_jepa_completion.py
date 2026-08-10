"""Create a compact Phase 4 comparison table from the four benchmark evaluations."""

from __future__ import annotations

import csv
import json
from pathlib import Path


EXPERIMENTS = ("exp_4A", "exp_4B", "exp_4C", "exp_4D")


def load_metrics(root: Path, experiment: str) -> dict:
    with (root / experiment / "metrics.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    root = Path("outputs/phase4_jepa")
    records = [load_metrics(root, experiment) for experiment in EXPERIMENTS]
    rows: list[dict[str, object]] = []
    for experiment, metrics in zip(EXPERIMENTS, records):
        cnn = metrics["cnn"]
        jepa = metrics["jepa"]
        paired = metrics["paired_difference"]
        complexity = metrics["complexity"]["groups"]
        rows.append(
            {
                "experiment": experiment,
                "mask_type": metrics["mask_type"],
                "missing_ratio": metrics["missing_ratio"],
                "cnn_masked_iou": cnn["cnn_masked_iou"],
                "jepa_masked_iou": jepa["jepa_masked_iou"],
                "delta_masked_iou": jepa["jepa_masked_iou"] - cnn["cnn_masked_iou"],
                "cnn_masked_dice": cnn["cnn_masked_dice"],
                "jepa_masked_dice": jepa["jepa_masked_dice"],
                "delta_masked_dice": jepa["jepa_masked_dice"] - cnn["cnn_masked_dice"],
                "paired_delta_mean": paired["mean"],
                "paired_delta_median": paired["median"],
                "jepa_wins_fraction": paired["jepa_wins_fraction"],
                "simple_delta_iou": complexity["simple"]["delta_masked_iou"],
                "medium_delta_iou": complexity["medium"]["delta_masked_iou"],
                "complex_delta_iou": complexity["complex"]["delta_masked_iou"],
                "jepa_context_std": metrics["latent_variance"]["context_mean_std"],
                "jepa_target_std": metrics["latent_variance"]["target_mean_std"],
                "jepa_pred_std": metrics["latent_variance"]["pred_mean_std"],
                "known_region_error_jepa_max": metrics["known_region_error"]["jepa_max"],
            }
        )

    fields = list(rows[0])
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Phase 4 JEPA vs CNN comparison",
        "",
        "Main benchmark runs use the JEPA + masked reconstruction variant.",
        "",
        "| Experiment | Mask | Missing | CNN masked IoU | JEPA masked IoU | Delta | JEPA wins |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['mask_type']} | {float(row['missing_ratio']):.2f} "
            f"| {float(row['cnn_masked_iou']):.4f} | {float(row['jepa_masked_iou']):.4f} "
            f"| {float(row['delta_masked_iou']):+.4f} | {float(row['jepa_wins_fraction']):.3f} |"
        )
    lines.extend(
        [
            "",
            "Complexity deltas are JEPA masked IoU minus CNN masked IoU:",
            "",
            "| Experiment | Simple | Medium | Complex |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {float(row['simple_delta_iou']):+.4f} "
            f"| {float(row['medium_delta_iou']):+.4f} | {float(row['complex_delta_iou']):+.4f} |"
        )
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((root / "comparison.md").as_posix())


if __name__ == "__main__":
    main()
