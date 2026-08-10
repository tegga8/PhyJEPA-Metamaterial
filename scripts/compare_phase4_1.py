"""Aggregate the four Phase 4.1 three-way evaluations."""

from __future__ import annotations

import csv
import json
from pathlib import Path


EXPERIMENTS = ("exp_4_1A", "exp_4_1B", "exp_4_1C", "exp_4_1D")


def main() -> None:
    root = Path("outputs/phase4_1")
    records = [json.loads((root / name / "metrics.json").read_text(encoding="utf-8")) for name in EXPERIMENTS]
    rows: list[dict[str, object]] = []
    for experiment, metrics in zip(EXPERIMENTS, records):
        cnn = metrics["cnn"]
        global_jepa = metrics["global_jepa"]
        spatial = metrics["spatial_jepa"]
        rows.append({
            "experiment": experiment, "mask_type": metrics["mask_type"], "missing_ratio": metrics["missing_ratio"],
            "cnn_masked_iou": cnn["cnn_masked_iou"], "global_jepa_masked_iou": global_jepa["global_jepa_masked_iou"], "spatial_jepa_masked_iou": spatial["spatial_jepa_masked_iou"],
            "spatial_minus_cnn": metrics["paired_spatial_minus_cnn"]["mean"], "spatial_minus_global": metrics["paired_spatial_minus_global"]["mean"],
            "spatial_wins_cnn": metrics["paired_spatial_minus_cnn"]["wins_fraction"], "spatial_wins_global": metrics["paired_spatial_minus_global"]["wins_fraction"],
            "simple_spatial_minus_cnn": metrics["complexity"]["groups"]["simple"]["spatial_minus_cnn_iou"], "medium_spatial_minus_cnn": metrics["complexity"]["groups"]["medium"]["spatial_minus_cnn_iou"], "complex_spatial_minus_cnn": metrics["complexity"]["groups"]["complex"]["spatial_minus_cnn_iou"],
            "simple_spatial_minus_global": metrics["complexity"]["groups"]["simple"]["spatial_minus_global_iou"], "medium_spatial_minus_global": metrics["complexity"]["groups"]["medium"]["spatial_minus_global_iou"], "complex_spatial_minus_global": metrics["complexity"]["groups"]["complex"]["spatial_minus_global_iou"],
        })
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Phase 4.1 three-way comparison", "", "Main runs use spatial JEPA + 0.1 masked reconstruction BCE.", "",
        "| Experiment | Mask | Missing | CNN | Global JEPA | Spatial JEPA | Spatial-CNN | Spatial-Global |", "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['mask_type']} | {float(row['missing_ratio']):.2f} | {float(row['cnn_masked_iou']):.4f} | {float(row['global_jepa_masked_iou']):.4f} | {float(row['spatial_jepa_masked_iou']):.4f} | {float(row['spatial_minus_cnn']):+.4f} | {float(row['spatial_minus_global']):+.4f} |")
    lines.extend(["", "Complexity deltas are masked-IoU differences:", "", "| Experiment | Simple S-C | Medium S-C | Complex S-C | Simple S-G | Medium S-G | Complex S-G |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        lines.append(f"| {row['experiment']} | {float(row['simple_spatial_minus_cnn']):+.4f} | {float(row['medium_spatial_minus_cnn']):+.4f} | {float(row['complex_spatial_minus_cnn']):+.4f} | {float(row['simple_spatial_minus_global']):+.4f} | {float(row['medium_spatial_minus_global']):+.4f} | {float(row['complex_spatial_minus_global']):+.4f} |")
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((root / "comparison.md").as_posix())


if __name__ == "__main__":
    main()
