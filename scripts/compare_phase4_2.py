"""Aggregate the four Phase 4.2 four-way evaluations."""

from __future__ import annotations

import csv
import json
from pathlib import Path


EXPERIMENTS = ("exp_4_2A", "exp_4_2B", "exp_4_2C", "exp_4_2D")


def main() -> None:
    root = Path("outputs/phase4_2")
    records = [json.loads((root / name / "metrics.json").read_text(encoding="utf-8")) for name in EXPERIMENTS]
    rows: list[dict[str, object]] = []
    for experiment, metrics in zip(EXPERIMENTS, records):
        rows.append({
            "experiment": experiment, "mask_type": metrics["mask_type"], "missing_ratio": metrics["missing_ratio"],
            "cnn_masked_iou": metrics["cnn"]["cnn_masked_iou"], "global_jepa_masked_iou": metrics["global_jepa"]["global_jepa_masked_iou"], "spatial_jepa_masked_iou": metrics["spatial_jepa"]["spatial_jepa_masked_iou"], "mask_aware_masked_iou": metrics["mask_aware_spatial_jepa"]["mask_aware_spatial_jepa_masked_iou"],
            "mask_aware_minus_spatial": metrics["paired_mask_aware_minus_spatial"]["mean"], "mask_aware_minus_cnn": metrics["paired_mask_aware_minus_cnn"]["mean"], "mask_aware_wins_spatial": metrics["paired_mask_aware_minus_spatial"]["wins_fraction"], "mask_aware_wins_cnn": metrics["paired_mask_aware_minus_cnn"]["wins_fraction"],
        })
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Phase 4.2 four-way comparison", "", "Main runs use W = 0.10 + 0.90*M8 and 0.1 masked reconstruction BCE.", "", "| Experiment | Mask | Missing | CNN | Global JEPA | Spatial JEPA | Mask-aware JEPA | M-A − Spatial | M-A − CNN |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['mask_type']} | {float(row['missing_ratio']):.2f} | {float(row['cnn_masked_iou']):.4f} | {float(row['global_jepa_masked_iou']):.4f} | {float(row['spatial_jepa_masked_iou']):.4f} | {float(row['mask_aware_masked_iou']):.4f} | {float(row['mask_aware_minus_spatial']):+.4f} | {float(row['mask_aware_minus_cnn']):+.4f} |")
    lines.extend(["", "Complexity deltas are mask-aware JEPA minus the indicated baseline masked IoU:", "", "| Experiment | Simple − Spatial | Medium − Spatial | Complex − Spatial | Simple − CNN | Medium − CNN | Complex − CNN |", "|---|---:|---:|---:|---:|---:|---:|"])
    for experiment, metrics in zip(EXPERIMENTS, records):
        groups = metrics["complexity"]["groups"]
        lines.append(f"| {experiment} | {float(groups['simple']['mask_aware_minus_spatial_iou']):+.4f} | {float(groups['medium']['mask_aware_minus_spatial_iou']):+.4f} | {float(groups['complex']['mask_aware_minus_spatial_iou']):+.4f} | {float(groups['simple']['mask_aware_minus_cnn_iou']):+.4f} | {float(groups['medium']['mask_aware_minus_cnn_iou']):+.4f} | {float(groups['complex']['mask_aware_minus_cnn_iou']):+.4f} |")
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((root / "comparison.md").as_posix())


if __name__ == "__main__":
    main()
