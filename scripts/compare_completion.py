"""Create a compact Phase 3 experiment comparison from saved metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ("full_accuracy", "masked_accuracy", "full_iou", "masked_iou", "full_dice", "masked_dice", "full_bce", "masked_bce", "known_region_error")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/phase3_completion"))
    args = parser.parse_args()
    experiments = {name: json.loads((args.root / name / "metrics.json").read_text(encoding="utf-8")) for name in ("exp_3A", "exp_3B", "exp_3C", "exp_3D")}
    with (args.root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", *experiments])
        writer.writeheader()
        for metric in METRICS:
            writer.writerow({"metric": metric, **{name: experiments[name][metric] for name in experiments}})
    lines = ["# Phase 3 completion experiment comparison", "", "| Metric | 3A central 25% | 3B central 50% | 3C random 25% | 3D random 50% |", "| --- | ---: | ---: | ---: | ---: |"]
    for metric in METRICS:
        values = [experiments[name][metric] for name in experiments]
        lines.append(f"| {metric} | " + " | ".join(f"{float(value):.6f}" for value in values) + " |")
    (args.root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {args.root / 'comparison.csv'}")


if __name__ == "__main__":
    main()
