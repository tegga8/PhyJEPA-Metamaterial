"""Regenerate a ranked forward-surrogate failure plot from evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_forward import save_prediction_grid
from src.dataset import SUTDPRCMDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--evaluation-dir", type=Path, default=Path("outputs/phase2_forward_evaluation"))
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    cache_path = args.evaluation_dir / "prediction_cache.npz"
    metrics_path = args.evaluation_dir / "per_sample_metrics.csv"
    if not cache_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("Run scripts/evaluate_forward.py first to create cache and per-sample metrics")
    cache = np.load(cache_path)
    with metrics_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    error = np.asarray([float(row["normalized_mse"]) for row in rows])
    indexes = np.argsort(error)[-min(args.count, len(error)):][::-1]
    output = args.evaluation_dir / "plots" / "failures" / "worst_predictions.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_prediction_grid(output, SUTDPRCMDataset(args.subset_root, "test"), cache["geometries"], cache["prediction"], cache["target"], indexes, error, "Worst test predictions ranked only by normalized MSE")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
