"""Calibrate only the uncertain substrate thickness against held-out stored data."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import SUTDPRCMDataset
from src.rcwa_solver import RCWAConfig, frequency_vector
from src.rcwa_validation import pack_modes, response_metrics, save_json, cached_solve, select_balanced_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase6_rcwa"))
    parser.add_argument("--per-complexity", type=int, default=12)
    parser.add_argument("--thicknesses", default="0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    parser.add_argument("--mappings", default="s_to_ty_p_to_rx,p_to_ty_s_to_rx,conj_s_to_ty_p_to_rx,conj_p_to_ty_s_to_rx")
    parser.add_argument("--fourier-order", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true", help="Use three frequencies and one structure/group only to check the pipeline.")
    args = parser.parse_args()
    dataset = SUTDPRCMDataset(args.subset_root, "test", normalize_response=False)
    selected, labels, scores = select_balanced_indices(np.asarray([dataset.geometries[int(index)] for index in dataset.indices]), 1 if args.quick else args.per_complexity)
    frequencies = np.array([2.0, 7.0, 12.0]) if args.quick else frequency_vector()
    stats = np.load(args.subset_root / "train_response_stats.npz"); mean, std = stats["mean"], stats["std"]
    thicknesses = [float(value) for value in args.thicknesses.split(",")]
    mappings = args.mappings.split(",")
    calibration = args.output_dir / "calibration"; calibration.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []; summary: list[dict[str, object]] = []
    for thickness in thicknesses:
        for mapping in mappings:
            candidate_rows = []
            for index in selected:
                geometry, target = dataset[int(index)]
                result = cached_solve(geometry.numpy(), frequencies, RCWAConfig(substrate_thickness_mm=thickness, fourier_order=args.fourier_order, device=args.device), args.output_dir / "cache")
                metrics = response_metrics(pack_modes(result, mapping), target.numpy()[:, :len(frequencies)], mean, std)
                row = {"source_id": dataset.source_id(int(index)), "test_index": int(index), "complexity": labels[int(index)], "complexity_score": float(scores[int(index)],), "substrate_thickness_mm": thickness, "channel_mapping": mapping, **metrics}
                all_rows.append(row); candidate_rows.append(row)
            summary.append({"substrate_thickness_mm": thickness, "channel_mapping": mapping, "mean_normalized_mse": float(np.mean([row["normalized_mse"] for row in candidate_rows])), "median_normalized_mse": float(np.median([row["normalized_mse"] for row in candidate_rows])), **{f"mean_{name}_normalized_mse": float(np.mean([row[f"{name}_normalized_mse"] for row in candidate_rows])) for name in ("Re(Ty)", "Im(Ty)", "Re(Rx)", "Im(Rx)")}})
    summary.sort(key=lambda row: row["mean_normalized_mse"])
    with (calibration / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    save_json(calibration / "metrics.json", {"frequency_ghz": frequencies, "sample_count": len(selected), "samples_per_complexity": 1 if args.quick else args.per_complexity, "fourier_order": args.fourier_order, "best_agreement_candidate_only": summary[0], "candidates": summary})
    save_json(args.output_dir / "config.json", {"frequency_ghz": frequencies, "fourier_order": args.fourier_order, "device": args.device, "calibration_candidates_mm": thicknesses, "selected_substrate_thickness_mm": summary[0]["substrate_thickness_mm"], "selected_channel_mapping": summary[0]["channel_mapping"], "interpretation": "best agreement with stored dataset, not a proven original substrate thickness or unverified polarization convention"})
    print(calibration / "metrics.json")


if __name__ == "__main__":
    main()
