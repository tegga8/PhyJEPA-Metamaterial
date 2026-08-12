"""Stage B/C of Phase 6.1: full-spectrum substrate and stored-data calibration."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import SUTDPRCMDataset
from src.rcwa_solver import RCWAConfig, frequency_vector
from src.rcwa_validation import (
    MAPPING_CANDIDATES,
    PHYSICAL_MAPPING,
    aggregate_metric_rows,
    cached_solve,
    channel_frequency_wise_normalized_mse,
    frequency_wise_normalized_mse,
    pack_modes,
    response_metrics,
    save_json,
)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 9:
        raise ValueError(f"Phase 6.1 requires exactly 9 manifest rows, found {len(rows)}")
    if {row["complexity_group"] for row in rows} != {"simple", "medium", "complex"}:
        raise ValueError("Manifest must contain all three complexity groups")
    if len({row["source_id"] for row in rows}) != len(rows):
        raise ValueError("Manifest source IDs must be unique")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase6_1"))
    parser.add_argument("--thicknesses", default="0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--allow-unconverged-diagnostic", action="store_true")
    args = parser.parse_args()

    config_path = args.output_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError("Run scripts/analyze_rcwa_convergence.py first")
    project_config = json.loads(config_path.read_text(encoding="utf-8"))
    convergence = project_config.get("convergence", {})
    selected_order = convergence.get("selected_order")
    if selected_order is None and not args.allow_unconverged_diagnostic:
        raise RuntimeError("NO CONVERGENCE ESTABLISHED: refusing physical calibration without a converged order")
    order = int(selected_order if selected_order is not None else convergence["orders"][-1])
    diagnostic_only = selected_order is None
    manifest = read_manifest(args.output_dir / "validation_geometry_manifest.csv")
    dataset = SUTDPRCMDataset(args.subset_root, "test", normalize_response=False)
    stats = np.load(args.subset_root / "train_response_stats.npz")
    mean, std = stats["mean"], stats["std"]
    frequencies = frequency_vector()
    thicknesses = [float(value) for value in args.thicknesses.split(",")]
    if thicknesses != [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        raise ValueError("The required Phase 6.1 substrate sweep is 0.10,0.15,0.20,0.25,0.30,0.40,0.50 mm")

    calibration = args.output_dir / "calibration"
    plots = calibration / "plots"
    calibration.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    frequency_error: dict[float, list[np.ndarray]] = {thickness: [] for thickness in thicknesses}
    channel_frequency_error: dict[float, list[np.ndarray]] = {thickness: [] for thickness in thicknesses}

    for thickness in thicknesses:
        config = RCWAConfig(substrate_thickness_mm=thickness, fourier_order=order, device=args.device, cpu_workers=args.cpu_workers)
        for item in manifest:
            local_index = int(item["test_index"])
            geometry, target = dataset[local_index]
            result = cached_solve(geometry.numpy(), frequencies, config, args.output_dir / "cache")
            physical_raw = pack_modes(result, PHYSICAL_MAPPING)
            metrics = response_metrics(physical_raw, target.numpy(), mean, std)
            row = {
                "source_id": item["source_id"],
                "test_index": local_index,
                "complexity_group": item["complexity_group"],
                "complexity_score": float(item["complexity_score"]),
                "substrate_thickness_mm": thickness,
                "fourier_order": order,
                "channel_mapping": PHYSICAL_MAPPING,
                "runtime_seconds": float(result.metadata["runtime_seconds"]),
                "cache_hit": bool(result.metadata.get("cache_hit", False)),
                **metrics,
            }
            rows.append(row)
            frequency_error[thickness].append(frequency_wise_normalized_mse(physical_raw, target.numpy(), mean, std))
            channel_frequency_error[thickness].append(channel_frequency_wise_normalized_mse(physical_raw, target.numpy(), mean, std))
            # Mode mappings are comparison diagnostics.  They are evaluated on
            # the exact same RCWA response but never substitute for the
            # physically documented mapping above.
            for mapping in MAPPING_CANDIDATES:
                mapping_rows.append({
                    "source_id": item["source_id"],
                    "substrate_thickness_mm": thickness,
                    "channel_mapping": mapping,
                    **response_metrics(pack_modes(result, mapping), target.numpy(), mean, std),
                })

    write_csv(calibration / "per_sample_metrics.csv", rows)
    write_csv(calibration / "polarization_mapping_diagnostics.csv", mapping_rows)
    sweep_rows: list[dict[str, object]] = []
    for thickness in thicknesses:
        selected = [row for row in rows if row["substrate_thickness_mm"] == thickness]
        aggregate = aggregate_metric_rows(selected)
        sweep_rows.append({
            "substrate_thickness_mm": thickness,
            "sample_count": len(selected),
            "mean_normalized_mse": aggregate["normalized_mse"],
            "median_normalized_mse": float(np.median([float(row["normalized_mse"]) for row in selected])),
            "mean_Re(Ty)_normalized_mse": aggregate["Re(Ty)_normalized_mse"],
            "mean_Im(Ty)_normalized_mse": aggregate["Im(Ty)_normalized_mse"],
            "mean_Re(Rx)_normalized_mse": aggregate["Re(Rx)_normalized_mse"],
            "mean_Im(Rx)_normalized_mse": aggregate["Im(Rx)_normalized_mse"],
            "mean_Ty_magnitude_mae": aggregate["Ty_magnitude_mae"],
            "mean_Rx_magnitude_mae": aggregate["Rx_magnitude_mae"],
        })
    write_csv(calibration / "substrate_sweep.csv", sweep_rows)
    best = min(sweep_rows, key=lambda row: float(row["mean_normalized_mse"]))
    tolerance = float(best["mean_normalized_mse"]) * 1.05
    near_optimal = [float(row["substrate_thickness_mm"]) for row in sweep_rows if float(row["mean_normalized_mse"]) <= tolerance]
    empirical_by_mapping = []
    for mapping in MAPPING_CANDIDATES:
        selected = [row for row in mapping_rows if row["channel_mapping"] == mapping and row["substrate_thickness_mm"] == best["substrate_thickness_mm"]]
        empirical_by_mapping.append({"channel_mapping": mapping, "mean_normalized_mse": float(np.mean([float(row["normalized_mse"]) for row in selected]))})
    empirical_best = min(empirical_by_mapping, key=lambda row: float(row["mean_normalized_mse"]))

    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.plot(thicknesses, [float(row["mean_normalized_mse"]) for row in sweep_rows], marker="o", label="mean normalized MSE")
    axis.axvline(float(best["substrate_thickness_mm"]), color="black", linestyle="--", linewidth=.8, label="minimum agreement")
    axis.set(xlabel="substrate thickness (mm)", ylabel="RCWA vs stored normalized MSE", title="Full-spectrum substrate calibration")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(calibration / "substrate_sweep.png", dpi=180)
    plt.close(figure)
    for thickness in thicknesses:
        np.savez_compressed(
            calibration / f"frequency_wise_error_{thickness:.2f}mm.npz",
            frequency_ghz=frequencies,
            normalized_mse=np.mean(frequency_error[thickness], axis=0),
            channel_normalized_squared_error=np.mean(channel_frequency_error[thickness], axis=0),
        )

    summary = {
        "frequency_ghz": frequencies,
        "frequency_points": len(frequencies),
        "sample_count": len(manifest),
        "fourier_order": order,
        "diagnostic_only_unconverged": diagnostic_only,
        "physical_mapping": PHYSICAL_MAPPING,
        "empirical_best_mapping_for_comparison": empirical_best,
        "physical_and_empirical_mapping_agree": empirical_best["channel_mapping"] == PHYSICAL_MAPPING,
        "thickness_selection": {
            "selected_for_agreement_mm": best["substrate_thickness_mm"],
            "best_row": best,
            "within_five_percent_of_minimum_mm": near_optimal,
            "interpretation": "Agreement-selected calibration parameter, not a claim of the original CST substrate thickness.",
        },
        "sweep": sweep_rows,
    }
    save_json(calibration / "aggregate_metrics.json", summary)
    project_config.update({
        "selected_substrate_thickness_mm": best["substrate_thickness_mm"],
        "selected_fourier_order": order if not diagnostic_only else None,
        "physical_channel_mapping": PHYSICAL_MAPPING,
        "empirical_best_mapping_for_comparison": empirical_best,
        "calibration_diagnostic_only_unconverged": diagnostic_only,
    })
    save_json(config_path, project_config)
    print(calibration / "aggregate_metrics.json")


if __name__ == "__main__":
    main()
