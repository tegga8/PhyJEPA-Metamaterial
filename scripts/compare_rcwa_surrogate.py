"""Stage C of Phase 6.1: full-spectrum stored dataset, CNN, and RCWA comparison."""
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import SUTDPRCMDataset
from src.forward_analysis import resonance_errors
from src.physics_consistency import load_frozen_forward_surrogate
from src.rcwa_solver import RCWAConfig, frequency_vector
from src.rcwa_validation import (
    PHYSICAL_MAPPING,
    aggregate_metric_rows,
    cached_solve,
    channel_frequency_wise_normalized_mse,
    pack_modes,
    response_metrics,
    save_json,
)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_full_comparison(path: Path, frequency: np.ndarray, target: np.ndarray, rcwa: np.ndarray, cnn: np.ndarray, title: str) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True)
    channels = ((0, "Re(Ty)"), (1, "Im(Ty)"), (2, "Re(Rx)"), (3, "Im(Rx)"))
    for axis, (channel, label) in zip(axes[:2].flat, channels):
        axis.plot(frequency, target[channel], color="black", label="stored dataset", linewidth=1.1)
        axis.plot(frequency, rcwa[channel], color="#1b9e77", label="RCWA", linewidth=.9)
        axis.plot(frequency, cnn[channel], color="#d95f02", label="CNN", linewidth=.9)
        axis.set_title(label)
        axis.legend(fontsize=7)
    for axis, start, label in ((axes[2, 0], 0, "|Ty|"), (axes[2, 1], 2, "|Rx|")):
        axis.plot(frequency, np.hypot(target[start], target[start + 1]), color="black", label="stored dataset")
        axis.plot(frequency, np.hypot(rcwa[start], rcwa[start + 1]), color="#1b9e77", label="RCWA")
        axis.plot(frequency, np.hypot(cnn[start], cnn[start + 1]), color="#d95f02", label="CNN")
        axis.set_title(label)
        axis.legend(fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("frequency (GHz)")
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, .97))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_error(path: Path, frequency: np.ndarray, target: np.ndarray, rcwa: np.ndarray, cnn: np.ndarray, mean: np.ndarray, std: np.ndarray, title: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 3.5))
    for name, left, right, color in (
        ("|CNN - dataset|", cnn, target, "#d95f02"),
        ("|RCWA - dataset|", rcwa, target, "#1b9e77"),
        ("|CNN - RCWA|", cnn, rcwa, "#7570b3"),
    ):
        axis.plot(frequency, np.sqrt(np.square((left - right) / std).mean(axis=0)), label=name, color=color)
    axis.set(xlabel="frequency (GHz)", ylabel="normalized RMSE", title=title)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/phase2_5/exp_A_5k_mse/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase6_1"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--allow-unconverged-diagnostic", action="store_true")
    args = parser.parse_args()

    project_config = json.loads((args.output_dir / "config.json").read_text(encoding="utf-8"))
    if project_config.get("selected_fourier_order") is None and not args.allow_unconverged_diagnostic:
        raise RuntimeError("NO CONVERGENCE ESTABLISHED: refusing a production CNN/RCWA comparison")
    order = int(project_config.get("selected_fourier_order") or project_config["convergence"]["orders"][-1])
    thickness = float(project_config["selected_substrate_thickness_mm"])
    diagnostic_only = project_config.get("selected_fourier_order") is None
    manifest = read_manifest(args.output_dir / "validation_geometry_manifest.csv")
    dataset = SUTDPRCMDataset(args.subset_root, "test", normalize_response=False)
    stats = np.load(args.subset_root / "train_response_stats.npz")
    mean, std = stats["mean"], stats["std"]
    frequencies = frequency_vector()
    model_device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    cnn, model_name = load_frozen_forward_surrogate(args.checkpoint, model_device)

    comparison = args.output_dir / "comparisons"
    plots = comparison / "plots"
    comparison.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    grouped_metrics = {"cnn_vs_dataset": [], "rcwa_vs_dataset": [], "cnn_vs_rcwa": []}
    frequency_errors = {key: [] for key in grouped_metrics}
    resonance_rows: list[dict[str, object]] = []
    representatives: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, str]] = {}
    solver_config = RCWAConfig(substrate_thickness_mm=thickness, fourier_order=order, device=args.device, cpu_workers=args.cpu_workers)
    for item in manifest:
        local_index = int(item["test_index"])
        geometry, target = dataset[local_index]
        target_raw = target.numpy()
        with torch.inference_mode():
            cnn_raw = (cnn(geometry.unsqueeze(0).to(model_device)).cpu().numpy()[0] * std + mean)
        result = cached_solve(geometry.numpy(), frequencies, solver_config, args.output_dir / "cache")
        rcwa_raw = pack_modes(result, PHYSICAL_MAPPING)
        metric_sets = {
            "cnn_vs_dataset": response_metrics(cnn_raw, target_raw, mean, std),
            "rcwa_vs_dataset": response_metrics(rcwa_raw, target_raw, mean, std),
            "cnn_vs_rcwa": response_metrics(cnn_raw, rcwa_raw, mean, std),
        }
        rows.append({
            "source_id": item["source_id"], "test_index": local_index, "complexity_group": item["complexity_group"],
            "complexity_score": float(item["complexity_score"]), "fourier_order": order,
            "substrate_thickness_mm": thickness, "physical_channel_mapping": PHYSICAL_MAPPING,
            **{f"{name}_{key}": value for name, values in metric_sets.items() for key, value in values.items()},
        })
        for name, values in metric_sets.items():
            grouped_metrics[name].append(values)
        frequency_errors["cnn_vs_dataset"].append(channel_frequency_wise_normalized_mse(cnn_raw, target_raw, mean, std))
        frequency_errors["rcwa_vs_dataset"].append(channel_frequency_wise_normalized_mse(rcwa_raw, target_raw, mean, std))
        frequency_errors["cnn_vs_rcwa"].append(channel_frequency_wise_normalized_mse(cnn_raw, rcwa_raw, mean, std))
        for response_name, candidate in (("rcwa_vs_dataset", rcwa_raw), ("cnn_vs_dataset", cnn_raw), ("cnn_vs_rcwa", cnn_raw)):
            reference = target_raw if response_name != "cnn_vs_rcwa" else rcwa_raw
            for start, component in ((0, "Ty"), (2, "Rx")):
                detail = resonance_errors(np.hypot(reference[start], reference[start + 1]), np.hypot(candidate[start], candidate[start + 1]), frequencies)
                for frequency_error in detail["frequency_errors_ghz"]:
                    resonance_rows.append({"source_id": item["source_id"], "comparison": response_name, "component": component, "frequency_error_ghz": frequency_error, "resonance_region_magnitude_mae": detail["resonance_region_magnitude_mae"]})
        representatives.setdefault(item["complexity_group"], (target_raw, rcwa_raw, cnn_raw, item["source_id"]))

    write_csv(comparison / "per_sample_metrics.csv", rows)
    for name in grouped_metrics:
        write_csv(comparison / f"{name}.csv", [{"source_id": row["source_id"], "complexity_group": row["complexity_group"], **{key[len(name)+1:]: value for key, value in row.items() if key.startswith(name + "_")}} for row in rows])
        np.savez_compressed(comparison / f"{name}_frequency_wise_error.npz", frequency_ghz=frequencies, channel_normalized_squared_error=np.mean(frequency_errors[name], axis=0))
    if resonance_rows:
        write_csv(comparison / "resonance_metrics.csv", resonance_rows)
    aggregate = {name: aggregate_metric_rows(values) for name, values in grouped_metrics.items()}
    complexity_summary = []
    for group in ("simple", "medium", "complex"):
        selected = [row for row in rows if row["complexity_group"] == group]
        complexity_summary.append({
            "complexity_group": group,
            "sample_count": len(selected),
            "mean_cnn_vs_rcwa_normalized_mse": float(np.mean([row["cnn_vs_rcwa_normalized_mse"] for row in selected])),
            "mean_rcwa_vs_dataset_normalized_mse": float(np.mean([row["rcwa_vs_dataset_normalized_mse"] for row in selected])),
            "mean_cnn_vs_dataset_normalized_mse": float(np.mean([row["cnn_vs_dataset_normalized_mse"] for row in selected])),
        })
    write_csv(comparison / "complexity_summary.csv", complexity_summary)
    save_json(comparison / "aggregate_metrics.json", {
        "frequency_ghz": frequencies, "frequency_points": len(frequencies), "sample_count": len(rows),
        "cnn_model": model_name, "cnn_device": str(model_device), "fourier_order": order,
        "substrate_thickness_mm": thickness, "physical_channel_mapping": PHYSICAL_MAPPING,
        "diagnostic_only_unconverged": diagnostic_only, "aggregate": aggregate,
    })
    for group, (target_raw, rcwa_raw, cnn_raw, source_id) in representatives.items():
        plot_full_comparison(plots / f"{group}_full_spectrum.png", frequencies, target_raw, rcwa_raw, cnn_raw, f"{group}: {source_id}")
        plot_error(plots / f"{group}_frequency_error.png", frequencies, target_raw, rcwa_raw, cnn_raw, mean, std, f"{group}: {source_id}")
    figure, axis = plt.subplots(figsize=(7, 3.5))
    for name, color in (("cnn_vs_dataset", "#d95f02"), ("rcwa_vs_dataset", "#1b9e77"), ("cnn_vs_rcwa", "#7570b3")):
        axis.plot(frequencies, np.mean(frequency_errors[name], axis=(0, 1)), label=name.replace("_", " "), color=color)
    axis.set(xlabel="frequency (GHz)", ylabel="normalized MSE", title="Nine-geometry full-spectrum error")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plots / "frequency_wise_error.png", dpi=180)
    plt.close(figure)
    print(comparison / "aggregate_metrics.json")


if __name__ == "__main__":
    main()
