"""Evaluate a trained forward surrogate with resonance-aware diagnostics.

Run from the repository root.  The evaluation never writes into a training
directory, and all targets are unnormalised with the subset's train-only stats.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import SUTDPRCMDataset
from src.forward_analysis import finite_summary, geometry_complexity, resonance_errors
from src.metrics import aggregate_forward_metrics, per_sample_forward_metrics, unnormalize_response
from src.models import build_forward_model


def set_seed(seed: int) -> None:
    """Seed the randomness used only for deterministic plotting and diagnostics."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return device


def load_predictions(
    subset_root: Path, checkpoint_path: Path, batch_size: int, device: torch.device, normalization_root: Path, model_name: str
) -> tuple[SUTDPRCMDataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load test predictions and return normalized/raw targets plus geometries."""
    dataset = SUTDPRCMDataset(subset_root, "test", normalize_response=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_forward_model(model_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stats = np.load(normalization_root / "train_response_stats.npz")
    mean = torch.from_numpy(stats["mean"]).to(device)
    std = torch.from_numpy(stats["std"]).to(device)
    normalized_predictions: list[np.ndarray] = []
    normalized_targets: list[np.ndarray] = []
    geometries: list[np.ndarray] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for first in range(0, len(dataset), batch_size):
            batch = [dataset[index] for index in range(first, min(first + batch_size, len(dataset)))]
            geometry = torch.stack([item[0] for item in batch]).to(device)
            raw_target = torch.stack([item[1] for item in batch]).to(device)
            target = (raw_target - mean) / std
            prediction = model(geometry)
            normalized_predictions.append(prediction.cpu().numpy())
            normalized_targets.append(target.cpu().numpy())
            geometries.append(geometry.cpu().numpy())
    elapsed = time.perf_counter() - start
    normalized_prediction = np.concatenate(normalized_predictions)
    normalized_target = np.concatenate(normalized_targets)
    raw_prediction = unnormalize_response(torch.from_numpy(normalized_prediction), mean.cpu(), std.cpu()).numpy()
    raw_target = np.stack(
        [np.asarray(dataset.responses[int(index)], dtype=np.float32) for index in dataset.indices]
    )
    return dataset, normalized_prediction, normalized_target, raw_prediction, raw_target, elapsed


def save_prediction_grid(
    path: Path,
    dataset: SUTDPRCMDataset,
    geometries: np.ndarray,
    predicted: np.ndarray,
    target: np.ndarray,
    indexes: np.ndarray,
    errors: np.ndarray,
    title: str,
    error_label: str = "normalized MSE",
) -> None:
    """Plot geometry, y-cross magnitude, and x-co magnitude for selected rows."""
    figure, axes = plt.subplots(len(indexes), 3, figsize=(13, max(3.2, 2.2 * len(indexes))))
    axes = np.atleast_2d(axes)
    frequency = dataset.frequency_ghz
    for row, index in enumerate(indexes):
        geometry_axis, y_axis, x_axis = axes[row]
        geometry_axis.imshow(geometries[index, 0], cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        geometry_axis.set_title(f"{dataset.source_id(int(index))}\n{error_label}={errors[index]:.4f}", fontsize=7)
        geometry_axis.set_xticks([])
        geometry_axis.set_yticks([])
        for axis, components, label in ((y_axis, (0, 1), "y-cross |T| reflection"), (x_axis, (2, 3), "x-co |R| reflection")):
            true_magnitude = np.hypot(target[index, components[0]], target[index, components[1]])
            predicted_magnitude = np.hypot(predicted[index, components[0]], predicted[index, components[1]])
            axis.plot(frequency, true_magnitude, color="black", linewidth=0.9, label="true")
            axis.plot(frequency, predicted_magnitude, color="#d95f02", linewidth=0.8, linestyle="--", label="predicted")
            axis.set_ylabel("magnitude", fontsize=7)
            axis.set_xlabel("frequency (GHz)", fontsize=7)
            axis.set_title(label, fontsize=8)
            axis.tick_params(labelsize=6)
            if row == 0:
                axis.legend(fontsize=6, loc="best")
    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_complexity_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    """Visualize direct geometry-descriptor/error associations without fitting a model."""
    descriptors = ("fill_ratio", "connected_components_4", "boundary_transitions_4")
    error = np.asarray([float(row["normalized_mse"]) for row in rows])
    figure, axes = plt.subplots(1, len(descriptors), figsize=(13, 3.8))
    for axis, descriptor in zip(axes, descriptors):
        values = np.asarray([float(row[descriptor]) for row in rows])
        correlation = np.corrcoef(values, error)[0, 1] if np.std(values) > 0 else float("nan")
        axis.scatter(values, error, s=13, alpha=0.55)
        axis.set_xlabel(descriptor.replace("_", " "))
        axis.set_ylabel("normalized MSE")
        axis.set_title(f"Pearson r={correlation:.3f}")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def gradient_sanity(
    checkpoint_path: Path,
    normalized_target: np.ndarray,
    geometry: np.ndarray,
    device: torch.device,
    output_path: Path,
    model_name: str,
) -> dict[str, float | bool]:
    """Backpropagate a response-MSE through a continuous [0,1] geometry."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_forward_model(model_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    continuous_geometry = torch.from_numpy(geometry[None].astype(np.float32)).to(device).requires_grad_(True)
    target = torch.from_numpy(normalized_target[None].astype(np.float32)).to(device)
    loss = torch.mean(torch.square(model(continuous_geometry) - target))
    loss.backward()
    gradient = continuous_geometry.grad.detach().cpu().numpy()[0, 0]
    absolute_gradient = np.abs(gradient)
    figure, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    axes[0].imshow(geometry[0], cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("continuous-relaxation start")
    image = axes[1].imshow(absolute_gradient, cmap="magma", interpolation="nearest")
    axes[1].set_title("|d response-MSE / dG|")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(image, ax=axes[1], fraction=0.046)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return {
        "loss": float(loss.item()), "all_finite": bool(np.isfinite(gradient).all()),
        "nonzero_fraction": float(np.mean(absolute_gradient > 1e-12)), "abs_mean": float(absolute_gradient.mean()),
        "abs_median": float(np.median(absolute_gradient)), "abs_p99": float(np.percentile(absolute_gradient, 99)),
        "abs_max": float(absolute_gradient.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/phase2_forward_75ep/best.pt"))
    parser.add_argument("--normalization-root", type=Path, default=None, help="Subset whose train-only response statistics were used to train the checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase2_forward_evaluation"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a PyTorch device string")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None, choices=("ForwardSurrogateCNN", "ResponseAwareSurrogateCNN"), help="Override the model name stored in the checkpoint")
    parser.add_argument("--resonance-prominence", type=float, default=0.03)
    parser.add_argument("--resonance-distance-points", type=int, default=10)
    parser.add_argument("--resonance-window-ghz", type=float, default=0.10)
    parser.add_argument("--plot-count", type=int, default=20)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.plot_count <= 0 or args.resonance_window_ghz <= 0:
        raise ValueError("Batch size, plot count, and resonance window must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    checkpoint_metadata = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = args.model or checkpoint_metadata.get("args", {}).get("model", "ForwardSurrogateCNN")
    normalization_root = args.normalization_root or args.subset_root
    if not (normalization_root / "train_response_stats.npz").is_file():
        raise FileNotFoundError(f"Normalization statistics not found beneath {normalization_root}")
    set_seed(args.seed)
    device = resolve_device(args.device)
    plots = args.output_dir / "plots"
    failures = plots / "failures"
    failures.mkdir(parents=True, exist_ok=True)

    dataset, prediction_n, target_n, prediction, target, inference_seconds = load_predictions(
        args.subset_root, args.checkpoint, args.batch_size, device, normalization_root, model_name
    )
    per_tensor_metrics = per_sample_forward_metrics(
        torch.from_numpy(prediction_n), torch.from_numpy(target_n), torch.from_numpy(prediction), torch.from_numpy(target)
    )
    aggregate = aggregate_forward_metrics(per_tensor_metrics)
    error_arrays = {name: values.numpy() for name, values in per_tensor_metrics.items()}
    rows: list[dict[str, Any]] = []
    all_resonance_errors: list[float] = []
    all_region_errors: list[float] = []
    by_polarization: dict[str, list[float]] = {"y_cross_reflection": [], "x_co_reflection": []}
    total_features = total_matched = 0
    for index in range(len(dataset)):
        row: dict[str, Any] = {"source_id": dataset.source_id(index)}
        row.update({name: float(values[index]) for name, values in error_arrays.items()})
        row.update(geometry_complexity(np.asarray(dataset.geometries[int(dataset.indices[index])])))
        channel_errors: list[float] = []
        channel_region_errors: list[float] = []
        feature_count = matched_count = 0
        for key, components in (("y_cross_reflection", (0, 1)), ("x_co_reflection", (2, 3))):
            detail = resonance_errors(
                np.hypot(target[index, components[0]], target[index, components[1]]),
                np.hypot(prediction[index, components[0]], prediction[index, components[1]]), dataset.frequency_ghz,
                args.resonance_prominence, args.resonance_distance_points, args.resonance_window_ghz,
            )
            errors = detail["frequency_errors_ghz"]
            channel_errors.extend(errors)
            by_polarization[key].extend(errors)
            channel_region_errors.append(float(detail["resonance_region_magnitude_mae"]))
            feature_count += int(detail["true_feature_count"])
            matched_count += int(detail["matched_feature_count"])
        total_features += feature_count
        total_matched += matched_count
        all_resonance_errors.extend(channel_errors)
        local_region = np.asarray(channel_region_errors, dtype=float)
        row["resonance_feature_count"] = feature_count
        row["matched_resonance_feature_count"] = matched_count
        row["resonance_frequency_error_ghz"] = float(np.mean(channel_errors)) if channel_errors else float("nan")
        row["resonance_region_magnitude_mae"] = float(np.nanmean(local_region)) if np.isfinite(local_region).any() else float("nan")
        row["magnitude_mae"] = float(np.mean([row["y_cross_reflection_magnitude_mae"], row["x_co_reflection_magnitude_mae"]]))
        all_region_errors.append(row["resonance_region_magnitude_mae"])
        rows.append(row)

    resonance_summary: dict[str, Any] = {
        "method": {
            "features": "SciPy find_peaks on magnitude and negative magnitude (peaks and dips)",
            "prominence": args.resonance_prominence, "distance_points": args.resonance_distance_points,
            "frequency_spacing_ghz": float(dataset.frequency_ghz[1] - dataset.frequency_ghz[0]),
            "window_ghz": args.resonance_window_ghz,
            "matching": "nearest predicted prominent extremum of the same kind; frequency error summary is conditional on a match",
        },
        "true_feature_count": total_features, "matched_feature_count": total_matched,
        "feature_match_rate": total_matched / total_features if total_features else None,
        "resonance_frequency_error_ghz": finite_summary(all_resonance_errors),
        "resonance_region_magnitude_mae": finite_summary(all_region_errors),
        "by_polarization": {key: {"frequency_error_ghz": finite_summary(values)} for key, values in by_polarization.items()},
    }
    aggregate["resonance_frequency_error_ghz"] = resonance_summary["resonance_frequency_error_ghz"]["mean"]
    aggregate["resonance_region_magnitude_mae"] = resonance_summary["resonance_region_magnitude_mae"]["mean"]
    aggregate["resonance_feature_match_rate"] = resonance_summary["feature_match_rate"]
    aggregate["test_samples"] = len(dataset)
    aggregate["inference_seconds"] = inference_seconds
    aggregate["inference_milliseconds_per_sample"] = inference_seconds * 1000 / len(dataset)
    training_metadata = json.loads((args.checkpoint.parent / "training_metadata.json").read_text(encoding="utf-8")) if (args.checkpoint.parent / "training_metadata.json").is_file() else {}
    if "training_seconds" in training_metadata:
        aggregate["training_seconds"] = float(training_metadata["training_seconds"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (args.output_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    ranked = sorted(rows, key=lambda row: float(row["normalized_mse"]))
    for filename, selection in (("easiest_samples.csv", ranked[:10]), ("hardest_samples.csv", ranked[-10:][::-1])):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selection)
    (args.output_dir / "metrics.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "resonance_metrics.json").write_text(json.dumps(resonance_summary, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.output_dir / "prediction_cache.npz", prediction=prediction, target=target,
        normalized_prediction=prediction_n, normalized_target=target_n, geometries=np.asarray([dataset.geometries[int(i)] for i in dataset.indices]),
    )
    normalized_mse = error_arrays["normalized_mse"]
    magnitude_mae = np.asarray([float(row["magnitude_mae"]) for row in rows])
    resonance_frequency_error = np.asarray([
        float(row["resonance_frequency_error_ghz"]) if np.isfinite(float(row["resonance_frequency_error_ghz"])) else -np.inf
        for row in rows
    ])
    generator = np.random.default_rng(args.seed)
    random_indexes = generator.choice(len(dataset), size=min(args.plot_count, len(dataset)), replace=False)
    worst_indexes = np.argsort(normalized_mse)[-min(args.plot_count, len(dataset)):][::-1]
    worst_magnitude_indexes = np.argsort(magnitude_mae)[-min(args.plot_count, len(dataset)):][::-1]
    finite_resonance_indexes = np.flatnonzero(np.isfinite(resonance_frequency_error) & (resonance_frequency_error >= 0))
    worst_resonance_indexes = finite_resonance_indexes[np.argsort(resonance_frequency_error[finite_resonance_indexes])[-min(args.plot_count, len(finite_resonance_indexes)):][::-1]] if len(finite_resonance_indexes) else worst_indexes
    geometries = np.asarray([dataset.geometries[int(i)] for i in dataset.indices])
    save_prediction_grid(plots / "random_predictions.png", dataset, geometries, prediction, target, random_indexes, normalized_mse, "Random deterministic test predictions")
    save_prediction_grid(failures / "worst_predictions.png", dataset, geometries, prediction, target, worst_indexes, normalized_mse, "Worst test predictions ranked by normalized MSE")
    save_prediction_grid(failures / "worst_magnitude_mae.png", dataset, geometries, prediction, target, worst_magnitude_indexes, magnitude_mae, "Worst test predictions ranked by magnitude MAE", "magnitude MAE")
    save_prediction_grid(failures / "worst_resonance_frequency_error.png", dataset, geometries, prediction, target, worst_resonance_indexes, resonance_frequency_error, "Worst test predictions ranked by resonance-frequency error", "resonance error (GHz)")
    save_complexity_plot(plots / "error_vs_geometry_complexity.png", rows)
    gradient = gradient_sanity(args.checkpoint, target_n[0], geometries[0], device, plots / "gradient_sanity.png", model_name)
    metadata = {
        "subset_root": str(args.subset_root), "normalization_root": str(normalization_root), "checkpoint": str(args.checkpoint), "checkpoint_epoch": int(torch.load(args.checkpoint, map_location="cpu", weights_only=False).get("epoch", -1)),
        "seed": args.seed, "device": str(device), "batch_size": args.batch_size, "split": "test",
        "model": model_name, "model_parameter_count": sum(parameter.numel() for parameter in build_forward_model(model_name).parameters()),
        "training_metadata": training_metadata,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
        "gradient_sanity": gradient,
    }
    (args.output_dir / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": aggregate, "resonance": resonance_summary["resonance_frequency_error_ghz"], "gradient": gradient}, indent=2))


if __name__ == "__main__":
    main()
