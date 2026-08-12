"""Plotting helpers for the Physics-JEPA experiment (Agg backend)."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 180


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return path


def read_history(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: value for key, value in row.items()} for row in csv.DictReader(handle)]


def plot_loss_curve(history: Path, output: Path) -> Path:
    rows = read_history(history)
    epochs = [float(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(figsize=(7, 4.5))
    axes.plot(epochs, [float(row["train_total_loss"]) for row in rows], label="train total")
    axes.plot(epochs, [float(row["val_total_loss"]) for row in rows], label="val total")
    axes.plot(epochs, [float(row["val_cross_loss"]) for row in rows], label="val cross (geometry->physics)")
    axes.set_xlabel("epoch")
    axes.set_ylabel("JEPA loss")
    axes.set_title("Physics-JEPA training")
    axes.legend()
    axes.grid(alpha=0.3)
    return _save(figure, output)


def plot_variance_diagnostics(history: Path, output: Path) -> Path:
    rows = read_history(history)
    epochs = [float(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(figsize=(7, 4.5))
    for key, label in (
        ("val_context_mean_std", "geometry latent std"),
        ("val_online_mean_std", "online spectrum std"),
        ("val_target_mean_std", "target spectrum std"),
        ("val_pred_mean_std", "predicted latent std"),
    ):
        axes.plot(epochs, [float(row[key]) for row in rows], label=label)
    axes.set_xlabel("epoch")
    axes.set_ylabel("mean per-dimension std")
    axes.set_title("Latent variance / collapse diagnostic")
    axes.legend()
    axes.grid(alpha=0.3)
    return _save(figure, output)


def plot_probe_response(predicted: np.ndarray, target: np.ndarray, baseline: np.ndarray, sample_index: int, output: Path) -> Path:
    figure, axes = plt.subplots(figsize=(8, 4))
    frequency = np.arange(1001)
    channel = 0
    axes.plot(frequency, target[sample_index, channel], label="true (normalized)", color="black", linewidth=1)
    axes.plot(frequency, predicted[sample_index, channel], label="linear probe", color="tab:red", linewidth=1)
    axes.plot(frequency, baseline[sample_index, channel], label="mean baseline", color="gray", linestyle="--", linewidth=1)
    axes.set_xlabel("frequency index")
    axes.set_ylabel("normalized response")
    axes.set_title(f"Response probe, sample {sample_index}, channel {channel}")
    axes.legend()
    axes.grid(alpha=0.3)
    return _save(figure, output)


def plot_resonance_probe(metrics: dict[str, float], output: Path) -> Path:
    labels = [key for key in metrics if key.endswith("_frequency_mae_ghz")]
    values = [metrics[key] for key in labels]
    figure, axes = plt.subplots(figsize=(7, 4))
    axes.bar(labels, values, color="tab:blue")
    axes.set_ylabel("frequency MAE (GHz)")
    axes.set_title("Resonance probe: predicted vs detected resonance frequencies")
    axes.grid(alpha=0.3, axis="y")
    return _save(figure, output)


def plot_distance_correlation(
    d_latent: np.ndarray,
    d_response: np.ndarray,
    d_geometry: np.ndarray,
    spearman_em: float,
    spearman_geometry: float,
    output: Path,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(d_response, d_latent, s=2, alpha=0.25, rasterized=True)
    axes[0].set_xlabel("EM response distance D(S_i, S_j)")
    axes[0].set_ylabel("latent distance D(z_i, z_j)")
    axes[0].set_title(f"latent vs EM (spearman {spearman_em:.3f})")
    axes[1].scatter(d_geometry, d_latent, s=2, alpha=0.25, rasterized=True)
    axes[1].set_xlabel("geometry distance D(G_i, G_j)")
    axes[1].set_ylabel("latent distance D(z_i, z_j)")
    axes[1].set_title(f"latent vs geometry (spearman {spearman_geometry:.3f})")
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.suptitle("Physics similarity: latent distance organization")
    return _save(figure, output)


def plot_correct_vs_shuffled(comparison: dict[str, float], output: Path) -> Path:
    figure, axes = plt.subplots(figsize=(7, 4.2))
    groups = [key for key in comparison if key.endswith("_latent_vs_response_spearman")]
    values = [comparison[key] for key in groups]
    axes.bar(groups, values, color=["tab:green", "tab:red", "tab:gray"])
    axes.set_ylabel("spearman rho(latent, EM distance)")
    axes.set_title("Correct-pair vs shuffled-pair training control")
    axes.grid(alpha=0.3, axis="y")
    return _save(figure, output)


def plot_within_cross_family(metrics: dict[str, float], output: Path) -> Path:
    figure, axes = plt.subplots(figsize=(7, 4.2))
    keys = [key for key in metrics if "latent_vs_response_spearman" in key and key.startswith(("within", "cross"))]
    axes.bar(keys, [metrics[key] for key in keys], color=["tab:blue", "tab:orange"])
    axes.set_ylabel("spearman rho(latent, EM distance)")
    axes.set_title("Within-family vs cross-family physics correlation")
    axes.grid(alpha=0.3, axis="y")
    return _save(figure, output)


def plot_latent_size_comparison(summary: dict[str, dict[str, float]], output: Path) -> Path:
    figure, axes = plt.subplots(figsize=(7, 4.2))
    sizes = sorted(summary.keys())
    metrics = ("latent_vs_response_spearman", "latent_vs_geometry_spearman", "response_r2_vs_mean")
    width = 0.25
    positions = np.arange(len(sizes))
    for offset, metric in enumerate(metrics):
        axes.bar(positions + offset * width, [summary[size][metric] for size in sizes], width=width, label=metric)
    axes.set_xticks(positions + width)
    axes.set_xticklabels([f"{size}-D" for size in sizes])
    axes.set_title("Latent size comparison (32-D vs 64-D)")
    axes.legend()
    axes.grid(alpha=0.3, axis="y")
    return _save(figure, output)


def plot_data_efficiency(percentages: list[float], values: dict[str, list[float]], output: Path) -> Path:
    figure, axes = plt.subplots(figsize=(7, 4.5))
    for label, series in values.items():
        axes.plot(percentages, series, marker="o", label=label)
    axes.set_xlabel("training data (% of full train split)")
    axes.set_ylabel("metric")
    axes.set_title("Data efficiency of Physics-JEPA representation")
    axes.legend()
    axes.grid(alpha=0.3)
    return _save(figure, output)


def plot_representative_pairs(
    geometries: np.ndarray,
    responses: np.ndarray,
    pairs: dict[str, np.ndarray],
    output_a: Path,
    output_b: Path,
) -> tuple[Path, Path]:
    return (
        _plot_pair_grid(geometries, responses, pairs, "Case A: different geometry, similar response", output_a),
        _plot_pair_grid(geometries, responses, pairs, "Case B: similar geometry, different response", output_b),
    )


def _plot_pair_grid(geometries: np.ndarray, responses: np.ndarray, pairs: dict[str, np.ndarray], title: str, output: Path) -> Path:
    count = min(3, len(pairs["first"]))
    figure, axes = plt.subplots(count, 3, figsize=(11, 3.2 * count))
    if count == 1:
        axes = axes[None, :]
    frequency = np.arange(1001)
    for row in range(count):
        first = int(pairs["first"][row])
        second = int(pairs["second"][row])
        axes[row, 0].imshow(geometries[first].squeeze(), cmap="gray_r", interpolation="nearest")
        axes[row, 0].set_title(f"geometry {first}")
        axes[row, 1].imshow(geometries[second].squeeze(), cmap="gray_r", interpolation="nearest")
        axes[row, 1].set_title(f"geometry {second}")
        axes[row, 2].plot(frequency, np.hypot(responses[first, 0], responses[first, 1]), color="tab:blue", alpha=0.8)
        axes[row, 2].plot(frequency, np.hypot(responses[second, 0], responses[second, 1]), color="tab:red", alpha=0.8)
        axes[row, 2].set_title(f"|Ty| response overlap (d_EM={float(pairs['d_response'][row]):.4f}, d_G={float(pairs['d_geometry'][row]):.4f})")
        for column in range(3):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 2].set_xticks([0, 500, 1000])
        axes[row, 2].set_yticks([])
    figure.suptitle(title)
    return _save(figure, output)
