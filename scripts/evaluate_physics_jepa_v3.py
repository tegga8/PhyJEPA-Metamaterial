"""Aggregate the Physics-JEPA v3 evaluation into final plots, metrics.json, summary.csv.

Assumes ``scripts/evaluate_physics_representation.py`` has already been run on
the correct and shuffled model directories (writing ``evaluation.json`` each),
and that the cached val/test latents exist in each model directory.  Uses
inputs collected from: correct evaluation, shuffled evaluation, latent health
computed from cached test latents, and regeneration of the v2 latent-health
figures from the frozen v2 latents for comparison.

Writes to ``outputs/physics_jepa_v3/``: ``metrics.json``, ``summary.csv``, and
the required plots under ``plots/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics_jepa_plots import _save, plot_loss_curve, plot_variance_diagnostics
from src.physics_representation_metrics import mine_representative_pairs, pair_distances, physics_similarity_evaluation

V2_DIR = Path("outputs/physics_jepa_v2/seed42_32d")


def load_cached_latents(directory: Path, split: str) -> dict[str, np.ndarray]:
    names = ("z_geometry", "z_online", "z_target", "z_pred", "geometry", "response")
    return {name: np.load(directory / f"{split}_{name}.npy") for name in names}


def load_evaluation(directory: Path) -> dict[str, object]:
    return json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))


def latency_structure(latent: np.ndarray) -> dict[str, float]:
    values = np.asarray(latent, dtype=np.float64)
    centrality = values - values.mean(axis=0, keepdims=True)
    covariance = centrality.T @ centrality / (values.shape[0] - 1)
    eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2.0)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = eigenvalues.sum()
    rank1 = float(eigenvalues[-1] / total) if total > 0 else 0.0
    effective = float(total**2 / np.square(eigenvalues).sum()) if np.square(eigenvalues).sum() > 0 else 0.0
    diagonal = np.diag(covariance)
    return {
        "rank1_fraction": rank1,
        "effective_rank": float(effective),
        "per_dim_variance_mean": float(diagonal.mean()),
        "per_dim_variance_min": float(diagonal.min()),
        "per_dim_variance_max": float(diagonal.max()),
        "covariance_offdiag_norm": float(np.sqrt(np.square(covariance - np.diag(diagonal)).sum()) / values.shape[1]),
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
    }


def latent_health(cached: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {name: latency_structure(cached[name]) for name in ("z_target", "z_pred", "z_online", "z_geometry")}


def plot_latent_health(health: dict[str, dict[str, float]], output: Path) -> Path:
    names = list(health)
    rank1 = [health[name]["rank1_fraction"] for name in names]
    effective = [health[name]["effective_rank"] for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, rank1, color="tab:blue")
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("rank-1 fraction (top eig / trace)")
    axes[0].set_title("Latent collapse: rank-1 fraction (lower = better)")
    axes[1].bar(names, effective, color="tab:orange")
    axes[1].set_ylabel("effective rank (participation ratio)")
    axes[1].set_title("Latent health: effective rank (32 maximum)")
    for axis in axes:
        axis.grid(alpha=0.3, axis="y")
        axis.tick_params(axis="x", rotation=15)
    figure.suptitle("Physics-JEPA v3 latent health (test split)")
    return _save(figure, output)


def plot_physics_vs_geometry(evaluation: dict[str, object], correct_health: dict[str, dict[str, float]], output: Path) -> Path:
    names = ("z_target", "z_pred", "z_geometry")
    em = [evaluation["similarity"][name]["latent_vs_response_spearman"] for name in names]
    geo = [evaluation["similarity"][name]["latent_vs_geometry_spearman"] for name in names]
    position = np.arange(len(names))
    width = 0.35
    figure, axes = plt.subplots(figsize=(8, 4.2))
    axes.bar(position - width / 2, em, width, label="rho(D_z, D_EM)", color="tab:blue")
    axes.bar(position + width / 2, geo, width, label="rho(D_z, D_G)", color="tab:red")
    axes.axhline(0.0, color="gray", linewidth=1)
    axes.set_xticks(position)
    axes.set_xticklabels(names)
    axes.set_ylabel("Spearman correlation")
    axes.set_title("Physics vs geometry distance correlation (v3 correct)")
    axes.legend()
    axes.grid(alpha=0.3, axis="y")
    return _save(figure, output)


def plot_neighborhood_examples(cached: dict[str, np.ndarray], evaluation: dict[str, object], output: Path) -> Path:
    """Three-row panel: Case A pair, Case B pair, and their latent distances."""
    pairs = evaluation["representative_pairs"]
    case_a = pairs["case_a"]
    case_b = pairs["case_b"]
    figure, axes = plt.subplots(3, 3, figsize=(11, 9))
    for row, (case, title) in enumerate(((case_a, "Case A: different geometry, similar response"), (case_b, "Case B: similar geometry, different response"))):
        first = int(case["first"][0])
        second = int(case["second"][0])
        axes[row, 0].imshow(cached["geometry"][first].squeeze(), cmap="gray_r", interpolation="nearest")
        axes[row, 0].set_title(f"geometry {first}")
        axes[row, 1].imshow(cached["geometry"][second].squeeze(), cmap="gray_r", interpolation="nearest")
        axes[row, 1].set_title(f"geometry {second}")
        frequency = np.arange(1001)
        axes[row, 2].plot(frequency, np.hypot(cached["response"][first, 0], cached["response"][first, 1]), color="tab:blue", alpha=0.8)
        axes[row, 2].plot(frequency, np.hypot(cached["response"][second, 0], cached["response"][second, 1]), color="tab:red", alpha=0.8)
        axes[row, 2].set_title(f"d_EM={float(case['d_response'][0]):.4f} d_G={float(case['d_geometry'][0]):.4f}")
        for column in range(3):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    cached_pred = cached["z_pred"]
    normalized = cached_pred / np.linalg.norm(cached_pred, axis=1, keepdims=True)
    d_z_a = float(np.linalg.norm(normalized[case_a["first"][0]] - normalized[case_a["second"][0]]))
    d_z_b = float(np.linalg.norm(normalized[case_b["first"][0]] - normalized[case_b["second"][0]]))
    figure.suptitle(f"Physics neighborhoods | z_pred: case A D_z={d_z_a:.3f} (want small), case B D_z={d_z_b:.3f} (want large)")
    return _save(figure, output)


def plot_probe_summary(evaluation: dict[str, object], output: Path) -> Path:
    names = ("z_target", "z_pred", "z_geometry")
    values = [evaluation["probes"][name]["response"]["response_r2_vs_mean"] for name in names]
    figure, axes = plt.subplots(figsize=(7, 4))
    axes.bar(names, values, color="tab:green")
    axes.set_ylabel("response probe R2 vs mean")
    axes.set_title("Linear response probe: information content of latents")
    axes.grid(alpha=0.3, axis="y")
    return _save(figure, output)


def plot_resonance_summary(evaluation: dict[str, object], output: Path) -> Path:
    names = ("z_target", "z_pred", "z_geometry")
    labels = ["ty_frequency_mae_ghz", "rx_frequency_mae_ghz", "feature_count_mae"]
    figure, axes = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(names))
    width = 0.25
    for offset, label in enumerate(labels):
        values = [evaluation["probes"][name]["resonance"].get(label, 0.0) for name in names]
        axes.bar(positions + (offset - 1) * width, values, width, label=label)
    axes.set_xticks(positions)
    axes.set_xticklabels(names)
    axes.set_ylabel("resonance probe error")
    axes.set_title("Resonance probe (lower = better)")
    axes.legend()
    axes.grid(alpha=0.3, axis="y")
    return _save(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/physics_jepa_v3"))
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--compare-dir", type=Path, default=None)
    parser.add_argument("--v2-dir", type=Path, default=V2_DIR)
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_30k"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-pairs", type=int, default=20000)
    args = parser.parse_args()

    root = args.root
    model_dir = args.model_dir or root / "models/correct"
    compare_dir = args.compare_dir or root / "models/shuffled"
    evaluation_dir = root / "evaluation"
    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    correct_eval = load_evaluation(evaluation_dir / "correct")
    shuffled_eval = load_evaluation(evaluation_dir / "shuffled")
    correct_cached = load_cached_latents(model_dir, "test")
    shuffled_cached = load_cached_latents(compare_dir, "test")

    correct_health = latent_health(correct_cached)
    shuffled_health = latent_health(shuffled_cached)
    v2_cached = load_cached_latents(args.v2_dir, "test")
    v2_health = latent_health(v2_cached)

    history = model_dir / "training_history.csv"
    if history.is_file():
        plot_loss_curve(history, plots_dir / "loss_curves.png")
        plot_variance_diagnostics(history, plots_dir / "latent_variance_diagnostics.png")
    plot_latent_health(correct_health, plots_dir / "latent_health.png")
    plot_physics_vs_geometry(correct_eval, correct_health, plots_dir / "physics_vs_geometry_distance.png")
    plot_neighborhood_examples(correct_cached, correct_eval, plots_dir / "physics_neighborhood_examples.png")
    plot_probe_summary(correct_eval, plots_dir / "probe_response.png")
    plot_resonance_summary(correct_eval, plots_dir / "probe_resonance.png")

    comparison = correct_eval.get("correct_vs_shuffled", {})
    comparison.setdefault("correct_z_pred_latent_vs_response_spearman", 0.0)
    comparison.setdefault("shuffled_z_pred_latent_vs_response_spearman", 0.0)
    comparison.setdefault("separation", 0.0)
    plot_correct_vs_shuffled = _bar_correct_vs_shuffled(comparison, plots_dir / "correct_vs_shuffled.png")

    metrics = {
        "experiment_label": "physics_jepa_v3_frequency_relational",
        "split": {
            "train": 24000,
            "val": 3000,
            "test": 3000,
            "seed": 42,
        },
        "representations": ["z_target", "z_pred", "z_geometry"],
        "similarity_correct": {name: correct_eval["similarity"][name] for name in ("z_target", "z_pred", "z_geometry")},
        "similarity_shuffled": {name: shuffled_eval["similarity"][name] for name in ("z_target", "z_pred", "z_geometry")},
        "correct_vs_shuffled": comparison,
        "latent_health_correct": correct_health,
        "latent_health_shuffled": shuffled_health,
        "latent_health_v2": v2_health,
        "probes_correct": {name: correct_eval["probes"][name] for name in ("z_target", "z_pred", "z_geometry")},
        "python": platform.python_version(),
    }
    metrics["probes_correct"] = {name: {k: v for k, v in metrics["probes_correct"][name].items() if k != "predictions"} for name in metrics["probes_correct"]}

    with (root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    summary_rows = [
        ["metric", "v1", "v2", "v3_correct", "v3_shuffled"],
        ["z_pred EM rho (norm)", "seev1fail", "-0.1057", f"{correct_eval['similarity']['z_pred']['latent_vs_response_spearman']:.4f}", f"{shuffled_eval['similarity']['z_pred']['latent_vs_response_spearman']:.4f}"],
        ["z_target EM rho (norm)", "-0.0405", "-0.0169", f"{correct_eval['similarity']['z_target']['latent_vs_response_spearman']:.4f}", f"{shuffled_eval['similarity']['z_target']['latent_vs_response_spearman']:.4f}"],
        ["z_pred geometry rho (norm)", "-0.1716", "0.0583", f"{correct_eval['similarity']['z_pred']['latent_vs_geometry_spearman']:.4f}", f"{shuffled_eval['similarity']['z_pred']['latent_vs_geometry_spearman']:.4f}"],
        ["correct-shuffled separation (z_pred)", "-", "-0.1107", f"{comparison['separation']:.4f}", "-"],
        ["z_pred rank-1 fraction", "0.9759", "0.7136", f"{correct_health['z_pred']['rank1_fraction']:.4f}", f"{shuffled_health['z_pred']['rank1_fraction']:.4f}"],
        ["z_target rank-1 fraction", "0.8911", "0.0378", f"{correct_health['z_target']['rank1_fraction']:.4f}", f"{shuffled_health['z_target']['rank1_fraction']:.4f}"],
        ["z_pred effective rank", "-", "-", f"{correct_health['z_pred']['effective_rank']:.2f}", f"{shuffled_health['z_pred']['effective_rank']:.2f}"],
        ["response probe R2 z_pred", "0.1585", "0.0424", f"{correct_eval['probes']['z_pred']['response']['response_r2_vs_mean']:.4f}", "-"],
        ["response probe R2 z_target", "0.2030", "0.3418", f"{correct_eval['probes']['z_target']['response']['response_r2_vs_mean']:.4f}", "-"],
        ["resonance Ty MAE GHz z_pred", "-", "2.93", f"{correct_eval['probes']['z_pred']['resonance'].get('ty_frequency_mae_ghz', float('nan')):.3f}", "-"],
        ["resonance Rx MAE GHz z_pred", "-", "2.70", f"{correct_eval['probes']['z_pred']['resonance'].get('rx_frequency_mae_ghz', float('nan')):.3f}", "-"],
        ["feature-count MAE z_pred", "-", "0.4311", f"{correct_eval['probes']['z_pred']['resonance'].get('feature_count_mae', float('nan')):.4f}", "-"],
    ]
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(summary_rows)

    print(json.dumps({"metrics": str(root / "metrics.json"), "summary": str(root / "summary.csv"), "plots": str(plots_dir)}))
    print(json.dumps({
        "correct_z_pred_rho": correct_eval["similarity"]["z_pred"]["latent_vs_response_spearman"],
        "shuffled_z_pred_rho": shuffled_eval["similarity"]["z_pred"]["latent_vs_response_spearman"],
        "separation": comparison["separation"],
        "correct_z_pred_geo_rho": correct_eval["similarity"]["z_pred"]["latent_vs_geometry_spearman"],
    }, indent=2))


def _bar_correct_vs_shuffled(comparison: dict[str, float], output: Path) -> Path:
    groups = [key for key in comparison if key.endswith("_latent_vs_response_spearman")]
    values = [comparison[key] for key in groups if isinstance(comparison[key], (int, float))]
    figure, axes = plt.subplots(figsize=(7, 4.2))
    axes.bar(groups, values, color=["tab:green", "tab:red"])
    axes.set_ylabel("spearman rho(z_pred, EM distance)")
    axes.set_title(f"Correct vs shuffled (separation {comparison.get('separation', 0.0):.4f})")
    axes.grid(alpha=0.3, axis="y")
    return _save(figure, output)


if __name__ == "__main__":
    main()