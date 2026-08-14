"""Aggregate the seed-123 reproduction evaluation for Physics-JEPA v3.

Assumes ``scripts/evaluate_physics_representation.py`` has been run on the
``correct`` and ``shuffled`` reproduction model directories (writing
``evaluation.json`` into each), and that cached val/test latents exist in each
model directory.  Compares the seed-123 result against the original seed-42 v3
run (``outputs/physics_jepa_v3/metrics.json``) and writes:

- ``outputs/physics_jepa_v3_repro/metrics.json``
- ``outputs/physics_jepa_v3_repro/summary.csv``
- ``outputs/physics_jepa_v3_repro/plots/v3_seed_reproduction.png``
- ``outputs/physics_jepa_v3_repro/plots/correct_vs_shuffled_seed123.png``
- standard v3 latent-health / similarity / probe plots under ``plots/``

The classification decision (REPRODUCED / WEAKLY / NOT) is written into
``metrics.json`` and printed.
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

from src.physics_jepa_plots import _save


def load_cached_latents(directory: Path, split: str) -> dict[str, np.ndarray]:
    names = ("z_geometry", "z_online", "z_target", "z_pred", "geometry", "response")
    return {name: np.load(directory / f"{split}_{name}.npy") for name in names}


def load_evaluation(directory: Path) -> dict[str, object]:
    return json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))


def load_original_metrics(root: Path) -> dict[str, object]:
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    correct = metrics["similarity_correct"]
    shuffled = metrics["similarity_shuffled"]
    health = metrics["latent_health_correct"]
    probes = metrics["probes_correct"]
    return {
        "em_rho_correct": correct["z_pred"]["latent_vs_response_spearman"],
        "geo_rho_correct": correct["z_pred"]["latent_vs_geometry_spearman"],
        "em_rho_shuffled": shuffled["z_pred"]["latent_vs_response_spearman"],
        "separation": metrics["correct_vs_shuffled"].get("separation", 0.0),
        "target_rank1": health["z_target"]["rank1_fraction"],
        "pred_rank1": health["z_pred"]["rank1_fraction"],
        "target_eff_rank": health["z_target"]["effective_rank"],
        "pred_eff_rank": health["z_pred"]["effective_rank"],
        "response_r2_pred": probes["z_pred"]["response"]["response_r2_vs_mean"],
        "response_r2_target": probes["z_target"]["response"]["response_r2_vs_mean"],
        "resonance_ty_mae": probes["z_pred"]["resonance"].get("ty_frequency_mae_ghz", float("nan")),
        "resonance_rx_mae": probes["z_pred"]["resonance"].get("rx_frequency_mae_ghz", float("nan")),
        "feature_count_mae": probes["z_pred"]["resonance"].get("feature_count_mae", float("nan")),
        "within_family_rho": None,
        "cross_family_rho": None,
    }


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


def plot_seed_reproduction(original: dict[str, object], current: dict[str, object], output: Path) -> Path:
    rows = [
        ("EM-distance rho (z_pred)", original["em_rho_correct"], current["em_rho_correct"]),
        ("geometry rho (z_pred)", original["geo_rho_correct"], current["geo_rho_correct"]),
        ("shuffled EM rho (z_pred)", original["em_rho_shuffled"], current["em_rho_shuffled"]),
        ("correct-shuffled gap", original["separation"], current["separation"]),
        ("target rank-1", original["target_rank1"], current["target_rank1"]),
        ("predictor rank-1", original["pred_rank1"], current["pred_rank1"]),
        ("response probe R2 (z_pred)", original["response_r2_pred"], current["response_r2_pred"]),
        ("response probe R2 (z_target)", original["response_r2_target"], current["response_r2_target"]),
    ]
    labels = [row[0] for row in rows]
    original_values = [row[1] for row in rows]
    current_values = [row[2] for row in rows]
    position = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(position - width / 2, original_values, width, label="original v3 (seed 42)", color="tab:blue")
    axis.bar(position + width / 2, current_values, width, label="reproduction (seed 123)", color="tab:orange")
    axis.axhline(0.0, color="gray", linewidth=1)
    axis.set_xticks(position)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylabel("value")
    axis.set_title("Physics-JEPA v3 seed reproduction: original (seed 42) vs seed 123")
    axis.legend()
    axis.grid(alpha=0.3, axis="y")
    return _save(figure, output)


def plot_correct_vs_shuffled(comparison: dict[str, float], output: Path) -> Path:
    groups = [key for key in comparison if key.endswith("_latent_vs_response_spearman")]
    values = [comparison[key] for key in groups if isinstance(comparison[key], (int, float))]
    figure, axis = plt.subplots(figsize=(7, 4.2))
    axis.bar(groups, values, color=["tab:green", "tab:red"])
    axis.set_ylabel("spearman rho(z_pred, EM distance)")
    axis.set_title(f"Correct vs shuffled (seed 123), separation {comparison.get('separation', 0.0):.4f}")
    axis.grid(alpha=0.3, axis="y")
    return _save(figure, output)


def plot_latent_health(health: dict[str, dict[str, float]], title: str, output: Path) -> Path:
    names = list(health)
    rank1 = [health[name]["rank1_fraction"] for name in names]
    effective = [health[name]["effective_rank"] for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, rank1, color="tab:blue")
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("rank-1 fraction")
    axes[0].set_title("rank-1 (lower = better)")
    axes[1].bar(names, effective, color="tab:orange")
    axes[1].set_ylabel("effective rank")
    axes[1].set_title("effective rank (32 max)")
    for axis in axes:
        axis.grid(alpha=0.3, axis="y")
        axis.tick_params(axis="x", rotation=15)
    figure.suptitle(title)
    return _save(figure, output)


def plot_physics_vs_geometry(evaluation: dict[str, object], output: Path) -> Path:
    names = ("z_target", "z_pred", "z_geometry")
    em = [evaluation["similarity"][name]["latent_vs_response_spearman"] for name in names]
    geo = [evaluation["similarity"][name]["latent_vs_geometry_spearman"] for name in names]
    position = np.arange(len(names))
    width = 0.35
    figure, axis = plt.subplots(figsize=(8, 4.2))
    axis.bar(position - width / 2, em, width, label="rho(D_z, D_EM)", color="tab:blue")
    axis.bar(position + width / 2, geo, width, label="rho(D_z, D_G)", color="tab:red")
    axis.axhline(0.0, color="gray", linewidth=1)
    axis.set_xticks(position)
    axis.set_xticklabels(names)
    axis.set_ylabel("Spearman correlation")
    axis.set_title("Physics vs geometry distance correlation (seed-123 correct)")
    axis.legend()
    axis.grid(alpha=0.3, axis="y")
    return _save(figure, output)


def plot_probe_summary(evaluation: dict[str, object], output: Path) -> Path:
    names = ("z_target", "z_pred", "z_geometry")
    values = [evaluation["probes"][name]["response"]["response_r2_vs_mean"] for name in names]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(names, values, color="tab:green")
    axis.set_ylabel("response probe R2 vs mean")
    axis.set_title("Linear response probe (seed-123 correct)")
    axis.grid(alpha=0.3, axis="y")
    return _save(figure, output)


def plot_resonance_summary(evaluation: dict[str, object], output: Path) -> Path:
    names = ("z_target", "z_pred", "z_geometry")
    labels = ["ty_frequency_mae_ghz", "rx_frequency_mae_ghz", "feature_count_mae"]
    figure, axis = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(names))
    width = 0.25
    for offset, label in enumerate(labels):
        values = [evaluation["probes"][name]["resonance"].get(label, 0.0) for name in names]
        axis.bar(positions + (offset - 1) * width, values, width, label=label)
    axis.set_xticks(positions)
    axis.set_xticklabels(names)
    axis.set_ylabel("resonance probe error")
    axis.set_title("Resonance probe (seed-123 correct, lower = better)")
    axis.legend()
    axis.grid(alpha=0.3, axis="y")
    return _save(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/physics_jepa_v3_repro"))
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--compare-dir", type=Path, default=None)
    parser.add_argument("--original-root", type=Path, default=Path("outputs/physics_jepa_v3"))
    args = parser.parse_args()

    root = args.root
    model_dir = args.model_dir or root / "correct"
    compare_dir = args.compare_dir or root / "shuffled"
    evaluation_dir = root / "evaluation"
    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    correct_eval = load_evaluation(model_dir)
    shuffled_eval = load_evaluation(compare_dir)
    correct_cached = load_cached_latents(model_dir, "test")
    shuffled_cached = load_cached_latents(compare_dir, "test")

    correct_health = latent_health(correct_cached)
    shuffled_health = latent_health(shuffled_cached)
    original = load_original_metrics(args.original_root)

    comparison = dict(correct_eval.get("correct_vs_shuffled", {}))
    comparison.setdefault("correct_z_pred_latent_vs_response_spearman", correct_eval["similarity"]["z_pred"]["latent_vs_response_spearman"])
    comparison.setdefault("shuffled_z_pred_latent_vs_response_spearman", shuffled_eval["similarity"]["z_pred"]["latent_vs_response_spearman"])
    comparison.setdefault("separation", comparison["correct_z_pred_latent_vs_response_spearman"] - comparison["shuffled_z_pred_latent_vs_response_spearman"])

    current = {
        "em_rho_correct": correct_eval["similarity"]["z_pred"]["latent_vs_response_spearman"],
        "geo_rho_correct": correct_eval["similarity"]["z_pred"]["latent_vs_geometry_spearman"],
        "em_rho_shuffled": shuffled_eval["similarity"]["z_pred"]["latent_vs_response_spearman"],
        "separation": comparison["separation"],
        "target_rank1": correct_health["z_target"]["rank1_fraction"],
        "pred_rank1": correct_health["z_pred"]["rank1_fraction"],
        "target_eff_rank": correct_health["z_target"]["effective_rank"],
        "pred_eff_rank": correct_health["z_pred"]["effective_rank"],
        "response_r2_pred": correct_eval["probes"]["z_pred"]["response"]["response_r2_vs_mean"],
        "response_r2_target": correct_eval["probes"]["z_target"]["response"]["response_r2_vs_mean"],
        "resonance_ty_mae": correct_eval["probes"]["z_pred"]["resonance"].get("ty_frequency_mae_ghz", float("nan")),
        "resonance_rx_mae": correct_eval["probes"]["z_pred"]["resonance"].get("rx_frequency_mae_ghz", float("nan")),
        "feature_count_mae": correct_eval["probes"]["z_pred"]["resonance"].get("feature_count_mae", float("nan")),
        "within_family_rho": correct_eval.get("within_cross_family", {}).get("within_family_latent_vs_response_spearman"),
        "cross_family_rho": correct_eval.get("within_cross_family", {}).get("cross_family_latent_vs_response_spearman"),
    }

    history = model_dir / "training_history.csv"
    if history.is_file():
        from src.physics_jepa_plots import plot_loss_curve, plot_variance_diagnostics

        plot_loss_curve(history, plots_dir / "loss_curves_seed123.png")
        plot_variance_diagnostics(history, plots_dir / "latent_variance_diagnostics_seed123.png")

    plot_seed_reproduction(original, current, plots_dir / "v3_seed_reproduction.png")
    plot_correct_vs_shuffled(comparison, plots_dir / "correct_vs_shuffled_seed123.png")
    plot_latent_health(correct_health, "Physics-JEPA v3 latent health (seed 123, test split)", plots_dir / "latent_health_seed123.png")
    plot_physics_vs_geometry(correct_eval, plots_dir / "physics_vs_geometry_distance_seed123.png")
    plot_probe_summary(correct_eval, plots_dir / "probe_response_seed123.png")
    plot_resonance_summary(correct_eval, plots_dir / "probe_resonance_seed123.png")

    # Reproduction classification per the task rules.
    reproduced = bool(current["em_rho_correct"] > 0.0 and current["separation"] > 0.05 and abs(current["geo_rho_correct"]) < 0.2 and correct_health["z_target"]["nan_count"] == 0 and correct_health["z_pred"]["nan_count"] == 0)
    weakly = bool(current["em_rho_correct"] > 0.0 and (current["separation"] <= 0.05 or abs(current["geo_rho_correct"]) >= 0.2))
    if reproduced:
        decision = "REPRODUCED"
    elif weakly:
        decision = "WEAKLY REPRODUCED"
    else:
        decision = "NOT REPRODUCED"

    metrics = {
        "experiment_label": "physics_jepa_v3_repro_seed123",
        "reproduction_seed": 123,
        "shuffle_seed": 456,
        "original": {"seed": 42, **original},
        "seed123": current,
        "correct_vs_shuffled": comparison,
        "latent_health_correct": correct_health,
        "latent_health_shuffled": shuffled_health,
        "probes_correct": {name: {k: v for k, v in correct_eval["probes"][name].items() if k != "predictions"} for name in ("z_target", "z_pred", "z_geometry")},
        "classification": decision,
        "classification_criteria": {
            "correct_positive": current["em_rho_correct"] > 0.0,
            "shuffled_lower": current["em_rho_correct"] > current["em_rho_shuffled"],
            "geometry_not_dominant": abs(current["geo_rho_correct"]) < 0.2,
            "latent_health_consistent": correct_health["z_target"]["nan_count"] == 0 and correct_health["z_pred"]["nan_count"] == 0,
            "separation_positive": current["separation"] > 0.0,
        },
        "python": platform.python_version(),
    }
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    summary_rows = [
        ["metric", "original_v3_seed42", "seed123"],
        ["EM-distance rho (z_pred)", f"{original['em_rho_correct']:.4f}", f"{current['em_rho_correct']:.4f}"],
        ["geometry-distance rho (z_pred)", f"{original['geo_rho_correct']:.4f}", f"{current['geo_rho_correct']:.4f}"],
        ["shuffled EM rho (z_pred)", f"{original['em_rho_shuffled']:.4f}", f"{current['em_rho_shuffled']:.4f}"],
        ["correct-vs-shuffled gap", f"{original['separation']:.4f}", f"{current['separation']:.4f}"],
        ["target rank-1", f"{original['target_rank1']:.4f}", f"{current['target_rank1']:.4f}"],
        ["predictor rank-1", f"{original['pred_rank1']:.4f}", f"{current['pred_rank1']:.4f}"],
        ["target effective rank", f"{original['target_eff_rank']:.2f}", f"{current['target_eff_rank']:.2f}"],
        ["predictor effective rank", f"{original['pred_eff_rank']:.2f}", f"{current['pred_eff_rank']:.2f}"],
        ["response probe R2 (z_pred)", f"{original['response_r2_pred']:.4f}", f"{current['response_r2_pred']:.4f}"],
        ["response probe R2 (z_target)", f"{original['response_r2_target']:.4f}", f"{current['response_r2_target']:.4f}"],
        ["resonance Ty MAE GHz (z_pred)", f"{original['resonance_ty_mae']:.3f}", f"{current['resonance_ty_mae']:.3f}"],
        ["resonance Rx MAE GHz (z_pred)", f"{original['resonance_rx_mae']:.3f}", f"{current['resonance_rx_mae']:.3f}"],
        ["feature-count MAE (z_pred)", f"{original['feature_count_mae']:.4f}", f"{current['feature_count_mae']:.4f}"],
        ["within-family rho (z_pred)", "-" if original["within_family_rho"] is None else f"{original['within_family_rho']:.4f}", "-" if current["within_family_rho"] is None else f"{current['within_family_rho']:.4f}"],
        ["cross-family rho (z_pred)", "-" if original["cross_family_rho"] is None else f"{original['cross_family_rho']:.4f}", "-" if current["cross_family_rho"] is None else f"{current['cross_family_rho']:.4f}"],
        ["classification", "-", decision],
    ]
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(summary_rows)

    print(json.dumps({"metrics": str(root / "metrics.json"), "summary": str(root / "summary.csv"), "plots": str(plots_dir)}))
    print(json.dumps({"seed123_z_pred_rho": current["em_rho_correct"], "shuffled_z_pred_rho": current["em_rho_shuffled"], "separation": current["separation"], "classification": decision}, indent=2))


if __name__ == "__main__":
    main()
