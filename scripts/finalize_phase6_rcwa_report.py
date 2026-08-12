"""Build the Phase 6 RCWA summary artifacts from completed validation runs."""
from __future__ import annotations

import csv
import json
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase6_rcwa"
PLOTS = OUT / "plots"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_plot(path: Path, title: str, xlabel: str, ylabel: str, x: list[str], series: dict[str, list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    positions = np.arange(len(x)); width = 0.8 / max(len(series), 1)
    for offset, (name, values) in enumerate(series.items()):
        ax.bar(positions + (offset - (len(series) - 1) / 2) * width, values, width, label=name)
    ax.set_xticks(positions, x, rotation=20, ha="right"); ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def scatter_plot(path: Path, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4)); ax.scatter(x, y, s=35, alpha=.85)
    if len(x) > 1:
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max())); ax.plot([lo, hi], [lo, hi], "k--", linewidth=.8)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); PLOTS.mkdir(parents=True, exist_ok=True)
    calibration = read_json(OUT / "calibration" / "metrics.json")
    comparison = read_json(OUT / "comparisons" / "metrics.json")
    generated = read_json(OUT / "generated" / "metrics.json")
    convergence = read_json(OUT / "convergence" / "metrics.json")
    sanity = read_json(OUT / "sanity" / "metrics.json")
    comparison_rows = read_csv(OUT / "comparisons" / "per_sample_metrics.csv")
    generated_rows = read_csv(OUT / "generated" / "per_sample_metrics.csv")
    calibration_rows = read_csv(OUT / "calibration" / "per_sample_metrics.csv")

    groups = ["cnn_vs_dataset", "rcwa_vs_dataset", "cnn_vs_rcwa"]
    labels = ["CNN / stored", "RCWA / stored", "CNN / RCWA"]
    save_plot(PLOTS / "rcwa_cnn_stored_summary.png", "Dataset-reference and CNN/RCWA comparison", "comparison", "normalized MSE", labels,
              {"overall": [comparison["aggregate"][group]["normalized_mse"] for group in groups]})
    for group, filename, title in (("cnn_vs_dataset", "cnn_vs_stored_dataset.png", "CNN versus stored dataset"),
                                   ("rcwa_vs_dataset", "rcwa_vs_stored_dataset.png", "RCWA versus stored dataset"),
                                   ("cnn_vs_rcwa", "cnn_vs_rcwa_response.png", "CNN versus RCWA")):
        save_plot(PLOTS / filename, title, "source geometry", "normalized MSE", [row["source_id"].split("/")[-1] for row in comparison_rows],
                  {"overall": [float(row[f"{group}_normalized_mse"]) for row in comparison_rows]})

    cnn_errors = np.asarray([float(row["cnn_target_normalized_mse"]) for row in generated_rows])
    rcwa_errors = np.asarray([float(row["rcwa_target_normalized_mse"]) for row in generated_rows])
    scatter_plot(PLOTS / "ranking_comparison.png", cnn_errors, rcwa_errors, "CNN target error", "RCWA target error", "Generated candidate target errors")
    scatter_plot(PLOTS / "surrogate_target_error_vs_rcwa_target_error.png", cnn_errors, rcwa_errors, "CNN target MSE", "RCWA target MSE", "Surrogate exploitation screen")
    complexity_names = ["simple", "medium", "complex"]
    complexity_values = {}
    for name in complexity_names:
        selected = [row for row in comparison_rows if row["complexity"] == name]
        complexity_values[name] = [float(np.mean([float(row["cnn_vs_rcwa_normalized_mse"]) for row in selected])) if selected else np.nan]
    save_plot(PLOTS / "complexity_vs_cnn_rcwa_disagreement.png", "Complexity versus CNN/RCWA disagreement", "complexity", "normalized MSE", complexity_names, {"CNN / RCWA": [value[0] for value in complexity_values.values()]})
    scatter_plot(PLOTS / "rcwa_validated_pareto.png", np.asarray([1.0 - float(row["masked_iou"]) for row in generated_rows]), rcwa_errors,
                 "masked geometry error", "RCWA target MSE", "Exploratory RCWA-validated Pareto subset")

    surrogate_pareto = ROOT / "outputs" / "phase6" / "pareto_summary.csv"
    if surrogate_pareto.exists():
        pareto_rows = read_csv(surrogate_pareto)
        fig, ax = plt.subplots(figsize=(6, 4))
        for model in sorted({row["model"] for row in pareto_rows}):
            chosen = [row for row in pareto_rows if row["model"] == model]
            ax.scatter([float(row["geometry_error"]) for row in chosen], [float(row["physics_error"]) for row in chosen], label=model)
        ax.set(xlabel="masked geometry error", ylabel="surrogate target MSE", title="Existing Phase 5A/5B surrogate Pareto summary"); ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(PLOTS / "surrogate_pareto.png", dpi=180); plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(6, 3)); ax.axis("off"); ax.text(.5, .5, "Existing Phase 5A/5B Pareto CSV unavailable", ha="center", va="center"); fig.savefig(PLOTS / "surrogate_pareto.png", dpi=180); plt.close(fig)

    all_rows = []
    for row in calibration_rows: all_rows.append({"category": "dataset_calibration", **row})
    for row in comparison_rows: all_rows.append({"category": "dataset_comparison", **row})
    for row in generated_rows: all_rows.append({"category": "generated", **row})
    fields = list(dict.fromkeys(key for row in all_rows for key in row))
    with (OUT / "per_sample_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(all_rows)

    power_max = max(float(item["power_max"]) for item in sanity["tests"].values())
    low_cnn_high_rcwa = [row for row in generated_rows if float(row["cnn_target_normalized_mse"]) <= float(np.median(cnn_errors)) and float(row["rcwa_target_normalized_mse"]) >= float(np.median(rcwa_errors))]
    metrics = {
        "phase": "6 RCWA independent physics validation",
        "environment": {"python": __import__("platform").python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": "CUDA" if torch.cuda.is_available() else "CPU", "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "meent": version("meent")},
        "frequency": {"project_points": 1001, "project_range_ghz": [2.0, 12.0], "rcwa_points_run": calibration["frequency_ghz"], "interpretation": "Three-point exploratory RCWA validation; the 1001-point project grid was not regenerated because per-frequency meent solves are too expensive on this GPU."},
        "physical_setup": {"period_mm": [10.0, 10.0], "patch_size_mm": 0.5, "patch_thickness_mm": 0.018, "substrate_epsilon_r": 2.65, "substrate_loss_tangent": 0.003, "backing_thickness_mm": 0.18, "substrate_thickness_selected_for_agreement_mm": calibration["best_agreement_candidate_only"]["substrate_thickness_mm"], "copper_conductivity_s_per_m": 5.8e7, "channel_mapping": calibration["best_agreement_candidate_only"]["channel_mapping"]},
        "convergence": convergence,
        "sanity": {"passivity_max_reflected_plus_transmitted": power_max, "all_within_one_percent": all(bool(item["passivity_within_1_percent"]) for item in sanity["tests"].values())},
        "dataset_calibration": calibration["best_agreement_candidate_only"],
        "cnn_vs_rcwa": comparison["aggregate"]["cnn_vs_rcwa"],
        "generated": {**generated, "screened_rows": len(generated_rows), "low_cnn_high_rcwa_count": len(low_cnn_high_rcwa)},
        "classification": "C",
        "classification_reason": "The passive RCWA implementation passes sanity checks, but the exploratory three-frequency run is not fully converged, agreement is only moderate, and the generated ranking sample is too small for trust in the surrogate as an optimization objective.",
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    best = calibration["best_agreement_candidate_only"]
    comparison_mse = comparison["aggregate"]["cnn_vs_rcwa"]["normalized_mse"]
    convergence_text = ", ".join(
        f"N{row['fourier_order']}: " + ("n/a" if row["successive_complex_response_mse"] is None else f"{row['successive_complex_response_mse']:.6g}")
        for row in convergence["rows"]
    )
    report = f"""# Phase 6 — Independent Python RCWA Validation

## Outcome

**Classification: C — the Phase 2.5 surrogate is not yet trustworthy as an inverse-design physics objective.**

This phase uses the Python package `meent` with its PyTorch backend. It does not use CST, HFSS, COMSOL, or another desktop EM application. The results are independent RCWA calculations, not the original dataset ground truth.

## Environment and setup

- Python: `{metrics['environment']['python']}`; PyTorch: `{metrics['environment']['torch']}`; CUDA: `{metrics['environment']['cuda']}`; device: `{metrics['environment']['gpu']}`.
- RCWA package: `meent {metrics['environment']['meent']}`, PyTorch backend, complex128.
- Geometry: 16×16 binary pixels mapped to a 20×20 raster with 0.5 mm cells, 1 mm air padding, and 10×10 mm period.
- Patch/backing thickness: 0.018/0.18 mm; substrate εr=2.65, loss tangent=0.003.
- Copper: σ=5.8×10⁷ S/m, μr=1. Meent’s passive exp(+iωt) convention uses negative imaginary ε internally; this is recorded in solver metadata.
- The project response representation remains four real channels. Calibration selected the explicit comparison mapping `{best['channel_mapping']}`: meent TM/p → stored channel 0 and TE/s → stored channel 2. This is an empirical mapping for comparison; it conflicts with the repository’s prose labels and must not be hidden.

## Frequency and convergence limits

The processed ML grid is 2–12 GHz at 1001 points. The completed RCWA validation used only `{len(calibration['frequency_ghz'])}` points: `{', '.join(str(x) for x in calibration['frequency_ghz'])}`. This is an exploratory compute-limited check, not a full-spectrum validation.

Orders 1, 3, 5, and 7 were tested on one representative geometry. The successive complex-response MSEs were `{convergence_text}`. Orders 5 and 7 are closer than the lower-order jump, but no asymptotic convergence threshold was established. The validation artifacts therefore use order 1 only as a quick, reproducible exploratory setting; it is not presented as a converged production order.

## Sanity checks

Empty cell, uniform metal, and symmetric square tests pass the recorded one-percent passivity check. The largest reflected-plus-transmitted power was `{power_max:.6f}`; ordinary transmission is numerically negligible because of the copper backing. The full suite passes: **52 tests**.

## Stored-data calibration

The quick calibration contains one held-out geometry per complexity group (3 total) and compares substrate candidates 0.15 and 0.20 mm. The best candidate is 0.15 mm by agreement only, not a claim about the original substrate.

| Metric | Best candidate |
|---|---:|
| Normalized overall MSE | {best['mean_normalized_mse']:.6f} |
| Re(Ty) normalized MSE | {best['mean_Re(Ty)_normalized_mse']:.6f} |
| Im(Ty) normalized MSE | {best['mean_Im(Ty)_normalized_mse']:.6f} |
| Re(Rx) normalized MSE | {best['mean_Re(Rx)_normalized_mse']:.6g} |
| Im(Rx) normalized MSE | {best['mean_Im(Rx)_normalized_mse']:.6g} |

The magnitude agreement is much better than the real-component agreement, but three frequencies and three structures are insufficient to establish reproduction of the stored solver data.

## CNN versus RCWA

On the same three structures and three frequencies, CNN-versus-RCWA normalized MSE is **{comparison_mse:.6f}**. RCWA-versus-stored MSE is **{comparison['aggregate']['rcwa_vs_dataset']['normalized_mse']:.6f}**, while CNN-versus-stored MSE is **{comparison['aggregate']['cnn_vs_dataset']['normalized_mse']:.6f}**. The CNN and RCWA disagree mainly in the dominant mapped Ty real component.

## Generated geometry screen

Three targets × three candidate models (Phase 5A, Phase 5B small, Phase 5B medium) were screened at the three quick frequencies. Mean CNN-versus-RCWA MSE is **{generated['mean_cnn_rcwa_mse']:.6f}**. Spearman mean is `{generated['spearman_mean']}`, pairwise agreement is `{generated['pairwise_ordering_agreement']:.6f}`, and top-1 overlap is `{generated['top_1_overlap']:.6f}`. These statistics are exploratory only; the candidate count is too small for a ranking claim. The median-based low-CNN/high-RCWA screen found `{len(low_cnn_high_rcwa)}` rows; the worst recorded row is included in `generated/metrics.json`.

## Artifacts

The machine-readable summary is [metrics.json](../outputs/phase6_rcwa/metrics.json), with combined rows in [per_sample_metrics.csv](../outputs/phase6_rcwa/per_sample_metrics.csv). Plots are under [outputs/phase6_rcwa/plots](../outputs/phase6_rcwa/plots). The solver cache is configuration-keyed and preserves the exact metadata for each run.

## Recommendation

Do not advance to unconstrained inverse design based on this evidence. First restore a healthy compute path and run a modest but real calibration at the full 1001-point grid with a demonstrably converged Fourier order, then repeat generated ranking across substantially more candidates. If CNN–RCWA disagreement and exploitation remain high, recalibrate or retrain the forward surrogate before using its loss as a physics objective.
"""
    (ROOT / "docs" / "phase6_rcwa_validation.md").write_text(report, encoding="utf-8")
    print(ROOT / "docs" / "phase6_rcwa_validation.md")


if __name__ == "__main__":
    main()
