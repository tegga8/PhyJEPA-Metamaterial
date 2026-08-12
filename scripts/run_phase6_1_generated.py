"""Stage D of Phase 6.1: RCWA audit of frozen Phase 5A/5B completions."""
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

from src.physics_conditioned_dataset import PhysicsCompletionDataset
from src.physics_conditioned_spatial_jepa import PhysicsConditionedSpatialJEPA
from src.physics_consistency import load_frozen_forward_surrogate
from src.rcwa_solver import RCWAConfig, frequency_vector
from src.rcwa_validation import (
    PHYSICAL_MAPPING,
    phase42_complexity_groups,
    cached_solve,
    pack_modes,
    ranking_statistics,
    response_metrics,
    save_json,
)
from src.spatial_jepa_completion_model import compose_binary_spatial_completion


MODELS = {
    "phase5a": Path("outputs/phase5a/physics_5aA/best.pt"),
    "phase5b_small": Path("outputs/phase5b/physics_5bA_small/best.pt"),
    "phase5b_medium": Path("outputs/phase5b/physics_5bA_medium/best.pt"),
}


def load_completion(path: Path, device: torch.device) -> PhysicsConditionedSpatialJEPA:
    config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
    model = PhysicsConditionedSpatialJEPA(
        config.get("latent_channels", 64), config.get("predictor_hidden_channels", 128),
        config.get("ema_decay", .996), config.get("physics_embedding_dim", 128),
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model_state_dict"])
    model.eval()
    return model


def masked_iou(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    selected = mask.astype(bool)
    predicted, target = prediction[selected].astype(bool), target[selected].astype(bool)
    union = np.logical_or(prediction, target).sum()
    return float(np.logical_and(prediction, target).sum() / union) if union else 1.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_exploitation(path: Path, frequency: np.ndarray, row: dict[str, object]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(10, 5.5))
    axes[0, 0].imshow(row["geometry"], cmap="gray_r")
    axes[0, 0].set_title(f"{row['model']} / {row['source_id']}")
    axes[0, 0].axis("off")
    target, cnn, rcwa = row["target_raw"], row["cnn_raw"], row["rcwa_raw"]
    for axis, channel, label in ((axes[0, 1], 0, "Re(Ty)"), (axes[0, 2], 2, "Re(Rx)"), (axes[1, 1], 1, "Im(Ty)"), (axes[1, 2], 3, "Im(Rx)")):
        axis.plot(frequency, target[channel], color="black", label="target")
        axis.plot(frequency, cnn[channel], color="#d95f02", label="CNN")
        axis.plot(frequency, rcwa[channel], color="#1b9e77", label="RCWA")
        axis.set_title(label)
        if channel in (0, 2):
            axis.legend(fontsize=7)
    axes[1, 0].axis("off")
    figure.suptitle(f"surrogate exploitation gap = {row['exploitation_gap']:.6g}")
    figure.tight_layout(rect=(0, 0, 1, .95))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--forward-checkpoint", type=Path, default=Path("outputs/phase2_5/exp_A_5k_mse/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase6_1"))
    parser.add_argument("--count-per-model", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--allow-unconverged-diagnostic", action="store_true")
    args = parser.parse_args()
    if args.count_per_model < 1:
        raise ValueError("--count-per-model must be positive")

    project_config = json.loads((args.output_dir / "config.json").read_text(encoding="utf-8"))
    if project_config.get("selected_fourier_order") is None and not args.allow_unconverged_diagnostic:
        raise RuntimeError("NO CONVERGENCE ESTABLISHED: generated-geometry physics validation is not permitted")
    order = int(project_config.get("selected_fourier_order") or project_config["convergence"]["orders"][-1])
    thickness = float(project_config["selected_substrate_thickness_mm"])
    diagnostic_only = project_config.get("selected_fourier_order") is None
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    frequencies = frequency_vector()
    stats = np.load(args.subset_root / "train_response_stats.npz")
    mean, std = stats["mean"], stats["std"]
    data = PhysicsCompletionDataset(args.subset_root, "test", "central_block", .25, 42)
    actual_count = min(args.count_per_model, len(data))
    cnn, cnn_name = load_frozen_forward_surrogate(args.forward_checkpoint, device)
    completions = {name: load_completion(path, device) for name, path in MODELS.items() if path.exists()}
    if len(completions) != 3:
        raise FileNotFoundError(f"Required Phase 5 checkpoints unavailable: {set(MODELS) - set(completions)}")
    output = args.output_dir / "generated"
    plots = output / "plots"
    cases_dir = output / "exploitation_cases"
    output.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)
    solver_config = RCWAConfig(substrate_thickness_mm=thickness, fourier_order=order, device=args.device, cpu_workers=args.cpu_workers)
    rows: list[dict[str, object]] = []
    for index in range(actual_count):
        item = data[index]
        inputs, target, mask, response = (item[name].unsqueeze(0).to(device) for name in ("input", "target", "mask", "response"))
        target_raw = item["response_raw"].numpy()
        for name, model in completions.items():
            with torch.inference_mode():
                probabilities = torch.sigmoid(model(inputs, response)["logits"])
                completed = compose_binary_spatial_completion(probabilities, inputs, mask)
                cnn_raw = cnn(completed).cpu().numpy()[0] * std + mean
            geometry = completed[0, 0].cpu().numpy()
            rcwa = cached_solve(geometry, frequencies, solver_config, args.output_dir / "cache")
            rcwa_raw = pack_modes(rcwa, PHYSICAL_MAPPING)
            complexity_score, complexity_label = phase42_complexity_groups(np.asarray([geometry]))
            cnn_target = response_metrics(cnn_raw, target_raw, mean, std)["normalized_mse"]
            rcwa_target = response_metrics(rcwa_raw, target_raw, mean, std)["normalized_mse"]
            cnn_rcwa = response_metrics(cnn_raw, rcwa_raw, mean, std)["normalized_mse"]
            rows.append({
                "source_id": item["sample_id"], "test_index": index, "model": name,
                "generated_complexity_group": complexity_label[0], "generated_complexity_score": float(complexity_score[0]),
                "masked_iou": masked_iou(geometry, target[0, 0].cpu().numpy(), mask[0, 0].cpu().numpy()),
                "cnn_target_normalized_mse": cnn_target, "rcwa_target_normalized_mse": rcwa_target,
                "cnn_rcwa_normalized_mse": cnn_rcwa, "exploitation_gap": rcwa_target - cnn_target,
                "geometry": geometry, "target_raw": target_raw, "cnn_raw": cnn_raw, "rcwa_raw": rcwa_raw,
            })
    public_rows = [{key: value for key, value in row.items() if key not in {"geometry", "target_raw", "cnn_raw", "rcwa_raw"}} for row in rows]
    write_csv(output / "per_sample_metrics.csv", public_rows)
    overall_stats = ranking_statistics([row["cnn_target_normalized_mse"] for row in rows], [row["rcwa_target_normalized_mse"] for row in rows])
    ranking_rows = []
    for index in range(actual_count):
        candidates = [row for row in rows if row["test_index"] == index]
        stats_for_target = ranking_statistics([row["cnn_target_normalized_mse"] for row in candidates], [row["rcwa_target_normalized_mse"] for row in candidates])
        ranking_rows.append({"source_id": candidates[0]["source_id"], "test_index": index, **stats_for_target})
    write_csv(output / "ranking_metrics.csv", ranking_rows)
    exploitation = sorted(rows, key=lambda row: float(row["exploitation_gap"]), reverse=True)
    exploitation_rows = []
    for rank, row in enumerate(exploitation[:10]):
        case_path = cases_dir / f"case_{rank:02d}.npz"
        np.savez_compressed(case_path, geometry=row["geometry"], target_raw=row["target_raw"], cnn_raw=row["cnn_raw"], rcwa_raw=row["rcwa_raw"], frequency_ghz=frequencies)
        plot_exploitation(plots / f"exploitation_{rank:02d}.png", frequencies, row)
        exploitation_rows.append({key: value for key, value in row.items() if key not in {"geometry", "target_raw", "cnn_raw", "rcwa_raw"}} | {"case_file": str(case_path)})
    write_csv(output / "exploitation_cases.csv", exploitation_rows)
    pareto_rows = []
    for model in MODELS:
        selected = [row for row in rows if row["model"] == model]
        pareto_rows.append({
            "model": model, "sample_count": len(selected), "geometry_error_mean": float(np.mean([1 - row["masked_iou"] for row in selected])),
            "cnn_physics_error_mean": float(np.mean([row["cnn_target_normalized_mse"] for row in selected])),
            "rcwa_physics_error_mean": float(np.mean([row["rcwa_target_normalized_mse"] for row in selected])),
        })
    write_csv(output / "pareto_summary.csv", pareto_rows)
    figure, axis = plt.subplots(figsize=(5, 4))
    for model in MODELS:
        selected = [row for row in rows if row["model"] == model]
        axis.scatter([row["cnn_target_normalized_mse"] for row in selected], [row["rcwa_target_normalized_mse"] for row in selected], s=24, label=model)
    axis.set(xlabel="CNN target MSE", ylabel="RCWA target MSE", title="Generated candidate target error")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(plots / "target_error_correlation.png", dpi=180)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(5, 4))
    for model in MODELS:
        selected = [row for row in pareto_rows if row["model"] == model][0]
        axis.scatter(selected["geometry_error_mean"], selected["cnn_physics_error_mean"], marker="o", label=f"{model} CNN")
        axis.scatter(selected["geometry_error_mean"], selected["rcwa_physics_error_mean"], marker="x", label=f"{model} RCWA")
    axis.set(xlabel="mean geometry error (1 - masked IoU)", ylabel="mean target MSE", title="Phase 5B physical Pareto subset")
    axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(plots / "physical_pareto.png", dpi=180)
    plt.close(figure)
    complexity_summary = {group: {
        "count": len(selected := [row for row in rows if row["generated_complexity_group"] == group]),
        "mean_cnn_rcwa_mse": float(np.mean([row["cnn_rcwa_normalized_mse"] for row in selected])) if selected else None,
        "mean_rcwa_target_mse": float(np.mean([row["rcwa_target_normalized_mse"] for row in selected])) if selected else None,
    } for group in ("simple", "medium", "complex")}
    save_json(output / "metrics.json", {
        "frequency_ghz": frequencies, "frequency_points": len(frequencies), "cnn_model": cnn_name,
        "sample_count_per_model": actual_count, "counts": {model: len([row for row in rows if row["model"] == model]) for model in MODELS},
        "fourier_order": order, "substrate_thickness_mm": thickness, "physical_channel_mapping": PHYSICAL_MAPPING,
        "diagnostic_only_unconverged": diagnostic_only, "target_error_relationship": overall_stats,
        "mean_per_target_ranking": {key: (float(np.nanmean([row[key] for row in ranking_rows if row[key] is not None])) if any(row[key] is not None for row in ranking_rows) else None) for key in ("pearson", "spearman", "kendall_tau", "pairwise_ordering_agreement", "top_1_overlap", "top_3_overlap", "top_5_overlap")},
        "worst_exploitation_gap": float(exploitation[0]["exploitation_gap"]), "complexity": complexity_summary,
    })
    print(output / "metrics.json")


if __name__ == "__main__":
    main()
