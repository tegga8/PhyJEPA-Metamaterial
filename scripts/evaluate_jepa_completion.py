"""Evaluate JEPA completion against the corresponding Phase 3 CNN checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.completion_dataset import CompletionDataset
from src.completion_losses import completion_loss_metrics
from src.jepa_completion_losses import jepa_loss, latent_variance_metrics, masked_reconstruction_bce
from src.completion_model import CompletionCNN, compose_binary_completion
from src.forward_analysis import geometry_complexity
from src.jepa_completion_model import JEPACompletionModel, compose_binary_jepa_completion


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def binary_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    if mask is not None:
        selection = mask.astype(bool)
        prediction, target = prediction[selection], target[selection]
    prediction, target = prediction.astype(bool), target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    predicted_count, target_count = prediction.sum(), target.sum()
    return {
        "accuracy": float(np.mean(prediction == target)),
        "iou": float(intersection / union) if union else 1.0,
        "dice": float(2 * intersection / (predicted_count + target_count)) if predicted_count + target_count else 1.0,
    }


def checkpoint_for_mask(mask_type: str, ratio: float) -> Path:
    name = {("central_block", 0.25): "exp_3A", ("central_block", 0.5): "exp_3B", ("random_holes", 0.25): "exp_3C", ("random_holes", 0.5): "exp_3D"}[(mask_type, ratio)]
    return Path("outputs/phase3_completion") / name / "best.pt"


def aggregate(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_{metric}": float(np.mean([row[f"{prefix}_{metric}"] for row in rows]))
        for metric in ("full_bce", "masked_bce", "full_accuracy", "full_iou", "full_dice", "masked_accuracy", "masked_iou", "masked_dice")
    }


def save_comparison_grid(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    figure, axes = plt.subplots(len(rows), 7, figsize=(16, max(4, 2.4 * len(rows))))
    axes = np.atleast_2d(axes)
    labels = ("target", "mask", "partial", "CNN", "JEPA", "CNN error", "JEPA error")
    for row_index, row in enumerate(rows):
        images = (row["target"], row["mask"], row["partial"], row["cnn_completion"], row["jepa_completion"], row["cnn_error"], row["jepa_error"])
        for axis, image, label in zip(axes[row_index], images, labels):
            axis.imshow(image, cmap="magma" if "error" in label else "gray_r", vmin=0, vmax=1, interpolation="nearest")
            axis.set_title(label, fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row_index, 0].set_ylabel(f"{row['sample_id']}\nJEPA-CNN={row['difference']:.3f}", fontsize=7)
    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cnn-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    config = json.loads((args.checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    cnn_checkpoint = args.cnn_checkpoint or checkpoint_for_mask(config["mask_type"], float(config["missing_ratio"]))
    dataset = CompletionDataset(args.subset_root, "test", config["mask_type"], config["missing_ratio"], config["mask_seed"])
    jepa = JEPACompletionModel(config["latent_dim"], config["predictor_hidden_dim"], config["ema_decay"]).to(device)
    jepa.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    jepa.eval()
    cnn = CompletionCNN().to(device)
    cnn.load_state_dict(torch.load(cnn_checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    cnn.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    rows: list[dict[str, Any]] = []
    visual: list[dict[str, Any]] = []
    context_latents: list[torch.Tensor] = []
    target_latents: list[torch.Tensor] = []
    pred_latents: list[torch.Tensor] = []
    for batch in loader:
        inputs, target, mask = batch["input"].to(device), batch["target"].to(device), batch["mask"].to(device)
        with torch.inference_mode():
            jepa_outputs = jepa(inputs, target)
            cnn_logits = cnn(inputs)
        jepa_probability = torch.sigmoid(jepa_outputs["logits"])
        cnn_probability = torch.sigmoid(cnn_logits)
        jepa_binary = compose_binary_jepa_completion(jepa_probability, inputs, mask, args.threshold)
        cnn_binary = compose_binary_completion(cnn_probability, inputs, mask, args.threshold)
        context_latents.append(jepa_outputs["z_context"].cpu())
        target_latents.append(jepa_outputs["z_target"].cpu())
        pred_latents.append(jepa_outputs["z_pred"].cpu())
        for local_index, sample_id in enumerate(batch["sample_id"]):
            target_np = target[local_index, 0].cpu().numpy()
            mask_np = mask[local_index, 0].cpu().numpy()
            partial_np = inputs[local_index, 0].cpu().numpy()
            jepa_np = jepa_binary[local_index, 0].cpu().numpy()
            cnn_np = cnn_binary[local_index, 0].cpu().numpy()
            jepa_metrics = binary_metrics(jepa_np, target_np)
            jepa_masked = binary_metrics(jepa_np, target_np, mask_np)
            cnn_metrics = binary_metrics(cnn_np, target_np)
            cnn_masked = binary_metrics(cnn_np, target_np, mask_np)
            jepa_sample_losses = completion_loss_metrics(jepa_outputs["logits"][local_index:local_index + 1], target[local_index:local_index + 1], mask[local_index:local_index + 1])
            cnn_sample_losses = completion_loss_metrics(cnn_logits[local_index:local_index + 1], target[local_index:local_index + 1], mask[local_index:local_index + 1])
            row: dict[str, Any] = {
                "sample_id": sample_id, "mask_type": config["mask_type"], "missing_ratio": config["missing_ratio"],
                "cnn_full_bce": float(cnn_sample_losses["full_bce"].item()), "cnn_masked_bce": float(cnn_sample_losses["masked_bce"].item()),
                "jepa_full_bce": float(jepa_sample_losses["full_bce"].item()), "jepa_masked_bce": float(jepa_sample_losses["masked_bce"].item()),
                "cnn_full_accuracy": cnn_metrics["accuracy"], "cnn_full_iou": cnn_metrics["iou"], "cnn_full_dice": cnn_metrics["dice"],
                "cnn_masked_accuracy": cnn_masked["accuracy"], "cnn_masked_iou": cnn_masked["iou"], "cnn_masked_dice": cnn_masked["dice"],
                "jepa_full_accuracy": jepa_metrics["accuracy"], "jepa_full_iou": jepa_metrics["iou"], "jepa_full_dice": jepa_metrics["dice"],
                "jepa_masked_accuracy": jepa_masked["accuracy"], "jepa_masked_iou": jepa_masked["iou"], "jepa_masked_dice": jepa_masked["dice"],
                "difference": jepa_masked["iou"] - cnn_masked["iou"], "known_region_error_cnn": float(np.mean(np.abs(cnn_np[mask_np == 0] - target_np[mask_np == 0]))), "known_region_error_jepa": float(np.mean(np.abs(jepa_np[mask_np == 0] - target_np[mask_np == 0]))),
            }
            row.update(geometry_complexity(target_np))
            rows.append(row)
            visual.append({"sample_id": sample_id, "target": target_np, "mask": mask_np, "partial": partial_np, "cnn_completion": cnn_np, "jepa_completion": jepa_np, "cnn_error": np.abs(cnn_np - target_np), "jepa_error": np.abs(jepa_np - target_np), "difference": row["difference"]})

    latent_metrics = latent_variance_metrics(torch.cat(context_latents), torch.cat(target_latents), torch.cat(pred_latents))
    complexity_score = np.asarray([row["connected_components_4"] + row["boundary_transitions_4"] / 32.0 for row in rows])
    q1, q2 = np.quantile(complexity_score, [1 / 3, 2 / 3])
    complexity_summary: dict[str, Any] = {"score_tertiles": [float(q1), float(q2)], "groups": {}}
    for group, selection in (("simple", complexity_score <= q1), ("medium", (complexity_score > q1) & (complexity_score <= q2)), ("complex", complexity_score > q2)):
        selected = [row for row, keep in zip(rows, selection) if keep]
        complexity_summary["groups"][group] = {
            "samples": len(selected),
            "cnn_masked_iou": float(np.mean([row["cnn_masked_iou"] for row in selected])), "jepa_masked_iou": float(np.mean([row["jepa_masked_iou"] for row in selected])), "delta_masked_iou": float(np.mean([row["difference"] for row in selected])),
            "cnn_masked_dice": float(np.mean([row["cnn_masked_dice"] for row in selected])), "jepa_masked_dice": float(np.mean([row["jepa_masked_dice"] for row in selected])),
            "cnn_masked_accuracy": float(np.mean([row["cnn_masked_accuracy"] for row in selected])), "jepa_masked_accuracy": float(np.mean([row["jepa_masked_accuracy"] for row in selected])),
        }

    differences = np.asarray([row["difference"] for row in rows])
    metrics = {
        "checkpoint": str(args.checkpoint), "cnn_checkpoint": str(cnn_checkpoint), "subset_root": str(args.subset_root), "device": str(device), "mask_type": config["mask_type"], "missing_ratio": config["missing_ratio"], "threshold": args.threshold,
        "samples": len(rows), "cnn": aggregate(rows, "cnn"), "jepa": aggregate(rows, "jepa"),
        "paired_difference": {"mean": float(differences.mean()), "median": float(np.median(differences)), "std": float(differences.std()), "p25": float(np.percentile(differences, 25)), "p75": float(np.percentile(differences, 75)), "jepa_wins_fraction": float(np.mean(differences > 0))},
        "known_region_error": {"cnn_max": float(max(row["known_region_error_cnn"] for row in rows)), "jepa_max": float(max(row["known_region_error_jepa"] for row in rows))},
        "complexity": complexity_summary, "latent_variance": latent_metrics,
        "jepa_parameter_count_total": sum(parameter.numel() for parameter in jepa.parameters()), "jepa_trainable_parameter_count": sum(parameter.numel() for parameter in jepa.trainable_parameters()), "cnn_parameter_count": sum(parameter.numel() for parameter in cnn.parameters()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots = args.output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.output_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (plots / "paired_iou_histogram.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("difference",))
        writer.writerows((value,) for value in differences)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(differences, bins=30, color="#386cb0", alpha=0.85)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("JEPA masked IoU - CNN masked IoU")
    axis.set_ylabel("test samples")
    figure.tight_layout()
    figure.savefig(plots / "paired_iou_histogram.png", dpi=180)
    plt.close(figure)
    generator = np.random.default_rng(args.seed)
    random_indexes = generator.choice(len(rows), size=min(10, len(rows)), replace=False)
    difficult_indexes = np.argsort(np.asarray([row["jepa_masked_iou"] for row in rows]))[:min(10, len(rows))]
    complex_indexes = np.flatnonzero(complexity_score > q2)
    complex_indexes = complex_indexes[np.argsort(np.asarray([rows[index]["jepa_masked_iou"] for index in complex_indexes]))[:min(10, len(complex_indexes))]]
    save_comparison_grid(plots / "random_comparisons.png", [visual[index] for index in random_indexes], "Random CNN vs JEPA completions")
    save_comparison_grid(plots / "difficult_comparisons.png", [visual[index] for index in difficult_indexes], "Difficult CNN vs JEPA completions")
    save_comparison_grid(plots / "complex_comparisons.png", [visual[index] for index in complex_indexes], "Complex CNN vs JEPA completions")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
