"""Evaluate a Phase 3 completion checkpoint and save visual diagnostics."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.completion_dataset import CompletionDataset
from src.completion_losses import completion_loss_metrics
from src.completion_model import CompletionCNN, compose_binary_completion, compose_completion
from src.forward_analysis import geometry_complexity


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
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    predicted_count = prediction.sum()
    target_count = target.sum()
    return {
        "accuracy": float(np.mean(prediction == target)),
        "iou": float(intersection / union) if union else 1.0,
        "dice": float(2 * intersection / (predicted_count + target_count)) if predicted_count + target_count else 1.0,
    }


def evaluate_baseline(targets: np.ndarray, partials: np.ndarray, masks: np.ndarray, fill_value: float) -> dict[str, float]:
    prediction = partials.copy()
    prediction[masks.astype(bool)] = fill_value
    full = binary_metrics(prediction, targets)
    masked = binary_metrics(prediction, targets, masks)
    return {f"full_{name}": value for name, value in full.items()} | {f"masked_{name}": value for name, value in masked.items()}


def save_prediction_grid(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    figure, axes = plt.subplots(len(rows), 6, figsize=(14, max(4, 2.4 * len(rows))))
    axes = np.atleast_2d(axes)
    labels = ("target", "mask", "partial", "probability", "thresholded", "completed")
    for row_index, row in enumerate(rows):
        images = (row["target"], row["mask"], row["partial"], row["probability"], row["thresholded"], row["completed"])
        for axis, image, label in zip(axes[row_index], images, labels):
            axis.imshow(image, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            axis.set_title(label, fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row_index, 0].set_ylabel(f"{row['sample_id']}\nIoU={row['masked_iou']:.3f}", fontsize=7)
    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    config = json.loads((args.checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    threshold = float(args.threshold if args.threshold is not None else config.get("threshold", 0.5))
    dataset = CompletionDataset(args.subset_root, "test", config["mask_type"], config["missing_ratio_requested"], config["mask_seed"])
    model = CompletionCNN().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[dict[str, Any]] = []
    targets: list[np.ndarray] = []
    partials: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    full_bce_total = masked_bce_total = 0.0
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            inputs = sample["input"].unsqueeze(0).to(device)
            target = sample["target"].unsqueeze(0).to(device)
            mask = sample["mask"].unsqueeze(0).to(device)
            logits = model(inputs)
            losses = completion_loss_metrics(logits, target, mask)
            full_bce_total += float(losses["full_bce"].item())
            masked_bce_total += float(losses["masked_bce"].item())
            probability = torch.sigmoid(logits)
            completed_probability = compose_completion(probability, inputs, mask)
            completed_binary = compose_binary_completion(probability, inputs, mask, threshold)
            target_np = target[0, 0].cpu().numpy()
            mask_np = mask[0, 0].cpu().numpy()
            partial_np = inputs[0, 0].cpu().numpy()
            probability_np = completed_probability[0, 0].cpu().numpy()
            completed_np = completed_binary[0, 0].cpu().numpy()
            full = binary_metrics(completed_np, target_np)
            masked = binary_metrics(completed_np, target_np, mask_np)
            known_error = float(np.mean(np.abs(completed_np[mask_np == 0] - target_np[mask_np == 0])))
            row = {
                "sample_id": sample["sample_id"], "index": index, "masked_pixels": int(mask_np.sum()),
                "full_accuracy": full["accuracy"], "full_iou": full["iou"], "full_dice": full["dice"],
                "masked_accuracy": masked["accuracy"], "masked_iou": masked["iou"], "masked_dice": masked["dice"],
                "known_region_error": known_error,
            }
            row.update(geometry_complexity(target_np))
            rows.append(row)
            targets.append(target_np)
            partials.append(partial_np)
            masks.append(mask_np)
            probabilities.append(probability_np)

    target_array = np.asarray(targets)
    partial_array = np.asarray(partials)
    mask_array = np.asarray(masks)
    train_dataset = CompletionDataset(args.subset_root, "train", config["mask_type"], config["missing_ratio_requested"], config["mask_seed"])
    training_occupancy = float(np.mean([sample["target"].numpy() for sample in train_dataset]))
    prior = training_occupancy
    baselines = {"zero_fill": evaluate_baseline(target_array, partial_array, mask_array, 0.0), "occupancy_prior": evaluate_baseline(target_array, partial_array, mask_array, float(prior >= threshold))}

    scores = np.asarray([row["masked_iou"] for row in rows])
    complexity = np.asarray([row["connected_components_4"] + row["boundary_transitions_4"] / 32.0 for row in rows])
    q1, q2 = np.quantile(complexity, [1 / 3, 2 / 3]) if len(complexity) else (0.0, 0.0)
    for row, score in zip(rows, complexity):
        row["complexity_group"] = "simple" if score <= q1 else "medium" if score <= q2 else "complex"
    group_metrics: dict[str, dict[str, float | int]] = {}
    for group in ("simple", "medium", "complex"):
        values = [row for row in rows if row["complexity_group"] == group]
        group_metrics[group] = {
            "samples": len(values), "mean_masked_iou": float(np.mean([row["masked_iou"] for row in values])) if values else float("nan"),
            "median_masked_iou": float(np.median([row["masked_iou"] for row in values])) if values else float("nan"),
            "mean_masked_dice": float(np.mean([row["masked_dice"] for row in values])) if values else float("nan"),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots = args.output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.output_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    random_indexes = np.random.default_rng(args.seed).choice(len(rows), size=min(20, len(rows)), replace=False)
    worst_indexes = np.argsort(scores)[:min(10, len(rows))]
    random_rows = [{"sample_id": rows[index]["sample_id"], "target": target_array[index], "mask": mask_array[index], "partial": partial_array[index], "probability": probabilities[index], "thresholded": (probabilities[index] >= threshold).astype(float), "completed": (partial_array[index] * (1 - mask_array[index]) + (probabilities[index] >= threshold).astype(float) * mask_array[index]), "masked_iou": rows[index]["masked_iou"]} for index in random_indexes]
    worst_rows = [{"sample_id": rows[index]["sample_id"], "target": target_array[index], "mask": mask_array[index], "partial": partial_array[index], "probability": probabilities[index], "thresholded": (probabilities[index] >= threshold).astype(float), "completed": (partial_array[index] * (1 - mask_array[index]) + (probabilities[index] >= threshold).astype(float) * mask_array[index]), "masked_iou": rows[index]["masked_iou"]} for index in worst_indexes]
    save_prediction_grid(plots / "random_predictions.png", random_rows, "Random completion predictions")
    save_prediction_grid(plots / "worst_masked_iou.png", worst_rows, "Worst completion predictions by masked IoU")

    aggregate = {
        "checkpoint": str(args.checkpoint), "subset_root": str(args.subset_root), "split": "test", "device": str(device),
        "model": "CompletionCNN", "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()), "threshold": threshold,
        "samples": len(rows), "full_bce": full_bce_total / len(rows), "masked_bce": masked_bce_total / len(rows), "training_occupancy_prior": prior,
        "full_accuracy": float(np.mean([row["full_accuracy"] for row in rows])), "full_iou": float(np.mean([row["full_iou"] for row in rows])), "full_dice": float(np.mean([row["full_dice"] for row in rows])),
        "masked_accuracy": float(np.mean([row["masked_accuracy"] for row in rows])), "masked_iou": float(np.mean([row["masked_iou"] for row in rows])), "masked_dice": float(np.mean([row["masked_dice"] for row in rows])),
        "known_region_error": float(np.max([row["known_region_error"] for row in rows])), "baselines": baselines,
        "complexity": {"score_tertiles": [float(q1), float(q2)], "groups": group_metrics},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
