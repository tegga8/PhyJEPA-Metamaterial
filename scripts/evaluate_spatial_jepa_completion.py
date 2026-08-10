"""Three-way Phase 4.1 evaluation: CNN vs global JEPA vs spatial JEPA."""

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
from src.completion_model import CompletionCNN, compose_binary_completion
from src.forward_analysis import geometry_complexity
from src.jepa_completion_model import JEPACompletionModel, compose_binary_jepa_completion
from src.spatial_jepa_completion_losses import spatial_latent_norm, spatial_latent_statistics
from src.spatial_jepa_completion_model import SpatialJEPACompletionModel, compose_binary_spatial_completion


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


def aggregate(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    metrics = ("full_bce", "masked_bce", "full_accuracy", "full_iou", "full_dice", "masked_accuracy", "masked_iou", "masked_dice")
    return {f"{prefix}_{metric}": float(np.mean([row[f"{prefix}_{metric}"] for row in rows])) for metric in metrics}


def checkpoint_for_mask(mask_type: str, ratio: float) -> Path:
    name = {("central_block", 0.25): "exp_4A", ("central_block", 0.5): "exp_4B", ("random_holes", 0.25): "exp_4C", ("random_holes", 0.5): "exp_4D"}[(mask_type, ratio)]
    return Path("outputs/phase4_jepa") / name / "best.pt"


def paired_summary(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(values.mean()), "median": float(np.median(values)), "std": float(values.std()), "p25": float(np.percentile(values, 25)), "p75": float(np.percentile(values, 75)), "wins_fraction": float(np.mean(values > 0))}


def save_three_way_grid(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    figure, axes = plt.subplots(len(rows), 9, figsize=(19, max(4, 2.4 * len(rows))))
    axes = np.atleast_2d(axes)
    labels = ("target", "mask", "partial", "CNN", "global JEPA", "spatial JEPA", "CNN error", "global error", "spatial error")
    for row_index, row in enumerate(rows):
        images = (row["target"], row["mask"], row["partial"], row["cnn"], row["global"], row["spatial"], row["cnn_error"], row["global_error"], row["spatial_error"])
        for axis, image, label in zip(axes[row_index], images, labels):
            axis.imshow(image, cmap="magma" if "error" in label else "gray_r", vmin=0, vmax=1, interpolation="nearest")
            axis.set_title(label, fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row_index, 0].set_ylabel(f"{row['sample_id']}\nS-C={row['spatial_minus_cnn']:.3f}", fontsize=7)
    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_latent_grid(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    figure, axes = plt.subplots(len(rows), 3, figsize=(7, max(4, 2.2 * len(rows))))
    axes = np.atleast_2d(axes)
    labels = ("context norm", "target norm", "predicted norm")
    for row_index, row in enumerate(rows):
        for axis, key, label in zip(axes[row_index], ("context_norm_map", "target_norm_map", "pred_norm_map"), labels):
            axis.imshow(row[key], cmap="viridis", interpolation="nearest")
            axis.set_title(label, fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row_index, 0].set_ylabel(str(row["sample_id"]), fontsize=7)
    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Spatial JEPA checkpoint")
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--global-checkpoint", type=Path, default=None)
    parser.add_argument("--cnn-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    spatial_config = load_json(args.checkpoint.parent / "config.json")
    global_checkpoint = args.global_checkpoint or checkpoint_for_mask(spatial_config["mask_type"], float(spatial_config["missing_ratio"]))
    cnn_checkpoint = args.cnn_checkpoint or Path("outputs/phase3_completion") / {("central_block", 0.25): "exp_3A", ("central_block", 0.5): "exp_3B", ("random_holes", 0.25): "exp_3C", ("random_holes", 0.5): "exp_3D"}[(spatial_config["mask_type"], float(spatial_config["missing_ratio"]))] / "best.pt"
    global_config = load_json(global_checkpoint.parent / "config.json")
    dataset = CompletionDataset(args.subset_root, "test", spatial_config["mask_type"], spatial_config["missing_ratio"], spatial_config["mask_seed"])
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    spatial = SpatialJEPACompletionModel(spatial_config["latent_channels"], spatial_config["predictor_hidden_channels"], spatial_config["ema_decay"]).to(device)
    spatial.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    spatial.eval()
    global_model = JEPACompletionModel(global_config["latent_dim"], global_config["predictor_hidden_dim"], global_config["ema_decay"]).to(device)
    global_model.load_state_dict(torch.load(global_checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    global_model.eval()
    cnn = CompletionCNN().to(device)
    cnn.load_state_dict(torch.load(cnn_checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    cnn.eval()

    rows: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    latent_visuals: list[dict[str, Any]] = []
    context_latents: list[torch.Tensor] = []
    target_latents: list[torch.Tensor] = []
    pred_latents: list[torch.Tensor] = []
    for batch in loader:
        inputs, target, mask = batch["input"].to(device), batch["target"].to(device), batch["mask"].to(device)
        with torch.inference_mode():
            spatial_outputs = spatial(inputs, target)
            global_outputs = global_model(inputs, target)
            cnn_logits = cnn(inputs)
        spatial_probability = torch.sigmoid(spatial_outputs["logits"])
        global_probability = torch.sigmoid(global_outputs["logits"])
        cnn_probability = torch.sigmoid(cnn_logits)
        spatial_binary = compose_binary_spatial_completion(spatial_probability, inputs, mask, args.threshold)
        global_binary = compose_binary_jepa_completion(global_probability, inputs, mask, args.threshold)
        cnn_binary = compose_binary_completion(cnn_probability, inputs, mask, args.threshold)
        context_latents.append(spatial_outputs["z_context"].cpu())
        target_latents.append(spatial_outputs["z_target"].cpu())
        pred_latents.append(spatial_outputs["z_pred"].cpu())
        context_maps = spatial_latent_norm(spatial_outputs["z_context"]).cpu().numpy()
        target_maps = spatial_latent_norm(spatial_outputs["z_target"]).cpu().numpy()
        pred_maps = spatial_latent_norm(spatial_outputs["z_pred"]).cpu().numpy()
        for local_index, sample_id in enumerate(batch["sample_id"]):
            target_np = target[local_index, 0].cpu().numpy()
            mask_np = mask[local_index, 0].cpu().numpy()
            partial_np = inputs[local_index, 0].cpu().numpy()
            predictions = {"cnn": cnn_binary[local_index, 0].cpu().numpy(), "global": global_binary[local_index, 0].cpu().numpy(), "spatial": spatial_binary[local_index, 0].cpu().numpy()}
            metrics: dict[str, dict[str, float]] = {}
            logits_by_name = {"cnn": cnn_logits, "global": global_outputs["logits"], "spatial": spatial_outputs["logits"]}
            for name, prediction in predictions.items():
                full = binary_metrics(prediction, target_np)
                masked = binary_metrics(prediction, target_np, mask_np)
                losses = completion_loss_metrics(logits_by_name[name][local_index:local_index + 1], target[local_index:local_index + 1], mask[local_index:local_index + 1])
                metrics[name] = {"full_bce": float(losses["full_bce"].item()), "masked_bce": float(losses["masked_bce"].item()), **{f"full_{key}": value for key, value in full.items()}, **{f"masked_{key}": value for key, value in masked.items()}}
            spatial_minus_cnn = metrics["spatial"]["masked_iou"] - metrics["cnn"]["masked_iou"]
            spatial_minus_global = metrics["spatial"]["masked_iou"] - metrics["global"]["masked_iou"]
            known_errors = {name: float(np.mean(np.abs(prediction[mask_np == 0] - target_np[mask_np == 0]))) for name, prediction in predictions.items()}
            row: dict[str, Any] = {"sample_id": sample_id, "mask_type": spatial_config["mask_type"], "missing_ratio": spatial_config["missing_ratio"], "spatial_minus_cnn": spatial_minus_cnn, "spatial_minus_global_jepa": spatial_minus_global, "known_region_error_cnn": known_errors["cnn"], "known_region_error_global_jepa": known_errors["global"], "known_region_error_spatial_jepa": known_errors["spatial"]}
            for name in ("cnn", "global", "spatial"):
                prefix = {"cnn": "cnn", "global": "global_jepa", "spatial": "spatial_jepa"}[name]
                row.update({f"{prefix}_{key}": value for key, value in metrics[name].items()})
            row.update(geometry_complexity(target_np))
            rows.append(row)
            visuals.append({"sample_id": sample_id, "target": target_np, "mask": mask_np, "partial": partial_np, "cnn": predictions["cnn"], "global": predictions["global"], "spatial": predictions["spatial"], "cnn_error": np.abs(predictions["cnn"] - target_np), "global_error": np.abs(predictions["global"] - target_np), "spatial_error": np.abs(predictions["spatial"] - target_np), "spatial_minus_cnn": spatial_minus_cnn})
            latent_visuals.append({"sample_id": sample_id, "context_norm_map": context_maps[local_index], "target_norm_map": target_maps[local_index], "pred_norm_map": pred_maps[local_index], "context_mean": float(spatial_outputs["z_context"][local_index].mean().item()), "target_mean": float(spatial_outputs["z_target"][local_index].mean().item()), "pred_mean": float(spatial_outputs["z_pred"][local_index].mean().item()), "context_std": float(spatial_outputs["z_context"][local_index].std().item()), "target_std": float(spatial_outputs["z_target"][local_index].std().item()), "pred_std": float(spatial_outputs["z_pred"][local_index].std().item()), "context_norm_mean": float(context_maps[local_index].mean()), "target_norm_mean": float(target_maps[local_index].mean()), "pred_norm_mean": float(pred_maps[local_index].mean())})

    spatial_latents = spatial_latent_statistics(torch.cat(context_latents), torch.cat(target_latents), torch.cat(pred_latents))
    complexity_score = np.asarray([row["connected_components_4"] + row["boundary_transitions_4"] / 32.0 for row in rows])
    q1, q2 = np.quantile(complexity_score, [1 / 3, 2 / 3])
    complexity_summary: dict[str, Any] = {"score_tertiles": [float(q1), float(q2)], "groups": {}}
    for group, selection in (("simple", complexity_score <= q1), ("medium", (complexity_score > q1) & (complexity_score <= q2)), ("complex", complexity_score > q2)):
        selected = [row for row, keep in zip(rows, selection) if keep]
        group_result: dict[str, Any] = {"samples": len(selected)}
        for name in ("cnn", "global_jepa", "spatial_jepa"):
            group_result[f"{name}_masked_iou"] = float(np.mean([row[f"{name}_masked_iou"] for row in selected]))
            group_result[f"{name}_masked_dice"] = float(np.mean([row[f"{name}_masked_dice"] for row in selected]))
            group_result[f"{name}_masked_accuracy"] = float(np.mean([row[f"{name}_masked_accuracy"] for row in selected]))
        group_result["spatial_minus_cnn_iou"] = group_result["spatial_jepa_masked_iou"] - group_result["cnn_masked_iou"]
        group_result["spatial_minus_global_iou"] = group_result["spatial_jepa_masked_iou"] - group_result["global_jepa_masked_iou"]
        complexity_summary["groups"][group] = group_result

    spatial_cnn_delta = np.asarray([row["spatial_minus_cnn"] for row in rows])
    spatial_global_delta = np.asarray([row["spatial_minus_global_jepa"] for row in rows])
    metrics = {
        "phase": "4.1", "checkpoint": str(args.checkpoint), "global_checkpoint": str(global_checkpoint), "cnn_checkpoint": str(cnn_checkpoint), "subset_root": str(args.subset_root), "device": str(device), "mask_type": spatial_config["mask_type"], "missing_ratio": spatial_config["missing_ratio"], "threshold": args.threshold, "samples": len(rows),
        "cnn": aggregate(rows, "cnn"), "global_jepa": aggregate(rows, "global_jepa"), "spatial_jepa": aggregate(rows, "spatial_jepa"),
        "paired_spatial_minus_cnn": paired_summary(spatial_cnn_delta), "paired_spatial_minus_global": paired_summary(spatial_global_delta),
        "known_region_error": {"cnn_max": max(row["known_region_error_cnn"] for row in rows), "global_jepa_max": max(row["known_region_error_global_jepa"] for row in rows), "spatial_jepa_max": max(row["known_region_error_spatial_jepa"] for row in rows)},
        "complexity": complexity_summary, "spatial_latent_statistics": spatial_latents,
        "cnn_parameter_count": sum(parameter.numel() for parameter in cnn.parameters()), "global_jepa_parameter_count_total": sum(parameter.numel() for parameter in global_model.parameters()), "global_jepa_trainable_parameter_count": sum(parameter.numel() for parameter in global_model.trainable_parameters()), "spatial_jepa_parameter_count_total": sum(parameter.numel() for parameter in spatial.parameters()), "spatial_jepa_trainable_parameter_count": sum(parameter.numel() for parameter in spatial.trainable_parameters()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots = args.output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "latent_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [key for key in latent_visuals[0] if not key.endswith("_map")]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key in fields} for row in latent_visuals)
    for values, filename, label in ((spatial_cnn_delta, "spatial_minus_cnn", "Spatial JEPA masked IoU - CNN masked IoU"), (spatial_global_delta, "spatial_minus_global", "Spatial JEPA masked IoU - global JEPA masked IoU")):
        with (plots / f"{filename}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("difference",))
            writer.writerows((value,) for value in values)
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.hist(values, bins=30, color="#386cb0", alpha=0.85)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel(label)
        axis.set_ylabel("test samples")
        figure.tight_layout()
        figure.savefig(plots / f"{filename}.png", dpi=180)
        plt.close(figure)
    generator = np.random.default_rng(args.seed)
    random_indexes = generator.choice(len(rows), size=min(10, len(rows)), replace=False)
    difficult_indexes = np.argsort(np.asarray([row["spatial_jepa_masked_iou"] for row in rows]))[:min(10, len(rows))]
    complex_indexes = np.flatnonzero(complexity_score > q2)
    complex_indexes = complex_indexes[np.argsort(np.asarray([rows[index]["spatial_jepa_masked_iou"] for index in complex_indexes]))[:min(10, len(complex_indexes))]]
    save_three_way_grid(plots / "random_comparisons.png", [visuals[index] for index in random_indexes], "Random CNN vs global JEPA vs spatial JEPA")
    save_three_way_grid(plots / "difficult_comparisons.png", [visuals[index] for index in difficult_indexes], "Difficult CNN vs global JEPA vs spatial JEPA")
    save_three_way_grid(plots / "complex_comparisons.png", [visuals[index] for index in complex_indexes], "Complex CNN vs global JEPA vs spatial JEPA")
    latent_indexes = random_indexes[:min(10, len(random_indexes))]
    save_latent_grid(plots / "latent_norm_maps.png", [latent_visuals[index] for index in latent_indexes], "Spatial latent norm maps")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
