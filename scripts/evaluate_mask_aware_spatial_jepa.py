"""Four-way Phase 4.2 evaluation with boundary and mask-alignment diagnostics."""

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
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.completion_dataset import CompletionDataset
from src.completion_losses import completion_loss_metrics
from src.completion_model import CompletionCNN, compose_binary_completion
from src.forward_analysis import geometry_complexity
from src.jepa_completion_model import JEPACompletionModel, compose_binary_jepa_completion
from src.mask_aware_spatial_jepa_losses import downsample_mask, latent_norm, mask_weight_map, spatial_latent_statistics
from src.spatial_jepa_completion_model import SpatialJEPACompletionModel, compose_binary_spatial_completion


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def binary_metrics(prediction: np.ndarray, target: np.ndarray, selection: np.ndarray | None = None) -> dict[str, float] | None:
    if selection is not None:
        if int(selection.sum()) == 0:
            return None
        prediction, target = prediction[selection.astype(bool)], target[selection.astype(bool)]
    prediction, target = prediction.astype(bool), target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    predicted_count, target_count = prediction.sum(), target.sum()
    return {"accuracy": float(np.mean(prediction == target)), "iou": float(intersection / union) if union else 1.0, "dice": float(2 * intersection / (predicted_count + target_count)) if predicted_count + target_count else 1.0}


def aggregate(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    metrics = ("full_bce", "masked_bce", "full_accuracy", "full_iou", "full_dice", "masked_accuracy", "masked_iou", "masked_dice")
    return {f"{prefix}_{metric}": float(np.mean([row[f"{prefix}_{metric}"] for row in rows])) for metric in metrics}


def model_checkpoint_paths(mask_type: str, ratio: float) -> tuple[Path, Path, Path]:
    phase_map = {("central_block", 0.25): ("exp_3A", "exp_4A", "exp_4_1A"), ("central_block", 0.5): ("exp_3B", "exp_4B", "exp_4_1B"), ("random_holes", 0.25): ("exp_3C", "exp_4C", "exp_4_1C"), ("random_holes", 0.5): ("exp_3D", "exp_4D", "exp_4_1D")}[mask_type, ratio]
    return Path("outputs/phase3_completion") / phase_map[0] / "best.pt", Path("outputs/phase4_jepa") / phase_map[1] / "best.pt", Path("outputs/phase4_1") / phase_map[2] / "best.pt"


def boundary_masks(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    known = 1.0 - mask
    padded = F.pad(known, (1, 1, 1, 1))
    neighbor_known = padded[:, :, 0:-2, 1:-1] + padded[:, :, 2:, 1:-1] + padded[:, :, 1:-1, 0:-2] + padded[:, :, 1:-1, 2:]
    boundary = (mask > 0) & (neighbor_known > 0)
    interior = (mask > 0) & ~boundary
    return boundary.float(), interior.float()


def paired_summary(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(values.mean()), "median": float(np.median(values)), "std": float(values.std()), "p25": float(np.percentile(values, 25)), "p75": float(np.percentile(values, 75)), "wins_fraction": float(np.mean(values > 0))}


def save_four_way_grid(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    figure, axes = plt.subplots(len(rows), 10, figsize=(21, max(4, 2.4 * len(rows))))
    axes = np.atleast_2d(axes)
    labels = ("target", "mask", "partial", "CNN", "global", "spatial", "mask-aware", "CNN error", "spatial error", "mask-aware error")
    for row_index, row in enumerate(rows):
        images = (row["target"], row["mask"], row["partial"], row["cnn"], row["global"], row["spatial"], row["mask_aware"], row["cnn_error"], row["spatial_error"], row["mask_aware_error"])
        for axis, image, label in zip(axes[row_index], images, labels):
            axis.imshow(image, cmap="magma" if "error" in label else "gray_r", vmin=0, vmax=1, interpolation="nearest")
            axis.set_title(label, fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row_index, 0].set_ylabel(f"{row['sample_id']}\nM-S={row['mask_aware_minus_spatial']:.3f}", fontsize=7)
    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_alignment_grid(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    figure, axes = plt.subplots(len(rows), 7, figsize=(15, max(4, 2.4 * len(rows))))
    axes = np.atleast_2d(axes)
    labels = ("original", "mask 16x16", "mask 8x8", "context norm", "target norm", "pred norm", "weight W")
    keys = ("target", "mask", "mask8", "context_norm", "target_norm", "pred_norm", "weights")
    for row_index, row in enumerate(rows):
        for axis, key, label in zip(axes[row_index], keys, labels):
            axis.imshow(row[key], cmap="viridis" if key not in ("target", "mask") else "gray_r", vmin=0, vmax=1 if key in ("target", "mask", "mask8", "weights") else None, interpolation="nearest")
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
    parser.add_argument("--checkpoint", type=Path, required=True, help="Mask-aware spatial JEPA checkpoint")
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spatial-checkpoint", type=Path, default=None)
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
    mask_config = load_json(args.checkpoint.parent / "config.json")
    default_cnn, default_global, default_spatial = model_checkpoint_paths(mask_config["mask_type"], float(mask_config["missing_ratio"]))
    cnn_checkpoint = args.cnn_checkpoint or default_cnn
    global_checkpoint = args.global_checkpoint or default_global
    spatial_checkpoint = args.spatial_checkpoint or default_spatial
    global_config = load_json(global_checkpoint.parent / "config.json")
    spatial_config = load_json(spatial_checkpoint.parent / "config.json")
    dataset = CompletionDataset(args.subset_root, "test", mask_config["mask_type"], mask_config["missing_ratio"], mask_config["mask_seed"])
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    mask_aware = SpatialJEPACompletionModel(mask_config["latent_channels"], mask_config["predictor_hidden_channels"], mask_config["ema_decay"]).to(device)
    mask_aware.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    mask_aware.eval()
    spatial = SpatialJEPACompletionModel(spatial_config["latent_channels"], spatial_config["predictor_hidden_channels"], spatial_config["ema_decay"]).to(device)
    spatial.load_state_dict(torch.load(spatial_checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    spatial.eval()
    global_model = JEPACompletionModel(global_config["latent_dim"], global_config["predictor_hidden_dim"], global_config["ema_decay"]).to(device)
    global_model.load_state_dict(torch.load(global_checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    global_model.eval()
    cnn = CompletionCNN().to(device)
    cnn.load_state_dict(torch.load(cnn_checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    cnn.eval()

    rows: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    alignments: list[dict[str, Any]] = []
    latent_context: list[torch.Tensor] = []
    latent_target: list[torch.Tensor] = []
    latent_pred: list[torch.Tensor] = []
    weight_rows: list[dict[str, Any]] = []
    model_names = ("cnn", "global_jepa", "spatial_jepa", "mask_aware_spatial_jepa")
    for batch in loader:
        inputs, target, mask = batch["input"].to(device), batch["target"].to(device), batch["mask"].to(device)
        with torch.inference_mode():
            mask_outputs = mask_aware(inputs, target)
            spatial_outputs = spatial(inputs, target)
            global_outputs = global_model(inputs, target)
            cnn_logits = cnn(inputs)
        mask_probability = torch.sigmoid(mask_outputs["logits"])
        spatial_probability = torch.sigmoid(spatial_outputs["logits"])
        global_probability = torch.sigmoid(global_outputs["logits"])
        cnn_probability = torch.sigmoid(cnn_logits)
        predictions = {"cnn": compose_binary_completion(cnn_probability, inputs, mask, args.threshold), "global_jepa": compose_binary_jepa_completion(global_probability, inputs, mask, args.threshold), "spatial_jepa": compose_binary_spatial_completion(spatial_probability, inputs, mask, args.threshold), "mask_aware_spatial_jepa": compose_binary_spatial_completion(mask_probability, inputs, mask, args.threshold)}
        logits_by_name = {"cnn": cnn_logits, "global_jepa": global_outputs["logits"], "spatial_jepa": spatial_outputs["logits"], "mask_aware_spatial_jepa": mask_outputs["logits"]}
        latent_context.append(mask_outputs["z_context"].cpu())
        latent_target.append(mask_outputs["z_target"].cpu())
        latent_pred.append(mask_outputs["z_pred"].cpu())
        mask8_batch = downsample_mask(mask).cpu().numpy()
        weights_batch = mask_weight_map(mask, mask_config["alpha"], mask_config["gamma"]).cpu().numpy()
        context_maps = latent_norm(mask_outputs["z_context"]).cpu().numpy()
        target_maps = latent_norm(mask_outputs["z_target"]).cpu().numpy()
        pred_maps = latent_norm(mask_outputs["z_pred"]).cpu().numpy()
        boundary_batch, interior_batch = boundary_masks(mask)
        for local_index, sample_id in enumerate(batch["sample_id"]):
            target_np, mask_np, partial_np = target[local_index, 0].cpu().numpy(), mask[local_index, 0].cpu().numpy(), inputs[local_index, 0].cpu().numpy()
            boundary_np, interior_np = boundary_batch[local_index, 0].cpu().numpy(), interior_batch[local_index, 0].cpu().numpy()
            row: dict[str, Any] = {"sample_id": sample_id, "mask_type": mask_config["mask_type"], "missing_ratio": mask_config["missing_ratio"], "mask_aware_minus_spatial": 0.0, "mask_aware_minus_cnn": 0.0}
            prediction_np: dict[str, np.ndarray] = {}
            all_metrics: dict[str, dict[str, float]] = {}
            for name in model_names:
                prediction_np[name] = predictions[name][local_index, 0].cpu().numpy()
                full = binary_metrics(prediction_np[name], target_np)
                masked = binary_metrics(prediction_np[name], target_np, mask_np)
                losses = completion_loss_metrics(logits_by_name[name][local_index:local_index + 1], target[local_index:local_index + 1], mask[local_index:local_index + 1])
                all_metrics[name] = {"full_bce": float(losses["full_bce"].item()), "masked_bce": float(losses["masked_bce"].item()), **{f"full_{key}": value for key, value in full.items()}, **{f"masked_{key}": value for key, value in masked.items()}}
                prefix = name
                row.update({f"{prefix}_{key}": value for key, value in all_metrics[name].items()})
                boundary_metrics = binary_metrics(prediction_np[name], target_np, boundary_np)
                interior_metrics = binary_metrics(prediction_np[name], target_np, interior_np)
                for region, region_metrics in (("boundary", boundary_metrics), ("interior", interior_metrics)):
                    row[f"{prefix}_{region}_iou"] = None if region_metrics is None else region_metrics["iou"]
                    row[f"{prefix}_{region}_dice"] = None if region_metrics is None else region_metrics["dice"]
                    row[f"{prefix}_{region}_accuracy"] = None if region_metrics is None else region_metrics["accuracy"]
            row.update({f"known_region_error_{name}": float(np.mean(np.abs(prediction_np[name][mask_np == 0] - target_np[mask_np == 0]))) for name in model_names})
            row["mask_aware_minus_spatial"] = row["mask_aware_spatial_jepa_masked_iou"] - row["spatial_jepa_masked_iou"]
            row["mask_aware_minus_cnn"] = row["mask_aware_spatial_jepa_masked_iou"] - row["cnn_masked_iou"]
            complexity = geometry_complexity(target_np)
            complexity_score = complexity["connected_components_4"] + complexity["boundary_transitions_4"] / 32.0
            row["complexity_score"] = complexity_score
            row.update(complexity)
            rows.append(row)
            visuals.append({"sample_id": sample_id, "target": target_np, "mask": mask_np, "partial": partial_np, "cnn": prediction_np["cnn"], "global": prediction_np["global_jepa"], "spatial": prediction_np["spatial_jepa"], "mask_aware": prediction_np["mask_aware_spatial_jepa"], "cnn_error": np.abs(prediction_np["cnn"] - target_np), "spatial_error": np.abs(prediction_np["spatial_jepa"] - target_np), "mask_aware_error": np.abs(prediction_np["mask_aware_spatial_jepa"] - target_np), "mask_aware_minus_spatial": row["mask_aware_minus_spatial"]})
            alignments.append({"sample_id": sample_id, "target": target_np, "mask": mask_np, "mask8": mask8_batch[local_index, 0], "context_norm": context_maps[local_index], "target_norm": target_maps[local_index], "pred_norm": pred_maps[local_index], "weights": weights_batch[local_index, 0]})
            weight_rows.append({"sample_id": sample_id, "mean_mask8": float(mask8_batch[local_index].mean()), "min_mask8": float(mask8_batch[local_index].min()), "max_mask8": float(mask8_batch[local_index].max()), "mean_weight": float(weights_batch[local_index].mean()), "min_weight": float(weights_batch[local_index].min()), "max_weight": float(weights_batch[local_index].max())})

    complexity_score = np.asarray([row["complexity_score"] for row in rows])
    q1, q2 = np.quantile(complexity_score, [1 / 3, 2 / 3])
    for row, score in zip(rows, complexity_score):
        row["complexity_group"] = "simple" if score <= q1 else "medium" if score <= q2 else "complex"
    complexity_summary: dict[str, Any] = {"score_tertiles": [float(q1), float(q2)], "groups": {}}
    for group, selection in (("simple", complexity_score <= q1), ("medium", (complexity_score > q1) & (complexity_score <= q2)), ("complex", complexity_score > q2)):
        selected = [row for row, keep in zip(rows, selection) if keep]
        summary: dict[str, Any] = {"samples": len(selected)}
        for name in model_names:
            summary[f"{name}_masked_iou"] = float(np.mean([row[f"{name}_masked_iou"] for row in selected]))
            summary[f"{name}_masked_dice"] = float(np.mean([row[f"{name}_masked_dice"] for row in selected]))
            summary[f"{name}_masked_accuracy"] = float(np.mean([row[f"{name}_masked_accuracy"] for row in selected]))
        summary["mask_aware_minus_spatial_iou"] = summary["mask_aware_spatial_jepa_masked_iou"] - summary["spatial_jepa_masked_iou"]
        summary["mask_aware_minus_cnn_iou"] = summary["mask_aware_spatial_jepa_masked_iou"] - summary["cnn_masked_iou"]
        complexity_summary["groups"][group] = summary

    boundary_summary: dict[str, Any] = {"models": {}}
    for region in ("boundary", "interior"):
        boundary_summary[region] = {"available_samples": int(sum(row[f"mask_aware_spatial_jepa_{region}_iou"] is not None for row in rows)), "models": {}}
        for name in model_names:
            region_summary: dict[str, Any] = {}
            for metric in ("iou", "dice", "accuracy"):
                values = [row[f"{name}_{region}_{metric}"] for row in rows if row[f"{name}_{region}_{metric}"] is not None]
                region_summary[metric] = None if not values else float(np.mean(values))
            boundary_summary[region]["models"][name] = region_summary
    spatial_delta = np.asarray([row["mask_aware_minus_spatial"] for row in rows])
    cnn_delta = np.asarray([row["mask_aware_minus_cnn"] for row in rows])
    latent_metrics = spatial_latent_statistics(torch.cat(latent_context), torch.cat(latent_target), torch.cat(latent_pred))
    metrics = {
        "phase": "4.2", "checkpoint": str(args.checkpoint), "spatial_checkpoint": str(spatial_checkpoint), "global_checkpoint": str(global_checkpoint), "cnn_checkpoint": str(cnn_checkpoint), "subset_root": str(args.subset_root), "device": str(device), "mask_type": mask_config["mask_type"], "missing_ratio": mask_config["missing_ratio"], "alpha": mask_config["alpha"], "gamma": mask_config["gamma"], "lambda_recon": mask_config["lambda_recon"], "threshold": args.threshold, "samples": len(rows),
        "cnn": aggregate(rows, "cnn"), "global_jepa": aggregate(rows, "global_jepa"), "spatial_jepa": aggregate(rows, "spatial_jepa"), "mask_aware_spatial_jepa": aggregate(rows, "mask_aware_spatial_jepa"),
        "paired_mask_aware_minus_spatial": paired_summary(spatial_delta), "paired_mask_aware_minus_cnn": paired_summary(cnn_delta), "known_region_error": {name: float(max(row[f"known_region_error_{name}"] for row in rows)) for name in model_names}, "complexity": complexity_summary, "boundary_analysis": boundary_summary, "spatial_latent_statistics": latent_metrics,
        "parameter_counts": {"cnn": sum(parameter.numel() for parameter in cnn.parameters()), "global_jepa_total": sum(parameter.numel() for parameter in global_model.parameters()), "global_jepa_trainable": sum(parameter.numel() for parameter in global_model.trainable_parameters()), "spatial_jepa_total": sum(parameter.numel() for parameter in spatial.parameters()), "spatial_jepa_trainable": sum(parameter.numel() for parameter in spatial.trainable_parameters()), "mask_aware_spatial_jepa_total": sum(parameter.numel() for parameter in mask_aware.parameters()), "mask_aware_spatial_jepa_trainable": sum(parameter.numel() for parameter in mask_aware.trainable_parameters())},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots = args.output_dir / "plots"
    alignment_dir = plots / "mask_latent_alignment"
    plots.mkdir(parents=True, exist_ok=True)
    alignment_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "mask_weight_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(weight_rows[0]))
        writer.writeheader()
        writer.writerows(weight_rows)
    for values, filename, label in ((spatial_delta, "mask_aware_minus_spatial", "Mask-aware spatial JEPA masked IoU - spatial JEPA masked IoU"), (cnn_delta, "mask_aware_minus_cnn", "Mask-aware spatial JEPA masked IoU - CNN masked IoU")):
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
    difficult_indexes = np.argsort(np.asarray([row["mask_aware_spatial_jepa_masked_iou"] for row in rows]))[:min(10, len(rows))]
    complex_indexes = np.flatnonzero(complexity_score > q2)
    complex_indexes = complex_indexes[np.argsort(np.asarray([rows[index]["mask_aware_spatial_jepa_masked_iou"] for index in complex_indexes]))[:min(10, len(complex_indexes))]]
    save_four_way_grid(plots / "random_comparisons.png", [visuals[index] for index in random_indexes], "CNN vs global JEPA vs spatial JEPA vs mask-aware JEPA")
    save_four_way_grid(plots / "difficult_comparisons.png", [visuals[index] for index in difficult_indexes], "Difficult four-way completion comparison")
    save_four_way_grid(plots / "complex_comparisons.png", [visuals[index] for index in complex_indexes], "Complex four-way completion comparison")
    save_alignment_grid(alignment_dir / "alignment_examples.png", [alignments[index] for index in random_indexes], "Mask-latent alignment")
    (args.output_dir / "latent_statistics.csv").write_text("sample_id,context_norm_mean,target_norm_mean,pred_norm_mean\n" + "\n".join(f"{row['sample_id']},{row['context_norm'].mean()},{row['target_norm'].mean()},{row['pred_norm'].mean()}" for row in alignments) + "\n", encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
