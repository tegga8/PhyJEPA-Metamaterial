"""Evaluate Phase 5A geometry and frozen-surrogate EM conditioning evidence."""

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

from src.completion_losses import completion_loss_metrics
from src.forward_analysis import geometry_complexity
from src.models import build_forward_model
from src.physics_conditioned_dataset import PhysicsCompletionDataset
from src.physics_conditioned_spatial_jepa import PhysicsConditionedSpatialJEPA
from src.spatial_jepa_completion_model import SpatialJEPACompletionModel, compose_binary_spatial_completion


CHANNEL_NAMES = ("re_t_y", "im_t_y", "re_r_x", "im_r_x")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def binary_metrics(prediction: np.ndarray, target: np.ndarray, selection: np.ndarray | None = None) -> dict[str, float]:
    if selection is not None:
        selected = selection.astype(bool)
        prediction, target = prediction[selected], target[selected]
    prediction, target = prediction.astype(bool), target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    predicted_count, target_count = prediction.sum(), target.sum()
    return {
        "accuracy": float(np.mean(prediction == target)),
        "iou": float(intersection / union) if union else 1.0,
        "dice": float(2 * intersection / (predicted_count + target_count)) if predicted_count + target_count else 1.0,
    }


def paired_summary(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(values.mean()), "median": float(np.median(values)), "std": float(values.std()), "p25": float(np.percentile(values, 25)), "p75": float(np.percentile(values, 75)), "wins_fraction": float(np.mean(values > 0))}


def response_metrics(prediction: torch.Tensor, target: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-sample normalized response MSE and raw coefficient/magnitude errors."""
    raw_prediction = prediction * std + mean
    raw_target = target * std + mean
    result = {"normalized_response_mse": torch.square(prediction - target).mean(dim=(1, 2))}
    component_mse = torch.square(prediction - target).mean(dim=-1)
    for index, name in enumerate(CHANNEL_NAMES):
        result[f"{name}_normalized_mse"] = component_mse[:, index]
    target_t, predicted_t = torch.hypot(raw_target[:, 0], raw_target[:, 1]), torch.hypot(raw_prediction[:, 0], raw_prediction[:, 1])
    target_r, predicted_r = torch.hypot(raw_target[:, 2], raw_target[:, 3]), torch.hypot(raw_prediction[:, 2], raw_prediction[:, 3])
    result["t_y_magnitude_mae"] = torch.abs(predicted_t - target_t).mean(dim=-1)
    result["r_x_magnitude_mae"] = torch.abs(predicted_r - target_r).mean(dim=-1)
    return result


def save_representative_grid(path: Path, rows: list[dict[str, Any]], frequency: np.ndarray) -> None:
    figure, axes = plt.subplots(len(rows), 6, figsize=(15, max(4, 2.5 * len(rows))))
    axes = np.atleast_2d(axes)
    for row_index, row in enumerate(rows):
        for axis, image, title in zip(axes[row_index, :4], (row["target"], row["partial"], row["baseline"], row["physics"]), ("complete target", "partial", "Phase 4.2 control", "Phase 5A")):
            axis.imshow(image, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            axis.set_title(title, fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
        for axis, components, label in ((axes[row_index, 4], (0, 1), "|T_y|"), (axes[row_index, 5], (2, 3), "|R_x|")):
            target = np.hypot(row["response_target"][components[0]], row["response_target"][components[1]])
            baseline = np.hypot(row["response_baseline"][components[0]], row["response_baseline"][components[1]])
            physics = np.hypot(row["response_physics"][components[0]], row["response_physics"][components[1]])
            axis.plot(frequency, target, color="black", linewidth=1.0, label="target")
            axis.plot(frequency, baseline, color="#7570b3", linewidth=0.8, linestyle="--", label="control")
            axis.plot(frequency, physics, color="#d95f02", linewidth=0.8, linestyle="-.", label="5A")
            axis.set_title(label, fontsize=8)
            axis.set_xlabel("GHz", fontsize=7)
            axis.set_ylabel("magnitude", fontsize=7)
            axis.tick_params(labelsize=6)
            if row_index == 0:
                axis.legend(fontsize=6)
        axes[row_index, 0].set_ylabel(f"{row['sample_id']}\nΔIoU={row['physics_minus_baseline_iou']:+.3f}", fontsize=7)
    figure.suptitle("Phase 4.2 control vs Phase 5A completion and frozen-surrogate response", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_sensitivity_grid(path: Path, rows: list[dict[str, Any]], frequency: np.ndarray) -> None:
    figure, axes = plt.subplots(len(rows), 6, figsize=(15, max(4, 2.5 * len(rows))))
    axes = np.atleast_2d(axes)
    for row_index, row in enumerate(rows):
        axes[row_index, 0].imshow(row["partial"], cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        axes[row_index, 0].set_title("same partial", fontsize=8)
        axes[row_index, 3].imshow(row["completion"], cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        axes[row_index, 3].set_title("Phase 5A completion", fontsize=8)
        for axis in (axes[row_index, 0], axes[row_index, 3]):
            axis.set_xticks([])
            axis.set_yticks([])
        for axis, components, label, response_key in ((axes[row_index, 1], (0, 1), "target |T_y|", "target_response"), (axes[row_index, 2], (2, 3), "target |R_x|", "target_response"), (axes[row_index, 4], (0, 1), "completion |T_y|", "completion_response"), (axes[row_index, 5], (2, 3), "completion |R_x|", "completion_response")):
            response = row[response_key]
            axis.plot(frequency, np.hypot(response[components[0]], response[components[1]]), color="#d95f02" if response_key == "completion_response" else "black", linewidth=0.9)
            axis.set_title(label, fontsize=8)
            axis.set_xlabel("GHz", fontsize=7)
            axis.tick_params(labelsize=6)
        axes[row_index, 0].set_ylabel(f"{row['base_sample_id']}\n{row['condition_label']}", fontsize=7)
    figure.suptitle("Same partial geometry and mask with different target EM responses", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--physics-checkpoint", type=Path, required=True)
    parser.add_argument("--forward-checkpoint", type=Path, default=Path("outputs/phase2_5/exp_A_5k_mse/best.pt"))
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--visual-count", type=int, default=4)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    baseline_config = json.loads((args.baseline_checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    physics_config = json.loads((args.physics_checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    if baseline_config["mask_type"] != physics_config["mask_type"] or float(baseline_config["missing_ratio"]) != float(physics_config["missing_ratio"]):
        raise ValueError("Baseline and Phase 5A checkpoints must use the same mask condition")
    dataset = PhysicsCompletionDataset(args.subset_root, "test", physics_config["mask_type"], float(physics_config["missing_ratio"]), int(physics_config["mask_seed"]))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    baseline = SpatialJEPACompletionModel(baseline_config["latent_channels"], baseline_config["predictor_hidden_channels"], baseline_config["ema_decay"]).to(device)
    baseline.load_state_dict(torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    baseline.eval()
    physics = PhysicsConditionedSpatialJEPA(physics_config["latent_channels"], physics_config["predictor_hidden_channels"], physics_config["ema_decay"], physics_config["physics_embedding_dim"]).to(device)
    physics.load_state_dict(torch.load(args.physics_checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    physics.eval()
    forward_checkpoint = torch.load(args.forward_checkpoint, map_location=device, weights_only=False)
    forward_name = forward_checkpoint.get("args", {}).get("model", "ForwardSurrogateCNN")
    forward = build_forward_model(forward_name).to(device)
    forward.load_state_dict(forward_checkpoint["model_state_dict"])
    forward.eval()
    stats = np.load(args.subset_root / "train_response_stats.npz")
    mean = torch.from_numpy(stats["mean"]).to(device)
    std = torch.from_numpy(stats["std"]).to(device)
    frequency = np.load(args.subset_root / "frequency_ghz.npy")
    selected_indexes = set(np.random.default_rng(args.seed).choice(len(dataset), size=min(args.visual_count, len(dataset)), replace=False).tolist())
    rows: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    global_index = 0
    with torch.inference_mode():
        for batch in loader:
            inputs, target, mask, response, response_raw = (batch["input"].to(device), batch["target"].to(device), batch["mask"].to(device), batch["response"].to(device), batch["response_raw"].to(device))
            base_outputs = baseline(inputs, target)
            physics_outputs = physics(inputs, response, target)
            base_binary = compose_binary_spatial_completion(torch.sigmoid(base_outputs["logits"]), inputs, mask, args.threshold)
            physics_binary = compose_binary_spatial_completion(torch.sigmoid(physics_outputs["logits"]), inputs, mask, args.threshold)
            permuted_response = torch.roll(response, shifts=1, dims=0)
            permuted_outputs = physics(inputs, permuted_response, target)
            permuted_binary = compose_binary_spatial_completion(torch.sigmoid(permuted_outputs["logits"]), inputs, mask, args.threshold)
            forward_base = forward(base_binary)
            forward_physics = forward(physics_binary)
            forward_permuted = forward(permuted_binary)
            base_response = response_metrics(forward_base, response, mean, std)
            physics_response = response_metrics(forward_physics, response, mean, std)
            permuted_response_metrics = response_metrics(forward_permuted, response, mean, std)
            for local_index, sample_id in enumerate(batch["sample_id"]):
                target_np = target[local_index, 0].cpu().numpy()
                mask_np = mask[local_index, 0].cpu().numpy()
                base_np = base_binary[local_index, 0].cpu().numpy()
                physics_np = physics_binary[local_index, 0].cpu().numpy()
                permuted_np = permuted_binary[local_index, 0].cpu().numpy()
                base_full, base_masked = binary_metrics(base_np, target_np), binary_metrics(base_np, target_np, mask_np)
                physics_full, physics_masked = binary_metrics(physics_np, target_np), binary_metrics(physics_np, target_np, mask_np)
                perm_full, perm_masked = binary_metrics(permuted_np, target_np), binary_metrics(permuted_np, target_np, mask_np)
                base_loss = completion_loss_metrics(base_outputs["logits"][local_index:local_index + 1], target[local_index:local_index + 1], mask[local_index:local_index + 1])
                physics_loss = completion_loss_metrics(physics_outputs["logits"][local_index:local_index + 1], target[local_index:local_index + 1], mask[local_index:local_index + 1])
                perm_loss = completion_loss_metrics(permuted_outputs["logits"][local_index:local_index + 1], target[local_index:local_index + 1], mask[local_index:local_index + 1])
                row: dict[str, Any] = {"sample_id": sample_id, "index": global_index + local_index, "mask_type": physics_config["mask_type"], "missing_ratio": physics_config["missing_ratio"]}
                for prefix, full, masked, losses, response_values, prediction in (("baseline", base_full, base_masked, base_loss, base_response, base_np), ("physics", physics_full, physics_masked, physics_loss, physics_response, physics_np), ("permuted", perm_full, perm_masked, perm_loss, permuted_response_metrics, permuted_np)):
                    row.update({f"{prefix}_full_{name}": value for name, value in full.items()})
                    row.update({f"{prefix}_masked_{name}": value for name, value in masked.items()})
                    row[f"{prefix}_full_bce"] = float(losses["full_bce"].item())
                    row[f"{prefix}_masked_bce"] = float(losses["masked_bce"].item())
                    row[f"{prefix}_known_region_error"] = float(np.mean(np.abs(prediction[mask_np == 0] - target_np[mask_np == 0])))
                    row.update({f"{prefix}_{name}": float(values[local_index].item()) for name, values in response_values.items()})
                row["physics_minus_baseline_iou"] = row["physics_masked_iou"] - row["baseline_masked_iou"]
                row["physics_minus_baseline_response_mse"] = row["physics_normalized_response_mse"] - row["baseline_normalized_response_mse"]
                row["permuted_minus_physics_iou"] = row["permuted_masked_iou"] - row["physics_masked_iou"]
                row["permuted_minus_physics_response_mse"] = row["permuted_normalized_response_mse"] - row["physics_normalized_response_mse"]
                row["permuted_masked_pixel_difference"] = float(np.mean(np.abs(permuted_np[mask_np.astype(bool)] - physics_np[mask_np.astype(bool)])))
                row["permuted_prediction_masked_iou"] = binary_metrics(permuted_np, physics_np, mask_np)["iou"]
                complexity = geometry_complexity(target_np)
                row.update(complexity)
                row["complexity_score"] = complexity["connected_components_4"] + complexity["boundary_transitions_4"] / 32.0
                rows.append(row)
                if global_index + local_index in selected_indexes:
                    visuals.append({"sample_id": sample_id, "target": target_np, "partial": inputs[local_index, 0].cpu().numpy(), "baseline": base_np, "physics": physics_np, "response_target": response_raw[local_index].cpu().numpy(), "response_baseline": (forward_base[local_index] * std + mean).cpu().numpy(), "response_physics": (forward_physics[local_index] * std + mean).cpu().numpy(), "physics_minus_baseline_iou": row["physics_minus_baseline_iou"]})
            global_index += inputs.shape[0]
    scores = np.asarray([row["complexity_score"] for row in rows])
    q1, q2 = np.quantile(scores, [1 / 3, 2 / 3])
    for row, score in zip(rows, scores):
        row["complexity_group"] = "simple" if score <= q1 else "medium" if score <= q2 else "complex"
    def summary(prefix: str, selected: list[dict[str, Any]]) -> dict[str, float]:
        names = ("full_bce", "masked_bce", "full_accuracy", "full_iou", "full_dice", "masked_accuracy", "masked_iou", "masked_dice", "known_region_error", "normalized_response_mse", "re_t_y_normalized_mse", "im_t_y_normalized_mse", "re_r_x_normalized_mse", "im_r_x_normalized_mse", "t_y_magnitude_mae", "r_x_magnitude_mae")
        return {name: float(np.mean([row[f"{prefix}_{name}"] for row in selected])) for name in names}
    complexity: dict[str, Any] = {"score_tertiles": [float(q1), float(q2)], "groups": {}}
    for group in ("simple", "medium", "complex"):
        selected = [row for row in rows if row["complexity_group"] == group]
        complexity["groups"][group] = {"samples": len(selected), "baseline": summary("baseline", selected), "physics": summary("physics", selected)}
    delta_iou = np.asarray([row["physics_minus_baseline_iou"] for row in rows])
    delta_mse = np.asarray([row["physics_minus_baseline_response_mse"] for row in rows])
    permutation = {
        "geometry_masked_iou_correct_minus_permuted": paired_summary(-np.asarray([row["permuted_minus_physics_iou"] for row in rows])),
        "response_mse_permuted_minus_correct": paired_summary(np.asarray([row["permuted_minus_physics_response_mse"] for row in rows])),
        "masked_pixel_difference": paired_summary(np.asarray([row["permuted_masked_pixel_difference"] for row in rows])),
        "prediction_to_prediction_masked_iou": paired_summary(np.asarray([row["permuted_prediction_masked_iou"] for row in rows])),
    }
    # Sensitivity diagnostic: three deterministic complete-spectrum alternatives per fixed partial geometry.
    sensitivity_rows: list[dict[str, Any]] = []
    sensitivity_visuals: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    for base_index in rng.choice(len(dataset), size=min(3, len(dataset)), replace=False):
        base = dataset[int(base_index)]
        alternate_indexes = [int(base_index), (int(base_index) + 137) % len(dataset), (int(base_index) + 271) % len(dataset)]
        inputs = base["input"].unsqueeze(0).to(device)
        mask = base["mask"].unsqueeze(0).to(device)
        base_response = base["response"].unsqueeze(0).to(device)
        reference_completion: np.ndarray | None = None
        reference_latent: torch.Tensor | None = None
        for condition_number, response_index in enumerate(alternate_indexes):
            condition = dataset[response_index]
            response = condition["response"].unsqueeze(0).to(device)
            with torch.inference_mode():
                outputs = physics(inputs, response)
                completion = compose_binary_spatial_completion(torch.sigmoid(outputs["logits"]), inputs, mask, args.threshold)
                predicted_response = forward(completion)
            completion_np = completion[0, 0].cpu().numpy()
            if reference_completion is None:
                reference_completion, reference_latent = completion_np, outputs["z_pred"].detach()
                pixel_difference, completion_iou, latent_distance = 0.0, 1.0, 0.0
            else:
                selected = base["mask"][0].numpy().astype(bool)
                pixel_difference = float(np.mean(np.abs(completion_np[selected] - reference_completion[selected])))
                completion_iou = binary_metrics(completion_np, reference_completion, base["mask"][0].numpy())["iou"]
                latent_distance = float(torch.mean(torch.square(outputs["z_pred"] - reference_latent)).item())
            sensitivity_rows.append({"base_sample_id": base["sample_id"], "condition_label": "correct" if condition_number == 0 else f"alternate_{condition_number}", "condition_sample_id": condition["sample_id"], "masked_pixel_difference_vs_correct": pixel_difference, "prediction_to_prediction_masked_iou_vs_correct": completion_iou, "latent_mse_vs_correct": latent_distance})
            sensitivity_visuals.append({"base_sample_id": base["sample_id"], "condition_label": "correct target" if condition_number == 0 else f"target from {condition['sample_id']}", "partial": base["input"][0].numpy(), "completion": completion_np, "target_response": condition["response_raw"].numpy(), "completion_response": (predicted_response[0] * std + mean).cpu().numpy()})
    alternate_sensitivity = [row for row in sensitivity_rows if row["condition_label"] != "correct"]
    sensitivity = {
        "same_partial_different_response": {
            "alternate_cases": len(alternate_sensitivity),
            "masked_pixel_difference": paired_summary(np.asarray([row["masked_pixel_difference_vs_correct"] for row in alternate_sensitivity])),
            "prediction_to_prediction_masked_iou": paired_summary(np.asarray([row["prediction_to_prediction_masked_iou_vs_correct"] for row in alternate_sensitivity])),
            "latent_mse": paired_summary(np.asarray([row["latent_mse_vs_correct"] for row in alternate_sensitivity])),
        }
    }
    sensitivity_pass = sensitivity["same_partial_different_response"]["masked_pixel_difference"]["mean"] > 0.01
    permutation_pass = permutation["masked_pixel_difference"]["mean"] > 0.01 and permutation["response_mse_permuted_minus_correct"]["mean"] > 0.0
    metrics = {
        "phase": "5A", "subset_root": str(args.subset_root), "samples": len(rows), "device": str(device), "mask_type": physics_config["mask_type"], "missing_ratio": physics_config["missing_ratio"], "threshold": args.threshold,
        "baseline_checkpoint": str(args.baseline_checkpoint), "physics_checkpoint": str(args.physics_checkpoint), "forward_checkpoint": str(args.forward_checkpoint), "forward_model": forward_name,
        "baseline": summary("baseline", rows), "physics": summary("physics", rows), "paired_physics_minus_baseline": {"masked_iou": paired_summary(delta_iou), "normalized_response_mse": paired_summary(delta_mse)},
        "known_region_error_max": {"baseline": float(max(row["baseline_known_region_error"] for row in rows)), "physics": float(max(row["physics_known_region_error"] for row in rows))},
        "complexity": complexity, "conditioning_sensitivity": sensitivity, "conditioning_sensitivity_pass": sensitivity_pass, "permutation_test": permutation, "permutation_test_pass": permutation_pass,
        "parameter_counts": {"baseline_total": sum(p.numel() for p in baseline.parameters()), "baseline_trainable": sum(p.numel() for p in baseline.trainable_parameters()), "physics_total": sum(p.numel() for p in physics.parameters()), "physics_trainable": sum(p.numel() for p in physics.trainable_parameters()), "em_encoder": sum(p.numel() for p in physics.em_encoder.parameters()), "film": sum(p.numel() for p in physics.film.parameters())},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots = args.output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "conditioning_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sensitivity_rows[0]))
        writer.writeheader()
        writer.writerows(sensitivity_rows)
    save_representative_grid(plots / "representative_geometry_and_em.png", visuals, frequency)
    save_sensitivity_grid(plots / "conditioning_sensitivity.png", sensitivity_visuals, frequency)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
