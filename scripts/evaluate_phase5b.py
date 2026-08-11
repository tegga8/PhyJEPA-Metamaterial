"""Evaluate Phase 5A versus small/moderate Phase 5B physics-loss runs."""

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
from src.physics_conditioned_dataset import PhysicsCompletionDataset
from src.physics_conditioned_spatial_jepa import PhysicsConditionedSpatialJEPA
from src.physics_consistency import continuous_completion, load_frozen_forward_surrogate, local_perturbation_diagnostics, physics_gradient_diagnostics
from src.spatial_jepa_completion_model import compose_binary_spatial_completion

CHANNELS = ("re_t_y", "im_t_y", "re_r_x", "im_r_x")


def device_for(requested: str) -> torch.device:
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)


def binary(prediction: np.ndarray, target: np.ndarray, selection: np.ndarray | None = None) -> dict[str, float]:
    if selection is not None: prediction, target = prediction[selection.astype(bool)], target[selection.astype(bool)]
    prediction, target = prediction.astype(bool), target.astype(bool); intersection, union = np.logical_and(prediction, target).sum(), np.logical_or(prediction, target).sum()
    return {"accuracy": float(np.mean(prediction == target)), "iou": float(intersection / union) if union else 1.0, "dice": float(2 * intersection / (prediction.sum() + target.sum())) if prediction.sum() + target.sum() else 1.0}


def response_values(prediction: torch.Tensor, target: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> dict[str, torch.Tensor]:
    result = {"normalized_response_mse": torch.square(prediction - target).mean((1, 2))}
    for index, name in enumerate(CHANNELS): result[f"{name}_normalized_mse"] = torch.square(prediction[:, index] - target[:, index]).mean(-1)
    raw_prediction, raw_target = prediction * std + mean, target * std + mean
    for prefix, indices in (("t_y", (0, 1)), ("r_x", (2, 3))):
        p, t = torch.hypot(raw_prediction[:, indices[0]], raw_prediction[:, indices[1]]), torch.hypot(raw_target[:, indices[0]], raw_target[:, indices[1]])
        result[f"{prefix}_magnitude_mse"], result[f"{prefix}_magnitude_mae"] = torch.square(p - t).mean(-1), torch.abs(p - t).mean(-1)
    return result


def summary(rows: list[dict[str, Any]], label: str) -> dict[str, float]:
    names = ("full_accuracy", "full_iou", "full_dice", "masked_accuracy", "masked_iou", "masked_dice", "full_bce", "masked_bce", "known_region_error", "pixel_difference_from_ground_truth", "normalized_response_mse", "re_t_y_normalized_mse", "im_t_y_normalized_mse", "re_r_x_normalized_mse", "im_r_x_normalized_mse", "t_y_magnitude_mse", "r_x_magnitude_mse", "t_y_magnitude_mae", "r_x_magnitude_mae")
    return {name: float(np.mean([row[f"{label}_{name}"] for row in rows])) for name in names}


def paired(values: list[float]) -> dict[str, float]:
    array = np.asarray(values); return {"mean": float(array.mean()), "median": float(np.median(array)), "wins_fraction": float(np.mean(array > 0))}


def plot(path: Path, visuals: list[dict[str, Any]], frequency: np.ndarray) -> None:
    figure, axes = plt.subplots(len(visuals), 7, figsize=(17, max(3, 2.6 * len(visuals)))); axes = np.atleast_2d(axes)
    for row, item in enumerate(visuals):
        for axis, image, title in zip(axes[row, :5], (item["target"], item["partial"], item["phase5a"], item["small"], item["medium"]), ("ground truth", "partial", "Phase 5A", "5B small", "5B medium")):
            axis.imshow(image, cmap="gray_r", vmin=0, vmax=1); axis.set_title(title, fontsize=8); axis.set_xticks([]); axis.set_yticks([])
        for axis, indices, label in ((axes[row, 5], (0, 1), "|T_y|"), (axes[row, 6], (2, 3), "|R_x|")):
            for values, name, style in ((item["target_response"], "target", "-"), (item["phase5a_response"], "5A", "--"), (item["small_response"], "5B small", "-."), (item["medium_response"], "5B medium", ":")):
                axis.plot(frequency, np.hypot(values[indices[0]], values[indices[1]]), linewidth=0.8, linestyle=style, label=name)
            axis.set_title(label, fontsize=8); axis.tick_params(labelsize=6)
            if row == 0: axis.legend(fontsize=6)
        axes[row, 0].set_ylabel(item["sample_id"], fontsize=7)
    figure.suptitle("Phase 5A vs Phase 5B: binary completions and frozen-surrogate responses", fontsize=11); figure.tight_layout(rect=(0, 0, 1, .96)); figure.savefig(path, dpi=180); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--phase5a-checkpoint", type=Path, required=True); parser.add_argument("--small-checkpoint", type=Path, required=True); parser.add_argument("--medium-checkpoint", type=Path, required=True); parser.add_argument("--forward-checkpoint", type=Path, required=True); parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k")); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="auto"); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--threshold", type=float, default=.5); parser.add_argument("--visual-count", type=int, default=4)
    args = parser.parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); device = device_for(args.device)
    configs = {name: json.loads((path.parent / "config.json").read_text()) for name, path in (("phase5a", args.phase5a_checkpoint), ("small", args.small_checkpoint), ("medium", args.medium_checkpoint))}
    reference = configs["phase5a"]
    if any((config["mask_type"], float(config["missing_ratio"]), config["mask_seed"]) != (reference["mask_type"], float(reference["missing_ratio"]), reference["mask_seed"]) for config in configs.values()): raise ValueError("All checkpoints must share one deterministic mask condition")
    dataset = PhysicsCompletionDataset(args.subset_root, "test", reference["mask_type"], float(reference["missing_ratio"]), int(reference["mask_seed"])); loader = DataLoader(dataset, batch_size=64, shuffle=False)
    models = {}
    for name, checkpoint in (("phase5a", args.phase5a_checkpoint), ("small", args.small_checkpoint), ("medium", args.medium_checkpoint)):
        config = configs[name]; model = PhysicsConditionedSpatialJEPA(config.get("latent_channels", 64), config.get("predictor_hidden_channels", 128), config.get("ema_decay", .996), config.get("physics_embedding_dim", 128)).to(device); model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"]); model.eval(); models[name] = model
    surrogate, surrogate_name = load_frozen_forward_surrogate(args.forward_checkpoint, device); stats = np.load(args.subset_root / "train_response_stats.npz"); mean, std = torch.from_numpy(stats["mean"]).to(device), torch.from_numpy(stats["std"]).to(device); frequency = np.load(args.subset_root / "frequency_ghz.npy")
    selected = set(np.random.default_rng(args.seed).choice(len(dataset), min(args.visual_count, len(dataset)), replace=False)); rows, visuals = [], []
    with torch.inference_mode():
        offset = 0
        for batch in loader:
            inputs, target, mask, response, raw = (batch[key].to(device) for key in ("input", "target", "mask", "response", "response_raw"))
            outputs = {name: model(inputs, response) for name, model in models.items()}; completions = {name: compose_binary_spatial_completion(torch.sigmoid(output["logits"]), inputs, mask, args.threshold) for name, output in outputs.items()}; spectra = {name: surrogate(completion) for name, completion in completions.items()}; metrics = {name: response_values(spectrum, response, mean, std) for name, spectrum in spectra.items()}
            for i, sample_id in enumerate(batch["sample_id"]):
                row: dict[str, Any] = {"sample_id": sample_id, "index": offset + i}
                target_np, mask_np = target[i, 0].cpu().numpy(), mask[i, 0].cpu().numpy()
                for name in models:
                    completion = completions[name][i, 0].cpu().numpy(); full, hidden = binary(completion, target_np), binary(completion, target_np, mask_np); losses = completion_loss_metrics(outputs[name]["logits"][i:i + 1], target[i:i + 1], mask[i:i + 1])
                    row.update({f"{name}_full_{key}": value for key, value in full.items()}); row.update({f"{name}_masked_{key}": value for key, value in hidden.items()}); row[f"{name}_full_bce"], row[f"{name}_masked_bce"] = float(losses["full_bce"]), float(losses["masked_bce"]); row[f"{name}_known_region_error"] = float(np.abs(completion[mask_np == 0] - target_np[mask_np == 0]).mean()); row[f"{name}_pixel_difference_from_ground_truth"] = float(np.abs(completion - target_np).mean()); row.update({f"{name}_{key}": float(value[i]) for key, value in metrics[name].items()})
                rows.append(row)
                if offset + i in selected: visuals.append({"sample_id": sample_id, "target": target_np, "partial": inputs[i, 0].cpu().numpy(), **{name: completions[name][i, 0].cpu().numpy() for name in models}, "target_response": raw[i].cpu().numpy(), **{f"{name}_response": (spectra[name][i] * std + mean).cpu().numpy() for name in models}})
            offset += inputs.shape[0]
    deltas = {name: {"masked_iou": paired([row[f"{name}_masked_iou"] - row["phase5a_masked_iou"] for row in rows]), "normalized_response_mse": paired([row["phase5a_normalized_response_mse"] - row[f"{name}_normalized_response_mse"] for row in rows])} for name in ("small", "medium")}
    test_batch = next(iter(loader)); inputs, mask, response = (test_batch[key].to(device) for key in ("input", "mask", "response")); gradient = physics_gradient_diagnostics(models["medium"], surrogate, inputs, mask, response)
    with torch.enable_grad():
        output = models["medium"](inputs, response); continuous = continuous_completion(torch.sigmoid(output["logits"]), inputs, mask); perturbation = local_perturbation_diagnostics(surrogate, continuous, response, mask)
    # Correct-vs-permuted response diagnostic for both Phase 5B weights.
    permutation: dict[str, Any] = {}
    with torch.inference_mode():
        for name in ("small", "medium"):
            changes, response_changes = [], []
            for batch in loader:
                inputs, target, mask, response = (batch[key].to(device) for key in ("input", "target", "mask", "response")); correct = compose_binary_spatial_completion(torch.sigmoid(models[name](inputs, response)["logits"]), inputs, mask, args.threshold); wrong = compose_binary_spatial_completion(torch.sigmoid(models[name](inputs, torch.roll(response, 1, 0))["logits"]), inputs, mask, args.threshold); changes.extend(torch.abs(correct - wrong)[mask.bool()].view(-1).cpu().tolist()); response_changes.extend((torch.square(surrogate(wrong) - response).mean((1, 2)) - torch.square(surrogate(correct) - response).mean((1, 2))).cpu().tolist())
            permutation[name] = {"masked_pixel_difference": paired(changes), "permuted_minus_correct_response_mse": paired(response_changes)}
    sensitivity_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for name in ("small", "medium"):
            for base_index in np.random.default_rng(args.seed).choice(len(dataset), min(3, len(dataset)), replace=False):
                base = dataset[int(base_index)]; inputs, mask = base["input"].unsqueeze(0).to(device), base["mask"].unsqueeze(0).to(device); reference_completion = None
                for number, response_index in enumerate((int(base_index), (int(base_index) + 137) % len(dataset), (int(base_index) + 271) % len(dataset))):
                    condition = dataset[response_index]; completion = compose_binary_spatial_completion(torch.sigmoid(models[name](inputs, condition["response"].unsqueeze(0).to(device))["logits"]), inputs, mask, args.threshold)[0, 0].cpu().numpy()
                    if reference_completion is None: reference_completion, difference = completion, 0.0
                    else: difference = float(np.abs(completion[base["mask"][0].numpy().astype(bool)] - reference_completion[base["mask"][0].numpy().astype(bool)]).mean())
                    sensitivity_rows.append({"model": name, "base_sample_id": base["sample_id"], "condition_sample_id": condition["sample_id"], "condition": "correct" if number == 0 else "alternate", "masked_pixel_difference_vs_correct": difference})
    sensitivity = {name: {"alternate_cases": sum(row["model"] == name and row["condition"] == "alternate" for row in sensitivity_rows), "masked_pixel_difference": paired([row["masked_pixel_difference_vs_correct"] for row in sensitivity_rows if row["model"] == name and row["condition"] == "alternate"])} for name in ("small", "medium")}
    args.output_dir.mkdir(parents=True, exist_ok=True); (args.output_dir / "plots").mkdir(exist_ok=True); plot(args.output_dir / "plots" / "representative_geometry_and_em.png", visuals, frequency)
    with (args.output_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / "conditioning_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=list(sensitivity_rows[0])); writer.writeheader(); writer.writerows(sensitivity_rows)
    metrics = {"phase": "5B", "samples": len(rows), "mask_type": reference["mask_type"], "missing_ratio": reference["missing_ratio"], "threshold": args.threshold, "surrogate_checkpoint": str(args.forward_checkpoint), "surrogate_model": surrogate_name, "models": {name: {"checkpoint": str(path), "lambda_physics": configs[name].get("lambda_physics", 0.0), "summary": summary(rows, name)} for name, path in (("phase5a", args.phase5a_checkpoint), ("small", args.small_checkpoint), ("medium", args.medium_checkpoint))}, "phase5b_minus_phase5a": deltas, "physics_gradient_diagnostics_medium": gradient, "local_perturbation_diagnostics_medium": perturbation, "conditioning_sensitivity": sensitivity, "permutation_test": permutation, "fem_validation": "deferred: no compatible solver is configured in this repository"}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8"); print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
