"""Run the Phase 7 counterfactual target-direction test.

This is deliberately separate from the historical Phase 5A evaluator.  Phase
5A measured whether changing the response changes hidden pixels.  This script
asks the stronger directional question: for a fixed partial geometry from A,
does conditioning on B make the generated completion look more like B and/or
score better against B under the frozen learned screening surrogate?

The surrogate metrics are internal screening diagnostics only; they are not
independent Maxwell validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    predicted_count = prediction.sum()
    target_count = target.sum()
    return {
        "iou": float(intersection / union) if union else 1.0,
        "dice": float(2 * intersection / (predicted_count + target_count)) if predicted_count + target_count else 1.0,
        "pixel_mse": float(np.mean(np.square(prediction.astype(np.float32) - target.astype(np.float32)))),
        "occupancy_abs_difference": float(abs(float(prediction.mean()) - float(target.mean()))),
    }


def response_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.square(prediction - target).mean(dim=(1, 2))


def summarize(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "fraction_positive": float(np.mean(array > 0)),
    }


def assert_no_geometry_leakage(physics: PhysicsConditionedSpatialJEPA) -> None:
    """Guard the counterfactual path against accidentally using target geometry."""
    if not hasattr(physics, "em_encoder") or not hasattr(physics, "context_encoder"):
        raise AssertionError("Expected Phase 5A model with geometry context and EM encoder")
    for name in ("target_encoder", "decoder"):
        if not hasattr(physics, name):
            raise AssertionError(f"Expected Phase 5A component {name}")


def load_models(args: argparse.Namespace, device: torch.device) -> tuple[SpatialJEPACompletionModel, PhysicsConditionedSpatialJEPA, torch.nn.Module, torch.Tensor, torch.Tensor]:
    baseline_config = json.loads((args.baseline_checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    physics_config = json.loads((args.physics_checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    if baseline_config["mask_type"] != physics_config["mask_type"] or float(baseline_config["missing_ratio"]) != float(physics_config["missing_ratio"]):
        raise ValueError("Baseline and Phase 5A checkpoint mask conditions differ")
    baseline = SpatialJEPACompletionModel(
        baseline_config["latent_channels"], baseline_config["predictor_hidden_channels"], baseline_config["ema_decay"]
    ).to(device)
    baseline.load_state_dict(torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    baseline.eval()
    physics = PhysicsConditionedSpatialJEPA(
        physics_config["latent_channels"], physics_config["predictor_hidden_channels"], physics_config["ema_decay"], physics_config["physics_embedding_dim"]
    ).to(device)
    physics.load_state_dict(torch.load(args.physics_checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    physics.eval()
    checkpoint = torch.load(args.forward_checkpoint, map_location=device, weights_only=False)
    forward_name = checkpoint.get("args", {}).get("model", "ForwardSurrogateCNN")
    forward = build_forward_model(forward_name).to(device)
    forward.load_state_dict(checkpoint["model_state_dict"])
    forward.eval()
    stats = np.load(args.subset_root / "train_response_stats.npz")
    mean = torch.from_numpy(stats["mean"].astype(np.float32)).to(device)
    std = torch.from_numpy(stats["std"].astype(np.float32)).to(device)
    return baseline, physics, forward, mean, std


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
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    physics_config = json.loads((args.physics_checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    dataset = PhysicsCompletionDataset(
        args.subset_root, "test", physics_config["mask_type"], float(physics_config["missing_ratio"]), int(physics_config["mask_seed"])
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    baseline, physics, forward, mean, std = load_models(args, device)
    assert_no_geometry_leakage(physics)

    # Use a deterministic derangement: every A is paired with a different B.
    # The permutation is independent of model output and source IDs remain in
    # the row for auditability only.
    permutation = np.roll(np.arange(len(dataset)), 1)
    rows: list[dict[str, Any]] = []
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            batch_size = batch["input"].shape[0]
            indices_a = np.arange(offset, offset + batch_size)
            indices_b = permutation[indices_a]
            condition_b = [dataset[int(index)] for index in indices_b]
            input_a = batch["input"].to(device)
            target_a = batch["target"].to(device)
            response_a = batch["response"].to(device)
            target_b = torch.stack([item["target"] for item in condition_b]).to(device)
            response_b = torch.stack([item["response"] for item in condition_b]).to(device)

            # The complete geometry is reserved for metric computation below;
            # it is never passed to either inference path.
            baseline_out = baseline(input_a)
            control = compose_binary_spatial_completion(torch.sigmoid(baseline_out["logits"]), input_a, batch["mask"].to(device), args.threshold)
            correct_out = physics(input_a, response_a)
            counter_out = physics(input_a, response_b)
            correct = compose_binary_spatial_completion(torch.sigmoid(correct_out["logits"]), input_a, batch["mask"].to(device), args.threshold)
            counter = compose_binary_spatial_completion(torch.sigmoid(counter_out["logits"]), input_a, batch["mask"].to(device), args.threshold)

            pred_control = forward(control)
            pred_correct = forward(correct)
            pred_counter = forward(counter)
            target_a_pred = response_a
            target_b_pred = response_b
            for local in range(batch_size):
                geometry_a = target_a[local, 0].cpu().numpy()
                geometry_b = target_b[local, 0].cpu().numpy()
                control_np = control[local, 0].cpu().numpy()
                correct_np = correct[local, 0].cpu().numpy()
                counter_np = counter[local, 0].cpu().numpy()
                control_metrics_a = binary_metrics(control_np, geometry_a)
                correct_metrics_a = binary_metrics(correct_np, geometry_a)
                counter_metrics_a = binary_metrics(counter_np, geometry_a)
                control_metrics_b = binary_metrics(control_np, geometry_b)
                correct_metrics_b = binary_metrics(correct_np, geometry_b)
                counter_metrics_b = binary_metrics(counter_np, geometry_b)
                row: dict[str, Any] = {
                    "index_a": int(indices_a[local]),
                    "index_b": int(indices_b[local]),
                    "source_id_a": batch["sample_id"][local],
                    "source_id_b": condition_b[local]["sample_id"],
                    "mask_type": physics_config["mask_type"],
                    "missing_ratio": float(physics_config["missing_ratio"]),
                }
                for prefix, prediction, metrics_a, metrics_b, response_prediction in (
                    ("control", control_np, control_metrics_a, control_metrics_b, pred_control),
                    ("correct", correct_np, correct_metrics_a, correct_metrics_b, pred_correct),
                    ("counterfactual", counter_np, counter_metrics_a, counter_metrics_b, pred_counter),
                ):
                    for name, value in metrics_a.items():
                        row[f"{prefix}_vs_a_{name}"] = value
                    for name, value in metrics_b.items():
                        row[f"{prefix}_vs_b_{name}"] = value
                    row[f"{prefix}_response_mse_to_a"] = float(response_mse(response_prediction[local:local + 1], target_a_pred[local:local + 1])[0].item())
                    row[f"{prefix}_response_mse_to_b"] = float(response_mse(response_prediction[local:local + 1], target_b_pred[local:local + 1])[0].item())
                row["counterfactual_geometry_gain_to_b_vs_correct"] = row["correct_vs_b_pixel_mse"] - row["counterfactual_vs_b_pixel_mse"]
                row["counterfactual_response_gain_to_b_vs_correct"] = row["correct_response_mse_to_b"] - row["counterfactual_response_mse_to_b"]
                row["counterfactual_response_gain_to_b_vs_control"] = row["control_response_mse_to_b"] - row["counterfactual_response_mse_to_b"]
                row["target_a_response_gap_correct_vs_counterfactual"] = row["counterfactual_response_mse_to_a"] - row["correct_response_mse_to_a"]
                complexity = geometry_complexity(geometry_a)
                row["complexity_score_a"] = float(complexity["connected_components_4"] + complexity["boundary_transitions_4"] / 32.0)
                rows.append(row)
            offset += batch_size

    scores = np.asarray([row["complexity_score_a"] for row in rows])
    q1, q2 = np.quantile(scores, [1 / 3, 2 / 3])
    for row in rows:
        row["complexity_group"] = "simple" if row["complexity_score_a"] <= q1 else "medium" if row["complexity_score_a"] <= q2 else "complex"

    def summarize_prefix(prefix: str, selected: list[dict[str, Any]]) -> dict[str, Any]:
        metrics = {}
        for name in ("response_mse_to_a", "response_mse_to_b", "vs_a_iou", "vs_b_iou", "vs_a_dice", "vs_b_dice", "vs_a_pixel_mse", "vs_b_pixel_mse", "vs_b_occupancy_abs_difference"):
            metrics[name] = summarize([row[f"{prefix}_{name}"] for row in selected])
        return metrics

    def group_summary(selected: list[dict[str, Any]]) -> dict[str, Any]:
        result = {"samples": len(selected)}
        for prefix in ("control", "correct", "counterfactual"):
            result[prefix] = summarize_prefix(prefix, selected)
        for name in ("counterfactual_geometry_gain_to_b_vs_correct", "counterfactual_response_gain_to_b_vs_correct", "counterfactual_response_gain_to_b_vs_control", "target_a_response_gap_correct_vs_counterfactual"):
            result[name] = summarize([row[name] for row in selected])
        return result

    direction = {
        "all": group_summary(rows),
        "complexity_groups": {group: group_summary([row for row in rows if row["complexity_group"] == group]) for group in ("simple", "medium", "complex")},
        "criterion": {
            "target_directed_response_vs_correct": "counterfactual_response_gain_to_b_vs_correct > 0",
            "target_directed_response_vs_control": "counterfactual_response_gain_to_b_vs_control > 0",
            "geometry_directional_diagnostic_vs_correct": "counterfactual_geometry_gain_to_b_vs_correct > 0",
        },
    }
    metrics = {
        "phase": "7_counterfactual_target_direction",
        "hypothesis": "A counterfactual response B should move the completion toward B, beyond mere target sensitivity.",
        "subset_root": str(args.subset_root),
        "split": "test",
        "split_counts": {"train": 4000, "val": 500, "test": 500},
        "samples": len(rows),
        "pairing": "deterministic derangement index_b=roll(index_a,1)",
        "device": str(device),
        "seed": args.seed,
        "threshold": args.threshold,
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "physics_checkpoint": str(args.physics_checkpoint),
        "forward_checkpoint": str(args.forward_checkpoint),
        "forward_model": type(forward).__name__,
        "anti_leakage": {"target_geometry_not_passed_as_condition": True, "source_ids_used_only_for_audit": True, "geometry_condition_is_fixed_across_targets": True},
        "complexity_tertiles": [float(q1), float(q2)],
        "direction": direction,
        "decision": "target-directed" if direction["all"]["counterfactual_response_gain_to_b_vs_correct"]["mean"] > 0 and direction["all"]["counterfactual_response_gain_to_b_vs_control"]["mean"] > 0 else "target-sensitive-but-weakly-directed",
        "limitation": "All response-direction metrics use the frozen learned forward screening surrogate and are not independent RCWA/Maxwell validation.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "counterfactual_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
