"""Phase 6 surrogate-calibration diagnostics without an EM solver dependency.

The repository does not contain a compatible FEM/FDTD/Maxwell solver.  This
script therefore keeps solver-derived quantities explicitly unavailable while
auditing (1) held-out stored-reference accuracy of the frozen Phase 2.5 model,
(2) the distributions of Phase 5A/5B generated geometries, and (3) the
geometry--surrogate-objective Pareto trade-off already observed in Phase 5B.
It never treats a stored dataset response as a fresh solver run for a generated
geometry.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import SUTDPRCMDataset
from src.forward_analysis import geometry_complexity, resonance_errors
from src.physics_conditioned_dataset import PhysicsCompletionDataset
from src.physics_conditioned_spatial_jepa import PhysicsConditionedSpatialJEPA
from src.physics_consistency import load_frozen_forward_surrogate
from src.spatial_jepa_completion_model import compose_binary_spatial_completion


CHANNELS = ("re_t_y", "im_t_y", "re_r_x", "im_r_x")
CONDITIONS = {
    "central_25": {
        "phase5a": Path("outputs/phase5a/physics_5aA/best.pt"),
        "small": Path("outputs/phase5b/physics_5bA_small/best.pt"),
        "medium": Path("outputs/phase5b/physics_5bA_medium/best.pt"),
        "evaluation": Path("outputs/phase5b/evaluation_5bA/per_sample_metrics.csv"),
    },
    "central_50": {
        "phase5a": Path("outputs/phase5a/physics_5aB/best.pt"),
        "small": Path("outputs/phase5b/physics_5bB_small/best.pt"),
        "medium": Path("outputs/phase5b/physics_5bB_medium/best.pt"),
        "evaluation": Path("outputs/phase5b/evaluation_5bB/per_sample_metrics.csv"),
    },
    "random_25": {
        "phase5a": Path("outputs/phase5a/physics_5aC/best.pt"),
        "small": Path("outputs/phase5b/physics_5bC_small/best.pt"),
        "medium": Path("outputs/phase5b/physics_5bC_medium/best.pt"),
        "evaluation": Path("outputs/phase5b/evaluation_5bC/per_sample_metrics.csv"),
    },
    "random_50": {
        "phase5a": Path("outputs/phase5a/physics_5aD/best.pt"),
        "small": Path("outputs/phase5b/physics_5bD_small/best.pt"),
        "medium": Path("outputs/phase5b/physics_5bD_medium/best.pt"),
        "evaluation": Path("outputs/phase5b/evaluation_5bD/per_sample_metrics.csv"),
    },
}
MODEL_LABELS = {"phase5a": "Phase 5A", "small": "Phase 5B small", "medium": "Phase 5B medium"}


def device_for(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def scalar_summary(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(len(array)), "mean": float(array.mean()), "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)), "max": float(array.max()),
    }


def complexity_group(score: float) -> str:
    if score <= 3.0625:
        return "simple"
    if score <= 13.291666666666664:
        return "medium"
    return "complex"


def choose_validation_indices(dataset: SUTDPRCMDataset, per_group: int, seed: int) -> list[int]:
    grouped: dict[str, list[int]] = {"simple": [], "medium": [], "complex": []}
    for index in range(len(dataset)):
        geometry, _ = dataset[index]
        score = geometry_complexity(geometry.numpy())["connected_components_4"] + geometry_complexity(geometry.numpy())["boundary_transitions_4"] / 32.0
        grouped[complexity_group(float(score))].append(index)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for group in ("simple", "medium", "complex"):
        if len(grouped[group]) < per_group:
            raise ValueError(f"Only {len(grouped[group])} {group} test geometries; need {per_group}")
        selected.extend(sorted(rng.choice(grouped[group], size=per_group, replace=False).tolist()))
    return sorted(selected)


def normalized_metrics(prediction: np.ndarray, target: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict[str, float]:
    error = prediction - target
    values = {"surrogate_target_normalized_mse": float(np.square(error).mean())}
    for channel, name in enumerate(CHANNELS):
        values[f"{name}_normalized_mse"] = float(np.square(error[channel]).mean())
    raw_prediction, raw_target = prediction * std + mean, target * std + mean
    for start, name in ((0, "t_y"), (2, "r_x")):
        values[f"{name}_magnitude_mae"] = float(np.abs(np.hypot(raw_prediction[start], raw_prediction[start + 1]) - np.hypot(raw_target[start], raw_target[start + 1])).mean())
    return values


def binary_iou(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    selected = mask.astype(bool)
    prediction, target = prediction[selected].astype(bool), target[selected].astype(bool)
    union = np.logical_or(prediction, target).sum()
    return float(np.logical_and(prediction, target).sum() / union) if union else 1.0


def load_completion_model(checkpoint: Path, device: torch.device) -> tuple[PhysicsConditionedSpatialJEPA, dict[str, Any]]:
    config = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    model = PhysicsConditionedSpatialJEPA(
        config.get("latent_channels", 64), config.get("predictor_hidden_channels", 128),
        config.get("ema_decay", 0.996), config.get("physics_embedding_dim", 128),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    model.eval()
    return model, config


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_response(path: Path, item: dict[str, Any], frequency: np.ndarray, title: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for axis, start, label in ((axes[0], 0, "|T_y|"), (axes[1], 2, "|R_x|")):
        target = np.hypot(item["target_raw"][start], item["target_raw"][start + 1])
        prediction = np.hypot(item["prediction_raw"][start], item["prediction_raw"][start + 1])
        axis.plot(frequency, target, color="black", linewidth=1.1, label="stored paired target")
        axis.plot(frequency, prediction, color="#d95f02", linewidth=0.9, label="frozen surrogate")
        axis.set_title(label); axis.set_xlabel("GHz"); axis.set_ylabel("magnitude"); axis.legend(fontsize=8)
    figure.suptitle(title + " (target is not an independent solver rerun)")
    figure.tight_layout(rect=(0, 0, 1, .91)); figure.savefig(path, dpi=180); plt.close(figure)


def unavailable_plot(path: Path, title: str, detail: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 3.2)); axis.axis("off")
    axis.text(.5, .58, title, ha="center", va="center", fontsize=13, fontweight="bold")
    axis.text(.5, .36, detail, ha="center", va="center", fontsize=10, wrap=True)
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def pareto_from_existing_evaluations() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition, paths in CONDITIONS.items():
        with paths["evaluation"].open(encoding="utf-8") as handle:
            source = list(csv.DictReader(handle))
        for model in ("phase5a", "small", "medium"):
            geometry_error = np.asarray([1.0 - float(row[f"{model}_masked_iou"]) for row in source])
            physics_error = np.asarray([float(row[f"{model}_normalized_response_mse"]) for row in source])
            rows.append({"condition": condition, "model": model, "model_label": MODEL_LABELS[model], "samples": len(source), "geometry_error": float(geometry_error.mean()), "physics_error": float(physics_error.mean())})
    summary = {f"{row['condition']}:{row['model']}": row for row in rows}
    return rows, summary


def plot_pareto(path: Path, rows: list[dict[str, Any]]) -> None:
    colors = {"phase5a": "#7570b3", "small": "#1b9e77", "medium": "#d95f02"}
    figure, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=False, sharey=False)
    for axis, condition in zip(axes.flat, CONDITIONS):
        for row in rows:
            if row["condition"] != condition:
                continue
            axis.scatter(row["geometry_error"], row["physics_error"], s=55, color=colors[row["model"]], label=row["model_label"])
            axis.annotate(row["model"], (row["geometry_error"], row["physics_error"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
        axis.set_title(condition.replace("_", " ")); axis.set_xlabel("masked geometry error (1 - IoU)"); axis.set_ylabel("surrogate target MSE")
        axis.grid(alpha=.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .955), ncol=3)
    figure.suptitle("Phase 5A/5B geometry--surrogate-objective Pareto summary (500 test samples/condition)", y=.995)
    figure.tight_layout(rect=(0, 0, 1, .86)); figure.savefig(path, dpi=180); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--forward-checkpoint", type=Path, default=Path("outputs/phase2_5/exp_A_5k_mse/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase6"))
    parser.add_argument("--per-complexity", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.per_complexity < 1:
        raise ValueError("--per-complexity must be positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = device_for(args.device); started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir, real_dir, plots = (args.output_dir / "surrogate_predictions", args.output_dir / "real_solver_predictions", args.output_dir / "plots")
    for directory in (prediction_dir, real_dir, plots): directory.mkdir(exist_ok=True)
    (real_dir / "UNAVAILABLE.txt").write_text("No compatible real EM solver or solver interface was found in this repository. No solver predictions were created.\n", encoding="utf-8")

    reference_dataset = SUTDPRCMDataset(args.subset_root, "test", normalize_response=True)
    selected_indices = choose_validation_indices(reference_dataset, args.per_complexity, args.seed)
    stats = np.load(args.subset_root / "train_response_stats.npz"); mean, std = stats["mean"].astype(np.float32), stats["std"].astype(np.float32)
    frequency = np.load(args.subset_root / "frequency_ghz.npy")
    surrogate, surrogate_name = load_frozen_forward_surrogate(args.forward_checkpoint, device)

    manifest: list[dict[str, Any]] = []
    for index in selected_indices:
        geometry, _ = reference_dataset[index]; descriptor = geometry_complexity(geometry.numpy())
        score = descriptor["connected_components_4"] + descriptor["boundary_transitions_4"] / 32.0
        manifest.append({"sample_id": reference_dataset.source_id(index), "test_index": index, "category": "dataset", "condition": "none", "target_complexity_group": complexity_group(score), "target_complexity_score": score, **descriptor})

    base_loader = DataLoader(Subset(reference_dataset, selected_indices), batch_size=args.batch_size, shuffle=False)
    dataset_items: list[dict[str, Any]] = []
    with torch.inference_mode():
        offset = 0
        for geometry, response in base_loader:
            prediction = surrogate(geometry.to(device)).cpu().numpy(); response_np = response.numpy()
            raw = response_np * std + mean
            for local in range(len(geometry)):
                dataset_items.append({"sample_id": reference_dataset.source_id(selected_indices[offset + local]), "target_raw": raw[local], "prediction_raw": prediction[local] * std + mean, "target": response_np[local], "prediction": prediction[local], "geometry": geometry[local, 0].numpy()})
            offset += len(geometry)

    per_sample: list[dict[str, Any]] = []
    dataset_frequency_error: list[np.ndarray] = []
    resonance_frequency_errors: list[float] = []
    resonance_local_errors: list[float] = []
    resonance_true_count = resonance_matched_count = 0
    for item in dataset_items:
        target_descriptor = geometry_complexity(item["geometry"])
        score = target_descriptor["connected_components_4"] + target_descriptor["boundary_transitions_4"] / 32.0
        metrics = normalized_metrics(item["prediction"], item["target"], mean, std)
        resonances = []
        for start in (0, 2):
            result = resonance_errors(np.hypot(item["target_raw"][start], item["target_raw"][start + 1]), np.hypot(item["prediction_raw"][start], item["prediction_raw"][start + 1]), frequency)
            resonance_true_count += result["true_feature_count"]; resonance_matched_count += result["matched_feature_count"]
            resonance_frequency_errors.extend(result["frequency_errors_ghz"])
            if np.isfinite(result["resonance_region_magnitude_mae"]): resonance_local_errors.append(result["resonance_region_magnitude_mae"])
            resonances.append(result)
        dataset_frequency_error.append(np.square(item["prediction"] - item["target"]).mean(axis=0))
        per_sample.append({"sample_id": item["sample_id"], "category": "dataset", "model": "dataset_geometry", "condition": "none", "target_complexity_group": complexity_group(score), "candidate_complexity_group": complexity_group(score), "candidate_complexity_score": score, "masked_iou": 1.0, "real_solver_available": False, **metrics})
    np.savez_compressed(prediction_dir / "dataset_geometries.npz", geometry=np.stack([item["geometry"] for item in dataset_items]), surrogate_normalized=np.stack([item["prediction"] for item in dataset_items]), target_normalized=np.stack([item["target"] for item in dataset_items]), sample_id=np.asarray([item["sample_id"] for item in dataset_items]))

    generated_frequency_errors: dict[str, list[np.ndarray]] = {name: [] for name in MODEL_LABELS}
    representative: dict[str, dict[str, Any]] = {"dataset": dataset_items[0]}
    for condition, paths in CONDITIONS.items():
        models: dict[str, PhysicsConditionedSpatialJEPA] = {}
        configs: dict[str, dict[str, Any]] = {}
        for name in MODEL_LABELS:
            models[name], configs[name] = load_completion_model(paths[name], device)
        config = configs["phase5a"]
        condition_dataset = PhysicsCompletionDataset(args.subset_root, "test", config["mask_type"], float(config["missing_ratio"]), int(config["mask_seed"]))
        loader = DataLoader(Subset(condition_dataset, selected_indices), batch_size=args.batch_size, shuffle=False)
        packed: dict[str, dict[str, list[Any]]] = {name: {"geometry": [], "surrogate": [], "target": [], "sample_id": []} for name in MODEL_LABELS}
        with torch.inference_mode():
            offset = 0
            for batch in loader:
                inputs, target, mask, response = (batch[key].to(device) for key in ("input", "target", "mask", "response"))
                outputs = {name: model(inputs, response) for name, model in models.items()}
                completions = {name: compose_binary_spatial_completion(torch.sigmoid(outputs[name]["logits"]), inputs, mask, args.threshold) for name in models}
                predictions = {name: surrogate(completion) for name, completion in completions.items()}
                for local, sample_id in enumerate(batch["sample_id"]):
                    base_index = selected_indices[offset + local]
                    target_np, mask_np, response_np = target[local, 0].cpu().numpy(), mask[local, 0].cpu().numpy(), response[local].cpu().numpy()
                    descriptor = geometry_complexity(target_np); score = descriptor["connected_components_4"] + descriptor["boundary_transitions_4"] / 32.0
                    for name in MODEL_LABELS:
                        completion_np, prediction_np = completions[name][local, 0].cpu().numpy(), predictions[name][local].cpu().numpy()
                        candidate_descriptor = geometry_complexity(completion_np); candidate_score = candidate_descriptor["connected_components_4"] + candidate_descriptor["boundary_transitions_4"] / 32.0
                        metrics = normalized_metrics(prediction_np, response_np, mean, std)
                        generated_frequency_errors[name].append(np.square(prediction_np - response_np).mean(axis=0))
                        per_sample.append({"sample_id": sample_id, "category": "generated", "model": name, "condition": condition, "target_complexity_group": complexity_group(score), "candidate_complexity_group": complexity_group(candidate_score), "candidate_complexity_score": candidate_score, "masked_iou": binary_iou(completion_np, target_np, mask_np), "real_solver_available": False, **metrics})
                        packed[name]["geometry"].append(completion_np); packed[name]["surrogate"].append(prediction_np); packed[name]["target"].append(response_np); packed[name]["sample_id"].append(sample_id)
                        if name not in representative:
                            representative[name] = {"target_raw": response_np * std + mean, "prediction_raw": prediction_np * std + mean}
                offset += len(batch["sample_id"])
        for name, values in packed.items():
            np.savez_compressed(prediction_dir / f"{condition}_{name}.npz", geometry=np.asarray(values["geometry"]), surrogate_normalized=np.asarray(values["surrogate"]), target_normalized=np.asarray(values["target"]), sample_id=np.asarray(values["sample_id"]))
            for index in selected_indices:
                target_geometry, _ = reference_dataset[index]; descriptor = geometry_complexity(target_geometry.numpy()); score = descriptor["connected_components_4"] + descriptor["boundary_transitions_4"] / 32.0
                manifest.append({"sample_id": reference_dataset.source_id(index), "test_index": index, "category": "generated", "model": name, "condition": condition, "target_complexity_group": complexity_group(score), "target_complexity_score": score})

    write_csv(args.output_dir / "validation_geometry_manifest.csv", manifest)
    write_csv(args.output_dir / "per_sample_metrics.csv", per_sample)
    plot_response(plots / "surrogate_vs_stored_reference_dataset.png", representative["dataset"], frequency, "Representative held-out dataset geometry")
    plot_response(plots / "surrogate_vs_target_phase5a_generated.png", representative["phase5a"], frequency, "Representative Phase 5A generated geometry")
    plot_response(plots / "surrogate_vs_target_phase5b_generated.png", representative["medium"], frequency, "Representative Phase 5B generated geometry")
    figure, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    axes[0, 0].plot(frequency, np.mean(dataset_frequency_error, axis=0), color="black", label="dataset: surrogate vs stored reference")
    axes[0, 0].set_title("Dataset reference error"); axes[0, 0].legend(fontsize=7)
    for axis, name in zip(axes.flat[1:], ("phase5a", "small", "medium")):
        axis.plot(frequency, np.mean(generated_frequency_errors[name], axis=0), label=f"{MODEL_LABELS[name]}: surrogate vs target")
        axis.set_title(MODEL_LABELS[name] + " target error"); axis.legend(fontsize=7)
    for axis in axes.flat: axis.set_xlabel("GHz"); axis.set_ylabel("mean normalized squared error"); axis.grid(alpha=.2)
    figure.suptitle("Frequency-wise surrogate diagnostics; generated curves have no solver reference")
    figure.tight_layout(rect=(0, 0, 1, .93)); figure.savefig(plots / "frequency_wise_error.png", dpi=180); plt.close(figure)
    unavailable_plot(plots / "surrogate_error_vs_real_solver_error.png", "Surrogate-vs-real target error unavailable", "No compatible solver is configured. Generated geometries were not assigned invented real EM responses.")
    unavailable_plot(plots / "surrogate_ranking_vs_real_ranking.png", "Surrogate ranking validation unavailable", "Ranking agreement requires real solver scores for the same candidate geometries.")
    unavailable_plot(plots / "surrogate_exploitation_examples.png", "Surrogate-exploitation test unavailable", "A low-surrogate/high-real failure case cannot be identified without real solver predictions.")
    pareto_rows, pareto_summary = pareto_from_existing_evaluations(); write_csv(args.output_dir / "pareto_summary.csv", pareto_rows); plot_pareto(plots / "phase5a_5b_pareto.png", pareto_rows)

    dataset_rows = [row for row in per_sample if row["category"] == "dataset"]
    generated_rows = [row for row in per_sample if row["category"] == "generated"]
    group_summary: dict[str, Any] = {}
    for group in ("simple", "medium", "complex"):
        selected = [row for row in dataset_rows if row["target_complexity_group"] == group]
        group_summary[group] = {"samples": len(selected), "stored_reference_normalized_mse": scalar_summary([row["surrogate_target_normalized_mse"] for row in selected])}
    generated_summary: dict[str, Any] = {}
    for name in MODEL_LABELS:
        selected = [row for row in generated_rows if row["model"] == name]
        generated_summary[name] = {"samples": len(selected), "surrogate_target_normalized_mse": scalar_summary([row["surrogate_target_normalized_mse"] for row in selected]), "masked_iou": scalar_summary([row["masked_iou"] for row in selected])}
    dataset_aggregate = {key: float(np.mean([row[key] for row in dataset_rows])) for key in ("surrogate_target_normalized_mse", *(f"{name}_normalized_mse" for name in CHANNELS), "t_y_magnitude_mae", "r_x_magnitude_mae")}
    metrics = {
        "phase": "6", "real_solver": {"available": False, "name": None, "reason": "No compatible FEM/FDTD/Maxwell solver or solver interface was found in the repository."},
        "validation": {"dataset_geometries": len(dataset_rows), "phase5a_geometries": len([row for row in generated_rows if row["model"] == "phase5a"]), "phase5b_geometries": len([row for row in generated_rows if row["model"] in {"small", "medium"}]), "total_geometry_records": len(manifest), "selection": f"{args.per_complexity} deterministic held-out test geometries per fixed complexity group"},
        "stored_reference_diagnostic_dataset_only": dataset_aggregate,
        "frequency_wise_error": {"dataset_reference_mean_normalized_squared_error": np.mean(dataset_frequency_error, axis=0).tolist()},
        "resonance_dataset_reference_only": {"frequency_error_ghz": scalar_summary(resonance_frequency_errors), "feature_match_rate": float(resonance_matched_count / resonance_true_count) if resonance_true_count else None, "resonance_region_magnitude_mae": scalar_summary(resonance_local_errors)},
        "generated_geometry_surrogate_target_diagnostics": generated_summary,
        "complexity_dataset_reference_only": group_summary,
        "ranking": {"spearman": None, "pairwise_agreement": None, "top_k_agreement": None, "reason": "Requires real-solver candidate scores; unavailable."},
        "surrogate_exploitation": {"found": None, "examples": 0, "reason": "Requires real-solver predictions for generated geometries; unavailable."},
        "physics_objective_validity": {"surrogate_vs_real_target_error_correlation": None, "reason": "No real solver predictions for generated geometries."},
        "pareto_from_existing_500_sample_evaluations": pareto_summary,
        "classification": "C", "classification_reason": "The frozen surrogate has useful held-out stored-reference diagnostics, but there is no independent solver calibration for generated geometries. It must not be trusted as an optimization objective yet.",
        "reproducibility": {"subset_root": str(args.subset_root), "surrogate_checkpoint": str(args.forward_checkpoint), "surrogate_model": surrogate_name, "response_representation": "Phase-2-normalized [Re(T_y), Im(T_y), Re(R_x), Im(R_x)] [4,1001]; raw magnitudes use saved train mean/std", "frequency_ghz": {"start": float(frequency[0]), "stop": float(frequency[-1]), "points": int(len(frequency))}, "threshold": args.threshold, "seed": args.seed, "device": str(device), "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "runtime_seconds": time.perf_counter() - started},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps({"arguments": vars(args) | {"subset_root": str(args.subset_root), "forward_checkpoint": str(args.forward_checkpoint), "output_dir": str(args.output_dir)}, "solver_search_result": metrics["real_solver"], "conditions": {key: {name: str(value) for name, value in paths.items()} for key, paths in CONDITIONS.items()}}, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
