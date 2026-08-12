"""Independently validate existing Phase 5A/5B completions with RCWA."""
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
from scipy.stats import spearmanr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics_conditioned_dataset import PhysicsCompletionDataset
from src.physics_conditioned_spatial_jepa import PhysicsConditionedSpatialJEPA
from src.physics_consistency import load_frozen_forward_surrogate
from src.rcwa_solver import RCWAConfig, frequency_vector
from src.rcwa_validation import cached_solve, pack_modes, response_metrics, save_json
from src.spatial_jepa_completion_model import compose_binary_spatial_completion


MODELS = {
    "phase5a": Path("outputs/phase5a/physics_5aA/best.pt"),
    "phase5b_small": Path("outputs/phase5b/physics_5bA_small/best.pt"),
    "phase5b_medium": Path("outputs/phase5b/physics_5bA_medium/best.pt"),
}


def load_completion(path: Path, device: torch.device) -> PhysicsConditionedSpatialJEPA:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = PhysicsConditionedSpatialJEPA().to(device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    return model


def masked_iou(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    selected = mask.astype(bool); pred, actual = prediction[selected].astype(bool), target[selected].astype(bool)
    union = np.logical_or(pred, actual).sum()
    return float(np.logical_and(pred, actual).sum() / union) if union else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--forward-checkpoint", type=Path, default=Path("outputs/phase2_5/exp_A_5k_mse/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase6_rcwa"))
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--fourier-order", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    settings = json.loads((args.output_dir / "config.json").read_text(encoding="utf-8")) if (args.output_dir / "config.json").exists() else {}
    thickness, order, mapping = float(settings.get("selected_substrate_thickness_mm", 0.20)), args.fourier_order or int(settings.get("fourier_order", 3)), settings.get("selected_channel_mapping", "s_to_ty_p_to_rx"); solver_device = args.device or settings.get("device", "auto")
    frequencies = np.array([2.0, 7.0, 12.0]) if args.quick else frequency_vector()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = PhysicsCompletionDataset(args.subset_root, "test", "central_block", 0.25, 42)
    stats = np.load(args.subset_root / "train_response_stats.npz"); mean, std = stats["mean"], stats["std"]
    cnn, cnn_name = load_frozen_forward_surrogate(args.forward_checkpoint, device)
    completions = {name: load_completion(path, device) for name, path in MODELS.items()}
    output = args.output_dir / "generated"; plots = args.output_dir / "plots"; output.mkdir(parents=True, exist_ok=True); plots.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index in range(min(args.samples, len(data))):
        item = data[index]; inputs, target, mask, response = (item[name].unsqueeze(0).to(device) for name in ("input", "target", "mask", "response"))
        target_raw = item["response_raw"].numpy()[:, :len(frequencies)]
        for name, model in completions.items():
            with torch.no_grad():
                probabilities = torch.sigmoid(model(inputs, response)["logits"])
                completed = compose_binary_spatial_completion(probabilities, inputs, mask)
                cnn_raw = (cnn(completed).cpu().numpy()[0] * std + mean)[:, :len(frequencies)]
            geometry = completed[0, 0].cpu().numpy()
            rcwa = cached_solve(geometry, frequencies, RCWAConfig(substrate_thickness_mm=thickness, fourier_order=order, device=solver_device), args.output_dir / "cache")
            rcwa_raw = pack_modes(rcwa, mapping)
            cnn_target = response_metrics(cnn_raw, target_raw, mean, std)["normalized_mse"]
            rcwa_target = response_metrics(rcwa_raw, target_raw, mean, std)["normalized_mse"]
            cnn_rcwa = response_metrics(cnn_raw, rcwa_raw, mean, std)["normalized_mse"]
            rows.append({"source_id": item["sample_id"], "test_index": index, "model": name, "masked_iou": masked_iou(geometry, target[0, 0].cpu().numpy(), mask[0, 0].cpu().numpy()), "cnn_target_normalized_mse": cnn_target, "rcwa_target_normalized_mse": rcwa_target, "cnn_rcwa_normalized_mse": cnn_rcwa, "geometry": geometry, "target_raw": target_raw, "cnn_raw": cnn_raw, "rcwa_raw": rcwa_raw})
    public_rows = [{key: value for key, value in row.items() if key not in {"geometry", "target_raw", "cnn_raw", "rcwa_raw"}} for row in rows]
    with (output / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public_rows[0])); writer.writeheader(); writer.writerows(public_rows)
    surrogate_errors, rcwa_errors, correlations, agreement = [], [], [], []
    for index in range(min(args.samples, len(data))):
        candidates = [row for row in rows if row["test_index"] == index]
        cnn_values = [row["cnn_target_normalized_mse"] for row in candidates]; rcwa_values = [row["rcwa_target_normalized_mse"] for row in candidates]
        rho = spearmanr(cnn_values, rcwa_values).statistic
        correlations.append(float(rho) if np.isfinite(rho) else np.nan)
        agreement.extend([(cnn_values[left] - cnn_values[right]) * (rcwa_values[left] - rcwa_values[right]) > 0 for left in range(len(candidates)) for right in range(left + 1, len(candidates))])
        surrogate_errors.extend(cnn_values); rcwa_errors.extend(rcwa_values)
    exploitation = sorted(rows, key=lambda row: float(row["rcwa_target_normalized_mse"]) / max(float(row["cnn_target_normalized_mse"]), 1e-12), reverse=True)
    for rank, row in enumerate(exploitation[:10]):
        figure, axes = plt.subplots(2, 3, figsize=(9, 5)); axes[0, 0].imshow(row["geometry"], cmap="gray_r"); axes[0, 0].set_title(f"geometry #{rank}"); axes[0, 0].axis("off")
        for axis, start, label in ((axes[0, 1], 0, "Ty"), (axes[0, 2], 2, "Rx")):
            axis.plot(frequencies, row["target_raw"][start], label="target"); axis.plot(frequencies, row["cnn_raw"][start], label="CNN"); axis.plot(frequencies, row["rcwa_raw"][start], label="RCWA"); axis.set_title(f"Re({label})"); axis.legend(fontsize=6)
        for axis, start, label in ((axes[1, 1], 1, "Ty"), (axes[1, 2], 3, "Rx")):
            axis.plot(frequencies, row["target_raw"][start]); axis.plot(frequencies, row["cnn_raw"][start]); axis.plot(frequencies, row["rcwa_raw"][start]); axis.set_title(f"Im({label})")
        axes[1, 0].axis("off"); figure.tight_layout(); figure.savefig(plots / f"surrogate_exploitation_{rank:02d}.png", dpi=150); plt.close(figure)
    figure, axis = plt.subplots(figsize=(5, 4)); axis.scatter(surrogate_errors, rcwa_errors, s=20); axis.set(xlabel="CNN target MSE", ylabel="RCWA target MSE", title="Generated-geometry target error"); figure.tight_layout(); figure.savefig(plots / "surrogate_vs_rcwa_target_error.png", dpi=160); plt.close(figure)
    worst_public = {key: value for key, value in exploitation[0].items() if key not in {"geometry", "target_raw", "cnn_raw", "rcwa_raw"}} if exploitation else None
    save_json(output / "metrics.json", {"cnn_model": cnn_name, "samples_per_model": args.samples, "frequency_ghz": frequencies, "solver_device": solver_device, "fourier_order": order, "substrate_thickness_mm": thickness, "channel_mapping": mapping, "mean_cnn_rcwa_mse": float(np.mean([row["cnn_rcwa_normalized_mse"] for row in rows])), "spearman_mean": float(np.nanmean(correlations)), "pairwise_ordering_agreement": float(np.mean(agreement)), "top_1_overlap": float(np.mean([np.argmin([row["cnn_target_normalized_mse"] for row in rows if row["test_index"] == index]) == np.argmin([row["rcwa_target_normalized_mse"] for row in rows if row["test_index"] == index]) for index in range(min(args.samples, len(data)))])), "worst_exploitation": worst_public})
    print(output / "metrics.json")


if __name__ == "__main__":
    main()
