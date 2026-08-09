"""Run gradient stability and local perturbation checks for a forward CNN."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import SUTDPRCMDataset
from src.models import build_forward_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--subset-root", type=Path, required=True)
    parser.add_argument("--normalization-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None, choices=("ForwardSurrogateCNN", "ResponseAwareSurrogateCNN"))
    args = parser.parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = args.model or checkpoint.get("args", {}).get("model", "ForwardSurrogateCNN")
    model = build_forward_model(model_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    normalization_root = args.normalization_root or args.subset_root
    stats = np.load(normalization_root / "train_response_stats.npz")
    mean = torch.from_numpy(stats["mean"]).to(device)
    std = torch.from_numpy(stats["std"]).to(device)
    dataset = SUTDPRCMDataset(args.subset_root, "test", normalize_response=False)
    rng = np.random.default_rng(args.seed)
    geometry_trials = rng.choice(len(dataset), size=min(5, len(dataset)), replace=False)
    target_trials = rng.choice(len(dataset), size=min(5, len(dataset)), replace=False)
    pixels = ((0, 0), (4, 4), (7, 9), (11, 3), (15, 15))
    epsilon = 1e-3
    gradient_rows: list[dict[str, float | int | bool]] = []
    perturbation_rows: list[dict[str, float | int | bool]] = []
    first_gradient: np.ndarray | None = None

    for geometry_trial, geometry_index in enumerate(geometry_trials):
        geometry_array = (0.1 + 0.8 * rng.random((1, 16, 16))).astype(np.float32)
        geometry = torch.from_numpy(geometry_array[None]).to(device)
        for target_trial, target_index in enumerate(target_trials):
            raw_target = torch.from_numpy(np.asarray(dataset.responses[int(dataset.indices[target_index])], dtype=np.float32).copy()).to(device)
            target = ((raw_target - mean) / std).unsqueeze(0)
            geometry_variable = geometry.clone().requires_grad_(True)
            loss = torch.mean(torch.square(model(geometry_variable) - target))
            model.zero_grad(set_to_none=True)
            loss.backward()
            gradient = geometry_variable.grad.detach()[0, 0]
            gradient_np = gradient.cpu().numpy()
            if first_gradient is None:
                first_gradient = gradient_np.copy()
            gradient_rows.append({
                "geometry_trial": geometry_trial, "target_trial": target_trial, "geometry_index": int(geometry_index), "target_index": int(target_index),
                "loss": float(loss.item()), "min": float(gradient.min().item()), "max": float(gradient.max().item()),
                "mean": float(gradient.mean().item()), "std": float(gradient.std().item()), "norm": float(torch.linalg.vector_norm(gradient).item()),
                "nonzero_fraction": float(torch.mean((torch.abs(gradient) > 1e-12).float()).item()),
                "all_finite": bool(torch.isfinite(gradient).all().item()),
            })
            base_geometry = geometry_variable.detach()
            base_loss = float(loss.item())
            for row, col in pixels:
                perturbed = base_geometry.clone()
                perturbed[0, 0, row, col] += epsilon
                with torch.inference_mode():
                    perturbed_loss = float(torch.mean(torch.square(model(perturbed) - target)).item())
                delta = perturbed_loss - base_loss
                predicted_delta = float(gradient[row, col].item()) * epsilon
                perturbation_rows.append({
                    "geometry_trial": geometry_trial, "target_trial": target_trial, "row": row, "col": col,
                    "epsilon": epsilon, "loss_delta": delta, "linearized_delta": predicted_delta,
                    "absolute_error": abs(delta - predicted_delta),
                    "sign_agreement": bool(np.sign(delta) == np.sign(predicted_delta)) if abs(predicted_delta) > 1e-12 else None,
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "gradient_stability.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gradient_rows[0]))
        writer.writeheader()
        writer.writerows(gradient_rows)
    with (args.output_dir / "local_perturbation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(perturbation_rows[0]))
        writer.writeheader()
        writer.writerows(perturbation_rows)

    gradient_values = np.asarray([[row["min"], row["max"], row["mean"], row["std"], row["norm"], row["nonzero_fraction"]] for row in gradient_rows], dtype=float)
    finite_norms = gradient_values[:, 4]
    summary = {
        "checkpoint": str(args.checkpoint), "subset_root": str(args.subset_root), "normalization_root": str(normalization_root),
        "model": model_name, "device": str(device), "seed": args.seed, "geometry_trials": len(geometry_trials), "target_trials": len(target_trials),
        "gradient_cases": len(gradient_rows), "gradient_all_finite": bool(all(row["all_finite"] for row in gradient_rows)),
        "gradient_norm": {"min": float(finite_norms.min()), "max": float(finite_norms.max()), "mean": float(finite_norms.mean()), "std": float(finite_norms.std())},
        "gradient_nonzero_fraction": {"min": float(gradient_values[:, 5].min()), "max": float(gradient_values[:, 5].max()), "mean": float(gradient_values[:, 5].mean())},
        "local_perturbation_cases": len(perturbation_rows),
        "local_perturbation_sign_agreement": float(np.mean([row["sign_agreement"] for row in perturbation_rows if row["sign_agreement"] is not None])),
        "local_perturbation_mean_absolute_error": float(np.mean([row["absolute_error"] for row in perturbation_rows])),
    }
    (args.output_dir / "gradient_stability.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if first_gradient is not None:
        figure, axes = plt.subplots(1, 2, figsize=(7, 3.2))
        axes[0].imshow(geometry_array[0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[0].set_title("continuous geometry")
        image = axes[1].imshow(np.abs(first_gradient), cmap="magma", interpolation="nearest")
        axes[1].set_title("absolute gradient")
        for axis in axes:
            axis.set_xticks([])
            axis.set_yticks([])
        figure.colorbar(image, ax=axes[1], fraction=0.046)
        figure.tight_layout()
        figure.savefig(args.output_dir / "gradient_map.png", dpi=180)
        plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
