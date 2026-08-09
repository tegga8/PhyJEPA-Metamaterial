"""Verify a trained Phase 2 forward surrogate against a no-learning baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import build_dataloaders
from src.metrics import reflection_magnitude_mae, unnormalize_response
from src.models import ForwardSurrogateCNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/phase2_forward_75ep"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    required = [args.run_dir / name for name in ("best.pt", "history.json", "test_metrics.json", "learning_curve.png", "test_predictions.png")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 2 artifacts: {missing}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_dataloaders(args.subset_root, batch_size=args.batch_size)
    checkpoint = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False)
    model = ForwardSurrogateCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stats = np.load(args.subset_root / "train_response_stats.npz")
    mean, std = torch.from_numpy(stats["mean"]).to(device), torch.from_numpy(stats["std"]).to(device)

    squared_error = zero_squared_error = 0.0
    elements = 0
    magnitude_sums = {"y_reflection_magnitude_mae": 0.0, "x_reflection_magnitude_mae": 0.0}
    samples = 0
    with torch.inference_mode():
        for geometry, target in loaders["test"]:
            geometry, target = geometry.to(device), target.to(device)
            prediction = model(geometry)
            squared_error += float(torch.sum((prediction - target) ** 2).item())
            zero_squared_error += float(torch.sum(target ** 2).item())
            elements += target.numel()
            metrics = reflection_magnitude_mae(
                unnormalize_response(prediction, mean, std),
                unnormalize_response(target, mean, std),
            )
            batch_size = geometry.shape[0]
            for name, value in metrics.items():
                magnitude_sums[name] += value * batch_size
            samples += batch_size
    model_mse = squared_error / elements
    zero_mse = zero_squared_error / elements
    if model_mse >= zero_mse:
        raise RuntimeError(f"Model MSE ({model_mse:.6f}) does not beat zero baseline ({zero_mse:.6f})")
    print("Phase 2 verification passed")
    print(f"- normalized test MSE: {model_mse:.6f}")
    print(f"- zero-response baseline MSE: {zero_mse:.6f}")
    print(f"- relative MSE reduction: {(1.0 - model_mse / zero_mse) * 100:.1f}%")
    for name, value in magnitude_sums.items():
        print(f"- {name}: {value / samples:.6f}")


if __name__ == "__main__":n+    main()
