"""Train and evaluate a compact forward EM surrogate on the Phase 1 subset."""

from __future__ import annotations

import argparse
import json
import random
import sys
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import SUTDPRCMDataset, build_dataloaders
from src.metrics import reflection_magnitude_mae, unnormalize_response
from src.losses import resonance_weighted_complex_loss
from src.models import build_forward_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loss_function(
    name: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    resonance_weight: float,
    magnitude_weight: float,
) -> nn.Module | callable:
    if name == "normalized_mse":
        return nn.MSELoss()
    if name == "resonance_weighted_complex":
        return lambda prediction, target: resonance_weighted_complex_loss(
            prediction, target, mean, std, resonance_weight, magnitude_weight
        )
    raise ValueError("Unknown loss; choose from normalized_mse or resonance_weighted_complex")


@torch.inference_mode()
def evaluate_loss(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_function: nn.Module | callable,
) -> float:
    model.eval()
    total = 0.0
    samples = 0
    for geometry, target in loader:
        geometry, target = geometry.to(device), target.to(device)
        total += float(loss_function(model(geometry), target).item()) * geometry.shape[0]
        samples += geometry.shape[0]
    return total / samples


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    total_squared_error = 0.0
    total_elements = 0
    magnitude_total = {"y_reflection_magnitude_mae": 0.0, "x_reflection_magnitude_mae": 0.0}
    samples = 0
    for geometry, target in loader:
        geometry, target = geometry.to(device), target.to(device)
        prediction = model(geometry)
        total_squared_error += float(torch.sum((prediction - target) ** 2).item())
        total_elements += target.numel()
        raw_prediction = unnormalize_response(prediction, mean, std)
        raw_target = unnormalize_response(target, mean, std)
        batch_metrics = reflection_magnitude_mae(raw_prediction, raw_target)
        batch_size = geometry.shape[0]
        for name, value in batch_metrics.items():
            magnitude_total[name] += value * batch_size
        samples += batch_size
    return {"normalized_mse": total_squared_error / total_elements, **{name: value / samples for name, value in magnitude_total.items()}}


def save_prediction_plot(
    model: nn.Module,
    dataset: SUTDPRCMDataset,
    device: torch.device,
    output_path: Path,
    count: int = 4,
) -> None:
    model.eval()
    stats = np.load(dataset.root / "train_response_stats.npz")
    mean = torch.from_numpy(stats["mean"]).to(device)
    std = torch.from_numpy(stats["std"]).to(device)
    selected = np.linspace(0, len(dataset) - 1, count, dtype=int)
    figure, axes = plt.subplots(count, 2, figsize=(12, 2.8 * count), sharex=True, sharey=True)
    with torch.inference_mode():
        for row, index in enumerate(selected):
            geometry, target = dataset[int(index)]
            prediction = model(geometry.unsqueeze(0).to(device))[0]
            prediction = unnormalize_response(prediction, mean, std).cpu()
            target = unnormalize_response(target.to(device), mean, std).cpu()
            for axis, real_index, imag_index, label in (
                (axes[row, 0], 0, 1, "y-polarized |T|"),
                (axes[row, 1], 2, 3, "x-polarized |R|"),
            ):
                axis.plot(dataset.frequency_ghz, torch.hypot(target[real_index], target[imag_index]), label="true", linewidth=1.2)
                axis.plot(dataset.frequency_ghz, torch.hypot(prediction[real_index], prediction[imag_index]), label="predicted", linewidth=1.0, linestyle="--")
                axis.set_title(f"{label}: {dataset.source_id(int(index))}", fontsize=8)
                axis.set_xlabel("frequency (GHz)")
                axis.set_ylabel("magnitude")
                axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase2_forward"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="ForwardSurrogateCNN", choices=("ForwardSurrogateCNN", "ResponseAwareSurrogateCNN"))
    parser.add_argument("--loss", default="normalized_mse", choices=("normalized_mse", "resonance_weighted_complex"))
    parser.add_argument("--resonance-weight", type=float, default=4.0)
    parser.add_argument("--magnitude-weight", type=float, default=0.15)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a PyTorch device string")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaders = build_dataloaders(args.subset_root, batch_size=args.batch_size)
    model = build_forward_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    stats = np.load(args.subset_root / "train_response_stats.npz")
    mean = torch.from_numpy(stats["mean"]).to(device)
    std = torch.from_numpy(stats["std"]).to(device)
    loss_function = make_loss_function(args.loss, mean, std, args.resonance_weight, args.magnitude_weight)

    best_validation = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    checkpoint_path = args.output_dir / "best.pt"
    training_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        items = 0
        for geometry, target in loaders["train"]:
            geometry, target = geometry.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(geometry), target)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * geometry.shape[0]
            items += geometry.shape[0]
        validation = evaluate(model, loaders["val"], device, mean, std)
        validation_loss = evaluate_loss(model, loaders["val"], device, loss_function)
        record = {"epoch": epoch, "train_loss": running_loss / items, "train_normalized_mse": running_loss / items if args.loss == "normalized_mse" else None, "validation_loss": validation_loss, **validation}
        history.append(record)
        print(f"epoch {epoch:03d} | train_loss={record['train_loss']:.6f} | val_loss={validation_loss:.6f} | val_mse={validation['normalized_mse']:.6f}")
        if validation_loss < best_validation:
            best_validation = validation_loss
            epochs_without_improvement = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "validation": validation, "validation_loss": validation_loss, "args": vars(args)}, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping after {epoch} epochs")
                break
        scheduler.step()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, loaders["test"], device, mean, std)
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "subset_root": str(args.subset_root), "split_counts": {name: len(loader.dataset) for name, loader in loaders.items()},
        "model": args.model, "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "epochs_requested": args.epochs, "epochs_completed": len(history), "best_epoch": checkpoint["epoch"],
        "best_validation": checkpoint["validation"], "batch_size": args.batch_size, "optimizer": "AdamW",
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "loss": args.loss,
        "resonance_weight": args.resonance_weight, "magnitude_weight": args.magnitude_weight,
        "seed": args.seed, "device": str(device), "training_seconds": time.perf_counter() - training_start,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
    }
    (args.output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot([entry["epoch"] for entry in history], [entry["train_loss"] for entry in history], label="train loss")
    axis.plot([entry["epoch"] for entry in history], [entry["validation_loss"] for entry in history], label="validation loss")
    axis.set_xlabel("epoch")
    axis.set_ylabel("normalized MSE")
    axis.set_title("Forward-surrogate learning curve")
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "learning_curve.png", dpi=180)
    plt.close(figure)
    save_prediction_plot(model, SUTDPRCMDataset(args.subset_root, "test"), device, args.output_dir / "test_predictions.png")
    print("test metrics:", json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
