"""Train and evaluate the complete-geometry Phase 8 autoencoder."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import SUTDPRCMDataset
from src.geometry_autoencoder import GeometryAutoencoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def reconstruction_metrics(probabilities: torch.Tensor, target: torch.Tensor, threshold: float) -> dict[str, torch.Tensor]:
    binary = probabilities >= threshold
    target_binary = target >= 0.5
    intersection = (binary & target_binary).flatten(1).sum(dim=1).float()
    union = (binary | target_binary).flatten(1).sum(dim=1).float()
    predicted_count = binary.flatten(1).sum(dim=1).float()
    target_count = target_binary.flatten(1).sum(dim=1).float()
    return {
        "bce": F.binary_cross_entropy(probabilities.clamp(1e-6, 1 - 1e-6), target, reduction="none").flatten(1).mean(dim=1),
        "iou": torch.where(union > 0, intersection / union, torch.ones_like(union)),
        "dice": torch.where(predicted_count + target_count > 0, 2 * intersection / (predicted_count + target_count), torch.ones_like(union)),
        "pixel_accuracy": (binary == target_binary).flatten(1).float().mean(dim=1),
        "occupancy_difference": binary.flatten(1).float().mean(dim=1) - target_binary.flatten(1).float().mean(dim=1),
        "predicted_occupancy": binary.flatten(1).float().mean(dim=1),
        "target_occupancy": target_binary.flatten(1).float().mean(dim=1),
    }


@torch.inference_mode()
def evaluate(model: GeometryAutoencoder, loader: DataLoader, device: torch.device, threshold: float, cache_path: Path | None = None) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    totals: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    latent_cache: list[np.ndarray] = []
    count = 0
    for geometries, _ in loader:
        geometries = geometries.to(device)
        outputs = model(geometries)
        probabilities = torch.sigmoid(outputs["logits"])
        values = reconstruction_metrics(probabilities, geometries, threshold)
        latent_cache.append(outputs["latent"].cpu().numpy().astype(np.float32))
        batch_size = geometries.shape[0]
        for index in range(batch_size):
            row = {name: float(value[index].item()) for name, value in values.items()}
            row["latent_finite"] = bool(torch.isfinite(outputs["latent"][index]).all().item())
            row["logits_finite"] = bool(torch.isfinite(outputs["logits"][index]).all().item())
            row["probability_finite"] = bool(torch.isfinite(probabilities[index]).all().item())
            rows.append(row)
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value.sum().item())
        count += batch_size
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, np.concatenate(latent_cache, axis=0))
    summary = {name: value / count for name, value in totals.items()}
    summary["samples"] = count
    summary["latent_shape"] = list(np.concatenate(latent_cache[:1], axis=0).shape[1:])
    summary["finite_latent_fraction"] = float(np.mean([row["latent_finite"] for row in rows]))
    summary["finite_logits_fraction"] = float(np.mean([row["logits_finite"] for row in rows]))
    summary["finite_probability_fraction"] = float(np.mean([row["probability_finite"] for row in rows]))
    summary["occupancy_abs_difference"] = float(np.mean(np.abs([row["occupancy_difference"] for row in rows])))
    return summary, rows


def train_epoch(model: GeometryAutoencoder, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total = 0.0
    count = 0
    for geometries, _ in loader:
        geometries = geometries.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(geometries)["logits"]
        loss = F.binary_cross_entropy_with_logits(logits, geometries)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * geometries.shape[0]
        count += geometries.shape[0]
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase8_geometry_autoencoder"))
    parser.add_argument("--latent-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    if args.latent_channels != 64:
        raise ValueError("Phase 8 starts with the existing 64-channel spatial latent")
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {split: SUTDPRCMDataset(args.subset_root, split, normalize_response=False) for split in ("train", "val", "test")}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train", generator=generator if split == "train" else None, num_workers=0) for split, dataset in datasets.items()}
    # Evaluation and latent caching must preserve the dataset manifest order.
    # The training loader remains shuffled, but cache loaders are never shuffled.
    cache_loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) for split, dataset in datasets.items()}
    model = GeometryAutoencoder(args.latent_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    checkpoint_path = args.output_dir / "best.pt"
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    start_epoch = 1
    resume_path = args.resume if args.resume is not None else (checkpoint_path if checkpoint_path.is_file() else None)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_metric", best_val))
        best_epoch = int(checkpoint.get("epoch", best_epoch))
        print(f"resuming from epoch {start_epoch - 1} with best validation BCE {best_val:.6f}")
    start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        train_bce = train_epoch(model, loaders["train"], optimizer, device)
        val_metrics, _ = evaluate(model, loaders["val"], device, args.threshold)
        record = {"epoch": epoch, "train_bce": train_bce, "val_bce": val_metrics["bce"], "val_iou": val_metrics["iou"], "val_dice": val_metrics["dice"], "val_pixel_accuracy": val_metrics["pixel_accuracy"]}
        history.append(record)
        print(f"epoch {epoch:03d} | train_bce={train_bce:.6f} | val_bce={record['val_bce']:.6f} | val_iou={record['val_iou']:.6f}")
        if record["val_bce"] < best_val:
            best_val = record["val_bce"]
            best_epoch = epoch
            stale = 0
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "best_metric": best_val, "config": vars(args)}, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping after {epoch} epochs")
                break
    elapsed = time.perf_counter() - start
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    summaries: dict[str, dict[str, float]] = {}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        summary, rows = evaluate(model, cache_loaders[split], device, args.threshold, args.output_dir / "latents" / f"{split}.npy")
        summaries[split] = summary
        all_rows[split] = rows
        (args.output_dir / "latents" / f"{split}_source_ids.txt").write_text("\n".join(datasets[split].source_id(index) for index in range(len(datasets[split]))) + "\n", encoding="utf-8")
        with (args.output_dir / f"{split}_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    config = {
        "phase": "8", "objective": "complete geometry autoencoder", "subset_root": str(args.subset_root), "split": {split: len(loader.dataset) for split, loader in loaders.items()},
        "model": "GeometryAutoencoder", "latent_channels": args.latent_channels, "latent_shape": [args.latent_channels, 8, 8], "input_shape": [1, 16, 16], "output_shape": [1, 16, 16],
        "uses_masks": False, "uses_em": False, "uses_jepa_loss": False, "uses_physics_loss": False, "optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "batch_size": args.batch_size, "epochs_requested": args.epochs, "epochs_completed": int(checkpoint["epoch"]), "patience": args.patience, "best_epoch": best_epoch, "best_validation_bce": best_val,
        "threshold": args.threshold, "seed": args.seed, "device": str(device), "training_seconds": elapsed, "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()), "python": platform.python_version(), "torch": torch.__version__, "git_commit": git_commit(),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps({"phase": "8", "config": config, "splits": summaries}, indent=2) + "\n", encoding="utf-8")
    if history:
        with (args.output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    print(json.dumps({"checkpoint": str(checkpoint_path), "config": config, "test": summaries["test"]}, indent=2))


if __name__ == "__main__":
    main()
