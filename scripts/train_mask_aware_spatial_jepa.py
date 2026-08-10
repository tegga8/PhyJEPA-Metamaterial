"""Train Phase 4.2 mask-aware spatial JEPA with a fixed Phase 4.1 model."""

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

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.completion_dataset import build_completion_dataloaders
from src.mask_aware_spatial_jepa_losses import downsample_mask, mask_aware_spatial_jepa_loss, masked_reconstruction_bce, spatial_latent_statistics
from src.spatial_jepa_completion_model import SpatialJEPACompletionModel


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


def weight_batch_stats(mask: torch.Tensor, alpha: float, gamma: float) -> dict[str, float]:
    mask8 = downsample_mask(mask)
    weights = alpha + (1.0 - alpha) * mask8.pow(gamma)
    return {"mean_mask8": float(mask8.mean().item()), "min_mask8": float(mask8.min().item()), "max_mask8": float(mask8.max().item()), "mean_weight": float(weights.mean().item()), "min_weight": float(weights.min().item()), "max_weight": float(weights.max().item())}


@torch.inference_mode()
def evaluate_epoch(model: SpatialJEPACompletionModel, loader: torch.utils.data.DataLoader, device: torch.device, alpha: float, gamma: float, lambda_recon: float) -> dict[str, float]:
    model.eval()
    model.target_encoder.eval()
    totals = {"jepa_loss": 0.0, "reconstruction_bce": 0.0, "total_loss": 0.0, "mean_mask8": 0.0, "min_mask8": 0.0, "max_mask8": 0.0, "mean_weight": 0.0, "min_weight": 0.0, "max_weight": 0.0}
    context_values: list[torch.Tensor] = []
    target_values: list[torch.Tensor] = []
    pred_values: list[torch.Tensor] = []
    count = 0
    for batch in loader:
        inputs, target, mask = batch["input"].to(device), batch["target"].to(device), batch["mask"].to(device)
        outputs = model(inputs, target)
        weights = alpha + (1.0 - alpha) * downsample_mask(mask).pow(gamma)
        latent = mask_aware_spatial_jepa_loss(outputs["z_pred"], outputs["z_target"], weights)
        reconstruction = masked_reconstruction_bce(outputs["logits"], target, mask)
        total = latent + lambda_recon * reconstruction
        size = inputs.shape[0]
        totals["jepa_loss"] += float(latent.item()) * size
        totals["reconstruction_bce"] += float(reconstruction.item()) * size
        totals["total_loss"] += float(total.item()) * size
        stats = weight_batch_stats(mask, alpha, gamma)
        for name, value in stats.items():
            totals[name] += value * size
        context_values.append(outputs["z_context"].cpu())
        target_values.append(outputs["z_target"].cpu())
        pred_values.append(outputs["z_pred"].cpu())
        count += size
    latent_metrics = spatial_latent_statistics(torch.cat(context_values), torch.cat(target_values), torch.cat(pred_values))
    result = {name: value / count for name, value in totals.items()} | latent_metrics
    result["collapse_flag"] = float(any(result[f"{name}_mean_std"] < 1e-4 for name in ("context", "target", "pred")))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-type", choices=("central_block", "random_holes"), required=True)
    parser.add_argument("--missing-ratio", type=float, required=True)
    parser.add_argument("--latent-channels", type=int, default=64)
    parser.add_argument("--predictor-hidden-channels", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lambda-recon", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    if args.latent_channels != 64:
        raise ValueError("Phase 4.2 keeps the Phase 4.1 64-channel spatial latent")
    if not 0.0 <= args.alpha <= 1.0 or args.gamma <= 0.0:
        raise ValueError("alpha must be in [0,1] and gamma must be positive")
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaders = build_completion_dataloaders(args.subset_root, args.mask_type, args.missing_ratio, args.seed, args.batch_size)
    model = SpatialJEPACompletionModel(args.latent_channels, args.predictor_hidden_channels, args.ema_decay).to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_epoch = 1
    best_validation = float("inf")
    history: list[dict[str, float | int]] = []
    checkpoint_path = args.output_dir / "best.pt"
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint.get("best_metric", best_validation))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    stale_epochs = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        model.target_encoder.eval()
        totals = {"jepa_loss": 0.0, "reconstruction_bce": 0.0, "total_loss": 0.0}
        items = 0
        for batch in loaders["train"]:
            inputs, target, mask = batch["input"].to(device), batch["target"].to(device), batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs, target)
            weights = args.alpha + (1.0 - args.alpha) * downsample_mask(mask).pow(args.gamma)
            latent = mask_aware_spatial_jepa_loss(outputs["z_pred"], outputs["z_target"], weights)
            reconstruction = masked_reconstruction_bce(outputs["logits"], target, mask)
            total = latent + args.lambda_recon * reconstruction
            total.backward()
            optimizer.step()
            model.update_target_encoder()
            size = inputs.shape[0]
            totals["jepa_loss"] += float(latent.item()) * size
            totals["reconstruction_bce"] += float(reconstruction.item()) * size
            totals["total_loss"] += float(total.item()) * size
            items += size
        validation = evaluate_epoch(model, loaders["val"], device, args.alpha, args.gamma, args.lambda_recon)
        record = {"epoch": epoch, "train_jepa_loss": totals["jepa_loss"] / items, "train_reconstruction_bce": totals["reconstruction_bce"] / items, "train_total_loss": totals["total_loss"] / items, **{f"val_{name}": value for name, value in validation.items()}}
        history.append(record)
        print(f"epoch {epoch:03d} | train_total={record['train_total_loss']:.6f} | val_total={record['val_total_loss']:.6f} | val_jepa={record['val_jepa_loss']:.6f} | latent_std={record['val_context_mean_std']:.6f}")
        if validation["collapse_flag"]:
            print("warning: spatial latent collapse threshold reached")
        if validation["total_loss"] < best_validation:
            best_validation = validation["total_loss"]
            stale_epochs = 0
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "best_metric": best_validation, "config": vars(args)}, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping after {epoch} epochs")
                break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    elapsed = time.perf_counter() - start
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    config = {
        "phase": "4.2", "subset_root": str(args.subset_root), "split": {name: len(loader.dataset) for name, loader in loaders.items()}, "mask_type": args.mask_type, "missing_ratio": args.missing_ratio, "mask_seed": args.seed,
        "model": "SpatialJEPACompletionModel", "latent_channels": args.latent_channels, "latent_spatial_shape": [args.latent_channels, 8, 8], "predictor_hidden_channels": args.predictor_hidden_channels, "ema_decay": args.ema_decay, "alpha": args.alpha, "gamma": args.gamma, "lambda_recon": args.lambda_recon,
        "optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "batch_size": args.batch_size, "epochs_requested": args.epochs, "epochs_completed": len(history), "patience": args.patience, "model_parameter_count_total": sum(parameter.numel() for parameter in model.parameters()), "trainable_parameter_count": sum(parameter.numel() for parameter in model.trainable_parameters()),
        "best_epoch": checkpoint["epoch"], "best_validation_total_loss": checkpoint["best_metric"], "device": str(device), "seed": args.seed, "training_seconds": elapsed, "peak_gpu_memory_bytes": peak_memory, "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "git_commit": git_commit(), "threshold": 0.5,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps({"checkpoint": str(checkpoint_path), "config": config}, indent=2))


if __name__ == "__main__":
    main()
