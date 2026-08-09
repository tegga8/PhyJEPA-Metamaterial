"""Train the Phase 3 supervised completion CNN."""

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
from src.completion_losses import completion_loss_metrics
from src.completion_model import CompletionCNN


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


@torch.inference_mode()
def evaluate_loss(model: CompletionCNN, loader: torch.utils.data.DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"full_bce": 0.0, "masked_bce": 0.0}
    samples = 0
    for batch in loader:
        inputs, target, mask = batch["input"].to(device), batch["target"].to(device), batch["mask"].to(device)
        metrics = completion_loss_metrics(model(inputs), target, mask)
        size = inputs.shape[0]
        for name, value in metrics.items():
            totals[name] += float(value.item()) * size
        samples += size
    return {name: value / samples for name, value in totals.items()}


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-type", choices=("central_block", "random_holes"), required=True)
    parser.add_argument("--missing-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaders = build_completion_dataloaders(args.subset_root, args.mask_type, args.missing_ratio, args.seed, args.batch_size)
    model = CompletionCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_validation = float("inf")
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    checkpoint_path = args.output_dir / "best.pt"
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        train_items = 0
        for batch in loaders["train"]:
            inputs, target, mask = batch["input"].to(device), batch["target"].to(device), batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = completion_loss_metrics(model(inputs), target, mask)["full_bce"]
            loss.backward()
            optimizer.step()
            train_total += float(loss.item()) * inputs.shape[0]
            train_items += inputs.shape[0]
        validation = evaluate_loss(model, loaders["val"], device)
        record = {"epoch": epoch, "train_full_bce": train_total / train_items, "val_full_bce": validation["full_bce"], "val_masked_bce": validation["masked_bce"]}
        history.append(record)
        print(f"epoch {epoch:03d} | train_bce={record['train_full_bce']:.6f} | val_bce={record['val_full_bce']:.6f} | val_masked_bce={record['val_masked_bce']:.6f}")
        if validation["full_bce"] < best_validation:
            best_validation = validation["full_bce"]
            stale_epochs = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "validation": validation}, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping after {epoch} epochs")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    elapsed = time.perf_counter() - start
    config = {
        "phase": "3", "subset_root": str(args.subset_root), "split": {name: len(loader.dataset) for name, loader in loaders.items()},
        "mask_type": args.mask_type, "missing_ratio_requested": args.missing_ratio, "mask_pixels": int(round(256 * args.missing_ratio)),
        "mask_seed": args.seed, "mask_seed_rule": "base_seed + split_offset + sample_index for random_holes; fixed split seed for central_block",
        "model": "CompletionCNN", "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "loss": "full_bce", "batch_size": args.batch_size, "optimizer": "AdamW", "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "epochs_requested": args.epochs, "epochs_completed": len(history),
        "early_stopping_patience": args.patience, "best_epoch": checkpoint["epoch"], "device": str(device), "seed": args.seed,
        "training_seconds": elapsed, "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "git_commit": git_commit(), "known_pixels_preserved_by_compositing": True, "threshold": 0.5,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps({"checkpoint": str(checkpoint_path), "config": config}, indent=2))


if __name__ == "__main__":
    main()
