"""Train the cross-modal Physics-JEPA model.

Supports three modes:

* ``full``: normal training on the full requested train split (early stopping,
  checkpointing, optional latent caching for downstream evaluation).
* ``overfit``: the required first milestone.  Trains a correct-pair model and
  a shuffled-pair model on the same tiny subset and verifies that the JEPA
  loss decreases, correct pairs align better than shuffled pairs, gradients
  are finite and nonzero, EMA target parameters update, and the target branch
  receives no gradient.  Writes ``smoke_report.json``.
* ``shuffled``: trains on randomly re-paired geometry/response samples for the
  shuffled-pair control experiment (Gate E).
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics_jepa import PhysicsJEPA
from src.physics_jepa_losses import physics_jepa_loss
from src.physics_jepa_training import (
    build_paired_dataloaders,
    cache_latents,
    evaluate_epoch,
    git_commit,
    load_response_stats,
    resolve_device,
    set_seed,
    train_epoch,
)


def save_checkpoint(path: Path, model: PhysicsJEPA, optimizer: torch.optim.Optimizer, epoch: int, metric: float, config: dict) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": metric,
            "config": config,
        },
        path,
    )


def gradient_diagnostics(model: PhysicsJEPA, geometry: torch.Tensor, response: torch.Tensor, device: torch.device, alpha: float, lambda_variance: float) -> dict[str, float | bool]:
    model.train()
    model.spectrum_target_encoder.eval()
    model.zero_grad(set_to_none=True)
    outputs = model(geometry.to(device), response.to(device))
    loss = physics_jepa_loss(
        outputs["z_pred"], outputs["z_target"], outputs["z_self"], alpha, lambda_variance,
        z_online=outputs["z_online"], z_geometry=outputs["z_geometry"],
    )
    loss.backward()

    def norm(module: torch.nn.Module) -> float:
        grads = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        if not grads:
            return 0.0
        return float(torch.linalg.vector_norm(torch.stack([grad.detach().norm() for grad in grads])).item())

    target_gradients = [parameter.grad for parameter in model.spectrum_target_encoder.parameters()]
    values = {
        "loss": float(loss.item()),
        "predictor_gradient_norm": norm(model.predictor),
        "geometry_encoder_gradient_norm": norm(model.geometry_encoder),
        "spectrum_encoder_gradient_norm": norm(model.spectrum_encoder),
        "spectrum_predictor_gradient_norm": norm(model.spectrum_predictor),
        "target_branch_has_gradients": any(gradient is not None for gradient in target_gradients),
    }
    values["gradients_finite_nonzero"] = bool(
        all(torch.isfinite(torch.tensor(v)).item() for k, v in values.items() if isinstance(v, float))
        and values["predictor_gradient_norm"] > 0
        and values["geometry_encoder_gradient_norm"] > 0
        and values["spectrum_encoder_gradient_norm"] > 0
    )
    return values


def train_run(args: argparse.Namespace, shuffled_pairs: bool, run_tag: str) -> dict[str, float]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    loaders = build_paired_dataloaders(args.subset_root, args.batch_size, args.seed, args.max_samples, shuffled_pairs=shuffled_pairs, shuffle_seed=args.shuffle_seed)
    model = PhysicsJEPA(args.latent_dim, args.ema_decay, args.num_tokens, args.token_dim, args.alpha, args.lambda_variance, args.target_centering, args.target_center_decay).to(device)
    target_before_init = next(model.spectrum_target_encoder.parameters()).detach().clone()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, loaders["train"], optimizer, device, args.alpha, args.lambda_variance, args.lambda_covariance)
        validation = evaluate_epoch(model, loaders["train"], device)
        record = {"epoch": epoch, **{f"train_{name}": value for name, value in train_metrics.items()}, **{f"train_subset_{name}": value for name, value in validation.items()}}
        history.append(record)
        print(f"[{run_tag}] epoch {epoch:03d} | train_total={record['train_total_loss']:.6f} | subset_total={record['train_subset_total_loss']:.6f}")
    target_changed = not torch.equal(target_before_init.cpu(), next(model.spectrum_target_encoder.parameters()).detach().cpu())
    first_epoch_total = float(history[0]["train_total_loss"])
    last_epoch_total = float(history[-1]["train_total_loss"])
    geometry, response, _, _ = next(iter(loaders["train"]))
    diagnostics = gradient_diagnostics(model, geometry[: min(8, geometry.shape[0])], response[: min(8, response.shape[0])], device, args.alpha, args.lambda_variance)
    return {
        "first_epoch_total_loss": first_epoch_total,
        "last_epoch_total_loss": last_epoch_total,
        "loss_decreased": last_epoch_total < first_epoch_total,
        "target_parameters_updated": target_changed,
        **{k: v for k, v in diagnostics.items()},
        "final_train_cross_loss": float(history[-1]["train_cross_loss"]),
        "final_subset_total_loss": float(history[-1]["train_subset_total_loss"]),
        "epochs": len(history),
        "seed": args.seed,
    }


def run_overfit(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    correct = train_run(args, shuffled_pairs=False, run_tag="correct")
    set_seed(args.seed)
    shuffled = train_run(args, shuffled_pairs=True, run_tag="shuffled")
    report = {
        "mode": "overfit",
        "correct": correct,
        "shuffled": shuffled,
        "correct_last_total_loss": correct["last_epoch_total_loss"],
        "shuffled_last_total_loss": shuffled["last_epoch_total_loss"],
        "shuffled_pairs_degrade_alignment": bool(shuffled["last_epoch_total_loss"] > correct["last_epoch_total_loss"]),
        "checks": {
            "jepa_loss_decreases": bool(correct["loss_decreased"]),
            "gradients_finite_nonzero": bool(correct["gradients_finite_nonzero"]),
            "target_parameters_update": bool(correct["target_parameters_updated"]),
            "target_branch_has_no_gradient": bool(not correct["target_branch_has_gradients"]),
        },
        "milestone_passed": bool(
            correct["loss_decreased"]
            and correct["gradients_finite_nonzero"]
            and correct["target_parameters_updated"]
            and not correct["target_branch_has_gradients"]
            and shuffled["last_epoch_total_loss"] > correct["last_epoch_total_loss"]
        ),
    }
    (args.output_dir / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["milestone_passed"]:
        raise SystemExit(1)


def run_full(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaders = build_paired_dataloaders(args.subset_root, args.batch_size, args.seed, args.max_samples, shuffled_pairs=args.shuffled_pairs, shuffle_seed=args.shuffle_seed)
    model = PhysicsJEPA(args.latent_dim, args.ema_decay, args.num_tokens, args.token_dim, args.alpha, args.lambda_variance, args.target_centering, args.target_center_decay).to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    checkpoint_path = args.output_dir / "best.pt"
    best_validation = float("inf")
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, loaders["train"], optimizer, device, args.alpha, args.lambda_variance, args.lambda_covariance)
        validation = evaluate_epoch(model, loaders["val"], device, alpha=args.alpha, lambda_variance=args.lambda_variance, lambda_covariance=args.lambda_covariance)
        record = {"epoch": epoch, **{f"train_{name}": value for name, value in train_metrics.items()}, **{f"val_{name}": value for name, value in validation.items()}}
        history.append(record)
        print(f"epoch {epoch:03d} | train_total={record['train_total_loss']:.6f} | val_total={record['val_total_loss']:.6f} | val_cross={record['val_cross_loss']:.6f}")
        if validation["collapse_flag"]:
            print("warning: latent collapse threshold reached")
        if validation["total_loss"] < best_validation:
            best_validation = validation["total_loss"]
            stale_epochs = 0
            save_checkpoint(checkpoint_path, model, optimizer, epoch, best_validation, vars(args))
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping after {epoch} epochs")
                break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    elapsed = time.perf_counter() - start
    mean, std = load_response_stats(args.subset_root)
    frequency_ghz = np.load(args.subset_root / "frequency_ghz.npy")
    cache_paths: dict[str, dict[str, Path]] = {}
    for split in args.cache_splits:
        cache_paths[split] = cache_latents(model, loaders[split], device, args.output_dir, split, (mean, std), frequency_ghz)
    config = {
        "phase": "physics_jepa", "experiment_label": args.experiment_label,
        "mode": "full" if not args.shuffled_pairs else "shuffled",
        "subset_root": str(args.subset_root), "split": {name: len(loader.dataset) for name, loader in loaders.items()},
        "model": "PhysicsJEPA", "objective": "cross-modal JEPA: geometry -> physical latent matches momentum spectrum latent",
        "loss": {"cross": "normalized latent MSE", "bootstrap": "normalized latent MSE with momentum target", "variance": "VICReg std regularizer", "covariance": "Barlow redundancy reduction", "alpha": args.alpha, "lambda_variance": args.lambda_variance, "lambda_covariance": args.lambda_covariance},
        "target_centering": args.target_centering, "target_center_decay": args.target_center_decay,
        "latent_dim": args.latent_dim, "num_tokens": args.num_tokens, "token_dim": args.token_dim, "ema_decay": args.ema_decay,
        "geometry_encoder": "SpatialGeometryEncoder(1,64) + avgpool + MLP -> d", "spectrum_encoder": "Conv1D 4->32->64, 12 tokens, self-attention, attention pool -> d",
        "predictor": f"MLP {args.latent_dim}->{2 * args.latent_dim}->{args.latent_dim}",
        "optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "epochs_requested": args.epochs, "epochs_completed": len(history), "patience": args.patience, "seed": args.seed,
        "best_epoch": int(checkpoint["epoch"]), "best_validation_total_loss": float(checkpoint["best_metric"]),
        "model_parameter_count_total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.trainable_parameters()),
        "device": str(device), "training_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "git_commit": git_commit(),
        "cached_splits": {split: {name: str(path) for name, path in paths.items()} for split, paths in cache_paths.items()},
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps({"checkpoint": str(checkpoint_path), "config": config}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_30k"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "overfit"), required=True)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--num-tokens", type=int, default=12)
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lambda-variance", type=float, default=0.1)
    parser.add_argument("--lambda-covariance", type=float, default=0.0)
    parser.add_argument("--target-centering", action="store_true")
    parser.add_argument("--target-center-decay", type=float, default=0.99)
    parser.add_argument("--experiment-label", default="physics_jepa_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shuffled-pairs", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=123)
    parser.add_argument("--cache-splits", nargs="*", default=["val", "test"])
    args = parser.parse_args()
    if args.latent_dim not in (32, 64):
        raise ValueError("latent_dim must be 32 or 64")
    if args.mode == "overfit":
        if args.max_samples is None:
            args.max_samples = 1024
        args.epochs = 60
        args.patience = 0
        run_overfit(args)
    else:
        run_full(args)


if __name__ == "__main__":
    main()
