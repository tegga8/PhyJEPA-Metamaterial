"""Train Phase 5A: Phase 4.2 spatial JEPA conditioned on paired EM targets."""

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

from src.mask_aware_spatial_jepa_losses import mask_aware_spatial_jepa_loss, mask_weight_map, masked_reconstruction_bce, spatial_latent_statistics
from src.physics_conditioned_dataset import build_physics_completion_dataloaders
from src.physics_conditioned_spatial_jepa import PhysicsConditionedSpatialJEPA


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


@torch.inference_mode()
def evaluate_epoch(
    model: PhysicsConditionedSpatialJEPA,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    alpha: float,
    gamma: float,
    lambda_recon: float,
) -> dict[str, float]:
    model.eval()
    model.target_encoder.eval()
    totals = {"jepa_loss": 0.0, "reconstruction_bce": 0.0, "total_loss": 0.0, "film_gamma_abs_deviation": 0.0, "film_beta_abs": 0.0}
    contexts: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    count = 0
    for batch in loader:
        inputs = batch["input"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        response = batch["response"].to(device)
        outputs = model(inputs, response, target)
        latent = mask_aware_spatial_jepa_loss(outputs["z_pred"], outputs["z_target"], mask_weight_map(mask, alpha, gamma))
        reconstruction = masked_reconstruction_bce(outputs["logits"], target, mask)
        total = latent + lambda_recon * reconstruction
        size = inputs.shape[0]
        totals["jepa_loss"] += float(latent.item()) * size
        totals["reconstruction_bce"] += float(reconstruction.item()) * size
        totals["total_loss"] += float(total.item()) * size
        totals["film_gamma_abs_deviation"] += float(torch.mean(torch.abs(outputs["film_gamma"] - 1.0)).item()) * size
        totals["film_beta_abs"] += float(torch.mean(torch.abs(outputs["film_beta"])).item()) * size
        contexts.append(outputs["z_context"].cpu())
        targets.append(outputs["z_target"].cpu())
        predictions.append(outputs["z_pred"].cpu())
        count += size
    result = {name: value / count for name, value in totals.items()} | spatial_latent_statistics(torch.cat(contexts), torch.cat(targets), torch.cat(predictions))
    result["collapse_flag"] = float(any(result[f"{name}_mean_std"] < 1e-4 for name in ("context", "target", "pred")))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--em-encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-reference", type=Path, default=None)
    parser.add_argument("--mask-type", choices=("central_block", "random_holes"), required=True)
    parser.add_argument("--missing-ratio", type=float, required=True)
    parser.add_argument("--latent-channels", type=int, default=64)
    parser.add_argument("--predictor-hidden-channels", type=int, default=128)
    parser.add_argument("--physics-embedding-dim", type=int, default=128)
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
    args = parser.parse_args()
    if args.latent_channels != 64 or args.predictor_hidden_channels != 128 or args.physics_embedding_dim != 128:
        raise ValueError("Phase 5A preserves the Phase 4.2 64-channel/128-hidden model and uses a 128-D EM embedding")
    if not 0.0 <= args.alpha <= 1.0 or args.gamma <= 0:
        raise ValueError("alpha must be in [0,1] and gamma must be positive")
    if not args.em_encoder_checkpoint.is_file():
        raise FileNotFoundError(f"EM encoder checkpoint not found: {args.em_encoder_checkpoint}")
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaders = build_physics_completion_dataloaders(args.subset_root, args.mask_type, args.missing_ratio, args.seed, args.batch_size)
    model = PhysicsConditionedSpatialJEPA(args.latent_channels, args.predictor_hidden_channels, args.ema_decay, args.physics_embedding_dim).to(device)
    em_checkpoint = torch.load(args.em_encoder_checkpoint, map_location=device, weights_only=False)
    model.em_encoder.load_state_dict(em_checkpoint["encoder_state_dict"])
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    checkpoint_path = args.output_dir / "best.pt"
    best_validation = float("inf")
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.target_encoder.eval()
        totals = {"jepa_loss": 0.0, "reconstruction_bce": 0.0, "total_loss": 0.0, "film_gamma_abs_deviation": 0.0, "film_beta_abs": 0.0}
        items = 0
        for batch in loaders["train"]:
            inputs = batch["input"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            response = batch["response"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs, response, target)
            latent = mask_aware_spatial_jepa_loss(outputs["z_pred"], outputs["z_target"], mask_weight_map(mask, args.alpha, args.gamma))
            reconstruction = masked_reconstruction_bce(outputs["logits"], target, mask)
            total = latent + args.lambda_recon * reconstruction
            total.backward()
            optimizer.step()
            model.update_target_encoder()
            size = inputs.shape[0]
            totals["jepa_loss"] += float(latent.item()) * size
            totals["reconstruction_bce"] += float(reconstruction.item()) * size
            totals["total_loss"] += float(total.item()) * size
            totals["film_gamma_abs_deviation"] += float(torch.mean(torch.abs(outputs["film_gamma"] - 1.0)).item()) * size
            totals["film_beta_abs"] += float(torch.mean(torch.abs(outputs["film_beta"])).item()) * size
            items += size
        validation = evaluate_epoch(model, loaders["val"], device, args.alpha, args.gamma, args.lambda_recon)
        record = {"epoch": epoch, **{f"train_{name}": value / items for name, value in totals.items()}, **{f"val_{name}": value for name, value in validation.items()}}
        history.append(record)
        print(f"epoch {epoch:03d} | train_total={record['train_total_loss']:.6f} | val_total={record['val_total_loss']:.6f} | film_delta={record['val_film_gamma_abs_deviation']:.6f}")
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
    em_config_path = args.em_encoder_checkpoint.parent / "config.json"
    em_config = json.loads(em_config_path.read_text(encoding="utf-8")) if em_config_path.is_file() else {}
    config = {
        "phase": "5A", "subset_root": str(args.subset_root), "split": {name: len(loader.dataset) for name, loader in loaders.items()},
        "mask_type": args.mask_type, "missing_ratio": args.missing_ratio, "mask_seed": args.seed,
        "model": "PhysicsConditionedSpatialJEPA", "phase4_2_components_preserved": {"context_encoder": "SpatialGeometryEncoder(2,64)", "target_encoder": "SpatialGeometryEncoder(1,64), EMA geometry-only", "predictor": "SpatialPredictor(64,128)", "decoder": "SpatialGeometryDecoder(64)", "loss": "mask-aware spatial JEPA + 0.1 masked BCE"},
        "em_representation": "Phase-2-normalized [Re(T_y), Im(T_y), Re(R_x), Im(R_x)] [4,1001]", "em_normalization_stats": str(args.subset_root / "train_response_stats.npz"), "em_encoder_checkpoint": str(args.em_encoder_checkpoint), "em_encoder_config": em_config.get("encoder"), "physics_embedding_dim": args.physics_embedding_dim,
        "conditioning": {"method": "FiLM before unchanged SpatialPredictor", "mapping": "128 -> 128 -> gamma,beta [64,64]", "initialization": "final FiLM linear weight/bias zero; gamma=1 and beta=0"},
        "latent_channels": args.latent_channels, "latent_spatial_shape": [args.latent_channels, 8, 8], "predictor_hidden_channels": args.predictor_hidden_channels, "ema_decay": args.ema_decay,
        "alpha": args.alpha, "gamma": args.gamma, "lambda_recon": args.lambda_recon, "optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "batch_size": args.batch_size, "epochs_requested": args.epochs, "epochs_completed": len(history), "patience": args.patience,
        "baseline_reference": str(args.baseline_reference) if args.baseline_reference is not None else None, "model_parameter_count_total": sum(parameter.numel() for parameter in model.parameters()), "trainable_parameter_count": sum(parameter.numel() for parameter in model.trainable_parameters()), "em_encoder_parameter_count": sum(parameter.numel() for parameter in model.em_encoder.parameters()), "film_parameter_count": sum(parameter.numel() for parameter in model.film.parameters()),
        "best_epoch": int(checkpoint["epoch"]), "best_validation_total_loss": float(checkpoint["best_metric"]), "device": str(device), "seed": args.seed, "training_seconds": elapsed, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "git_commit": git_commit(), "threshold": 0.5,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps({"checkpoint": str(checkpoint_path), "config": config}, indent=2))


if __name__ == "__main__":
    main()
