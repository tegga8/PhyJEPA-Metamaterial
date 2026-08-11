"""Train Phase 5B: Phase 5A completion plus frozen-surrogate consistency."""

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
from src.completion_dataset import CompletionDataset
from src.physics_conditioned_dataset import build_physics_completion_dataloaders
from src.physics_conditioned_spatial_jepa import PhysicsConditionedSpatialJEPA
from src.physics_consistency import continuous_completion, load_frozen_forward_surrogate, physics_consistency_loss, physics_gradient_diagnostics


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


def resolve_device(requested: str) -> torch.device:
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but is unavailable")
    return device


def git_commit() -> str | None:
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError): return None


def save_mask_manifests(root: Path, subset_root: Path, mask_type: str, missing_ratio: float, seed: int) -> None:
    """Persist the deterministic Phase-4.2 masks beside every Phase-5B run."""
    for split in ("train", "val", "test"):
        dataset = CompletionDataset(subset_root, split, mask_type, missing_ratio, seed)
        with (root / f"mask_manifest_{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("sample_id", "index", "mask_seed", "masked_pixels", "mask_ratio")); writer.writeheader()
            for index in range(len(dataset)):
                sample = dataset[index]; mask = sample["mask"]
                writer.writerow({"sample_id": sample["sample_id"], "index": index, "mask_seed": dataset.mask_seed(index), "masked_pixels": int(mask.sum()), "mask_ratio": float(mask.mean())})
    (root / "mask_manifest_config.json").write_text(json.dumps({"subset_root": str(subset_root), "mask_type": mask_type, "missing_ratio": missing_ratio, "base_seed": seed, "height": 16, "width": 16}, indent=2) + "\n", encoding="utf-8")


def loss_terms(model, surrogate, batch, device, alpha, gamma):
    inputs, target, mask, response = (batch[key].to(device) for key in ("input", "target", "mask", "response"))
    outputs = model(inputs, response, target)
    jepa = mask_aware_spatial_jepa_loss(outputs["z_pred"], outputs["z_target"], mask_weight_map(mask, alpha, gamma))
    reconstruction = masked_reconstruction_bce(outputs["logits"], target, mask)
    probabilities = torch.sigmoid(outputs["logits"])
    physics, geometry = physics_consistency_loss(probabilities, inputs, mask, response, surrogate)
    return inputs, target, mask, response, outputs, jepa, reconstruction, physics, geometry


@torch.inference_mode()
def evaluate_epoch(model, surrogate, loader, device, alpha, gamma, lambda_recon, lambda_physics):
    model.eval(); model.target_encoder.eval(); surrogate.eval()
    names = ("jepa_loss", "reconstruction_bce", "physics_mse", "total_loss", "film_gamma_abs_deviation", "film_beta_abs")
    totals = dict.fromkeys(names, 0.0); contexts, targets, predictions = [], [], []; count = 0
    for batch in loader:
        inputs, target, mask, response, outputs, jepa, reconstruction, physics, _ = loss_terms(model, surrogate, batch, device, alpha, gamma)
        total = jepa + lambda_recon * reconstruction + lambda_physics * physics
        size = inputs.shape[0]
        values = (jepa, reconstruction, physics, total, torch.abs(outputs["film_gamma"] - 1).mean(), torch.abs(outputs["film_beta"]).mean())
        for name, value in zip(names, values): totals[name] += float(value.item()) * size
        contexts.append(outputs["z_context"].cpu()); targets.append(outputs["z_target"].cpu()); predictions.append(outputs["z_pred"].cpu()); count += size
    result = {name: value / count for name, value in totals.items()} | spatial_latent_statistics(torch.cat(contexts), torch.cat(targets), torch.cat(predictions))
    result["collapse_flag"] = float(any(result[f"{name}_mean_std"] < 1e-4 for name in ("context", "target", "pred")))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k")); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--em-encoder-checkpoint", type=Path, required=True); parser.add_argument("--forward-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-reference", type=Path, default=None); parser.add_argument("--mask-type", choices=("central_block", "random_holes"), required=True); parser.add_argument("--missing-ratio", type=float, required=True)
    parser.add_argument("--lambda-physics", type=float, required=True); parser.add_argument("--alpha", type=float, default=0.10); parser.add_argument("--gamma", type=float, default=1.0); parser.add_argument("--lambda-recon", type=float, default=0.1); parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--epochs", type=int, default=75); parser.add_argument("--patience", type=int, default=10); parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.lambda_physics < 0 or not 0 <= args.alpha <= 1 or args.gamma <= 0: raise ValueError("Invalid loss weights")
    for path in (args.em_encoder_checkpoint, args.forward_checkpoint):
        if not path.is_file(): raise FileNotFoundError(path)
    set_seed(args.seed); device = resolve_device(args.device); args.output_dir.mkdir(parents=True, exist_ok=True); save_mask_manifests(args.output_dir, args.subset_root, args.mask_type, args.missing_ratio, args.seed)
    loaders = build_physics_completion_dataloaders(args.subset_root, args.mask_type, args.missing_ratio, args.seed, args.batch_size)
    model = PhysicsConditionedSpatialJEPA(64, 128, args.ema_decay, 128).to(device)
    model.em_encoder.load_state_dict(torch.load(args.em_encoder_checkpoint, map_location=device, weights_only=False)["encoder_state_dict"])
    surrogate, forward_name = load_frozen_forward_surrogate(args.forward_checkpoint, device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    diagnostic_batch = next(iter(loaders["train"]))
    initial_scales = {key: float(value.item()) for key, value in zip(("jepa_loss", "reconstruction_bce", "physics_mse"), loss_terms(model, surrogate, diagnostic_batch, device, args.alpha, args.gamma)[5:8])}
    gradient_diagnostic = physics_gradient_diagnostics(model, surrogate, *(diagnostic_batch[key].to(device) for key in ("input", "mask", "response")))
    model.zero_grad(set_to_none=True)
    if not gradient_diagnostic["gradient_finite"] or not gradient_diagnostic["gradient_reaches_decoder_predictor"] or not gradient_diagnostic["surrogate_all_requires_grad_false"] or not gradient_diagnostic["surrogate_has_no_parameter_grad"]:
        raise RuntimeError(f"Phase 5B gradient verification failed: {gradient_diagnostic}")
    best, stale, history, checkpoint_path = float("inf"), 0, [], args.output_dir / "best.pt"
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); model.target_encoder.eval(); totals = dict.fromkeys(("jepa_loss", "reconstruction_bce", "physics_mse", "total_loss", "film_gamma_abs_deviation", "film_beta_abs"), 0.0); items = 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            inputs, target, mask, response, outputs, jepa, reconstruction, physics, _ = loss_terms(model, surrogate, batch, device, args.alpha, args.gamma)
            total = jepa + args.lambda_recon * reconstruction + args.lambda_physics * physics
            total.backward(); optimizer.step(); model.update_target_encoder()
            size = inputs.shape[0]; values = (jepa, reconstruction, physics, total, torch.abs(outputs["film_gamma"] - 1).mean(), torch.abs(outputs["film_beta"]).mean())
            for name, value in zip(totals, values): totals[name] += float(value.item()) * size
            items += size
        validation = evaluate_epoch(model, surrogate, loaders["val"], device, args.alpha, args.gamma, args.lambda_recon, args.lambda_physics)
        record = {"epoch": epoch, **{f"train_{name}": value / items for name, value in totals.items()}, **{f"val_{name}": value for name, value in validation.items()}}; history.append(record)
        print(f"epoch {epoch:03d} | train_total={record['train_total_loss']:.6f} | val_total={record['val_total_loss']:.6f} | val_physics={record['val_physics_mse']:.6f}")
        if validation["total_loss"] < best:
            best, stale = validation["total_loss"], 0
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "best_metric": best, "config": vars(args)}, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience: print(f"early stopping after {epoch} epochs"); break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False); model.load_state_dict(checkpoint["model_state_dict"]); final_gradient_diagnostic = physics_gradient_diagnostics(model, surrogate, *(diagnostic_batch[key].to(device) for key in ("input", "mask", "response"))); elapsed = time.perf_counter() - start
    config = {"phase": "5B", "phase5a_preserved": True, "subset_root": str(args.subset_root), "split": {name: len(loader.dataset) for name, loader in loaders.items()}, "mask_type": args.mask_type, "missing_ratio": args.missing_ratio, "mask_seed": args.seed, "model": "PhysicsConditionedSpatialJEPA", "latent_channels": 64, "predictor_hidden_channels": 128, "physics_embedding_dim": 128, "ema_decay": args.ema_decay, "em_encoder_checkpoint": str(args.em_encoder_checkpoint), "forward_surrogate_checkpoint": str(args.forward_checkpoint), "forward_surrogate_model": forward_name, "forward_surrogate_frozen": True, "physics_geometry_representation": "continuous sigmoid(logits), composited with observed input; no threshold/detach/round", "response_representation": "Phase-2-normalized [Re(T_y), Im(T_y), Re(R_x), Im(R_x)] [4,1001]", "loss": "mask-aware spatial JEPA + lambda_recon*masked BCE + lambda_physics*normalized surrogate MSE", "alpha": args.alpha, "gamma": args.gamma, "lambda_recon": args.lambda_recon, "lambda_physics": args.lambda_physics, "initial_loss_scales": initial_scales, "initial_physics_gradient_diagnostics": gradient_diagnostic, "final_physics_gradient_diagnostics": final_gradient_diagnostic, "optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "batch_size": args.batch_size, "epochs_requested": args.epochs, "epochs_completed": len(history), "patience": args.patience, "best_epoch": int(checkpoint["epoch"]), "best_validation_total_loss": float(checkpoint["best_metric"]), "baseline_reference": str(args.baseline_reference) if args.baseline_reference else None, "device": str(device), "seed": args.seed, "training_seconds": elapsed, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None, "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "git_commit": git_commit(), "threshold": 0.5}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    print(json.dumps({"checkpoint": str(checkpoint_path), "config": config}, indent=2))


if __name__ == "__main__": main()
