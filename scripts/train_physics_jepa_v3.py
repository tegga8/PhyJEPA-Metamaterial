"""Train Physics-JEPA v3 (frequency-aware + relational).

Label: ``physics_jepa_v3_frequency_relational``.

Change A: the spectrum target/online encoder is the frequency-aware
``FrequencySpectrumEncoder`` (multiscale dilated Conv1D at full 1001-point
resolution, sinusoidal frequency-position embedding, small self-attention).
Change B: a small relational margin-ranking loss enforces EM-response
similarity ordering of the latents, in addition to the v2 JEPA objective
(cross + bootstrap + variance + covariance) with the v2 EMA target.

Modes:

* ``overfit``: the required smoke test on a tiny subset (correct + shuffled).
  Verifies JEPA loss decreases, gradients are finite, the target branch has no
  gradients, EMA target parameters update, relational triplets are valid,
  the relational loss is finite, and the frequency positions align with the
  1001-point grid.  Writes ``smoke_report.json`` and a loss-scale diagnostic.
* ``full``: trains one run (``--tag correct`` or ``--tag shuffled``) with the
  v2 training budget/settings, early stopping, checkpointing, and val/test
  latent caching in the same layout the canonical evaluator expects.
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

from src.physics_jepa import PhysicsJEPAFrequencyRelational
from src.physics_jepa_losses import build_response_triplets, normalized_latent_distance_matrix, v3_loss_with_parts
from src.physics_jepa_training import (
    build_paired_dataloaders,
    cache_latents,
    git_commit,
    load_response_stats,
    resolve_device,
    set_seed,
)

MARGIN = 0.2
LAMBDA_RELATIONAL = 0.1


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metric: float, config: dict) -> None:
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


def build_triplets(response: torch.Tensor, num_triplets: int, seed: int) -> dict[str, torch.Tensor]:
    return build_response_triplets(response, num_triplets=num_triplets, seed=seed)


def v3_step(
    model: torch.nn.Module,
    geometry: torch.Tensor,
    response: torch.Tensor,
    alpha: float,
    lambda_variance: float,
    lambda_covariance: float,
    lambda_relational: float,
    num_triplets: int,
    margin: float,
    seed: int,
    update_target: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Forward + optional backward for one batch, returning (loss, parts, triplets)."""
    outputs = model(geometry, response)
    triplets = build_triplets(response, num_triplets, seed)
    total, parts = v3_loss_with_parts(
        outputs["z_pred"],
        outputs["z_target"],
        outputs["z_self"],
        alpha,
        lambda_variance,
        lambda_covariance,
        z_online=outputs["z_online"],
        z_geometry=outputs["z_geometry"],
        z_relational=outputs["z_online"],
        lambda_relational=lambda_relational,
        triplets=triplets,
        margin=margin,
    )
    if update_target:
        total.backward()
    return total, parts, triplets


def train_epoch_v3(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float,
    lambda_variance: float,
    lambda_covariance: float,
    lambda_relational: float,
    num_triplets: int,
    margin: float,
    seed: int,
) -> dict[str, float]:
    model.train()
    model.spectrum_target_encoder.eval()
    totals = {
        "cross_loss": 0.0,
        "bootstrap_loss": 0.0,
        "variance_loss": 0.0,
        "covariance_loss": 0.0,
        "relational_loss": 0.0,
        "total_loss": 0.0,
    }
    items = 0
    for geometry, response, _, _ in loader:
        geometry = geometry.to(device)
        response = response.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, parts, _ = v3_step(
            model, geometry, response, alpha, lambda_variance, lambda_covariance, lambda_relational, num_triplets, margin, seed, update_target=True
        )
        optimizer.step()
        model.update_target_encoder()
        size = geometry.shape[0]
        for name, value in parts.items():
            totals[name] += float(value.item()) * size
        totals["total_loss"] += float(loss.item()) * size
        items += size
    return {name: value / items for name, value in totals.items()}


def evaluate_epoch_v3(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    alpha: float,
    lambda_variance: float,
    lambda_covariance: float,
    lambda_relational: float,
    num_triplets: int,
    margin: float,
    seed: int,
) -> dict[str, float]:
    model.eval()
    model.spectrum_target_encoder.eval()
    totals = {
        "cross_loss": 0.0,
        "bootstrap_loss": 0.0,
        "variance_loss": 0.0,
        "covariance_loss": 0.0,
        "relational_loss": 0.0,
        "total_loss": 0.0,
    }
    latents = {"z_geometry": [], "z_online": [], "z_target": [], "z_pred": []}
    count = 0
    with torch.inference_mode():
        for geometry, response, _, _ in loader:
            geometry = geometry.to(device)
            response = response.to(device)
            loss, parts, _ = v3_step(
                model, geometry, response, alpha, lambda_variance, lambda_covariance, lambda_relational, num_triplets, margin, seed, update_target=False
            )
            size = geometry.shape[0]
            for name, value in parts.items():
                totals[name] += float(value.item()) * size
            totals["total_loss"] += float(loss.item()) * size
            for name, values in latents.items():
                values.append(model(geometry, response)[name].cpu())
            count += size
    result = {name: value / count for name, value in totals.items()}
    latents_cat = {name: torch.cat(values) for name, values in latents.items()}
    from src.physics_jepa_losses import physics_latent_variance_metrics

    result.update(
        physics_latent_variance_metrics(latents_cat["z_geometry"], latents_cat["z_online"], latents_cat["z_target"], latents_cat["z_pred"])
    )
    result["collapse_flag"] = float(
        any(result[f"{name}_mean_std"] < 1e-4 for name in ("context", "online", "target", "pred"))
    )
    return result


def gradient_diagnostics_v3(
    model: torch.nn.Module,
    geometry: torch.Tensor,
    response: torch.Tensor,
    device: torch.device,
    alpha: float,
    lambda_variance: float,
    lambda_covariance: float,
    lambda_relational: float,
    num_triplets: int,
    margin: float,
    seed: int,
) -> dict[str, float | bool]:
    model.train()
    model.spectrum_target_encoder.eval()
    model.zero_grad(set_to_none=True)
    outputs = model(geometry.to(device), response.to(device))
    triplets = build_triplets(response, num_triplets, seed)
    loss, parts = v3_loss_with_parts(
        outputs["z_pred"],
        outputs["z_target"],
        outputs["z_self"],
        alpha,
        lambda_variance,
        lambda_covariance,
        z_online=outputs["z_online"],
        z_geometry=outputs["z_geometry"],
        z_relational=outputs["z_online"],
        lambda_relational=lambda_relational,
        triplets=triplets,
        margin=margin,
    )
    loss.backward()

    def norm(module: torch.nn.Module) -> float:
        grads = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        if not grads:
            return 0.0
        return float(torch.linalg.vector_norm(torch.stack([grad.detach().norm() for grad in grads])).item())

    target_gradients = [parameter.grad for parameter in model.spectrum_target_encoder.parameters()]
    values: dict[str, float | bool] = {
        "loss": float(loss.item()),
        "relational_loss": float(parts["relational_loss"].item()),
        "predictor_gradient_norm": norm(model.predictor),
        "geometry_encoder_gradient_norm": norm(model.geometry_encoder),
        "spectrum_encoder_gradient_norm": norm(model.spectrum_encoder),
        "spectrum_predictor_gradient_norm": norm(model.spectrum_predictor),
        "target_branch_has_gradients": any(gradient is not None for gradient in target_gradients),
    }
    values["gradients_finite_nonzero"] = bool(
        all(torch.isfinite(torch.tensor(v)).item() for k, v in values.items() if isinstance(v, float))
        and float(values["predictor_gradient_norm"]) > 0
        and float(values["geometry_encoder_gradient_norm"]) > 0
        and float(values["spectrum_encoder_gradient_norm"]) > 0
    )
    return values


def triplet_validity(response: torch.Tensor, num_triplets: int, seed: int) -> dict[str, object]:
    """Validate relational triplets on one batch: distinct indices, pos < neg in D_S."""
    triplets = build_response_triplets(response, num_triplets=num_triplets, seed=seed)
    flat = response.reshape(response.shape[0], -1)
    d2 = torch.cdist(flat, flat)
    d_s = d2 / np.sqrt(flat.shape[1])
    anchors = triplets["anchors"].numpy()
    positives = triplets["positives"].numpy()
    negatives = triplets["negatives"].numpy()
    distinct = all(
        a != p and a != n and p != n for a, p, n in zip(anchors, positives, negatives)
    )
    ordered = all(
        float(d_s[a, p]) < float(d_s[a, n]) for a, p, n in zip(anchors, positives, negatives)
    )
    distances = {
        "positive_min": float(d_s[anchors, positives].min().item()),
        "positive_max": float(d_s[anchors, positives].max().item()),
        "negative_min": float(d_s[anchors, negatives].min().item()),
        "negative_max": float(d_s[anchors, negatives].max().item()),
    }
    return {"num_triplets": int(len(anchors)), "all_distinct": bool(distinct), "positive_before_negative": bool(ordered), "distances": distances}


def frequency_grid_check(model: torch.nn.Module, subset_root: Path) -> dict[str, object]:
    freq_ghz = np.load(subset_root / "frequency_ghz.npy")
    tilde_f = (freq_ghz - 2.0) / 10.0
    code = model.spectrum_encoder.frequency_code.detach().cpu().numpy()
    return {
        "grid_points": int(len(freq_ghz)),
        "expected_points": 1001,
        "grid_aligns": bool(len(freq_ghz) == 1001 and len(code) == 1001),
        "tilde_f_min": float(tilde_f.min()),
        "tilde_f_max": float(tilde_f.max()),
        "frequency_code_shape": list(code.shape),
    }


def loss_scale_diagnostic(
    model: torch.nn.Module,
    geometry: torch.Tensor,
    response: torch.Tensor,
    device: torch.device,
    alpha: float,
    lambda_variance: float,
    lambda_covariance: float,
    lambda_relational: float,
    num_triplets: int,
    margin: float,
    seed: int,
) -> dict[str, float]:
    """Typical contributions of all four loss terms at initialization."""
    model.eval()
    model.spectrum_target_encoder.eval()
    outputs = model(geometry.to(device), response.to(device))
    triplets = build_triplets(response, num_triplets, seed)
    _, parts = v3_loss_with_parts(
        outputs["z_pred"],
        outputs["z_target"],
        outputs["z_self"],
        alpha,
        lambda_variance,
        lambda_covariance,
        z_online=outputs["z_online"],
        z_geometry=outputs["z_geometry"],
        z_relational=outputs["z_online"],
        lambda_relational=lambda_relational,
        triplets=triplets,
        margin=margin,
    )
    contributions = {name: float(value.item()) for name, value in parts.items()}
    scaled = {
        "cross_scaled": contributions["cross_loss"],
        "bootstrap_scaled": alpha * contributions["bootstrap_loss"],
        "variance_scaled": lambda_variance * contributions["variance_loss"],
        "covariance_scaled": lambda_covariance * contributions["covariance_loss"],
        "relational_scaled": lambda_relational * contributions["relational_loss"],
    }
    total = sum(scaled.values())
    return {
        "unscaled": contributions,
        "scaled": scaled,
        "total": float(total),
        "fractions": {name: float(value / total) for name, value in scaled.items()},
    }


def run_smoke(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    set_seed(args.seed)

    loaders = build_paired_dataloaders(args.subset_root, args.batch_size, args.seed, max_samples=args.max_samples, shuffled_pairs=False)
    model = PhysicsJEPAFrequencyRelational(
        args.latent_dim, args.ema_decay, args.channels, args.token_dim, args.num_heads, args.num_harmonics, args.kernel_size, tuple(args.dilations), args.token_stride, args.alpha, args.target_centering, args.target_center_decay
    ).to(device)

    geometry, response, _, _ = next(iter(loaders["train"]))
    batch_geometry = geometry[: args.num_triplets].to(device)
    batch_response = response[: args.num_triplets].to(device)

    frequency = frequency_grid_check(model, args.subset_root)
    triplets = triplet_validity(response[: args.num_triplets], args.num_triplets, args.seed)
    scale = loss_scale_diagnostic(
        model, batch_geometry, batch_response, device, args.alpha, args.lambda_variance, args.lambda_covariance, args.lambda_relational, args.num_triplets, args.margin, args.seed
    )

    def train_run(shuffled_pairs: bool, run_tag: str) -> dict[str, float | bool]:
        set_seed(args.seed)
        model = PhysicsJEPAFrequencyRelational(
            args.latent_dim, args.ema_decay, args.channels, args.token_dim, args.num_heads, args.num_harmonics, args.kernel_size, tuple(args.dilations), args.token_stride, args.alpha, args.target_centering, args.target_center_decay
        ).to(device)
        run_loaders = build_paired_dataloaders(args.subset_root, args.batch_size, args.seed, max_samples=args.max_samples, shuffled_pairs=shuffled_pairs, shuffle_seed=args.shuffle_seed)
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        target_before_init = next(model.spectrum_target_encoder.parameters()).detach().clone()
        first_epoch_total = None
        last_epoch_total = None
        for epoch in range(1, args.smoke_epochs + 1):
            metrics = train_epoch_v3(
                model, run_loaders["train"], optimizer, device, args.alpha, args.lambda_variance, args.lambda_covariance, args.lambda_relational, args.num_triplets, args.margin, args.seed
            )
            if first_epoch_total is None:
                first_epoch_total = metrics["total_loss"]
            last_epoch_total = metrics["total_loss"]
        target_changed = not torch.equal(target_before_init.cpu(), next(model.spectrum_target_encoder.parameters()).detach().cpu())
        grad_geometry, grad_response, _, _ = next(iter(run_loaders["train"]))
        diagnostics = gradient_diagnostics_v3(
            model, grad_geometry[:8], grad_response[:8], device, args.alpha, args.lambda_variance, args.lambda_covariance, args.lambda_relational, args.num_triplets, args.margin, args.seed
        )
        return {
            "first_epoch_total_loss": float(first_epoch_total),
            "last_epoch_total_loss": float(last_epoch_total),
            "loss_decreased": float(last_epoch_total) < float(first_epoch_total),
            "target_parameters_updated": bool(target_changed),
            **diagnostics,
            "relational_loss_finite": bool(np.isfinite(float(diagnostics["relational_loss"]))),
        }

    correct = train_run(False, "correct")
    set_seed(args.seed)
    shuffled = train_run(True, "shuffled")

    report = {
        "mode": "overfit",
        "experiment_label": args.experiment_label,
        "frequency_grid": frequency,
        "triplet_validity": triplets,
        "loss_scale_diagnostic": scale,
        "correct": correct,
        "shuffled": shuffled,
        "shuffled_pairs_degrade_alignment": bool(shuffled["last_epoch_total_loss"] > correct["last_epoch_total_loss"]),
        "checks": {
            "jepa_loss_decreases": bool(correct["loss_decreased"]),
            "gradients_finite_nonzero": bool(correct["gradients_finite_nonzero"]),
            "target_parameters_update": bool(correct["target_parameters_updated"]),
            "target_branch_has_no_gradient": bool(not correct["target_branch_has_gradients"]),
            "relational_triplets_valid": bool(triplets["all_distinct"] and triplets["positive_before_negative"]),
            "relational_loss_finite": bool(correct["relational_loss_finite"]),
            "frequency_positions_align": bool(frequency["grid_aligns"]),
        },
        "milestone_passed": bool(
            correct["loss_decreased"]
            and correct["gradients_finite_nonzero"]
            and correct["target_parameters_updated"]
            and not correct["target_branch_has_gradients"]
            and triplets["all_distinct"]
            and triplets["positive_before_negative"]
            and correct["relational_loss_finite"]
            and frequency["grid_aligns"]
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
    loaders = build_paired_dataloaders(
        args.subset_root, args.batch_size, args.seed, args.max_samples, shuffled_pairs=args.shuffled_pairs, shuffle_seed=args.shuffle_seed
    )
    model = PhysicsJEPAFrequencyRelational(
        args.latent_dim, args.ema_decay, args.channels, args.token_dim, args.num_heads, args.num_harmonics, args.kernel_size, tuple(args.dilations), args.token_stride, args.alpha, args.target_centering, args.target_center_decay
    ).to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    checkpoint_path = args.output_dir / "best.pt"
    best_validation = float("inf")
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch_v3(
            model, loaders["train"], optimizer, device, args.alpha, args.lambda_variance, args.lambda_covariance, args.lambda_relational, args.num_triplets, args.margin, args.seed
        )
        validation = evaluate_epoch_v3(
            model, loaders["val"], device, args.alpha, args.lambda_variance, args.lambda_covariance, args.lambda_relational, args.num_triplets, args.margin, args.seed
        )
        record = {"epoch": epoch, **{f"train_{name}": value for name, value in train_metrics.items()}, **{f"val_{name}": value for name, value in validation.items()}}
        history.append(record)
        print(
            f"epoch {epoch:03d} | train_total={record['train_total_loss']:.6f} | val_total={record['val_total_loss']:.6f} | "
            f"val_cross={record['val_cross_loss']:.6f} | val_rel={record['val_relational_loss']:.6f}"
        )
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
        "phase": "physics_jepa_v3",
        "experiment_label": args.experiment_label,
        "mode": "full" if not args.shuffled_pairs else "shuffled",
        "tag": args.tag,
        "subset_root": str(args.subset_root),
        "split": {name: len(loader.dataset) for name, loader in loaders.items()},
        "model": "PhysicsJEPAFrequencyRelational",
        "objective": "cross-modal JEPA (frequency-aware target) + relational EM-similarity ordering",
        "loss": {
            "cross": "normalized latent MSE",
            "bootstrap": "normalized latent MSE with momentum target",
            "variance": "VICReg std regularizer",
            "covariance": "Barlow redundancy reduction",
            "relational": "margin ranking max(0, Dz(a,p) - Dz(a,n) + margin)",
            "alpha": args.alpha,
            "lambda_variance": args.lambda_variance,
            "lambda_covariance": args.lambda_covariance,
            "lambda_relational": args.lambda_relational,
            "margin": args.margin,
            "num_triplets": args.num_triplets,
            "triplet_seed": args.seed,
            "triplet_neg_ratio": 2.0,
        },
        "target_centering": args.target_centering,
        "target_center_decay": args.target_center_decay,
        "latent_dim": args.latent_dim,
        "channels": args.channels,
        "token_dim": args.token_dim,
        "num_heads": args.num_heads,
        "num_harmonics": args.num_harmonics,
        "kernel_size": args.kernel_size,
        "dilations": list(args.dilations),
        "token_stride": args.token_stride,
        "num_tokens": model.spectrum_encoder.num_tokens,
        "ema_decay": args.ema_decay,
        "geometry_encoder": "SpatialGeometryEncoder(1,64) + avgpool + MLP -> d (v2, unchanged)",
        "spectrum_encoder": (
            "FrequencySpectrumEncoder: stem Conv1d 4->48 k9 + sinusoidal freq PE (2-12 GHz, "
            "tilde=(f-2)/10, 8 harmonics) + multiscale dilated Conv1d (1,4,16) @1001pt + "
            "mix 1x1 + strided tokenizer k17/s16 -> 63 tokens + per-token pos + 4-head "
            "self-attention + attention pool -> d"
        ),
        "predictor": f"MLP {args.latent_dim}->{2 * args.latent_dim}->{args.latent_dim} (v2, unchanged)",
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "patience": args.patience,
        "seed": args.seed,
        "shuffle_seed": args.shuffle_seed,
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_total_loss": float(checkpoint["best_metric"]),
        "model_parameter_count_total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.trainable_parameters()),
        "device": str(device),
        "training_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "git_commit": git_commit(),
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
    parser.add_argument("--tag", choices=("correct", "shuffled"), default="correct")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-harmonics", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--token-stride", type=int, default=16)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lambda-variance", type=float, default=0.5)
    parser.add_argument("--lambda-covariance", type=float, default=0.05)
    parser.add_argument("--lambda-relational", type=float, default=LAMBDA_RELATIONAL)
    parser.add_argument("--margin", type=float, default=MARGIN)
    parser.add_argument("--num-triplets", type=int, default=32)
    parser.add_argument("--target-centering", action="store_true")
    parser.add_argument("--target-center-decay", type=float, default=0.99)
    parser.add_argument("--experiment-label", default="physics_jepa_v3_frequency_relational")
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
    parser.add_argument("--smoke-epochs", type=int, default=20)
    parser.add_argument("--cache-splits", nargs="*", default=["val", "test"])
    args = parser.parse_args()
    if args.latent_dim not in (32, 64):
        raise ValueError("latent_dim must be 32 or 64")
    if args.mode == "overfit":
        if args.max_samples is None:
            args.max_samples = 1024
        run_smoke(args)
    else:
        if args.shuffled_pairs:
            args.tag = "shuffled"
        run_full(args)


if __name__ == "__main__":
    main()
