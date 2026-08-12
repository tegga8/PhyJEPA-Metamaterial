"""Train and evaluate a controlled stochastic latent inverse generator."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_phase9_inverse_baselines import InverseDataset, load_geometry_autoencoder, load_forward, resolve_device, set_seed
from src.conditional_latent_vae import ConditionalLatentVAE


def kl_normal(q_mean: torch.Tensor, q_logvar: torch.Tensor, p_mean: torch.Tensor, p_logvar: torch.Tensor) -> torch.Tensor:
    value = p_logvar - q_logvar + (torch.exp(q_logvar) + torch.square(q_mean - p_mean)) / torch.exp(p_logvar) - 1.0
    return 0.5 * value.mean()


def topology(geometry: np.ndarray) -> dict[str, float]:
    occupied = np.asarray(geometry, dtype=bool).reshape(16, 16)
    seen = np.zeros_like(occupied, dtype=bool)
    components = 0
    for position in zip(*np.nonzero(occupied)):
        if seen[position]:
            continue
        components += 1
        stack = [position]
        seen[position] = True
        while stack:
            row, col = stack.pop()
            for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if 0 <= next_row < 16 and 0 <= next_col < 16 and occupied[next_row, next_col] and not seen[next_row, next_col]:
                    seen[next_row, next_col] = True
                    stack.append((next_row, next_col))
    boundaries = int(np.count_nonzero(occupied[:, 1:] != occupied[:, :-1]) + np.count_nonzero(occupied[1:, :] != occupied[:-1, :]))
    return {"occupancy": float(occupied.mean()), "connected_components_4": float(components), "boundary_transitions_4": float(boundaries)}


def geometry_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    p_count, t_count = prediction.sum(), target.sum()
    return {
        "iou": float(intersection / union) if union else 1.0,
        "dice": float(2 * intersection / (p_count + t_count)) if p_count + t_count else 1.0,
        "pixel_hamming": float(np.mean(prediction != target)),
        "occupancy_abs_difference": float(abs(float(prediction.mean()) - float(target.mean()))),
    }


def pairwise_upper(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.empty(0, dtype=np.float32)
    rows, cols = np.triu_indices(values.shape[0], k=1)
    return values[rows, cols]


def validity_limits(train_dataset: InverseDataset) -> dict[str, float]:
    values = [topology(train_dataset[index]["geometry"].numpy()[0]) for index in range(len(train_dataset))]
    return {
        "occupancy_min": float(min(value["occupancy"] for value in values)),
        "occupancy_max": float(max(value["occupancy"] for value in values)),
        "connected_components_max": float(max(value["connected_components_4"] for value in values)),
        "boundary_transitions_max": float(max(value["boundary_transitions_4"] for value in values)),
    }


def complexity_group(geometry: np.ndarray) -> tuple[float, str]:
    topo = topology(geometry)
    score = topo["connected_components_4"] + topo["boundary_transitions_4"] / 32.0
    group = "simple" if score <= 3.0625 else "medium" if score <= 13.291666666666664 else "complex"
    return float(score), group


def nearest_latent_mse(query_latents: np.ndarray, train_latents: np.ndarray, block_size: int = 128) -> np.ndarray:
    train_flat = train_latents.reshape(len(train_latents), -1).astype(np.float32, copy=False)
    train_norms = np.square(train_flat).mean(axis=1)
    values = np.empty(query_latents.shape[0], dtype=np.float32)
    for start in range(0, query_latents.shape[0], block_size):
        block = query_latents[start:start + block_size].reshape(query_latents[start:start + block_size].shape[0], -1).astype(np.float32, copy=False)
        distances = np.square(block).mean(axis=1, keepdims=True) + train_norms[None, :] - 2.0 * (block @ train_flat.T) / train_flat.shape[1]
        values[start:start + block.shape[0]] = distances.min(axis=1)
    return values


def is_valid(geometry: np.ndarray, limits: dict[str, float], finite: bool) -> tuple[bool, dict[str, float]]:
    info = topology(geometry)
    valid = bool(
        finite
        and 0.0 < info["occupancy"] < 1.0
        and limits["occupancy_min"] <= info["occupancy"] <= limits["occupancy_max"]
        and info["connected_components_4"] <= limits["connected_components_max"]
        and info["boundary_transitions_4"] <= limits["boundary_transitions_max"]
    )
    return valid, info


def summarize_candidate_rows(rows: list[dict[str, Any]], response_threshold: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in ("simple", "medium", "complex"):
        selected = [row for row in rows if row["complexity_group"] == group]
        best_by_target: dict[int, float] = {}
        successful_targets: dict[int, int] = {}
        for row in selected:
            target_index = int(row["target_index"])
            best_by_target[target_index] = min(best_by_target.get(target_index, float("inf")), float(row["response_mse"]))
            if bool(row["valid"]) and float(row["response_mse"]) <= response_threshold:
                successful_targets[target_index] = successful_targets.get(target_index, 0) + 1
        result[group] = {
            "targets": len(best_by_target),
            "candidates": len(selected),
            "validity_rate": float(np.mean([bool(row["valid"]) for row in selected])) if selected else None,
            "response_mse_best_candidate_mean": float(np.mean(list(best_by_target.values()))) if best_by_target else None,
            "multi_solution_success_fraction": float(np.mean([successful_targets.get(target_index, 0) >= 2 for target_index in best_by_target])) if best_by_target else None,
            "nearest_train_pixel_hamming_mean": float(np.mean([float(row["nearest_train_pixel_hamming"]) for row in selected])) if selected else None,
            "nearest_train_latent_mse_mean": float(np.mean([float(row["nearest_train_latent_mse"]) for row in selected])) if selected else None,
            "candidate_target_pixel_hamming_mean": float(np.mean([float(row["pixel_hamming"]) for row in selected])) if selected else None,
        }
    return result


def train(model: ConditionalLatentVAE, loaders: dict[str, DataLoader], decoder: torch.nn.Module, latent_mean: torch.Tensor, latent_std: torch.Tensor, device: torch.device, beta_kl: float, prior_weight: float, epochs: int, patience: int, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    checkpoint_path = output_dir / "best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in loaders["train"]:
            response = batch["response"].to(device)
            geometry = batch["geometry"].to(device)
            target = ((batch["latent"].to(device) - latent_mean) / latent_std).flatten(1)
            output = model(response, target)
            latent_loss = F.mse_loss(output["posterior_mean"], target)
            prior_loss = F.mse_loss(output["prior_mean"], target)
            sampled_raw = (output["posterior_sample"].view(-1, 64, 8, 8) * latent_std.view(1, 64, 8, 8) + latent_mean.view(1, 64, 8, 8))
            geometry_loss = F.binary_cross_entropy_with_logits(decoder(sampled_raw), geometry)
            kl = kl_normal(output["posterior_mean"], output["posterior_logvar"], output["prior_mean"], output["prior_logvar"])
            loss = latent_loss + prior_weight * prior_loss + geometry_loss + beta_kl * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item()) * response.shape[0]
            count += response.shape[0]
        validation = evaluate_validation(model, loaders["val"], decoder, latent_mean, latent_std, device)
        record = {"epoch": epoch, "train_loss": total / count, **validation}
        history.append(record)
        print(f"epoch {epoch:03d} | train={record['train_loss']:.6f} | val={record['val_loss']:.6f} | val_iou={record['val_iou']:.6f} | prior_std={record['prior_std']:.6f}")
        if record["val_loss"] < best:
            best = record["val_loss"]
            best_epoch = epoch
            stale = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_metric": best}, checkpoint_path)
        else:
            stale += 1
            if stale >= patience:
                break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return {"checkpoint": str(checkpoint_path), "best_epoch": best_epoch, "best_validation_loss": best, "history": history}


@torch.inference_mode()
def evaluate_validation(model: ConditionalLatentVAE, loader: DataLoader, decoder: torch.nn.Module, latent_mean: torch.Tensor, latent_std: torch.Tensor, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"val_loss": 0.0, "val_latent_mse": 0.0, "val_geometry_bce": 0.0, "val_iou": 0.0, "prior_std": 0.0}
    count = 0
    for batch in loader:
        response = batch["response"].to(device)
        geometry = batch["geometry"].to(device)
        target = ((batch["latent"].to(device) - latent_mean) / latent_std).flatten(1)
        output = model(response)
        raw_mean = output["prior_mean"].view(-1, 64, 8, 8) * latent_std.view(1, 64, 8, 8) + latent_mean.view(1, 64, 8, 8)
        logits = decoder(raw_mean)
        latent_loss = torch.square(output["prior_mean"] - target).flatten(1).mean(dim=1)
        geometry_loss = F.binary_cross_entropy_with_logits(logits, geometry, reduction="none").flatten(1).mean(dim=1)
        binary = torch.sigmoid(logits) >= 0.5
        target_binary = geometry >= 0.5
        intersection = (binary & target_binary).flatten(1).sum(dim=1).float()
        union = (binary | target_binary).flatten(1).sum(dim=1).float()
        iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
        loss = latent_loss + geometry_loss
        for name, values in (("val_loss", loss), ("val_latent_mse", latent_loss), ("val_geometry_bce", geometry_loss), ("val_iou", iou), ("prior_std", torch.exp(0.5 * output["prior_logvar"]).mean(dim=1))):
            totals[name] += float(values.sum().item())
        count += response.shape[0]
    return {name: value / count for name, value in totals.items()}


@torch.inference_mode()
def generate_candidates(model: ConditionalLatentVAE, dataset: InverseDataset, autoencoder: torch.nn.Module, forward: torch.nn.Module, forward_mean: torch.Tensor, forward_std: torch.Tensor, latent_mean: torch.Tensor, latent_std: torch.Tensor, train_latents: np.ndarray, train_geometries: np.ndarray, limits: dict[str, float], device: torch.device, candidates_per_target: int, seed: int, response_threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    model.eval()
    autoencoder.eval()
    decoder = autoencoder.decoder
    generator = torch.Generator(device=device).manual_seed(seed)
    all_binary = np.zeros((len(dataset), candidates_per_target, 1, 16, 16), dtype=np.uint8)
    all_encoded_latents = np.zeros((len(dataset), candidates_per_target, 64, 8, 8), dtype=np.float32)
    all_noise = []
    rows: list[dict[str, Any]] = []
    target_response_raw = np.stack([dataset[index]["response_raw"].numpy() for index in range(len(dataset))])
    target_response = torch.from_numpy(target_response_raw).to(device)
    target_response_screened = (target_response - forward_mean.view(1, 4, 1)) / forward_std.view(1, 4, 1)
    for start in range(0, len(dataset), 32):
        stop = min(start + 32, len(dataset))
        response = torch.stack([dataset[index]["response"] for index in range(start, stop)]).to(device)
        noise = torch.randn((stop - start, candidates_per_target, model.latent_dim), generator=generator, device=device)
        repeated_response = response[:, None].expand(-1, candidates_per_target, -1, -1).reshape(-1, 4, 1001)
        sampled = model.sample(repeated_response, noise.reshape(-1, model.latent_dim))
        raw_latent = sampled.view(-1, 64, 8, 8) * latent_std.view(1, 64, 8, 8) + latent_mean.view(1, 64, 8, 8)
        logits = decoder(raw_latent)
        probabilities = torch.sigmoid(logits)
        binary = (probabilities >= 0.5).to(torch.uint8).view(stop - start, candidates_per_target, 1, 16, 16)
        all_binary[start:stop] = binary.cpu().numpy()
        all_noise.append(noise.cpu().numpy())
        encoded = autoencoder.encoder(binary.float().flatten(0, 1)).view(stop - start, candidates_per_target, 64, 8, 8)
        all_encoded_latents[start:stop] = encoded.cpu().numpy()
    candidate_responses = []
    for start in range(0, len(dataset), 32):
        block = torch.from_numpy(all_binary[start:start + 32].astype(np.float32)).to(device).flatten(0, 1)
        candidate_responses.append(forward(block).view(block.shape[0] // candidates_per_target, candidates_per_target, 4, 1001).cpu().numpy())
    predicted_responses = np.concatenate(candidate_responses, axis=0)
    nearest_latent_values = nearest_latent_mse(all_encoded_latents.reshape(-1, 64, 8, 8), train_latents).reshape(len(dataset), candidates_per_target)
    valid_count = []
    successes = []
    diversity_values = []
    novelty_pixel_values = []
    novelty_latent_values = []
    duplicate_counts = []
    response_best_values = []
    response_valid_best_values = []
    for target_index in range(len(dataset)):
        target_geometry = dataset[target_index]["geometry"].numpy()[0]
        candidate_geometries = all_binary[target_index, :, 0]
        candidate_valid = []
        candidate_errors = []
        candidate_novelty_pixel = []
        candidate_novelty_latent = []
        candidate_rows = []
        target_complexity_score, target_complexity_group = complexity_group(target_geometry)
        for candidate_index, geometry in enumerate(candidate_geometries):
            valid, topo = is_valid(geometry, limits, True)
            error = float(np.square(predicted_responses[target_index, candidate_index] - target_response_screened[target_index].cpu().numpy()).mean())
            hamming_to_train = np.mean(train_geometries[:, 0] != geometry[None, :, :], axis=(1, 2))
            nearest_pixel = float(hamming_to_train.min())
            nearest_latent = float(nearest_latent_values[target_index, candidate_index])
            candidate_valid.append(valid)
            candidate_errors.append(error)
            candidate_novelty_pixel.append(nearest_pixel)
            candidate_novelty_latent.append(nearest_latent)
            candidate_rows.append({"candidate_index": candidate_index, "valid": valid, "response_mse": error, "nearest_train_pixel_hamming": nearest_pixel, "nearest_train_latent_mse": nearest_latent, **topo, **geometry_metrics(geometry, target_geometry)})
        unique = np.unique(candidate_geometries.reshape(candidates_per_target, -1), axis=0).shape[0]
        pairwise = pairwise_upper(np.mean(candidate_geometries[:, None] != candidate_geometries[None, :], axis=(2, 3)))
        valid_errors = [error for error, valid in zip(candidate_errors, candidate_valid) if valid]
        valid_count.append(sum(candidate_valid))
        successes.append(sum(error <= response_threshold and valid for error, valid in zip(candidate_errors, candidate_valid)) >= 2)
        diversity_values.append(float(pairwise.mean()) if len(pairwise) else 0.0)
        novelty_pixel_values.append(float(np.mean(candidate_novelty_pixel)))
        novelty_latent_values.append(float(np.mean(candidate_novelty_latent)))
        duplicate_counts.append(candidates_per_target - unique)
        response_best_values.append(float(min(candidate_errors)))
        response_valid_best_values.append(float(min(valid_errors)) if valid_errors else None)
        for candidate_index, row in enumerate(candidate_rows):
            row.update({"target_index": target_index, "target_sample_id": dataset[target_index]["sample_id"], "target_complexity_score": target_complexity_score, "complexity_group": target_complexity_group})
            rows.append(row)
    overall = {
        "targets": len(dataset),
        "candidates_per_target": candidates_per_target,
        "total_candidates": int(len(dataset) * candidates_per_target),
        "validity_rate": float(np.mean(valid_count) / candidates_per_target),
        "valid_candidates_per_target_mean": float(np.mean(valid_count)),
        "response_mse_best_candidate_mean": float(np.mean(response_best_values)),
        "response_mse_best_valid_candidate_mean": float(np.mean([value for value in response_valid_best_values if value is not None])) if any(value is not None for value in response_valid_best_values) else None,
        "response_threshold": response_threshold,
        "multi_solution_success_fraction": float(np.mean(successes)),
        "multi_solution_definition": "at least 2 valid candidates with screening response MSE <= threshold",
        "pairwise_geometry_hamming_mean": float(np.mean(diversity_values)),
        "nearest_train_pixel_hamming_mean": float(np.mean(novelty_pixel_values)),
        "nearest_train_latent_mse_mean": float(np.mean(novelty_latent_values)),
        "duplicate_candidates_per_target_mean": float(np.mean(duplicate_counts)),
    }
    return overall, rows, all_binary, predicted_responses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--geometry-autoencoder", type=Path, default=Path("outputs/phase8_geometry_autoencoder/best.pt"))
    parser.add_argument("--latent-root", type=Path, default=Path("outputs/phase8_geometry_autoencoder/latents"))
    parser.add_argument("--forward-checkpoint", type=Path, default=Path("outputs/phase2_5/exp_C_30k_resonance/best.pt"))
    parser.add_argument("--forward-subset-root", type=Path, default=Path("data/processed/sutd_prcm_30k"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase10_stochastic_inverse_design"))
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--beta-kl", type=float, default=1e-4)
    parser.add_argument("--prior-weight", type=float, default=0.5)
    parser.add_argument("--candidates-per-target", type=int, default=8)
    parser.add_argument("--response-threshold", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {split: InverseDataset(args.subset_root, split, args.latent_root) for split in ("train", "val", "test")}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {split: DataLoader(dataset, batch_size=64, shuffle=split == "train", generator=generator if split == "train" else None, num_workers=0) for split, dataset in datasets.items()}
    autoencoder = load_geometry_autoencoder(args.geometry_autoencoder, device)
    decoder = autoencoder.decoder
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)
    forward, forward_mean, forward_std = load_forward(args, device)
    train_latents_raw = np.asarray(datasets["train"].latents, dtype=np.float32)
    latent_mean_np = train_latents_raw.mean(axis=0).astype(np.float32)
    latent_std_np = train_latents_raw.std(axis=0).clip(1e-2).astype(np.float32)
    latent_mean = torch.from_numpy(latent_mean_np).to(device)
    latent_std = torch.from_numpy(latent_std_np).to(device)
    model = ConditionalLatentVAE(args.hidden_dim).to(device)
    checkpoint_path = args.output_dir / "generator" / "best.pt"
    if args.eval_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing generator checkpoint for --eval-only: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        previous_metrics = args.output_dir / "metrics.json"
        if previous_metrics.exists():
            training = json.loads(previous_metrics.read_text(encoding="utf-8"))["training"]
        else:
            training = {"checkpoint": str(checkpoint_path), "best_epoch": checkpoint.get("epoch"), "best_validation_loss": checkpoint.get("best_metric"), "history": []}
    else:
        training = train(model, loaders, decoder, latent_mean, latent_std, device, args.beta_kl, args.prior_weight, args.epochs, args.patience, args.output_dir / "generator")
    train_geometries = np.stack([datasets["train"][index]["geometry"].numpy() for index in range(len(datasets["train"]))])
    limits = validity_limits(datasets["train"])
    metrics, rows, candidates, candidate_responses = generate_candidates(model, datasets["test"], autoencoder, forward, forward_mean, forward_std, latent_mean, latent_std, train_latents_raw, train_geometries, limits, device, args.candidates_per_target, args.seed + 1000, args.response_threshold)
    np.save(args.output_dir / "test_candidates_binary.npy", candidates)
    np.save(args.output_dir / "test_candidate_responses_normalized_30k.npy", candidate_responses.astype(np.float32))
    np.savez(args.output_dir / "latent_standardization.npz", mean=latent_mean_np, std=latent_std_np)
    (args.output_dir / "validity_limits.json").write_text(json.dumps(limits, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "candidate_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"phase": "10", "objective": "one-to-many stochastic spectrum to geometry", "split": {"train": 4000, "val": 500, "test": 500}, "seed": args.seed, "device": str(device), "anti_leakage": {"generator_inference_inputs": ["normalized_target_response", "optional_noise"], "partial_geometry": False, "source_id": False, "original_geometry": False, "mask": False, "geometry_complexity_metadata": False, "target_geometry_latent": False}, "model": {"name": "ConditionalLatentVAE", "hidden_dim": args.hidden_dim, "latent_shape": [64, 8, 8], "beta_kl": args.beta_kl, "prior_weight": args.prior_weight, "candidates_per_target": args.candidates_per_target}, "geometry_autoencoder": str(args.geometry_autoencoder), "forward_screening_surrogate": str(args.forward_checkpoint), "training": training, "validity_limits": limits, "evaluation": metrics, "complexity_summaries": summarize_candidate_rows(rows, args.response_threshold), "scientific_decision": "assess whether stochastic candidates add valid diversity and response satisfaction; independent physical validation remains unavailable"}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"evaluation": metrics, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
