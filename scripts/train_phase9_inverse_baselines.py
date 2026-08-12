"""Train/evaluate Phase 9 deterministic inverse-design baselines."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import SUTDPRCMDataset
from src.geometry_autoencoder import GeometryAutoencoder
from src.models import build_forward_model
from src.spectrum_inverse_models import SpectrumToGeometryDirectMLP, SpectrumToGeometryLatent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


class InverseDataset(Dataset[dict[str, Any]]):
    def __init__(self, root: Path, split: str, latent_root: Path) -> None:
        self.source = SUTDPRCMDataset(root, split, normalize_response=False)
        stats = np.load(root / "train_response_stats.npz")
        self.mean = stats["mean"].astype(np.float32)
        self.std = stats["std"].astype(np.float32)
        self.latents = np.load(latent_root / f"{split}.npy", mmap_mode="r")
        cached_ids = (latent_root / f"{split}_source_ids.txt").read_text(encoding="utf-8").splitlines()
        source_ids = [self.source.source_id(index) for index in range(len(self.source))]
        if cached_ids != source_ids:
            raise AssertionError(f"Geometry latent cache/source-ID mismatch for {split}")
        if self.latents.shape != (len(self.source), 64, 8, 8):
            raise ValueError(f"Expected cached latents [{len(self.source)},64,8,8], got {self.latents.shape}")

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        geometry, response_raw = self.source[index]
        response = (response_raw.numpy() - self.mean) / self.std
        return {
            "response": torch.from_numpy(response.astype(np.float32, copy=False)),
            "response_raw": torch.from_numpy(response_raw.numpy().astype(np.float32, copy=False)),
            "geometry": geometry,
            "latent": torch.from_numpy(np.asarray(self.latents[index], dtype=np.float32).copy()),
            "sample_id": self.source.source_id(index),
        }


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    p_count, t_count = prediction.sum(), target.sum()
    return {
        "iou": float(intersection / union) if union else 1.0,
        "dice": float(2 * intersection / (p_count + t_count)) if p_count + t_count else 1.0,
        "pixel_accuracy": float(np.mean(prediction == target)),
        "occupancy_abs_difference": float(abs(float(prediction.mean()) - float(target.mean()))),
    }


def response_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.square(prediction - target).mean(dim=(1, 2))


def load_geometry_autoencoder(checkpoint_path: Path, device: torch.device) -> GeometryAutoencoder:
    model = GeometryAutoencoder().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_forward(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    checkpoint = torch.load(args.forward_checkpoint, map_location=device, weights_only=False)
    model_name = checkpoint.get("args", {}).get("model", "ForwardSurrogateCNN")
    model = build_forward_model(model_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stats = np.load(args.forward_subset_root / "train_response_stats.npz")
    return model, torch.from_numpy(stats["mean"].astype(np.float32)).to(device), torch.from_numpy(stats["std"].astype(np.float32)).to(device)


def train_model(model: torch.nn.Module, loaders: dict[str, DataLoader], decoder: torch.nn.Module, device: torch.device, latent_mode: bool, lambda_geometry: float, epochs: int, patience: int, learning_rate: float, weight_decay: float, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
            target_latent = batch["latent"].to(device)
            optimizer.zero_grad(set_to_none=True)
            if latent_mode:
                prediction = model(response)
                latent_loss = F.mse_loss(prediction, target_latent)
                geometry_logits = decoder(prediction)
                geometry_loss = F.binary_cross_entropy_with_logits(geometry_logits, geometry)
                loss = latent_loss + lambda_geometry * geometry_loss
            else:
                geometry_logits = model(response)
                geometry_loss = F.binary_cross_entropy_with_logits(geometry_logits, geometry)
                latent_loss = torch.zeros((), device=device)
                loss = geometry_loss
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * response.shape[0]
            count += response.shape[0]
        validation = evaluate_learned(model, loaders["val"], decoder, device, latent_mode, threshold=0.5, forward=None, forward_mean=None, forward_std=None)
        record = {"epoch": epoch, "train_loss": total / count, "val_loss": validation["loss"], "val_latent_mse": validation["latent_mse"], "val_geometry_bce": validation["geometry_bce"], "val_iou": validation["iou"]}
        history.append(record)
        print(f"epoch {epoch:03d} | train={record['train_loss']:.6f} | val={record['val_loss']:.6f} | val_iou={record['val_iou']:.6f}")
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
def evaluate_learned(model: torch.nn.Module, loader: DataLoader, decoder: GeometryAutoencoder, device: torch.device, latent_mode: bool, threshold: float, forward: torch.nn.Module | None, forward_mean: torch.Tensor | None, forward_std: torch.Tensor | None) -> dict[str, Any]:
    model.eval()
    totals = {"loss": 0.0, "latent_mse": 0.0, "geometry_bce": 0.0, "iou": 0.0, "dice": 0.0, "pixel_accuracy": 0.0, "occupancy_abs_difference": 0.0, "response_mse": 0.0}
    rows: list[dict[str, Any]] = []
    count = 0
    for batch in loader:
        response = batch["response"].to(device)
        geometry = batch["geometry"].to(device)
        if latent_mode:
            latent = model(response)
            logits = decoder(latent)
            latent_mse = torch.square(latent - batch["latent"].to(device)).flatten(1).mean(dim=1)
        else:
            logits = model(response)
            latent_mse = torch.zeros(response.shape[0], device=device)
        probabilities = torch.sigmoid(logits)
        geometry_bce = F.binary_cross_entropy_with_logits(logits, geometry, reduction="none").flatten(1).mean(dim=1)
        binary = probabilities >= threshold
        target_binary = geometry >= 0.5
        intersection = (binary & target_binary).flatten(1).sum(dim=1).float()
        union = (binary | target_binary).flatten(1).sum(dim=1).float()
        iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
        dice = torch.where(binary.flatten(1).sum(dim=1) + target_binary.flatten(1).sum(dim=1) > 0, 2 * intersection / (binary.flatten(1).sum(dim=1) + target_binary.flatten(1).sum(dim=1)).float(), torch.ones_like(iou))
        pixel_accuracy = (binary == target_binary).flatten(1).float().mean(dim=1)
        occupancy = torch.abs(binary.flatten(1).float().mean(dim=1) - target_binary.flatten(1).float().mean(dim=1))
        loss = latent_mse + geometry_bce if latent_mode else geometry_bce
        if forward is not None:
            predicted_response = forward(binary.float())
            target_screened = (batch["response_raw"].to(device) - forward_mean) / forward_std
            response_error = response_mse(predicted_response, target_screened)
        else:
            response_error = torch.zeros(response.shape[0], device=device)
        for index in range(response.shape[0]):
            rows.append({"sample_id": batch["sample_id"][index], "latent_mse": float(latent_mse[index].item()), "geometry_bce": float(geometry_bce[index].item()), "iou": float(iou[index].item()), "dice": float(dice[index].item()), "pixel_accuracy": float(pixel_accuracy[index].item()), "occupancy_abs_difference": float(occupancy[index].item()), "response_mse": float(response_error[index].item()), "binary_finite": bool(torch.isfinite(binary[index]).all().item()), "logits_finite": bool(torch.isfinite(logits[index]).all().item())})
        for name, values in (("loss", loss), ("latent_mse", latent_mse), ("geometry_bce", geometry_bce), ("iou", iou), ("dice", dice), ("pixel_accuracy", pixel_accuracy), ("occupancy_abs_difference", occupancy), ("response_mse", response_error)):
            totals[name] += float(values.sum().item())
        count += response.shape[0]
    return {name: value / count for name, value in totals.items()} | {"samples": count, "rows": rows}


def nearest_neighbor_predictions(train: InverseDataset, test: InverseDataset, batch_size: int = 64) -> tuple[np.ndarray, list[str]]:
    """Return train-only nearest-neighbor geometries for each test spectrum."""
    train_responses = np.stack([train[index]["response"].numpy() for index in range(len(train))]).reshape(len(train), -1)
    test_responses = np.stack([test[index]["response"].numpy() for index in range(len(test))]).reshape(len(test), -1)
    if train_responses.shape[0] != 4000:
        raise AssertionError("Nearest-neighbor index must contain exactly the 4000 training samples")
    train_norms = np.square(train_responses).mean(axis=1)
    selected_ids: list[str] = []
    geometries: list[np.ndarray] = []
    for start in range(0, len(test), batch_size):
        block = test_responses[start:start + batch_size]
        # ||x-y||² = ||x||² + ||y||² - 2 x·y.  This avoids materializing
        # a [batch, 4000, 4004] difference tensor.
        block_distances = np.square(block).mean(axis=1, keepdims=True) + train_norms[None, :] - 2.0 * (block @ train_responses.T) / train_responses.shape[1]
        indexes = np.argmin(block_distances, axis=1)
        for index in indexes:
            item = train[int(index)]
            geometries.append(item["geometry"].numpy())
            selected_ids.append(item["sample_id"])
    return np.stack(geometries), selected_ids


def summarize_geometry_rows(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name in ("latent_mse", "geometry_bce", "iou", "dice", "pixel_accuracy", "occupancy_abs_difference", "response_mse"):
        values = [row[name] for row in rows if row.get(name) is not None and np.isfinite(row[name])]
        result[name] = float(np.mean(values)) if values else None
    return result


def add_complexity_groups(rows: list[dict[str, Any]], dataset: InverseDataset) -> None:
    for index, row in enumerate(rows):
        geometry = dataset[index]["geometry"].numpy()[0]
        occupied = geometry.astype(bool)
        seen = np.zeros_like(occupied, dtype=bool)
        components = 0
        for position in zip(*np.nonzero(occupied)):
            if seen[position]:
                continue
            components += 1
            stack = [position]
            seen[position] = True
            while stack:
                current_row, current_col = stack.pop()
                for next_row, next_col in ((current_row - 1, current_col), (current_row + 1, current_col), (current_row, current_col - 1), (current_row, current_col + 1)):
                    if 0 <= next_row < 16 and 0 <= next_col < 16 and occupied[next_row, next_col] and not seen[next_row, next_col]:
                        seen[next_row, next_col] = True
                        stack.append((next_row, next_col))
        boundaries = int(np.count_nonzero(occupied[:, 1:] != occupied[:, :-1]) + np.count_nonzero(occupied[1:, :] != occupied[:-1, :]))
        score = components + boundaries / 32.0
        row["complexity_score"] = float(score)
        row["complexity_group"] = "simple" if score <= 3.0625 else "medium" if score <= 13.291666666666664 else "complex"


def complexity_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in ("simple", "medium", "complex"):
        selected = [row for row in rows if row.get("complexity_group") == group]
        result[group] = {"samples": len(selected), **summarize_geometry_rows(selected)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--geometry-autoencoder", type=Path, default=Path("outputs/phase8_geometry_autoencoder/best.pt"))
    parser.add_argument("--latent-root", type=Path, default=Path("outputs/phase8_geometry_autoencoder/latents"))
    parser.add_argument("--forward-checkpoint", type=Path, default=Path("outputs/phase2_5/exp_C_30k_resonance/best.pt"))
    parser.add_argument("--forward-subset-root", type=Path, default=Path("data/processed/sutd_prcm_30k"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase9_inverse_baselines"))
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-geometry", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {split: InverseDataset(args.subset_root, split, args.latent_root) for split in ("train", "val", "test")}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train", generator=generator if split == "train" else None, num_workers=0) for split, dataset in datasets.items()}
    decoder = load_geometry_autoencoder(args.geometry_autoencoder, device).decoder
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    forward, forward_mean, forward_std = load_forward(args, device)
    # The model signatures accept exactly one tensor: target response.
    latent_model = SpectrumToGeometryLatent(args.hidden_dim).to(device)
    direct_model = SpectrumToGeometryDirectMLP(args.hidden_dim).to(device)
    latent_result = train_model(latent_model, loaders, decoder, device, True, args.lambda_geometry, args.epochs, args.patience, args.learning_rate, args.weight_decay, args.output_dir / "latent_predictor")
    # Replace the temporary decoder reference used by train_model/evaluation
    # with the validated frozen decoder by training directly through a wrapper.
    # (The wrapper below is intentionally the same decoder instance for both.)
    # Re-run latent training is not permitted; the temporary object is avoided
    # by loading the checkpoint and evaluating with the validated decoder.
    direct_result = train_model(direct_model, loaders, decoder, device, False, args.lambda_geometry, args.epochs, args.patience, args.learning_rate, args.weight_decay, args.output_dir / "direct_mlp")
    # The train loss uses only latent labels/geometry. Evaluation with the
    # validated decoder is the authoritative Stage 9 metric path.
    latent_model.load_state_dict(torch.load(latent_result["checkpoint"], map_location=device, weights_only=False)["model_state_dict"])
    direct_model.load_state_dict(torch.load(direct_result["checkpoint"], map_location=device, weights_only=False)["model_state_dict"])
    latent_eval = evaluate_learned(latent_model, loaders["test"], decoder, device, True, args.threshold, forward, forward_mean, forward_std)
    direct_eval = evaluate_learned(direct_model, loaders["test"], decoder, device, False, args.threshold, forward, forward_mean, forward_std)
    neighbor_geometries, neighbor_ids = nearest_neighbor_predictions(datasets["train"], datasets["test"])
    train_index_by_id = {datasets["train"][index]["sample_id"]: index for index in range(len(datasets["train"]))}
    target_geometries = np.stack([datasets["test"][index]["geometry"].numpy() for index in range(len(datasets["test"]))])
    test_response_raw = torch.stack([datasets["test"][index]["response_raw"] for index in range(len(datasets["test"]))]).to(device)
    with torch.inference_mode():
        neighbor_tensor = torch.from_numpy(neighbor_geometries).to(device)
        neighbor_screened = forward(neighbor_tensor)
        neighbor_target_screened = (test_response_raw - forward_mean) / forward_std
        neighbor_response_errors = response_mse(neighbor_screened, neighbor_target_screened).cpu().numpy()
    neighbor_rows = []
    for index, (prediction, target) in enumerate(zip(neighbor_geometries, target_geometries)):
        row = binary_metrics(prediction[0], target[0])
        row["geometry_bce"] = None
        nearest_latent = datasets["train"][train_index_by_id[neighbor_ids[index]]]["latent"].numpy()
        target_latent = datasets["test"][index]["latent"].numpy()
        row.update({"sample_id": datasets["test"][index]["sample_id"], "nearest_train_sample_id": neighbor_ids[index], "latent_mse": float(np.mean(np.square(nearest_latent - target_latent))), "response_mse": float(neighbor_response_errors[index])})
        neighbor_rows.append(row)
    # Save rows and summary without exposing any geometry to the model calls.
    summaries = {"nearest_neighbor": summarize_geometry_rows(neighbor_rows), "direct_mlp": {key: direct_eval[key] for key in ("latent_mse", "geometry_bce", "iou", "dice", "pixel_accuracy", "occupancy_abs_difference", "response_mse")}, "latent_predictor": {key: latent_eval[key] for key in ("latent_mse", "geometry_bce", "iou", "dice", "pixel_accuracy", "occupancy_abs_difference", "response_mse")}}
    for name, rows in (("nearest_neighbor", neighbor_rows), ("direct_mlp", direct_eval["rows"]), ("latent_predictor", latent_eval["rows"])):
        add_complexity_groups(rows, datasets["test"])
        with (args.output_dir / f"{name}_per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {"phase": "9", "objective": "deterministic spectrum to geometry", "anti_leakage": {"inverse_model_inputs": ["normalized_target_response"], "partial_geometry": False, "source_id": False, "original_geometry": False, "mask": False, "complexity_metadata": False, "target_geometry_latent_as_input": False}, "split": {"train": 4000, "val": 500, "test": 500}, "seed": args.seed, "device": str(device), "geometry_autoencoder": str(args.geometry_autoencoder), "forward_screening_surrogate": str(args.forward_checkpoint), "training": {"hidden_dim": args.hidden_dim, "epochs": args.epochs, "patience": args.patience, "lambda_geometry": args.lambda_geometry}, "training_runs": {"latent_predictor": latent_result, "direct_mlp": direct_result}, "test_summaries": summaries, "complexity_summaries": {"nearest_neighbor": complexity_summaries(neighbor_rows), "direct_mlp": complexity_summaries(direct_eval["rows"]), "latent_predictor": complexity_summaries(latent_eval["rows"])}, "nearest_neighbor_train_only": True, "decision": "deterministic baseline evidence collected; assess multimodality before stochastic generation"}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps({"args": vars(args), "inverse_models_receive_only_response": True}, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"test_summaries": summaries, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
