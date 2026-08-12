"""Shared training helpers for the Physics-JEPA experiment.

Keeps the thin experiment scripts focused on their own configuration while
centralizing the deterministic seeding, paired data loading, training loop,
and latent caching that every Physics-JEPA stage needs.
"""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.dataset import SUTDPRCMDataset
from src.physics_jepa_losses import physics_jepa_loss, physics_latent_variance_metrics


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


class PairedDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str, int]]):
    """Geometry/response pairs with source ids, optional subsetting and shuffling.

    In shuffled-pair mode ``response_permutation`` re-pairs each geometry with
    the response of a different sample, implementing the shuffled-pair control.
    """

    def __init__(
        self,
        base: SUTDPRCMDataset,
        max_samples: int | None = None,
        seed: int = 42,
        response_permutation: np.ndarray | None = None,
    ) -> None:
        self.base = base
        self.indices = np.arange(len(base), dtype=np.int64)
        if max_samples is not None and max_samples < len(self.indices):
            rng = np.random.default_rng(seed)
            self.indices = np.sort(rng.choice(self.indices, size=int(max_samples), replace=False))
        if response_permutation is not None:
            if len(response_permutation) != len(self.indices):
                raise ValueError("response_permutation must match the number of sampled indices")
        self.response_permutation = response_permutation

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, int]:
        position = int(self.indices[index])
        geometry, response = self.base[position]
        if self.response_permutation is not None:
            response = self.base[int(self.indices[int(self.response_permutation[index])])][1]
        return geometry, response, self.base.source_id(position), int(self.indices[index])


def shuffle_response_permutation(count: int, seed: int) -> np.ndarray:
    """Deterministic derangement-like permutation of response pairing."""
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(count)
    if count > 1 and (permutation == np.arange(count)).all():
        permutation[0], permutation[1] = permutation[1], permutation[0]
    return permutation


def build_paired_dataloaders(
    subset_root: str | Path,
    batch_size: int = 64,
    seed: int = 42,
    max_samples: int | None = None,
    shuffled_pairs: bool = False,
    shuffle_seed: int = 123,
) -> dict[str, DataLoader]:
    root = Path(subset_root)
    permutation = None
    if shuffled_pairs:
        base = SUTDPRCMDataset(root, "train", normalize_response=True)
        count = len(base) if max_samples is None else min(max_samples, len(base))
        permutation = shuffle_response_permutation(count, shuffle_seed)
    datasets = {
        "train": PairedDataset(SUTDPRCMDataset(root, "train", normalize_response=True), max_samples, seed, permutation),
        "val": PairedDataset(SUTDPRCMDataset(root, "val", normalize_response=True)),
        "test": PairedDataset(SUTDPRCMDataset(root, "test", normalize_response=True)),
    }
    generator = torch.Generator().manual_seed(seed)
    return {
        name: DataLoader(dataset, batch_size=batch_size, shuffle=name == "train", generator=generator, num_workers=0)
        for name, dataset in datasets.items()
    }


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float,
    lambda_variance: float,
    lambda_covariance: float = 0.0,
) -> dict[str, float]:
    model.train()
    model.spectrum_target_encoder.eval()
    totals = {"cross_loss": 0.0, "bootstrap_loss": 0.0, "variance_loss": 0.0, "covariance_loss": 0.0, "total_loss": 0.0}
    items = 0
    for geometry, response, _, _ in loader:
        geometry = geometry.to(device)
        response = response.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(geometry, response)
        model.update_target_center(outputs["z_target_raw"])
        loss, parts = physics_jepa_loss_with_parts(
            outputs["z_pred"], outputs["z_target"], outputs["z_self"], alpha, lambda_variance,
            z_online=outputs["z_online"], z_geometry=outputs["z_geometry"], lambda_covariance=lambda_covariance,
        )
        loss.backward()
        optimizer.step()
        model.update_target_encoder()
        size = geometry.shape[0]
        for name, value in parts.items():
            totals[name] += float(value.item()) * size
        totals["total_loss"] += float(loss.item()) * size
        items += size
    return {name: value / items for name, value in totals.items()}


def physics_jepa_loss_with_parts(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    z_self: torch.Tensor,
    alpha: float,
    lambda_variance: float,
    z_online: torch.Tensor | None = None,
    z_geometry: torch.Tensor | None = None,
    lambda_covariance: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Like ``physics_jepa_loss`` but also returns each additive component."""
    from src.physics_jepa_losses import covariance_regularization, jepa_loss, variance_regularization

    cross = jepa_loss(z_pred, z_target)
    bootstrap = jepa_loss(z_self, z_target)
    variance = torch.zeros_like(cross)
    covariance = torch.zeros_like(cross)
    terms = [z for z in (z_online, z_geometry) if z is not None]
    if lambda_variance > 0 and terms:
        variance = sum(variance_regularization(z) for z in terms) / len(terms)
    if lambda_covariance > 0 and terms:
        covariance = sum(covariance_regularization(z) for z in terms) / len(terms)
    total = cross + alpha * bootstrap + lambda_variance * variance + lambda_covariance * covariance
    return total, {"cross_loss": cross, "bootstrap_loss": bootstrap, "variance_loss": variance, "covariance_loss": covariance}


@torch.inference_mode()
def evaluate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    alpha: float = 1.0,
    lambda_variance: float = 0.0,
    lambda_covariance: float = 0.0,
) -> dict[str, float]:
    model.eval()
    model.spectrum_target_encoder.eval()
    totals = {"cross_loss": 0.0, "bootstrap_loss": 0.0, "variance_loss": 0.0, "covariance_loss": 0.0, "total_loss": 0.0}
    latents = {"z_geometry": [], "z_online": [], "z_target": [], "z_pred": []}
    count = 0
    for geometry, response, _, _ in loader:
        geometry = geometry.to(device)
        response = response.to(device)
        outputs = model(geometry, response)
        loss, parts = physics_jepa_loss_with_parts(
            outputs["z_pred"], outputs["z_target"], outputs["z_self"], alpha, lambda_variance,
            z_online=outputs["z_online"], z_geometry=outputs["z_geometry"], lambda_covariance=lambda_covariance,
        )
        size = geometry.shape[0]
        for name, value in parts.items():
            totals[name] += float(value.item()) * size
        totals["total_loss"] += float(loss.item()) * size
        for name, values in latents.items():
            values.append(outputs[name].cpu())
        count += size
    result = {name: value / count for name, value in totals.items()}
    latents_cat = {name: torch.cat(values) for name, values in latents.items()}
    result.update(physics_latent_variance_metrics(latents_cat["z_geometry"], latents_cat["z_online"], latents_cat["z_target"], latents_cat["z_pred"]))
    result["collapse_flag"] = float(any(result[f"{name}_mean_std"] < 1e-4 for name in ("context", "online", "target", "pred")))
    return result


@torch.inference_mode()
def cache_latents(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    split: str,
    response_stats: tuple[np.ndarray, np.ndarray] | None = None,
    frequency_ghz: np.ndarray | None = None,
) -> dict[str, Path]:
    """Save latent embeddings, geometries, responses, resonance targets, and ids."""
    model.eval()
    model.spectrum_target_encoder.eval()
    collected: dict[str, list[np.ndarray]] = {
        "z_geometry": [], "z_online": [], "z_target": [], "z_pred": [], "geometry": [], "response": [], "resonance_targets": [], "source_id": [],
    }
    for geometry, response, source_id, _ in loader:
        geometry = geometry.to(device)
        response = response.to(device)
        outputs = model(geometry, response)
        collected["z_geometry"].append(outputs["z_geometry"].cpu().numpy())
        collected["z_online"].append(outputs["z_online"].cpu().numpy())
        collected["z_target"].append(outputs["z_target"].cpu().numpy())
        collected["z_pred"].append(outputs["z_pred"].cpu().numpy())
        collected["geometry"].append(geometry.cpu().numpy())
        collected["response"].append(response.cpu().numpy())
        collected["source_id"].append(source_id)
    arrays = {
        name: np.concatenate(values)
        for name, values in collected.items()
        if name not in ("resonance_targets", "source_id")
    }
    arrays["source_id"] = np.asarray([item for group in collected["source_id"] for item in group])
    if response_stats is not None and frequency_ghz is not None:
        mean, std = response_stats
        unnormalized = arrays["response"] * std[None] + mean[None]
        arrays["resonance_targets"] = resonance_targets(unnormalized, frequency_ghz)
    for name, values in arrays.items():
        if name == "source_id":
            Path(output_dir / f"{split}_source_ids.txt").write_text("\n".join(values) + "\n", encoding="utf-8")
        else:
            np.save(output_dir / f"{split}_{name}.npy", values)
    return {name: Path(output_dir / f"{split}_{name}.npy") for name in ("z_geometry", "z_online", "z_target", "z_pred", "geometry", "response")}


def load_response_stats(subset_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    stats = np.load(Path(subset_root) / "train_response_stats.npz")
    return stats["mean"].astype(np.float32), stats["std"].astype(np.float32)


def resonance_targets(unnormalized: np.ndarray, frequency_ghz: np.ndarray) -> np.ndarray:
    from src.physics_representation_metrics import resonance_targets as _resonance_targets

    return _resonance_targets(unnormalized, frequency_ghz)
