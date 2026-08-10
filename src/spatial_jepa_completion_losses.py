"""Spatial JEPA and masked reconstruction losses for Phase 4.1."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def normalize_spatial_latent(latent: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize feature channels independently at each spatial location."""
    if latent.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], got {tuple(latent.shape)}")
    return latent / torch.linalg.vector_norm(latent, dim=1, keepdim=True).clamp_min(eps)


def spatial_jepa_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """Aligned spatial normalized MSE with a stop-gradient target map."""
    predicted = normalize_spatial_latent(z_pred)
    target = normalize_spatial_latent(z_target).detach()
    return torch.mean(torch.square(predicted - target))


def masked_reconstruction_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    errors = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (errors * mask).sum() / mask.sum().clamp_min(1.0)


def spatial_latent_statistics(z_context: torch.Tensor, z_target: torch.Tensor, z_pred: torch.Tensor) -> dict[str, float]:
    """Return global and spatial spread diagnostics over batch/location values."""
    values = {"context": z_context, "target": z_target, "pred": z_pred}
    metrics: dict[str, float] = {}
    for name, latent in values.items():
        variance = latent.var(dim=0, unbiased=False)
        std = torch.sqrt(variance.clamp_min(0))
        metrics[f"{name}_mean_variance"] = float(variance.mean().item())
        metrics[f"{name}_mean_std"] = float(std.mean().item())
        metrics[f"{name}_min_std"] = float(std.min().item())
        metrics[f"{name}_max_std"] = float(std.max().item())
    return metrics


def spatial_latent_norm(latent: torch.Tensor) -> torch.Tensor:
    """Collapse channels only for visualization, preserving the HxW map."""
    return torch.linalg.vector_norm(latent, dim=1)
