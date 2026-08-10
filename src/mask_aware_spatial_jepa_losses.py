"""Soft mask-aware spatial JEPA objective for Phase 4.2."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def downsample_mask(mask: torch.Tensor) -> torch.Tensor:
    """Area-average a [B,1,16,16] hidden mask into [B,1,8,8]."""
    if mask.ndim != 4 or mask.shape[1:] != (1, 16, 16):
        raise ValueError(f"Expected [B, 1, 16, 16], got {tuple(mask.shape)}")
    return F.avg_pool2d(mask, kernel_size=2, stride=2)


def mask_weight_map(mask: torch.Tensor, alpha: float = 0.10, gamma: float = 1.0) -> torch.Tensor:
    """Build W = alpha + (1-alpha) * M8**gamma."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    mask8 = downsample_mask(mask)
    return alpha + (1.0 - alpha) * mask8.pow(gamma)


def spatial_jepa_distance(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """Return channel-normalized squared distance per [B,H,W] location."""
    if z_pred.shape != z_target.shape or z_pred.ndim != 4:
        raise ValueError(f"Expected matching [B,C,H,W] latents, got {tuple(z_pred.shape)} and {tuple(z_target.shape)}")
    predicted = z_pred / torch.linalg.vector_norm(z_pred, dim=1, keepdim=True).clamp_min(1e-8)
    target = (z_target / torch.linalg.vector_norm(z_target, dim=1, keepdim=True).clamp_min(1e-8)).detach()
    return torch.square(predicted - target).mean(dim=1)


def mask_aware_spatial_jepa_loss(z_pred: torch.Tensor, z_target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute weighted normalized spatial JEPA MSE without target gradients."""
    distances = spatial_jepa_distance(z_pred, z_target)
    if weights.ndim != 4 or weights.shape[1] != 1 or weights.shape[-2:] != distances.shape[-2:]:
        raise ValueError(f"Expected weights [B,1,{distances.shape[-2]},{distances.shape[-1]}], got {tuple(weights.shape)}")
    if weights.shape[0] != distances.shape[0]:
        raise ValueError("Weight batch size must match latent batch size")
    weights = weights[:, 0]
    return (distances * weights).sum() / weights.sum().clamp_min(1e-8)


def masked_reconstruction_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    errors = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (errors * mask).sum() / mask.sum().clamp_min(1.0)


def mask_weight_statistics(mask: torch.Tensor, alpha: float = 0.10, gamma: float = 1.0) -> dict[str, float]:
    mask8 = downsample_mask(mask)
    weights = mask_weight_map(mask, alpha, gamma)
    return {
        "mean_mask8": float(mask8.mean().item()),
        "min_mask8": float(mask8.min().item()),
        "max_mask8": float(mask8.max().item()),
        "mean_weight": float(weights.mean().item()),
        "min_weight": float(weights.min().item()),
        "max_weight": float(weights.max().item()),
    }


def spatial_latent_statistics(z_context: torch.Tensor, z_target: torch.Tensor, z_pred: torch.Tensor) -> dict[str, float]:
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


def latent_norm(latent: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(latent, dim=1)
