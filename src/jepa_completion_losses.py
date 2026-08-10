"""JEPA and optional reconstruction losses for Phase 4."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def normalize_latent(latent: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return latent / torch.linalg.vector_norm(latent, dim=-1, keepdim=True).clamp_min(eps)


def jepa_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """Normalized latent MSE with stop-gradient target representation."""
    predicted = normalize_latent(z_pred)
    target = normalize_latent(z_target).detach()
    return torch.mean(torch.square(predicted - target))


def masked_reconstruction_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    errors = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (errors * mask).sum() / mask.sum().clamp_min(1.0)


def latent_variance_metrics(z_context: torch.Tensor, z_target: torch.Tensor, z_pred: torch.Tensor) -> dict[str, float]:
    variances = {"context": z_context.var(dim=0, unbiased=False), "target": z_target.var(dim=0, unbiased=False), "pred": z_pred.var(dim=0, unbiased=False)}
    return {
        "context_mean_variance": float(variances["context"].mean().item()),
        "target_mean_variance": float(variances["target"].mean().item()),
        "pred_mean_variance": float(variances["pred"].mean().item()),
        "context_mean_std": float(torch.sqrt(variances["context"].clamp_min(0)).mean().item()),
        "target_mean_std": float(torch.sqrt(variances["target"].clamp_min(0)).mean().item()),
        "pred_mean_std": float(torch.sqrt(variances["pred"].clamp_min(0)).mean().item()),
    }
