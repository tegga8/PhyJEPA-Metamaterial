"""Physics-JEPA loss terms and latent collapse diagnostics.

All loss terms are JEPA-style: normalized latent MSE against a stop-gradient
target.  No reconstruction, geometry BCE, or masked reconstruction is used in
the core objective.
"""

from __future__ import annotations

import torch

from src.jepa_completion_losses import jepa_loss, latent_variance_metrics


def variance_regularization(latent: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """VICReg-style per-dimension std regularizer guarding against collapse."""
    if latent.ndim != 2:
        raise ValueError(f"Expected latent [B, D], got {tuple(latent.shape)}")
    std = torch.sqrt(latent.var(dim=0, unbiased=False) + eps)
    return torch.mean(torch.relu(1.0 - std))


def covariance_regularization(latent: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """VICReg-style covariance penalty on the centered latent.

    Penalizes the off-diagonal entries of the batch covariance matrix,
    encouraging the latent dimensions to be decorrelated (full-rank) rather
    than directionally collapsed.  This is the collapse guard the per-dimension
    std term alone cannot provide: per-dimension std can be satisfied while
    dimensions remain highly correlated, i.e. the directional / low-rank
    collapse diagnosed in the M3 gate failure.
    """
    if latent.ndim != 2:
        raise ValueError(f"Expected latent [B, D], got {tuple(latent.shape)}")
    centered = latent - latent.mean(dim=0, keepdim=True)
    covariance = (centered.T @ centered) / (latent.shape[0] - 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    return off_diagonal.pow(2).sum() / latent.shape[1]


def physics_jepa_loss(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    z_self: torch.Tensor,
    alpha: float = 0.5,
    lambda_variance: float = 0.1,
    z_online: torch.Tensor | None = None,
    z_geometry: torch.Tensor | None = None,
    lambda_covariance: float = 0.0,
) -> torch.Tensor:
    """Cross-modal JEPA loss with a spectrum bootstrap term.

    ``z_target`` must be the stop-gradient momentum spectrum latent.  The first
    term trains the geometry encoder and predictor; the bootstrap term trains
    the online spectrum encoder against its own momentum target; a variance
    term and, when enabled, a redundancy-reduction (covariance) term discourage
    latent collapse.  ``lambda_covariance > 0`` is the v2 collapse-fix
    formulation (labeled experiment ``physics_jepa_v2_collapse_fix``).
    """
    cross = jepa_loss(z_pred, z_target)
    bootstrap = jepa_loss(z_self, z_target)
    variance = torch.zeros_like(cross)
    covariance = torch.zeros_like(cross)
    terms = [z for z in (z_online, z_geometry) if z is not None]
    if lambda_variance > 0 and terms:
        variance = sum(variance_regularization(z) for z in terms) / len(terms)
    if lambda_covariance > 0 and terms:
        covariance = sum(covariance_regularization(z) for z in terms) / len(terms)
    return cross + alpha * bootstrap + lambda_variance * variance + lambda_covariance * covariance


def physics_latent_variance_metrics(
    z_geometry: torch.Tensor,
    z_online: torch.Tensor,
    z_target: torch.Tensor,
    z_pred: torch.Tensor,
) -> dict[str, float]:
    """Per-latent variance/std diagnostics including the online spectrum latent."""
    metrics = latent_variance_metrics(z_geometry, z_target, z_pred)
    online_variance = z_online.var(dim=0, unbiased=False)
    metrics["online_mean_variance"] = float(online_variance.mean().item())
    metrics["online_mean_std"] = float(torch.sqrt(online_variance.clamp_min(0)).mean().item())
    return metrics
