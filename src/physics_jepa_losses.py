"""Physics-JEPA loss terms and latent collapse diagnostics.

All loss terms are JEPA-style: normalized latent MSE against a stop-gradient
target.  No reconstruction, geometry BCE, or masked reconstruction is used in
the core objective.

``relational_margin_loss`` and ``build_response_triplets`` implement the
Physics-JEPA v3 change B (label ``physics_jepa_v3_frequency_relational``): a
small margin-ranking objective that requires latent distances to respect the
EM-response similarity ordering, using deterministic seed-42 triplets and the
existing normalized-response MSE as the response distance ``D_S``.
"""

from __future__ import annotations

import math

import numpy as np
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


def normalized_latent_distance_matrix(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pairwise Euclidean distance between L2-normalized latent rows (canonical convention)."""
    if z.ndim != 2:
        raise ValueError(f"Expected latent [B, D], got {tuple(z.shape)}")
    normalized = z / (z.norm(dim=-1, keepdim=True) + eps)
    return torch.cdist(normalized, normalized, p=2.0)


def build_response_triplets(
    response: torch.Tensor,
    num_triplets: int = 32,
    seed: int = 42,
    neg_ratio: float = 2.0,
    min_separation: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Deterministic anchor/positive/negative index selection from one response batch.

    ``D_S`` is the MSE of the Phase-2 normalized ``[B, 4, 1001]`` response
    (pairwise squared-Euclidean divided by the channel-point product).  For each
    anchor: the positive is the index (excluding the anchor) with the smallest
    response distance; the negative is the valid candidate with a response
    distance at least ``neg_ratio`` times the positive distance (closest such
    candidate, to keep the margin gradient meaningful), skipping source-identical
    duplicates.  Selection draws from a fixed numpy RNG seeded with ``seed``, so
    the triplets are deterministic for identical batches.
    """
    if response.ndim != 3 or response.shape[1:] != (4, 1001):
        raise ValueError(f"Expected normalized response [B, 4, 1001], got {tuple(response.shape)}")
    batch = response.shape[0]
    rng = np.random.default_rng(seed)
    count = min(int(num_triplets), batch)
    anchors = np.sort(rng.choice(batch, size=count, replace=False))
    flat = response.reshape(batch, -1).detach().cpu().numpy()
    squared = np.square(flat).sum(axis=-1, keepdims=True)
    response_distance = squared + squared.T - 2.0 * (flat @ flat.T)
    response_distance = np.clip(response_distance, 0.0, None)
    np.fill_diagonal(response_distance, np.inf)
    positive = np.argmin(response_distance[anchors], axis=1)
    positive_mask = np.zeros((count, batch), dtype=bool)
    positive_mask[np.arange(count), anchors] = True
    positive_mask[np.arange(count), positive] = True
    positive_distance = response_distance[anchors, positive]
    candidate = np.where(
        np.logical_and(~positive_mask, response_distance[anchors] >= neg_ratio * positive_distance[:, None] + min_separation),
        response_distance[anchors],
        np.inf,
    )
    has_valid = np.isfinite(candidate).any(axis=1)
    nearest_negative = np.argmin(candidate, axis=1)
    negative = np.where(has_valid, nearest_negative, np.argmax(response_distance[anchors], axis=1))
    return {
        "anchors": torch.as_tensor(anchors, dtype=torch.long),
        "positives": torch.as_tensor(positive, dtype=torch.long),
        "negatives": torch.as_tensor(negative, dtype=torch.long),
    }


def relational_margin_loss(
    z: torch.Tensor,
    triplets: dict[str, torch.Tensor],
    margin: float = 0.2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Margin-ranking loss enforcing EM-response-similarity ordering in ``z``.

    For triplets where ``D_S(S_a, S_p) < D_S(S_a, S_n)``, encourage
    ``D_z(z_a, z_p) < D_z(z_a, z_n)``:

    ``mean(max(0, D_z(z_a, z_p) - D_z(z_a, z_n) + margin))``.

    Distances use the canonical L2-normalized latent Euclidean distance.
    """
    if z.ndim != 2:
        raise ValueError(f"Expected latent [B, D], got {tuple(z.shape)}")
    normalized = z / (z.norm(dim=-1, keepdim=True) + eps)
    anchor = normalized[triplets["anchors"]]
    positive = normalized[triplets["positives"]]
    negative = normalized[triplets["negatives"]]
    d_ap = (anchor - positive).norm(dim=-1, p=2.0)
    d_an = (anchor - negative).norm(dim=-1, p=2.0)
    margin_loss = torch.relu(d_ap - d_an + margin).mean()
    return margin_loss


def v3_loss_with_parts(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    z_self: torch.Tensor,
    alpha: float,
    lambda_variance: float,
    lambda_covariance: float,
    z_online: torch.Tensor | None,
    z_geometry: torch.Tensor | None,
    z_relational: torch.Tensor | None,
    lambda_relational: float,
    triplets: dict[str, torch.Tensor],
    margin: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Physics-JEPA v3 loss: JEPA (cross + bootstrap) plus collapse guards plus relational term.

    ``L = L_JEPA + lambda_v * L_variance + lambda_c * L_covariance + lambda_r * L_rel``,
    where the relational term averages the margin-ranking loss over the online
    spectrum latent and the geometry-derived predictor latent (each with the
    same batch-local response triplets).
    """
    cross = jepa_loss(z_pred, z_target)
    bootstrap = jepa_loss(z_self, z_target)
    variance = torch.zeros_like(cross)
    covariance = torch.zeros_like(cross)
    relational = torch.zeros_like(cross)
    guard_terms = [z for z in (z_online, z_geometry) if z is not None]
    if lambda_variance > 0 and guard_terms:
        variance = sum(variance_regularization(z) for z in guard_terms) / len(guard_terms)
    if lambda_covariance > 0 and guard_terms:
        covariance = sum(covariance_regularization(z) for z in guard_terms) / len(guard_terms)
    if lambda_relational > 0 and z_relational is not None:
        relational = relational_margin_loss(z_relational, triplets, margin=margin)
    total = cross + alpha * bootstrap + lambda_variance * variance + lambda_covariance * covariance + lambda_relational * relational
    return total, {
        "cross_loss": cross,
        "bootstrap_loss": bootstrap,
        "variance_loss": variance,
        "covariance_loss": covariance,
        "relational_loss": relational,
    }
