"""Losses for response-aware forward-surrogate experiments."""

from __future__ import annotations

import torch


def _stable_magnitude(real: torch.Tensor, imaginary: torch.Tensor) -> torch.Tensor:
    """Avoid undefined magnitude derivatives when both components are zero."""
    return torch.sqrt(real.square() + imaginary.square() + 1e-8)


def resonance_weighted_complex_loss(
    normalized_prediction: torch.Tensor,
    normalized_target: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    resonance_weight: float = 4.0,
    magnitude_weight: float = 0.15,
) -> torch.Tensor:
    """Weight complex error near target spectral features.

    The target's magnitude curvature is used only to assign frequency weights;
    no predicted feature locations are used during training.  The main term is
    still normalized complex-component MSE, while the small magnitude term
    encourages the two physical reflection magnitudes to agree.
    """
    if normalized_prediction.shape != normalized_target.shape or normalized_prediction.ndim != 3 or normalized_prediction.shape[1] != 4:
        raise ValueError("Expected matching [B, 4, F] prediction and target tensors")
    if resonance_weight < 0 or magnitude_weight < 0:
        raise ValueError("Loss weights must be non-negative")

    target = normalized_target * std + mean
    prediction = normalized_prediction * std + mean
    target_magnitude = torch.stack(
        (_stable_magnitude(target[:, 0], target[:, 1]), _stable_magnitude(target[:, 2], target[:, 3])), dim=1
    )
    curvature = torch.zeros_like(target_magnitude)
    curvature[..., 1:-1] = torch.abs(
        target_magnitude[..., 2:] - 2 * target_magnitude[..., 1:-1] + target_magnitude[..., :-2]
    )
    scale = curvature.amax(dim=-1, keepdim=True).clamp_min(1e-6)
    feature_weight = 1.0 + resonance_weight * (curvature / scale).clamp(max=1.0)
    component_weight = torch.stack((feature_weight[:, 0], feature_weight[:, 0], feature_weight[:, 1], feature_weight[:, 1]), dim=1)
    component_error = torch.square(normalized_prediction - normalized_target)
    component_loss = (component_error * component_weight).mean() / component_weight.mean().clamp_min(1e-6)

    if magnitude_weight == 0:
        return component_loss
    prediction_magnitude = torch.stack(
        (_stable_magnitude(prediction[:, 0], prediction[:, 1]), _stable_magnitude(prediction[:, 2], prediction[:, 3])), dim=1
    )
    magnitude_scale = torch.stack(
        (torch.hypot(std[0], std[1]), torch.hypot(std[2], std[3]))
    ).view(1, 2, 1).clamp_min(1e-6)
    magnitude_loss = (
        torch.square((prediction_magnitude - target_magnitude) / magnitude_scale) * feature_weight
    ).mean() / feature_weight.mean().clamp_min(1e-6)
    return component_loss + magnitude_weight * magnitude_loss
