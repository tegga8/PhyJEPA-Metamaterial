"""Metrics for complex polarized reflection prediction."""

from __future__ import annotations

import torch


def unnormalize_response(response: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Convert normalized `[B, 4, F]` responses back to physical components."""
    return response * std + mean


def reflection_magnitude_mae(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Return MAE for y/T and x/R reflection magnitudes in physical units."""
    t_pred = torch.hypot(prediction[:, 0], prediction[:, 1])
    t_true = torch.hypot(target[:, 0], target[:, 1])
    r_pred = torch.hypot(prediction[:, 2], prediction[:, 3])
    r_true = torch.hypot(target[:, 2], target[:, 3])
    return {
        "y_reflection_magnitude_mae": float(torch.mean(torch.abs(t_pred - t_true)).item()),
        "x_reflection_magnitude_mae": float(torch.mean(torch.abs(r_pred - r_true)).item()),
    }
