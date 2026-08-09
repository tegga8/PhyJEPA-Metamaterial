"""Metrics for the two complex reflected-field coefficients.

The fixed channel layout is ``[Re(T_y), Im(T_y), Re(R_x), Im(R_x)]``.  ``T``
and ``R`` are the dataset's names for cross- and co-polarized *reflection*
coefficients; neither channel is a transmission coefficient.
"""

from __future__ import annotations

from typing import Final

import torch


CHANNEL_NAMES: Final[tuple[str, str, str, str]] = (
    "y_cross_reflection_real",
    "y_cross_reflection_imag",
    "x_co_reflection_real",
    "x_co_reflection_imag",
)
POLARIZATION_NAMES: Final[tuple[str, str]] = ("y_cross_reflection", "x_co_reflection")


def unnormalize_response(response: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Convert normalized ``[B, 4, F]`` responses back to physical components."""
    return response * std + mean


def reflection_magnitudes(response: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return magnitudes of y-cross (T) and x-co (R) reflection coefficients."""
    if response.ndim != 3 or response.shape[1] != 4:
        raise ValueError(f"Expected response [B, 4, F], got {tuple(response.shape)}")
    return torch.hypot(response[:, 0], response[:, 1]), torch.hypot(response[:, 2], response[:, 3])


def _pearson_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pearson correlation along frequency, returning zero for constant spectra."""
    pred_centered = prediction - prediction.mean(dim=-1, keepdim=True)
    target_centered = target - target.mean(dim=-1, keepdim=True)
    denominator = torch.sqrt(
        torch.sum(pred_centered.square(), dim=-1) * torch.sum(target_centered.square(), dim=-1)
    )
    correlation = torch.sum(pred_centered * target_centered, dim=-1) / denominator.clamp_min(1e-12)
    return torch.where(denominator > 1e-12, correlation, torch.zeros_like(correlation))


def per_sample_forward_metrics(
    normalized_prediction: torch.Tensor,
    normalized_target: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute frequency-averaged metrics for every item in a batch.

    Complex-component errors are in the dataset's unnormalised coefficient
    units. Correlation is calculated separately for every magnitude spectrum,
    then can be summarized across structures without flattening spectra.
    """
    if prediction.shape != target.shape or prediction.shape[1:] != (4, prediction.shape[-1]):
        raise ValueError("Prediction and target must have matching [B, 4, F] shapes")
    y_prediction, x_prediction = reflection_magnitudes(prediction)
    y_target, x_target = reflection_magnitudes(target)
    component_abs = torch.abs(prediction - target).mean(dim=-1)
    component_rmse = torch.sqrt(torch.square(prediction - target).mean(dim=-1))
    return {
        "normalized_mse": torch.square(normalized_prediction - normalized_target).mean(dim=(1, 2)),
        "complex_mae": component_abs.mean(dim=1),
        **{f"{name}_mae": component_abs[:, index] for index, name in enumerate(CHANNEL_NAMES)},
        **{f"{name}_rmse": component_rmse[:, index] for index, name in enumerate(CHANNEL_NAMES)},
        "y_cross_reflection_magnitude_mae": torch.abs(y_prediction - y_target).mean(dim=-1),
        "x_co_reflection_magnitude_mae": torch.abs(x_prediction - x_target).mean(dim=-1),
        "y_cross_reflection_magnitude_rmse": torch.sqrt(torch.square(y_prediction - y_target).mean(dim=-1)),
        "x_co_reflection_magnitude_rmse": torch.sqrt(torch.square(x_prediction - x_target).mean(dim=-1)),
        "y_cross_reflection_correlation": _pearson_per_sample(y_prediction, y_target),
        "x_co_reflection_correlation": _pearson_per_sample(x_prediction, x_target),
    }


def aggregate_forward_metrics(per_sample: dict[str, torch.Tensor]) -> dict[str, float]:
    """Mean every per-sample metric and add median magnitude correlations."""
    result = {name: float(values.mean().item()) for name, values in per_sample.items()}
    for name in ("y_cross_reflection_correlation", "x_co_reflection_correlation"):
        result[f"{name}_median"] = float(per_sample[name].median().item())
    return result


def reflection_magnitude_mae(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Backward-compatible aggregate magnitude MAE used by the Phase 2 baseline."""
    y_prediction, x_prediction = reflection_magnitudes(prediction)
    y_target, x_target = reflection_magnitudes(target)
    return {
        "y_reflection_magnitude_mae": float(torch.mean(torch.abs(y_prediction - y_target)).item()),
        "x_reflection_magnitude_mae": float(torch.mean(torch.abs(x_prediction - x_target)).item()),
    }
