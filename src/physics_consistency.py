"""Differentiable frozen-surrogate utilities for Phase 5B.

The geometry passed to the surrogate is a *continuous* completion: observed
pixels are copied from the input and hidden pixels are decoder probabilities.
Thresholding is reserved for evaluation artifacts only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.models import build_forward_model


def continuous_completion(probabilities: torch.Tensor, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Composite decoder probabilities into a differentiable [B,1,16,16] geometry."""
    if probabilities.shape != mask.shape or inputs.shape[1:] != (2, 16, 16):
        raise ValueError("Expected probabilities/mask [B,1,16,16] and inputs [B,2,16,16]")
    return inputs[:, :1] * (1.0 - mask) + probabilities * mask


def load_frozen_forward_surrogate(checkpoint_path: str | Path, device: torch.device) -> tuple[nn.Module, str]:
    """Load a Phase-2 surrogate with immutable weights but differentiable input."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args: Any = checkpoint.get("args", {})
    model_name = args.get("model", "ForwardSurrogateCNN") if isinstance(args, dict) else getattr(args, "model", "ForwardSurrogateCNN")
    surrogate = build_forward_model(model_name).to(device)
    surrogate.load_state_dict(checkpoint["model_state_dict"])
    surrogate.eval()
    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)
    return surrogate, model_name


def physics_consistency_loss(
    probabilities: torch.Tensor,
    inputs: torch.Tensor,
    mask: torch.Tensor,
    response_target: torch.Tensor,
    surrogate: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized response MSE and its continuous geometry input."""
    geometry = continuous_completion(probabilities, inputs, mask)
    response_prediction = surrogate(geometry)
    return torch.nn.functional.mse_loss(response_prediction, response_target), geometry


def _grad_norm(parameters: list[nn.Parameter]) -> float:
    norms = [parameter.grad.detach().norm() for parameter in parameters if parameter.grad is not None]
    return float(torch.linalg.vector_norm(torch.stack(norms)).item()) if norms else 0.0


def physics_gradient_diagnostics(
    model: nn.Module,
    surrogate: nn.Module,
    inputs: torch.Tensor,
    mask: torch.Tensor,
    response: torch.Tensor,
) -> dict[str, float | bool]:
    """Verify a nonzero finite physics gradient reaches each intended Phase-5A path."""
    model.zero_grad(set_to_none=True)
    surrogate.zero_grad(set_to_none=True)
    outputs = model(inputs, response)
    probabilities = torch.sigmoid(outputs["logits"])
    loss, geometry = physics_consistency_loss(probabilities, inputs, mask, response, surrogate)
    geometry.retain_grad()
    loss.backward()
    result: dict[str, float | bool] = {
        "physics_loss": float(loss.item()),
        "geometry_gradient_norm": float(geometry.grad.norm().item()) if geometry.grad is not None else 0.0,
        "decoder_gradient_norm": _grad_norm(list(model.decoder.parameters())),
        "predictor_gradient_norm": _grad_norm(list(model.predictor.parameters())),
        "film_gradient_norm": _grad_norm(list(model.film.parameters())),
        "em_encoder_gradient_norm": _grad_norm(list(model.em_encoder.parameters())),
        "surrogate_all_requires_grad_false": all(not parameter.requires_grad for parameter in surrogate.parameters()),
        "surrogate_has_no_parameter_grad": all(parameter.grad is None for parameter in surrogate.parameters()),
    }
    result["gradient_finite"] = bool(
        torch.isfinite(loss).item()
        and all(torch.isfinite(torch.tensor(value)).item() for key, value in result.items() if key.endswith("_norm"))
    )
    result["gradient_reaches_decoder_predictor"] = bool(result["decoder_gradient_norm"] > 0 and result["predictor_gradient_norm"] > 0)
    return result


def local_perturbation_diagnostics(
    surrogate: nn.Module,
    geometry: torch.Tensor,
    response_target: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 1e-3,
    directions: int = 8,
) -> dict[str, float | int]:
    """Compare directional finite differences with autograd on hidden pixels."""
    base = geometry.detach().clone().requires_grad_(True)
    loss = torch.nn.functional.mse_loss(surrogate(base), response_target)
    gradient = torch.autograd.grad(loss, base)[0]
    generator = torch.Generator(device=base.device).manual_seed(17)
    differences: list[float] = []
    derivatives: list[float] = []
    for _ in range(directions):
        direction = torch.randn(base.shape, device=base.device, dtype=base.dtype, generator=generator) * mask
        direction = direction / direction.norm().clamp_min(1e-12)
        with torch.no_grad():
            plus = torch.nn.functional.mse_loss(surrogate(base + epsilon * direction), response_target)
            minus = torch.nn.functional.mse_loss(surrogate(base - epsilon * direction), response_target)
        differences.append(float(((plus - minus) / (2 * epsilon)).item()))
        derivatives.append(float((gradient * direction).sum().item()))
    fd, autograd = torch.tensor(differences), torch.tensor(derivatives)
    valid = (fd.abs() > 1e-10) & (autograd.abs() > 1e-10)
    return {
        "epsilon": epsilon,
        "directions": directions,
        "finite_difference_abs_mean": float(fd.abs().mean().item()),
        "autograd_directional_abs_mean": float(autograd.abs().mean().item()),
        "sign_agreement": float((torch.sign(fd[valid]) == torch.sign(autograd[valid])).float().mean().item()) if valid.any() else 0.0,
        "valid_test_fraction": float(valid.float().mean().item()),
    }
