from __future__ import annotations

import torch

from src.models import ForwardSurrogateCNN
from src.physics_conditioned_spatial_jepa import PhysicsConditionedSpatialJEPA
from src.physics_consistency import continuous_completion, local_perturbation_diagnostics, physics_consistency_loss, physics_gradient_diagnostics


def test_continuous_completion_preserves_observed_pixels_and_gradients():
    probabilities = torch.rand(2, 1, 16, 16, requires_grad=True)
    inputs = torch.rand(2, 2, 16, 16)
    mask = torch.zeros(2, 1, 16, 16); mask[:, :, 4:12, 4:12] = 1
    geometry = continuous_completion(probabilities, inputs, mask)
    assert torch.equal(geometry[mask == 0], inputs[:, :1][mask == 0])
    geometry.sum().backward()
    assert probabilities.grad is not None and torch.all(probabilities.grad[mask == 0] == 0)


def test_frozen_surrogate_loss_has_input_gradients_and_no_weight_gradients():
    torch.manual_seed(11)
    model = PhysicsConditionedSpatialJEPA()
    surrogate = ForwardSurrogateCNN()
    surrogate.eval()
    for parameter in surrogate.parameters(): parameter.requires_grad_(False)
    inputs = torch.rand(2, 2, 16, 16)
    mask = torch.zeros(2, 1, 16, 16); mask[:, :, 4:12, 4:12] = 1
    response = torch.randn(2, 4, 1001)
    diagnostics = physics_gradient_diagnostics(model, surrogate, inputs, mask, response)
    assert diagnostics["gradient_finite"]
    assert diagnostics["gradient_reaches_decoder_predictor"]
    assert diagnostics["geometry_gradient_norm"] > 0
    assert diagnostics["surrogate_all_requires_grad_false"]
    assert diagnostics["surrogate_has_no_parameter_grad"]


def test_local_perturbation_matches_autograd_direction():
    torch.manual_seed(12)
    surrogate = ForwardSurrogateCNN().eval()
    for parameter in surrogate.parameters(): parameter.requires_grad_(False)
    geometry = torch.rand(2, 1, 16, 16)
    mask = torch.zeros_like(geometry); mask[:, :, 4:12, 4:12] = 1
    report = local_perturbation_diagnostics(surrogate, geometry, torch.randn(2, 4, 1001), mask)
    assert report["finite_difference_abs_mean"] > 0
    assert report["autograd_directional_abs_mean"] > 0
    assert report["valid_test_fraction"] > 0
