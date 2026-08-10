import pytest
import torch

from src.mask_aware_spatial_jepa_losses import (
    downsample_mask,
    mask_aware_spatial_jepa_loss,
    mask_weight_map,
    spatial_jepa_distance,
)
from src.spatial_jepa_completion_model import SpatialJEPACompletionModel, compose_binary_spatial_completion


def test_average_pool_mask_maps_top_left_block_exactly():
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, :8, :8] = 1
    mask8 = downsample_mask(mask)
    assert mask8.shape == (1, 1, 8, 8)
    assert torch.equal(mask8[:, :, :4, :4], torch.ones(1, 1, 4, 4))
    assert torch.equal(mask8[:, :, 4:, :], torch.zeros(1, 1, 4, 8))
    assert torch.equal(mask8[:, :, :, 4:], torch.zeros(1, 1, 8, 4))


def test_average_pool_single_hidden_pixel_is_quarter():
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, 0, 0] = 1
    mask8 = downsample_mask(mask)
    assert mask8[0, 0, 0, 0].item() == 0.25
    assert mask8.max().item() == 0.25


def test_weight_map_has_soft_alpha_bounds_and_responds_to_mask():
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, :8, :8] = 1
    weights = mask_weight_map(mask, alpha=0.1)
    assert weights.min().item() >= 0.1
    assert weights.max().item() <= 1.0
    assert weights[0, 0, 0, 0].item() == 1.0
    assert weights[0, 0, 7, 7].item() == pytest.approx(0.1)


def test_mask_aware_loss_is_finite_nonnegative_and_target_stopped():
    predicted = torch.randn(2, 64, 8, 8, requires_grad=True)
    target = torch.randn(2, 64, 8, 8, requires_grad=True)
    weights = torch.ones(2, 1, 8, 8)
    loss = mask_aware_spatial_jepa_loss(predicted, target, weights)
    assert torch.isfinite(loss)
    assert loss.item() >= 0
    loss.backward()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()
    assert target.grad is None


def test_higher_weight_increases_same_location_contribution():
    predicted = torch.randn(1, 64, 8, 8)
    target = torch.randn(1, 64, 8, 8)
    distances = spatial_jepa_distance(predicted, target)
    weights = torch.full((1, 1, 8, 8), 0.1)
    weights[:, :, 0, 0] = 1.0
    weighted = distances * weights[:, 0]
    assert torch.allclose(weighted[0, 0, 0], distances[0, 0, 0])
    assert torch.allclose(weighted[0, 0, 1], distances[0, 0, 1] * 0.1)


def test_mask_aware_model_gradient_ema_and_compositing():
    model = SpatialJEPACompletionModel()
    inputs = torch.rand(2, 2, 16, 16)
    target = torch.rand(2, 1, 16, 16)
    mask = torch.zeros(2, 1, 16, 16)
    mask[:, :, 4:12, 4:12] = 1
    outputs = model(inputs, target)
    loss = mask_aware_spatial_jepa_loss(outputs["z_pred"], outputs["z_target"], mask_weight_map(mask))
    loss = loss + 0.1 * torch.nn.functional.binary_cross_entropy_with_logits(outputs["logits"], target)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.context_encoder.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.predictor.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.decoder.parameters())
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    before = next(model.target_encoder.parameters()).detach().clone()
    model.update_target_encoder()
    assert not torch.equal(before, next(model.target_encoder.parameters()).detach())
    completed = compose_binary_spatial_completion(torch.ones(2, 1, 16, 16), inputs, mask)
    assert torch.equal(completed[mask == 0], inputs[:, :1][mask == 0])
