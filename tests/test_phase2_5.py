from __future__ import annotations

import numpy as np
import torch

from src.forward_analysis import detect_resonance_features, geometry_complexity, resonance_errors
from src.losses import resonance_weighted_complex_loss
from src.metrics import per_sample_forward_metrics, reflection_magnitudes, unnormalize_response
from src.models import ForwardSurrogateCNN
from scripts.compare_forward_scales import METRICS, evaluate_run


def test_magnitudes_unnormalization_and_metrics_are_finite():
    response = torch.zeros(2, 4, 3)
    response[:, 0] = 3
    response[:, 1] = 4
    y_magnitude, x_magnitude = reflection_magnitudes(response)
    assert torch.allclose(y_magnitude, torch.full((2, 3), 5.0))
    assert torch.allclose(unnormalize_response(torch.ones(1, 4, 3), torch.ones(4, 1), 2 * torch.ones(4, 1)), 3 * torch.ones(1, 4, 3))
    metrics = per_sample_forward_metrics(response, torch.zeros_like(response), response, torch.zeros_like(response))
    assert all(torch.isfinite(value).all() for value in metrics.values())


def test_resonance_detection_handles_features_and_flat_spectrum():
    frequency = np.linspace(2.0, 12.0, 1001)
    spectrum = np.exp(-((frequency - 7.0) / 0.08) ** 2)
    assert detect_resonance_features(spectrum, frequency, prominence=0.2, distance_points=10)
    detail = resonance_errors(spectrum, spectrum, frequency, prominence=0.2, distance_points=10)
    assert detail["frequency_errors_ghz"] == [0.0]
    flat = resonance_errors(np.ones_like(frequency), np.ones_like(frequency), frequency)
    assert flat["true_feature_count"] == 0
    assert np.isnan(flat["resonance_region_magnitude_mae"])


def test_binary_geometry_complexity_uses_four_connectivity():
    geometry = np.zeros((16, 16), dtype=np.uint8)
    geometry[1, 1] = geometry[2, 1] = geometry[4, 4] = 1
    detail = geometry_complexity(geometry)
    assert detail["fill_ratio"] == 3 / 256
    assert detail["connected_components_4"] == 2
    assert detail["boundary_transitions_4"] == 10


def test_continuous_geometry_gradient_is_finite_and_nonzero():
    model = ForwardSurrogateCNN()
    geometry = torch.rand(1, 1, 16, 16, requires_grad=True)
    loss = model(geometry).square().mean()
    loss.backward()
    assert geometry.grad is not None
    assert torch.isfinite(geometry.grad).all()
    assert torch.count_nonzero(geometry.grad) > 0


def test_per_sample_metrics_keep_one_row_per_input():
    prediction = torch.zeros(3, 4, 1001)
    metrics = per_sample_forward_metrics(prediction, prediction, prediction, prediction)
    assert all(values.shape == (3,) for values in metrics.values())


def test_resonance_weighted_loss_is_finite_and_differentiable():
    prediction = torch.zeros(2, 4, 1001, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[:, 0, 500] = 2.0
    mean = torch.zeros(4, 1)
    std = torch.ones(4, 1)
    loss = resonance_weighted_complex_loss(prediction, target, mean, std)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) > 0


def test_scale_comparison_tolerates_missing_optional_artifacts(tmp_path):
    assert evaluate_run(tmp_path) == {metric: None for metric in METRICS}
