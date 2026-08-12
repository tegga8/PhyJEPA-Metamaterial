import numpy as np
import pytest
import torch

from scripts.evaluate_phase7_counterfactual import binary_metrics, summarize
from src.physics_conditioned_spatial_jepa import PhysicsConditionedSpatialJEPA


def test_binary_metrics_reports_exact_match_and_occupancy():
    geometry = np.array([[0, 1], [1, 0]], dtype=np.float32)
    metrics = binary_metrics(geometry, geometry)
    assert metrics["iou"] == 1.0
    assert metrics["dice"] == 1.0
    assert metrics["pixel_mse"] == 0.0
    assert metrics["occupancy_abs_difference"] == 0.0


def test_summarize_is_finite_and_reproducible():
    result = summarize([0.0, 1.0, 2.0])
    assert result["mean"] == pytest.approx(1.0)
    assert result["median"] == pytest.approx(1.0)
    assert result["fraction_positive"] == pytest.approx(2 / 3)


def test_counterfactual_model_condition_has_no_geometry_target_input():
    model = PhysicsConditionedSpatialJEPA()
    inputs = torch.zeros(2, 2, 16, 16)
    response = torch.zeros(2, 4, 1001)
    output_without_target = model(inputs, response)
    output_with_target = model(inputs, response, torch.ones(2, 1, 16, 16))
    assert output_without_target["logits"].shape == (2, 1, 16, 16)
    assert torch.allclose(output_without_target["logits"], output_with_target["logits"])
