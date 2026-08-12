import pytest

from scripts.select_forward_screening_surrogate import complexity_group, select_candidate


def test_complexity_group_uses_documented_tertiles():
    assert complexity_group(0.0) == "simple"
    assert complexity_group(3.0625) == "simple"
    assert complexity_group(3.0626) == "medium"
    assert complexity_group(13.291666666666664) == "medium"
    assert complexity_group(13.2917) == "complex"


def test_selection_prioritizes_shared_broad_error_then_resonance():
    base = {
        "shared": {
            "normalized_mse": 0.2,
            "resonance_frequency_error_ghz": 0.2,
            "resonance_region_magnitude_mae": 0.2,
        }
    }
    candidates = [
        {"name": "broad", **base},
        {"name": "resonance", "shared": {**base["shared"], "normalized_mse": 0.3, "resonance_frequency_error_ghz": 0.01}},
    ]
    name, rationale = select_candidate(candidates)
    assert name == "broad"
    assert rationale["runner_up"] == "resonance"


def test_selection_uses_resonance_as_tie_breaker():
    common = {"normalized_mse": 0.2, "resonance_region_magnitude_mae": 0.2}
    candidates = [
        {"name": "slow", "shared": {**common, "resonance_frequency_error_ghz": 0.2}},
        {"name": "sharp", "shared": {**common, "resonance_frequency_error_ghz": 0.1}},
    ]
    name, _ = select_candidate(candidates)
    assert name == "sharp"
