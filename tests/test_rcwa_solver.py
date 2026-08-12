from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from src.rcwa_solver import (
    RCWAConfig,
    copper_relative_permittivity,
    convergence_mse,
    frequency_vector,
    geometry_to_physical_pattern,
    pack_response,
    resolve_device,
    validate_frequency_vector,
    solve_geometry,
)
from src.rcwa_validation import (
    PHYSICAL_MAPPING,
    cache_key,
    convergence_row,
    frequency_wise_normalized_mse,
    pack_modes,
    phase42_complexity_group,
    ranking_statistics,
)


def test_rcwa_import():
    assert importlib.util.find_spec("meent") is not None


def test_frequency_vector():
    frequency = frequency_vector()
    assert frequency.shape == (1001,)
    assert frequency[0] == pytest.approx(2.0)
    assert frequency[-1] == pytest.approx(12.0)
    assert np.diff(frequency).mean() == pytest.approx(0.01)


def test_geometry_to_physical_pattern():
    geometry = np.zeros((16, 16), dtype=np.uint8); geometry[0, 0] = geometry[-1, -1] = 1
    pattern = geometry_to_physical_pattern(geometry)
    assert pattern.shape == (20, 20)
    assert pattern[2, 2] and pattern[17, 17]
    assert not pattern[0].any() and not pattern[:, 0].any()


def test_unit_cell_dimensions():
    config = RCWAConfig()
    assert config.period_mm == (10.0, 10.0)
    assert 16 * config.patch_size_mm + 2 * config.padding_mm == config.period_mm[0]


def test_response_shape():
    value = pack_response(np.ones(1001, dtype=np.complex128), 1j * np.ones(1001, dtype=np.complex128))
    assert value.shape == (4, 1001)


def test_complex_response():
    with pytest.raises(ValueError):
        pack_response(np.ones(3), np.ones(3, dtype=np.complex128))


def test_channel_packing():
    packed = pack_response(np.array([1 + 2j]), np.array([3 + 4j]))
    assert np.array_equal(packed[:, 0], np.array([1, 2, 3, 4], dtype=np.float32))


def test_normal_incidence():
    config = RCWAConfig()
    assert config.period_mm == (10.0, 10.0)  # theta=phi=0 is fixed by solve_geometry.


def test_passive_material_convention():
    assert copper_relative_permittivity(7.0).imag < 0


def test_convergence():
    values = np.array([[1 + 2j, 3 + 4j]])
    assert convergence_mse(values, values.copy()) == 0.0


def test_energy_sanity():
    reflected = np.array([0.7, 0.8]); transmitted = np.array([0.1, 0.05])
    assert np.all(reflected + transmitted <= 1.0 + 1e-12)


def test_determinism_and_device_resolution():
    assert np.array_equal(validate_frequency_vector([2.0, 3.0]), validate_frequency_vector([2.0, 3.0]))
    device, _ = resolve_device("auto")
    assert device in {"cpu", "cuda"}


def test_phase42_complexity_bins_are_fixed():
    assert phase42_complexity_group(3.0625) == "simple"
    assert phase42_complexity_group(3.0626) == "medium"
    assert phase42_complexity_group(13.291666666666664) == "medium"
    assert phase42_complexity_group(13.3) == "complex"


def test_full_spectrum_frequency_wise_error_and_convergence_channels():
    target = np.zeros((4, 1001), dtype=np.float32)
    prediction = target.copy(); prediction[0] = 1
    values = frequency_wise_normalized_mse(prediction, target, np.zeros((4, 1)), np.ones((4, 1)))
    assert values.shape == (1001,)
    assert np.allclose(values, .25)
    detail = convergence_row(prediction, target)
    assert detail["complex_response_mse"] == pytest.approx(.25)
    assert detail["Re(Ty)_mse"] == pytest.approx(1.0)


def test_ranking_statistics_and_cache_keys_are_deterministic():
    stats = ranking_statistics([.1, .2, .3], [.11, .21, .31])
    assert stats["spearman"] == pytest.approx(1.0)
    assert stats["top_1_overlap"] == pytest.approx(1.0)
    geometry = np.zeros((16, 16), dtype=np.float32)
    config = RCWAConfig(fourier_order=3, substrate_thickness_mm=.2, device="cpu")
    assert cache_key(geometry, frequency_vector(), config) == cache_key(geometry.copy(), frequency_vector(), config)
    assert PHYSICAL_MAPPING == "s_to_ty_p_to_rx"


def test_cpu_frequency_parallelism_preserves_response_order():
    geometry = np.zeros((16, 16), dtype=np.float32)
    frequencies = np.array([2.0, 7.0, 12.0])
    serial = solve_geometry(geometry, frequencies, config=RCWAConfig(fourier_order=1, device="cpu", cpu_workers=1))
    parallel = solve_geometry(geometry, frequencies, config=RCWAConfig(fourier_order=1, device="cpu", cpu_workers=2))
    assert np.allclose(serial.ty, parallel.ty)
    assert np.allclose(serial.rx, parallel.rx)
    assert parallel.metadata["parallel_frequency_chunks"] == 2
