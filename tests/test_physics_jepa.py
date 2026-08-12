"""Tests for the cross-modal Physics-JEPA experiment (Phase: physics_jepa)."""

import numpy as np
import torch

from src.jepa_completion_losses import jepa_loss
from src.physics_jepa import PhysicsJEPA
from src.physics_jepa_encoders import GeometryLatentEncoder, PhysicsPredictor, PhysicsSpectrumEncoder
from src.physics_jepa_losses import covariance_regularization, physics_jepa_loss
from src.physics_jepa_training import set_seed
from src.spectral_masks import apply_mask, random_contiguous_masks, validate_spectral_mask


def make_pairs(count: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    geometry = torch.from_numpy(rng.integers(0, 2, size=(count, 1, 16, 16)).astype(np.float32))
    response = torch.from_numpy(rng.standard_normal((count, 4, 1001)).astype(np.float32))
    return geometry, response


def test_geometry_encoder_shape():
    model = GeometryLatentEncoder(latent_dim=64)
    geometry = torch.rand(3, 1, 16, 16)
    assert model(geometry).shape == (3, 64)


def test_spectrum_encoder_shape():
    model = PhysicsSpectrumEncoder(latent_dim=32)
    response = torch.rand(3, 4, 1001)
    assert model(response).shape == (3, 32)


def test_predictor_shape():
    model = PhysicsPredictor(latent_dim=32)
    latent = torch.rand(5, 32)
    assert model(latent).shape == (5, 32)


def test_model_latent_dimension_and_output_shapes():
    model = PhysicsJEPA(latent_dim=32)
    geometry, response = make_pairs(4)
    outputs = model(geometry, response)
    assert outputs["z_geometry"].shape == (4, 32)
    assert outputs["z_online"].shape == (4, 32)
    assert outputs["z_self"].shape == (4, 32)
    assert outputs["z_pred"].shape == (4, 32)
    assert outputs["z_target"].shape == (4, 32)


def test_target_encoder_initialized_from_online():
    model = PhysicsJEPA(latent_dim=64)
    online = model.spectrum_encoder.state_dict()
    target = model.spectrum_target_encoder.state_dict()
    for name, value in target.items():
        assert torch.equal(value, online[name])


def test_ema_target_update_changes_target_parameters():
    model = PhysicsJEPA(latent_dim=32)
    before = next(model.spectrum_target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.spectrum_encoder.parameters()).add_(1.0)
    model.update_target_encoder()
    after = next(model.spectrum_target_encoder.parameters()).detach().clone()
    assert not torch.equal(before, after)
    model.update_target_encoder()
    after_second = next(model.spectrum_target_encoder.parameters()).detach().clone()
    assert not torch.equal(after, after_second)


def test_target_encoder_parameters_are_not_trainable():
    model = PhysicsJEPA(latent_dim=32)
    trainable_ids = {id(parameter) for parameter in model.trainable_parameters()}
    for parameter in model.spectrum_target_encoder.parameters():
        assert id(parameter) not in trainable_ids
        assert parameter.requires_grad is False


def test_stop_gradient_target_branch_and_gradient_flow():
    model = PhysicsJEPA(latent_dim=32)
    geometry, response = make_pairs(4)
    outputs = model(geometry, response)
    loss = physics_jepa_loss(
        outputs["z_pred"], outputs["z_target"], outputs["z_self"], 0.5, 0.1,
        z_online=outputs["z_online"], z_geometry=outputs["z_geometry"],
    )
    loss.backward()
    assert all(parameter.grad is None for parameter in model.spectrum_target_encoder.parameters())
    predictor_grads = [parameter.grad for parameter in model.predictor.parameters() if parameter.grad is not None]
    assert predictor_grads and all(torch.isfinite(g).all() for g in predictor_grads)
    geometry_grads = [parameter.grad for parameter in model.geometry_encoder.parameters() if parameter.grad is not None]
    assert geometry_grads and all(torch.isfinite(g).all() for g in geometry_grads)


def test_jepa_loss_stop_gradient_target():
    predicted = torch.randn(4, 32, requires_grad=True)
    target = torch.randn(4, 32, requires_grad=True)
    loss = jepa_loss(predicted, target)
    loss.backward()
    assert predicted.grad is not None
    assert target.grad is None


def test_outputs_are_finite():
    model = PhysicsJEPA(latent_dim=64)
    geometry, response = make_pairs(4)
    outputs = model(geometry, response)
    for value in outputs.values():
        assert torch.isfinite(value).all()


def test_determinism():
    geometry, response = make_pairs(4, seed=11)
    set_seed(1)
    first = PhysicsJEPA(latent_dim=32)
    first_outputs = first(geometry, response)
    set_seed(1)
    second = PhysicsJEPA(latent_dim=32)
    second_outputs = second(geometry, response)
    for name in first_outputs:
        assert torch.equal(first_outputs[name], second_outputs[name])


def test_shuffled_pairs_do_not_align_equally_well():
    """Correct geometry/response pairs train a better cross-modal alignment than shuffled pairs."""
    import pytest

    from src.dataset import SUTDPRCMDataset

    subset_root = "data/processed/sutd_prcm_30k"
    if not (__import__("pathlib").Path(subset_root) / "metadata.json").is_file():
        pytest.skip("processed subset not available")

    def train(model: PhysicsJEPA, geometry: torch.Tensor, response: torch.Tensor, steps: int) -> None:
        optimizer = torch.optim.Adam(model.trainable_parameters(), lr=1e-3)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(geometry, response)
            loss = physics_jepa_loss(
                outputs["z_pred"], outputs["z_target"], outputs["z_self"], 0.5, 0.1,
                z_online=outputs["z_online"], z_geometry=outputs["z_geometry"],
            )
            loss.backward()
            optimizer.step()
            model.update_target_encoder()

    def cross_alignment(model: PhysicsJEPA, geometry: torch.Tensor, response: torch.Tensor) -> float:
        model.eval()
        model.spectrum_target_encoder.eval()
        with torch.no_grad():
            outputs = model(geometry, response)
            return float(jepa_loss(outputs["z_pred"], outputs["z_target"]).item())

    set_seed(7)
    base = SUTDPRCMDataset(subset_root, "train", normalize_response=True)
    rng = np.random.default_rng(5)
    indices = rng.choice(len(base), 160, replace=False)
    pairs = [base[int(index)] for index in indices]
    geometry = torch.stack([pair[0] for pair in pairs])
    response = torch.stack([pair[1] for pair in pairs])
    permutation = rng.permutation(160)
    correct = PhysicsJEPA(latent_dim=32, ema_decay=0.99)
    shuffled = PhysicsJEPA(latent_dim=32, ema_decay=0.99)
    train(correct, geometry, response, steps=25)
    train(shuffled, geometry, response[permutation], steps=25)
    correct_alignment = cross_alignment(correct, geometry, response)
    shuffled_alignment = cross_alignment(shuffled, geometry, response)
    assert correct_alignment < shuffled_alignment


def test_spectral_mask_validity():
    masks = random_contiguous_masks(1001, 8, keep_fraction=0.5, num_intervals=2, seed=5)
    assert masks.shape == (8, 1001)
    for mask in masks:
        info = validate_spectral_mask(mask)
        assert info["valid"]
    response = np.ones((2, 4, 1001), dtype=np.float32)
    masked = apply_mask(response, masks[0])
    keep = masks[0] > 0
    assert (masked[:, :, ~keep] == 0.0).all()
    assert (masked[:, :, keep] == 1.0).all()


def test_covariance_regularization_small_for_decorrelated_latents():
    rng = np.random.default_rng(3)
    latent = torch.from_numpy(rng.standard_normal((256, 32)).astype(np.float32))
    value = covariance_regularization(latent)
    assert float(value) < 0.3


def test_covariance_regularization_positive_for_correlated_latents():
    rng = np.random.default_rng(3)
    factor = torch.from_numpy(rng.standard_normal((8, 32)).astype(np.float32))
    latent = torch.from_numpy(rng.standard_normal((256, 8)).astype(np.float32)) @ factor
    correlated = covariance_regularization(latent)
    rng = np.random.default_rng(4)
    uncorrelated = covariance_regularization(torch.from_numpy(rng.standard_normal((256, 32)).astype(np.float32)))
    assert float(correlated) > 10.0
    assert float(correlated) > 50 * float(uncorrelated)


def test_lambda_covariance_penalizes_correlated_latents():
    rng = np.random.default_rng(5)
    factor = torch.from_numpy(rng.standard_normal((32, 8)).astype(np.float32))
    z_online = torch.from_numpy(rng.standard_normal((256, 8)).astype(np.float32)) @ torch.from_numpy(rng.standard_normal((8, 32)).astype(np.float32))
    z_geometry = torch.from_numpy(rng.standard_normal((256, 32)).astype(np.float32))
    target = torch.from_numpy(rng.standard_normal((256, 32)).astype(np.float32))
    base = physics_jepa_loss(z_geometry, target, z_geometry, 0.5, 0.1, z_online=z_online, z_geometry=z_geometry, lambda_covariance=0.0)
    with_cov = physics_jepa_loss(z_geometry, target, z_geometry, 0.5, 0.1, z_online=z_online, z_geometry=z_geometry, lambda_covariance=1.0)
    assert float(with_cov) > float(base)


def test_target_centering_centers_target_and_updates_buffer():
    model = PhysicsJEPA(latent_dim=32, target_centering=True, target_center_decay=0.99)
    assert float(model.target_center.abs().sum()) == 0.0
    geometry, response = make_pairs(8, seed=2)
    outputs = model(geometry, response)
    assert "z_target_raw" in outputs
    assert torch.allclose(outputs["z_target"], outputs["z_target_raw"] - model.target_center)
    model.update_target_center(outputs["z_target_raw"])
    assert float(model.target_center.abs().sum()) > 0.0
    outputs_after = model(geometry, response)
    assert not torch.allclose(outputs_after["z_target"], outputs_after["z_target_raw"])


def test_target_centering_disabled_by_default():
    model = PhysicsJEPA(latent_dim=32)
    geometry, response = make_pairs(4)
    outputs = model(geometry, response)
    assert torch.equal(outputs["z_target"], outputs["z_target_raw"])
