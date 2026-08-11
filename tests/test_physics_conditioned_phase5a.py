from __future__ import annotations

import json

import numpy as np
import torch

from src.mask_aware_spatial_jepa_losses import mask_aware_spatial_jepa_loss, mask_weight_map
from src.physics_conditioned_dataset import PhysicsCompletionDataset
from src.physics_conditioned_spatial_jepa import EMResponseDecoder, EMResponseEncoder, PhysicsConditionedSpatialJEPA


def _write_subset(root):
    root.mkdir()
    geometries = np.zeros((6, 1, 16, 16), dtype=np.uint8)
    geometries[:, 0, 2:6, 2:6] = 1
    responses = np.arange(6 * 4 * 1001, dtype=np.float32).reshape(6, 4, 1001)
    np.save(root / "geometries.npy", geometries)
    np.save(root / "responses.npy", responses)
    np.save(root / "frequency_ghz.npy", np.linspace(2.0, 12.0, 1001, dtype=np.float32))
    ids = [f"RDN/Data_001/{i:06d}" for i in range(6)]
    (root / "source_ids.txt").write_text("\n".join(ids) + "\n")
    splits = root / "splits"
    splits.mkdir()
    for name, indexes in {"train": range(4), "val": range(4, 5), "test": range(5, 6)}.items():
        (splits / f"{name}.txt").write_text("\n".join(ids[i] for i in indexes) + "\n")
    mean = responses[:4].mean(axis=(0, 2), keepdims=False).astype(np.float32)[:, None]
    std = responses[:4].std(axis=(0, 2), keepdims=False).astype(np.float32)[:, None]
    np.savez_compressed(root / "train_response_stats.npz", mean=mean, std=np.maximum(std, 1e-8))
    (root / "metadata.json").write_text(json.dumps({"subset_size": 6}))
    return responses, mean, std


def test_physics_dataset_preserves_geometry_response_pairing(tmp_path):
    root = tmp_path / "subset"
    responses, mean, std = _write_subset(root)
    dataset = PhysicsCompletionDataset(root, "train", "random_holes", 0.25, 42)
    sample = dataset[2]
    assert sample["response"].shape == (4, 1001)
    assert sample["response_raw"].shape == (4, 1001)
    assert torch.equal(sample["response_raw"], torch.from_numpy(responses[2]))
    assert torch.allclose(sample["response"], torch.from_numpy((responses[2] - mean) / std))
    assert sample["sample_id"] == "RDN/Data_001/000002"


def test_em_embedding_and_decoder_shapes_and_gradients():
    encoder = EMResponseEncoder()
    decoder = EMResponseDecoder()
    response = torch.randn(3, 4, 1001, requires_grad=True)
    reconstruction = decoder(encoder(response))
    assert reconstruction.shape == response.shape
    reconstruction.square().mean().backward()
    assert response.grad is not None and torch.isfinite(response.grad).all()


def test_film_is_identity_at_initialization_and_conditioned_model_has_expected_grads():
    torch.manual_seed(7)
    model = PhysicsConditionedSpatialJEPA()
    inputs = torch.rand(2, 2, 16, 16)
    response = torch.randn(2, 4, 1001)
    target = torch.rand(2, 1, 16, 16)
    mask = torch.zeros(2, 1, 16, 16)
    mask[:, :, 4:12, 4:12] = 1
    outputs = model(inputs, response, target)
    assert outputs["z_context"].shape == (2, 64, 8, 8)
    assert outputs["z_phys"].shape == (2, 128)
    assert torch.allclose(outputs["conditioned_context"], outputs["z_context"])
    assert torch.allclose(outputs["film_gamma"], torch.ones_like(outputs["film_gamma"]))
    assert torch.allclose(outputs["film_beta"], torch.zeros_like(outputs["film_beta"]))
    loss = mask_aware_spatial_jepa_loss(outputs["z_pred"], outputs["z_target"], mask_weight_map(mask))
    loss = loss + 0.1 * torch.nn.functional.binary_cross_entropy_with_logits(outputs["logits"], target)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.context_encoder.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.predictor.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.decoder.parameters())
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.film.parameters())
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
