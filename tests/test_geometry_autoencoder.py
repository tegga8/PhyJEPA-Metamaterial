import torch

from src.geometry_autoencoder import GeometryAutoencoder


def test_geometry_autoencoder_shapes_and_finite_outputs():
    model = GeometryAutoencoder()
    geometry = torch.rand(3, 1, 16, 16)
    outputs = model(geometry)
    assert outputs["latent"].shape == (3, 64, 8, 8)
    assert outputs["logits"].shape == (3, 1, 16, 16)
    assert torch.isfinite(outputs["latent"]).all()
    assert torch.isfinite(outputs["logits"]).all()


def test_geometry_autoencoder_encode_decode_path_is_deterministic():
    model = GeometryAutoencoder().eval()
    geometry = torch.rand(2, 1, 16, 16)
    with torch.inference_mode():
        first = model(geometry)
        latent = model.encode(geometry)
        second_logits = model.decode(latent)
    assert torch.equal(first["latent"], latent)
    assert torch.equal(first["logits"], second_logits)

